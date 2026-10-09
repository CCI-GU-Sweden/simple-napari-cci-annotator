"""Volume axes, frozen acquisition selection, and parity with 2D conversion."""

import json
from dataclasses import replace
from types import SimpleNamespace

import dask.array as da
from dask import delayed
import numpy as np
import pytest

from simple_napari_cci_annotator._image_adapter import (
    IMAGE_FILTERS,
    ImageAdapter,
    ImageConversionError,
    ImageProcessingSettings,
    PlaneSelection,
)
from simple_napari_cci_annotator._volume_adapter import (
    VolumeAdapter,
    VolumeAxes,
    VolumeInputError,
)


def layer(data, axes=None, **kwargs):
    return SimpleNamespace(
        data=data, name="volume", metadata={} if axes is None else {"axes": axes},
        **kwargs,
    )


def processing(multichannel=False, **kwargs):
    channels = (0, 2, 1, None) if multichannel else (None,) * 4
    return ImageProcessingSettings(*channels, "min_max", **kwargs)


@pytest.mark.parametrize("layout", ["ZCYX", "CZYX", "ZYXC", "XYZC"])
def test_channel_axis_moves_without_changing_project_recipe(layout):
    source = np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 4, 5, 6)
    data = source.transpose(tuple("ZCYX".index(axis) for axis in layout))
    settings = processing(True)
    volume = VolumeAdapter().prepare(layer(data, layout), settings)
    viewer = SimpleNamespace(dims=SimpleNamespace(current_step=(0, 0, 0)))
    expected = ImageAdapter().convert(layer(source[1], "CYX"), viewer, settings)
    converted = volume.convert_slice(1)

    np.testing.assert_array_equal(converted.image.data, expected.data)
    assert converted.image.normalization_stats == expected.normalization_stats
    assert converted.z_index == 1
    assert converted.image.sample_id == "volume__z001"
    assert volume.geometry.shape == (3, 5, 6)
    assert volume.processing.channel_axis == layout.index("C")
    assert settings.channel_axis == 0
    assert volume.project_processing is settings
    assert np.all(converted.image.data[..., 2] == 0)


@pytest.mark.parametrize("method,lower,upper", [
    ("min_max", None, None), ("simple_max", None, None),
    ("percentile", 1, 99), ("z_score", -2, 2),
    ("fixed_range", 0, 500), ("dtype_range", None, None),
])
@pytest.mark.parametrize("filter_method", list(IMAGE_FILTERS))
def test_volume_preprocessing_matches_2d(method, lower, upper, filter_method):
    rng = np.random.default_rng(17)
    data = rng.integers(0, 600, (2, 3, 9, 11), dtype=np.uint16)
    settings = replace(
        processing(True), normalization=method, lower=lower, upper=upper,
        filter_method=filter_method, filter_radius=2,
    )
    volume = VolumeAdapter().prepare(layer(data, "ZCYX"), settings, invert=True)
    viewer = SimpleNamespace(dims=SimpleNamespace(current_step=(0, 0, 0)))
    expected = ImageAdapter().convert(
        layer(data[1], "CYX"), viewer, settings, invert=True,
    )
    actual = volume.convert_slice(1).image
    np.testing.assert_array_equal(actual.data, expected.data)
    assert actual.normalization_stats == expected.normalization_stats
    assert actual.inverted
    assert actual.data.dtype == np.uint8
    assert actual.data.flags.c_contiguous


def test_grayscale_selection_and_geometry_preserve_source_coordinates():
    data = np.arange(2 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 4, 5, 6)
    indices = {0: 1}
    source = layer(
        data, "TZYX", scale=(7, 2.5, 0.3, 0.4), translate=(10, 20, 30, 40),
        units=("second", "micrometer", "micrometer", "micrometer"),
    )
    volume = VolumeAdapter().prepare(
        source, processing(), fixed_indices=indices, z_range=(1, 3),
    )
    indices[0] = 0
    source.data = np.zeros_like(data)
    source.scale = (1, 1, 1, 1)
    source.metadata["axes"] = "ABCD"
    result = list(volume.iter_slices())
    assert [item.z_index for item in result] == [1, 2]
    assert result[0].image.plane.non_spatial_indices == {0: 1, 1: 1}
    expected = ImageAdapter().convert_plane(
        data[1, 1], PlaneSelection((None, None), (0, 1), ("Y", "X")),
        processing(),
    ).data
    np.testing.assert_array_equal(result[0].image.data, expected)
    np.testing.assert_array_equal(result[0].image.data[..., 0], result[0].image.data[..., 1])
    assert volume.geometry.shape == (2, 5, 6)
    assert volume.geometry.scale == (2.5, 0.3, 0.4)
    assert volume.geometry.translate == (22.5, 30, 40)
    assert volume.geometry.units == ("micrometer",) * 3
    assert json.loads(json.dumps(volume.to_mapping()))["selection"]["z_range"] == [1, 3]
    with pytest.raises(VolumeInputError, match="outside"):
        volume.convert_slice(0)


@pytest.mark.parametrize("channels", [3, 4])
def test_rgb_transform_vectors_exclude_channel_dimension(channels):
    data = np.arange(2 * 4 * 5 * channels, dtype=np.uint8).reshape(2, 4, 5, channels)
    settings = ImageProcessingSettings(2, 0, 1, 2, "dtype_range")
    volume = VolumeAdapter().prepare(layer(
        data, "ZYXC", rgb=True, scale=(2, 0.5, 0.25), translate=(3, 4, 5),
        rotate=np.eye(3), shear=np.zeros(3),
        affine=SimpleNamespace(affine_matrix=np.eye(4)),
    ), settings)
    np.testing.assert_array_equal(volume.convert_slice(1).image.data, data[1, ..., :3])
    assert volume.geometry.scale == (2, 0.5, 0.25)
    assert volume.geometry.translate == (3, 4, 5)
    assert volume.processing.channel_axis == 3
    assert settings.channel_axis == 2


def test_explicit_mapping_overrides_ambiguous_source_without_guessing():
    data = np.zeros((4, 5, 6), dtype=np.uint8)
    for source in (layer(data), layer(data, "AYX"), layer(data, "CYX")):
        with pytest.raises(VolumeInputError, match="explicit"):
            VolumeAdapter().prepare(source, processing())
        volume = VolumeAdapter().prepare(source, processing(), axes=VolumeAxes(0, 1, 2))
        assert volume.selection.axis_labels == ("Z", "Y", "X")


def test_named_axes_and_reordered_geometry():
    source = layer(
        np.zeros((5, 6, 4)), ["row", "column", "depth"],
        scale=(0.3, 0.4, 2), translate=(10, 20, 30),
    )
    volume = VolumeAdapter().prepare(source, processing(), z_range=(2, 4))
    assert volume.selection.axes == VolumeAxes(2, 0, 1)
    assert volume.geometry.shape == (2, 5, 6)
    assert volume.geometry.scale == (2, 0.3, 0.4)
    assert volume.geometry.translate == (34, 10, 20)


def test_rgb_layer_axis_labels_omit_channel_dimension():
    source = layer(
        np.zeros((2, 4, 5, 3), dtype=np.uint8), rgb=True,
        axis_labels=("Z", "Y", "X"),
    )
    volume = VolumeAdapter().prepare(source, processing(True))
    assert volume.selection.axes == VolumeAxes(0, 1, 2, 3)
    assert volume.geometry.shape == (2, 4, 5)


@pytest.mark.parametrize("labels", ["ZZX", "ZY", 42])
def test_invalid_declared_axes_are_rejected(labels):
    with pytest.raises(VolumeInputError):
        VolumeAdapter().prepare(layer(np.zeros((2, 4, 5)), labels), processing())


@pytest.mark.parametrize("axes", [
    VolumeAxes(0, 0, 2), VolumeAxes(0, 1, 3), VolumeAxes(-1, 1, 2),
    VolumeAxes(True, 1, 2), VolumeAxes(0.5, 1, 2), VolumeAxes(0, 1, 2, 2),
])
def test_invalid_axis_mapping_is_rejected(axes):
    with pytest.raises(VolumeInputError):
        VolumeAdapter().prepare(layer(np.zeros((2, 3, 4))), processing(), axes=axes)


@pytest.mark.parametrize("fixed", [None, {0: 2}, {0: -1}, {0: True}, {1: 0}, {4: 0}])
def test_extra_axes_require_valid_explicit_indices(fixed):
    with pytest.raises(VolumeInputError):
        VolumeAdapter().prepare(layer(np.zeros((2, 3, 4, 5)), "TZYX"), processing(), fixed_indices=fixed)


@pytest.mark.parametrize("z_range", [(0, 0), (-1, 2), (0, 4), (2, 1), (True, 2), (0, 1.5)])
def test_invalid_z_range_is_rejected(z_range):
    with pytest.raises(VolumeInputError):
        VolumeAdapter().prepare(layer(np.zeros((3, 4, 5)), "ZYX"), processing(), z_range=z_range)


def test_channel_compatibility_and_availability_are_checked():
    with pytest.raises(VolumeInputError, match="channel layout"):
        VolumeAdapter().prepare(layer(np.zeros((3, 4, 5)), "ZYX"), processing(True))
    with pytest.raises(VolumeInputError, match="channel layout"):
        VolumeAdapter().prepare(layer(np.zeros((3, 2, 4, 5)), "ZCYX"), processing())
    with pytest.raises(ImageConversionError, match="Channel 2"):
        VolumeAdapter().prepare(layer(np.zeros((3, 2, 4, 5)), "ZCYX"), processing(True))


@pytest.mark.parametrize("kwargs", [
    {"rotate": 45}, {"shear": (0.1, 0, 0)}, {"affine": np.diag([2, 1, 1, 1])},
    {"scale": (1, 0, 1)}, {"scale": (1, np.nan, 1)},
    {"translate": (0, np.inf, 0)}, {"scale": (1, 1)}, {"units": ("um",)},
])
def test_unsupported_or_invalid_geometry_is_rejected(kwargs):
    with pytest.raises(VolumeInputError):
        VolumeAdapter().prepare(layer(np.zeros((3, 4, 5)), "ZYX", **kwargs), processing())


def test_dask_preparation_is_lazy_and_conversion_reads_only_selected_slice():
    reads = []

    @delayed
    def read_slice(z):
        reads.append(z)
        return np.arange(20, dtype=np.uint16).reshape(4, 5) + z * 100

    data = da.stack([da.from_delayed(read_slice(z), shape=(4, 5), dtype=np.uint16) for z in range(3)])
    volume = VolumeAdapter().prepare(layer(data, "ZYX"), processing())
    json.dumps(volume.to_mapping())
    assert reads == []
    converted = volume.convert_slice(1)
    assert set(reads) == {1}
    assert converted.image.data.shape == (4, 5, 3)


def test_memory_mapped_volume_remains_backed_by_source(tmp_path):
    path = tmp_path / "source.npy"
    source = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(2, 4, 5))
    source[:] = np.arange(40).reshape(2, 4, 5)
    source.flush()
    volume = VolumeAdapter().prepare(layer(source, "ZYX"), processing())
    assert volume.data is source
    assert isinstance(volume.data, np.memmap)
    assert volume.convert_slice(1).image.data.shape == (4, 5, 3)


def test_only_requested_channels_and_z_are_extracted():
    class SliceOnlyArray:
        shape = (3, 4, 5, 6)
        dtype = np.dtype("uint16")

        def __init__(self):
            self.reads = []

        def __array__(self, *args, **kwargs):
            raise AssertionError("The entire volume must never be materialized")

        def __getitem__(self, index):
            self.reads.append(index)
            z, channel, y, x = index
            assert (z, channel) in {(1, 1), (1, 2)}
            assert y == x == slice(None)
            return np.arange(30, dtype=np.uint16).reshape(5, 6)

    source = SliceOnlyArray()
    volume = VolumeAdapter().prepare(layer(source, "ZCYX"), processing(True))
    assert source.reads == []
    volume.convert_slice(1)
    assert len(source.reads) == 2


def test_source_shape_change_is_detected():
    data = np.zeros((2, 4, 5), dtype=np.uint8)
    volume = VolumeAdapter().prepare(layer(data, "ZYX"), processing())
    data.shape = (4, 2, 5)
    with pytest.raises(VolumeInputError, match="shape changed"):
        volume.convert_slice(1)


@pytest.mark.parametrize("dtype", ["complex64", "object"])
def test_nonreal_volume_pixels_are_rejected_without_conversion(dtype):
    with pytest.raises(VolumeInputError, match="real numeric"):
        VolumeAdapter().prepare(
            layer(np.zeros((2, 4, 5), dtype=dtype), "ZYX"), processing(),
        )


@pytest.mark.parametrize("plane", [
    PlaneSelection((None, None, None), (1, 2), ("Z", "Y", "X")),
    PlaneSelection((9, None, None), (1, 2), ("Z", "Y", "X")),
    PlaneSelection((0, None, None), (2, 2), ("Z", "Y", "X")),
])
def test_explicit_2d_conversion_rejects_unresolved_or_invalid_indices(plane):
    with pytest.raises(ImageConversionError):
        ImageAdapter().convert_plane(np.zeros((2, 4, 5)), plane, processing())
