"""Exact parity for the experimental chunked conversion building block."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from simple_napari_cci_annotator._image_adapter import ImageConversionError, ImageProcessingSettings
from simple_napari_cci_annotator._tiled_inference import InferenceCancelled
from simple_napari_cci_annotator._volume_adapter import VolumeAdapter
from simple_napari_cci_annotator._volume_chunked_conversion import convert_slice_to_chunks


@pytest.mark.parametrize("method,lower,upper", [
    ("min_max", None, None), ("simple_max", None, None),
    ("fixed_range", 20, 200), ("dtype_range", None, None),
])
@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("multi", [False, True])
def test_chunked_conversion_matches_full_slice(tmp_path, method, lower, upper, invert, multi):
    zarr = pytest.importorskip("zarr", minversion="3")
    rng = np.random.default_rng(42)
    data = rng.integers(0, 255, size=(2, 3, 15, 17), dtype=np.uint16)
    if multi:
        image = SimpleNamespace(data=data, name="input", metadata={"axes": "ZCYX"})
        settings = ImageProcessingSettings(1, 2, None, 0, method, lower=lower, upper=upper)
    else:
        image = SimpleNamespace(data=data[:, 0], name="input", metadata={"axes": "ZYX"})
        settings = ImageProcessingSettings(None, None, None, None, method, lower=lower, upper=upper)
    volume = VolumeAdapter().prepare(image, settings, invert=invert)
    expected = volume.convert_slice(1).image
    output = zarr.create_array(store=str(tmp_path / "rgb.zarr"), shape=expected.data.shape,
                               chunks=(5, 6, 3), dtype="uint8", zarr_format=3)
    stats = convert_slice_to_chunks(volume, 1, output, chunk_xy=5)
    np.testing.assert_array_equal(output[:], expected.data)
    assert stats == expected.normalization_stats


def test_empty_nan_and_unsupported_recipes(tmp_path):
    data = np.full((1, 4, 5), np.nan, dtype=np.float32)
    image = SimpleNamespace(data=data, name="input", metadata={"axes": "ZYX"})
    settings = ImageProcessingSettings(None, None, None, None, "min_max")
    volume = VolumeAdapter().prepare(image, settings, invert=True)
    target = np.empty((4, 5, 3), dtype=np.uint8)
    stats = convert_slice_to_chunks(volume, 0, target, chunk_xy=2)
    np.testing.assert_array_equal(target, volume.convert_slice(0).image.data)
    assert stats == volume.convert_slice(0).image.normalization_stats
    unsupported = replace(volume, processing=replace(settings, normalization="percentile", lower=1, upper=99))
    with pytest.raises(ImageConversionError, match="percentile requires"):
        convert_slice_to_chunks(unsupported, 0, target)
    with pytest.raises(InferenceCancelled):
        convert_slice_to_chunks(volume, 0, target, cancelled=lambda: True)


@pytest.mark.parametrize("method", ["median", "mean", "gaussian", "white_tophat"])
def test_local_filter_halos_match_full_plane(method):
    rng = np.random.default_rng(11)
    data = rng.integers(0, 65535, (1, 19, 23), dtype=np.uint16)
    image = SimpleNamespace(data=data, name="source", metadata={"axes": "ZYX"})
    settings = ImageProcessingSettings(None, None, None, None, "min_max",
                                       filter_method=method, filter_radius=2)
    volume = VolumeAdapter().prepare(image, settings)
    expected = volume.convert_slice(0).image
    output = np.empty(expected.data.shape, dtype=np.uint8)
    stats = convert_slice_to_chunks(volume, 0, output, chunk_xy=5)
    np.testing.assert_array_equal(output, expected.data)
    assert stats == expected.normalization_stats


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
@pytest.mark.parametrize("lower,upper", [(1, 99), (3.2, 97.1), (0, 100)])
def test_integer_percentile_matches_full_plane(dtype, lower, upper):
    rng = np.random.default_rng(123)
    limits = np.iinfo(dtype)
    values = rng.integers(max(limits.min, -1000), min(limits.max, 3000),
                          size=(1, 29, 31), dtype=dtype)
    image = SimpleNamespace(data=values, name="source", metadata={"axes": "ZYX"})
    settings = ImageProcessingSettings(None, None, None, None, "percentile", lower=lower, upper=upper)
    volume = VolumeAdapter().prepare(image, settings)
    expected = volume.convert_slice(0).image
    output = np.empty(expected.data.shape, dtype=np.uint8)
    stats = convert_slice_to_chunks(volume, 0, output, chunk_xy=7)
    np.testing.assert_array_equal(output, expected.data)
    assert stats == expected.normalization_stats
