"""Lossless chunked Zarr v3 export and lazy reopening."""

from dataclasses import replace
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import tifffile
zarr = pytest.importorskip("zarr", minversion="3.0")

from simple_napari_cci_annotator._image_adapter import ImageProcessingSettings
from simple_napari_cci_annotator._tiled_inference import InferenceCancelled
from simple_napari_cci_annotator._volume_adapter import VolumeAdapter, VolumeAxes
from simple_napari_cci_annotator._volume_zarr import (
    VolumeZarrError, open_volume_zarr, write_volume_zarr,
)


def test_grayscale_export_preserves_selected_pixels_geometry_and_chunks(tmp_path):
    original = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
    data = da.from_array(original.transpose(2, 0, 1), chunks=(3, 1, 3))
    layer = SimpleNamespace(data=data, name="em", metadata={"axes": "XZY"},
                            scale=(0.4, 2, 0.3), translate=(30, 10, 20),
                            units=("um", "um", "um"))
    settings = ImageProcessingSettings(None, None, None, None, "min_max")
    volume = VolumeAdapter().prepare(layer, settings, z_range=(1, 4))
    path = tmp_path / "em.zarr"
    progress = []
    write_volume_zarr(volume, path, chunk_xy=3, progress=lambda n, total: progress.append((n, total)))
    lazy, info = open_volume_zarr(path)
    assert isinstance(lazy, da.Array)
    assert zarr.open_group(str(path), mode="r")["0"].metadata.zarr_format == 3
    assert lazy.chunksize == (1, 3, 3)
    np.testing.assert_array_equal(lazy.compute(), original[1:4])
    assert info["axes"] == "ZYX"
    assert info["scale"] == [2, 0.3, 0.4]
    assert info["translate"] == [12, 20, 30]
    assert progress[-1] == (12, 12)
    with pytest.raises(FileExistsError):
        write_volume_zarr(volume, path)


def test_multichannel_and_fixed_acquisition_copy_all_raw_channels(tmp_path):
    raw = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 3, 4, 5, 6)
    layer = SimpleNamespace(data=raw, name="multi", metadata={"axes": "TCZXY"})
    settings = ImageProcessingSettings(1, 0, 1, None, "min_max")
    volume = VolumeAdapter().prepare(layer, settings, axes=VolumeAxes(2, 4, 3, 1),
                                     fixed_indices={0: 1}, z_range=(1, 3))
    path = tmp_path / "multi.zarr"
    write_volume_zarr(volume, path, chunk_xy=3)
    array, info = open_volume_zarr(path)
    np.testing.assert_array_equal(array.compute(), raw[1, :, 1:3].transpose(1, 3, 2, 0))
    assert info["axes"] == "ZYXC"
    assert array.chunksize == (1, 3, 3, 1)


def test_cancellation_and_failed_copy_are_not_opened_as_complete(tmp_path):
    data = np.zeros((2, 4, 4), dtype=np.uint8)
    layer = SimpleNamespace(data=data, name="em", metadata={"axes": "ZYX"})
    volume = VolumeAdapter().prepare(layer, ImageProcessingSettings(None, None, None, None, "min_max"))
    path = tmp_path / "cancelled.zarr"
    with pytest.raises(InferenceCancelled):
        write_volume_zarr(volume, path, cancelled=lambda: True)
    with pytest.raises(VolumeZarrError, match="incomplete"):
        open_volume_zarr(path)
    with pytest.raises(ValueError, match="chunk"):
        write_volume_zarr(replace(volume), tmp_path / "bad.zarr", chunk_xy=0)


def test_copy_requests_only_one_xy_chunk_at_a_time(tmp_path):
    class Source:
        def __init__(self):
            self.values = np.arange(2 * 7 * 9, dtype=np.uint16).reshape(2, 7, 9)
            self.shape = self.values.shape
            self.dtype = self.values.dtype
            self.requests = []

        def __getitem__(self, index):
            self.requests.append(index)
            return self.values[index]

    data = Source()
    layer = SimpleNamespace(data=data, name="em", metadata={"axes": "ZYX"})
    volume = VolumeAdapter().prepare(layer, ImageProcessingSettings(None, None, None, None, "min_max"))
    write_volume_zarr(volume, tmp_path / "chunked.zarr", chunk_xy=3)
    assert len(data.requests) == 18
    assert all(isinstance(z, int) and y.stop - y.start <= 3 and x.stop - x.start <= 3
               for z, y, x in data.requests)


def test_tiff_backed_volume_round_trip(tmp_path):
    pixels = np.arange(3 * 12 * 13, dtype=np.uint16).reshape(3, 12, 13)
    path = tmp_path / "source.tif"
    tifffile.imwrite(path, pixels, photometric="minisblack")
    source = tifffile.memmap(path)
    image = SimpleNamespace(data=source, name="source.tif", metadata={"axes": "ZYX"})
    volume = VolumeAdapter().prepare(image, ImageProcessingSettings(None, None, None, None, "min_max"))
    destination = tmp_path / "source.zarr"
    write_volume_zarr(volume, destination, chunk_xy=5)
    lazy, _ = open_volume_zarr(destination)
    np.testing.assert_array_equal(lazy.compute(), pixels)
