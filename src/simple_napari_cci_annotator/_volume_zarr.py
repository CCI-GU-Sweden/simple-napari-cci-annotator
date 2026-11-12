"""Chunked, lossless Zarr v3 copies of selected source volumes."""

from pathlib import Path

import dask.array as da
import numpy as np
try:
    import zarr
except ImportError:
    zarr = None

from ._tiled_inference import InferenceCancelled


class VolumeZarrError(ValueError):
    """A Zarr conversion or its metadata is invalid."""


def _require_zarr():
    if zarr is None or int(zarr.__version__.split(".")[0]) < 3:
        raise VolumeZarrError("Zarr v3 requires Python 3.11 or later and the zarr package.")


def write_volume_zarr(volume, destination, *, chunk_xy=512, cancelled=None, progress=None):
    """Copy selected raw ZYX(C) pixels without materializing a full plane.

    The destination is created exclusively. An incomplete directory is kept on
    error or cancellation and is rejected by ``open_volume_zarr``.
    """
    _require_zarr()
    if type(chunk_xy) is not int or chunk_xy < 1:
        raise VolumeZarrError("XY chunk size must be a positive integer.")
    destination = Path(destination).expanduser().resolve()
    if destination.suffix.lower() != ".zarr":
        raise VolumeZarrError("Choose a destination ending in .zarr.")
    destination.mkdir(parents=False, exist_ok=False)
    group = zarr.open_group(str(destination), mode="a", zarr_format=3)
    group.attrs["cci_status"] = "incomplete"
    axes = volume.selection.axes
    depth, height, width = volume.geometry.shape
    channels = None if axes.channel is None else volume.source_shape[axes.channel]
    shape = (depth, height, width) + (() if channels is None else (channels,))
    chunks = (1, min(chunk_xy, height), min(chunk_xy, width)) + (() if channels is None else (1,))
    target = group.create_array("0", shape=shape, chunks=chunks, dtype=np.dtype(volume.source_dtype))
    total = depth * ((height + chunk_xy - 1) // chunk_xy) * ((width + chunk_xy - 1) // chunk_xy) * (channels or 1)
    completed = 0
    for output_z, source_z in enumerate(volume.selection.z_indices):
        for y in range(0, height, chunk_xy):
            for x in range(0, width, chunk_xy):
                for channel in range(channels or 1):
                    if cancelled is not None and cancelled():
                        raise InferenceCancelled("Zarr conversion cancelled.")
                    index = list(volume.selection.fixed_indices)
                    index[axes.z] = source_z
                    index[axes.y] = slice(y, min(y + chunk_xy, height))
                    index[axes.x] = slice(x, min(x + chunk_xy, width))
                    if axes.channel is not None:
                        index[axes.channel] = channel
                    remaining = [i for i, item in enumerate(index) if isinstance(item, slice)]
                    block = np.asarray(volume.data[tuple(index)])
                    block = block.transpose(remaining.index(axes.y), remaining.index(axes.x))
                    output = (output_z, slice(y, min(y + chunk_xy, height)),
                              slice(x, min(x + chunk_xy, width)))
                    if channels is not None:
                        output += (channel,)
                    target[output] = block
                    completed += 1
                    if progress is not None:
                        progress(completed, total)
    if cancelled is not None and cancelled():
        raise InferenceCancelled("Zarr conversion cancelled.")
    group.attrs.update({
        "cci_volume": {
            "version": 1, "axes": "ZYX" if channels is None else "ZYXC",
            "scale": list(volume.geometry.scale),
            "translate": list(volume.geometry.translate),
            "units": list(volume.geometry.units),
            "source_name": volume.source_name,
            "selection": volume.selection.to_mapping(),
        },
        "cci_status": "complete",
    })
    return destination


def open_volume_zarr(path):
    """Open one completed plugin-created Zarr volume as a lazy Dask array."""
    _require_zarr()
    try:
        group = zarr.open_group(str(Path(path).expanduser().resolve()), mode="r")
        if group.attrs.get("cci_status") != "complete":
            raise VolumeZarrError("Zarr conversion is incomplete or not a plugin-created volume.")
        info = dict(group.attrs["cci_volume"])
        array = group["0"]
        axes = info["axes"]
        if (info["version"] != 1 or axes not in {"ZYX", "ZYXC"} or
                array.ndim != len(axes) or array.metadata.zarr_format != 3 or
                any(len(info[name]) != 3 for name in ("scale", "translate", "units"))):
            raise VolumeZarrError("Zarr volume metadata is invalid.")
        return da.from_zarr(array), info
    except VolumeZarrError:
        raise
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise VolumeZarrError(f"Cannot open Zarr volume: {exc}") from exc
