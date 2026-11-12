"""Disk-backed single-slice conversion, tile inference and mask assembly."""

import tempfile
from pathlib import Path

import numpy as np

from ._image_adapter import ImageAdapter
from ._instance_mask import ComposedInstances
from ._volume_chunked_components import split_components_to_chunks
from ._volume_chunked_conversion import convert_slice_to_chunks
from ._volume_chunked_fusion import fuse_tile_chunks
from ._volume_chunked_tiles import predict_tiles_to_chunks
from ._volume_zarr import _require_zarr


def predict_chunked_slice(volume, z_index, predictor, classes, settings, store,
                          *, cancelled=None, progress=None):
    """Predict one plane using temporary Zarr arrays and commit it to store."""
    _require_zarr()
    import zarr

    height, width = volume.geometry.shape[1:]
    inference = settings.inference
    core = inference.tile_size - 2 * inference.overlap
    tile_mode = max(height, width) > inference.tile_size
    chunk = 512
    chunks = (min(chunk, height), min(chunk, width))
    with tempfile.TemporaryDirectory(prefix=".cci-chunks-", dir=Path(store.root)) as work:
        group = zarr.open_group(work, mode="w", zarr_format=3)
        rgb = group.create_array("rgb", shape=(height, width, 3),
                                 chunks=(*chunks, 3), dtype="uint8")
        stats = convert_slice_to_chunks(
            volume, z_index, rgb, chunk_xy=chunk, cancelled=cancelled,
            progress=(lambda n, total: progress(n, total, "Converting chunks")) if progress else None,
        )
        fused = group.create_array("fused", shape=(height, width), chunks=chunks, dtype="uint32")
        if tile_mode:
            padded = (height + (-height) % core, width + (-width) % core)
            temporary = group.create_array("tile_masks", shape=padded,
                                           chunks=(core, core), dtype="uint32")
            registry, tile_info = predict_tiles_to_chunks(
                rgb, temporary, predictor, inference, classes,
                component_policy=settings.component_policy, cancelled=cancelled,
                progress=progress,
            )
            if progress is not None:
                progress(0, 1, "Fusing tile seams")
            records, mapping, pairs, conflicts, ambiguous = fuse_tile_chunks(
                temporary, fused, core, registry.records, classes,
                chunk_xy=chunk, cancelled=cancelled,
            )
            cleanup = tuple(item for tile_id in sorted(registry.cleanup) for item in registry.cleanup[tile_id])
            provenance = {
                "mode": "dask_tiled", "tile_size": inference.tile_size,
                "overlap": inference.overlap, **tile_info,
                "temporary_to_final": {str(key): value for key, value in sorted(mapping.items())},
                "equivalence_pairs": [list(pair) for pair in pairs],
                "class_conflicts": conflicts, "ambiguous_seam_ids": sorted(ambiguous),
                "cleared_border_ids": [],
                "component_removals": [
                    {"tile_id": tile_id,
                     "removed_components": sum(item.removed_components for item in registry.cleanup[tile_id]),
                     "removed_pixels": sum(item.removed_pixels for item in registry.cleanup[tile_id])}
                    for tile_id in sorted(registry.cleanup)
                ],
            }
            if settings.component_policy != "largest":
                provenance["component_policy"] = settings.component_policy
        else:
            kwargs = ({"component_policy": settings.component_policy}
                      if settings.component_policy != "largest" else {})
            result = predictor.predict_image(np.asarray(rgb[:]), inference, classes, **kwargs)
            fused[:] = result.mask
            records, cleanup, provenance = result.instances, result.cleanup, dict(result.provenance)
        components = group.create_array("components", shape=(height, width), chunks=chunks, dtype="uint64")
        output = group.create_array("mask", shape=(height, width), chunks=chunks, dtype="uint32")
        if progress is not None:
            progress(0, 1, "Splitting connected components")
        nodes, component_mapping = split_components_to_chunks(
            fused, components, output, records, chunk_xy=chunk, cancelled=cancelled,
        )
        provenance["slice_components"] = component_mapping
        plane = volume.selection.plane(z_index)
        store.write_slice(z_index, ComposedInstances(output, nodes, cleanup, provenance), {
            "sample_id": ImageAdapter().sample_id(volume.source_name, plane, volume.processing.channel_axis),
            "plane_indices": list(plane.axis_indices),
            "normalization_stats": list(stats), "inverted": volume.invert,
        })
