"""Experimental tile inference into a chunked temporary label array.

The returned IDs still require seam fusion and connected-component assembly;
this module is not yet connected to the volume inference runner.
"""

import dask.array as da
import numpy as np

from ._segmentation_io import InstanceRecord
from ._segmentation_tiling import _TileRegistry
from ._tiled_inference import InferenceCancelled, InferenceError


def predict_tiles_to_chunks(image, output, predictor, settings, classes, *,
                            component_policy="preserve", cancelled=None, progress=None):
    """Run one halo tile at a time, writing unmerged tile IDs to ``output``."""
    settings.validate()
    if component_policy not in {"largest", "preserve"}:
        raise InferenceError("Component policy must be largest or preserve.")
    if len(image.shape) != 3 or image.shape[2] != 3 or np.dtype(image.dtype) != np.dtype("uint8"):
        raise InferenceError("Chunked tile inference requires a YX3 uint8 image.")
    core = settings.tile_size - 2 * settings.overlap
    if core <= 0:
        raise InferenceError("Segmentation tile overlap must be smaller than half the tile size.")
    height, width = image.shape[:2]
    bottom, right = (-height) % core, (-width) % core
    padded_shape = (height + bottom, width + right)
    if tuple(output.shape) != padded_shape or np.dtype(output.dtype) != np.dtype("uint32"):
        raise InferenceError("Tile output must be a padded YX uint32 array.")
    rows, columns = padded_shape[0] // core, padded_shape[1] // core
    count = rows * columns
    if count * settings.max_detections >= np.iinfo(np.uint32).max:
        raise InferenceError("The tile grid exceeds uint32 instance-ID capacity.")
    registry = _TileRegistry(count, progress)
    source = da.from_array(image, chunks=(core, core, 3), asarray=False, name=False)
    if bottom or right:
        source = da.pad(source, ((0, bottom), (0, right), (0, 0)),
                        mode="reflect" if height > 1 and width > 1 else "edge")
    source = source.rechunk((core, core, 3))

    def predict(block, block_id=None):
        if cancelled is not None and cancelled():
            raise InferenceCancelled("Chunked tile inference cancelled.")
        row, column = (block_id or (0, 0, 0))[:2]
        tile_id = int(row) * columns + int(column)
        kwargs = {"component_policy": component_policy} if component_policy != "largest" else {}
        result = predictor.predict_image(np.asarray(block), settings, dict(classes), **kwargs)
        mask = np.asarray(result.mask, dtype=np.uint32)
        if int(mask.max(initial=0)) > settings.max_detections:
            raise InferenceError(f"Tile {tile_id} returned more than max detections.")
        lut = np.zeros(settings.max_detections + 1, dtype=np.uint32)
        records = {}
        for local, record in sorted(result.instances.items()):
            if local < 1 or local > settings.max_detections:
                raise InferenceError(f"Tile {tile_id} returned an invalid instance ID.")
            identifier = tile_id * settings.max_detections + int(local)
            lut[local] = identifier
            records[identifier] = InstanceRecord(
                instance_id=identifier, class_id=record.class_id,
                class_name=record.class_name, confidence=record.confidence,
                source="prediction", status="predicted", bbox=(0, 0, 0, 0),
                area=0, lineage=(identifier,),
            )
        registry.store(tile_id, records, result.cleanup)
        return lut[mask][..., np.newaxis]

    mapped = da.map_overlap(
        predict, source, depth={0: settings.overlap, 1: settings.overlap, 2: 0},
        boundary={0: "reflect", 1: "reflect", 2: "none"}, trim=True,
        chunks=(core, core, 1), dtype=np.uint32,
        meta=np.empty((0, 0, 0), dtype=np.uint32), allow_rechunk=False,
    )
    # A bounded chunk at a time also avoids Dask sharing a mutable Zarr target
    # between tasks and permits cancellation between writes.
    labels = mapped[:, :, 0]
    for row in range(rows):
        for column in range(columns):
            if cancelled is not None and cancelled():
                raise InferenceCancelled("Chunked tile inference cancelled.")
            y, x = row * core, column * core
            output[y:y + core, x:x + core] = labels.blocks[row, column].compute(scheduler="single-threaded")
    if cancelled is not None and cancelled():
        raise InferenceCancelled("Chunked tile inference cancelled.")
    return registry, {"core_size": core, "grid": [rows, columns],
                      "padding": {"bottom": bottom, "right": right}}
