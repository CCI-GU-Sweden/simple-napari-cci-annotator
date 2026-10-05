from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ._segmentation_io import InstanceRecord, SegmentationError, refresh_instance_records
from ._training_crop import CropBounds, TrainingCropError


@dataclass(frozen=True)
class SegmentationCrop:
    mask: np.ndarray
    instances: dict[int, InstanceRecord]
    partial_instance_ids: tuple[int, ...]


def crop_instance_mask(
    mask: np.ndarray,
    instances: Mapping[int, InstanceRecord],
    classes: Mapping[int, str],
    bounds: CropBounds,
) -> SegmentationCrop:
    """Crop and deterministically reindex an instance map without resizing."""
    source = np.asarray(mask)
    if source.ndim != 2 or source.shape != (
        bounds.source_height,
        bounds.source_width,
    ):
        raise TrainingCropError("Instance mask and crop source dimensions differ.")
    if not np.issubdtype(source.dtype, np.integer) or np.any(source < 0):
        raise TrainingCropError("Instance masks must contain non-negative integers.")

    source_crop = source[
        bounds.y0 : bounds.y0 + bounds.valid_height,
        bounds.x0 : bounds.x0 + bounds.valid_width,
    ]
    output = np.zeros((bounds.size, bounds.size), dtype=np.uint32)
    present = sorted(int(value) for value in np.unique(source_crop) if value)
    missing = sorted(set(present) - set(instances))
    if missing:
        raise SegmentationError(f"Crop instance IDs missing metadata: {missing}")

    partial: list[int] = []
    records: dict[int, InstanceRecord] = {}
    for new_id, old_id in enumerate(present, start=1):
        selected = source_crop == old_id
        output[: bounds.valid_height, : bounds.valid_width][selected] = new_id
        source_area = int(np.count_nonzero(source == old_id))
        if int(np.count_nonzero(selected)) < source_area:
            partial.append(new_id)
        old = instances[old_id]
        lineage = tuple(dict.fromkeys((*old.lineage, old_id)))
        records[new_id] = InstanceRecord(
            instance_id=new_id,
            class_id=old.class_id,
            class_name=classes[old.class_id],
            confidence=old.confidence,
            source=old.source,
            status="cropped_partial" if new_id in partial else old.status,
            bbox=(0, 0, 0, 0),
            area=0,
            lineage=lineage,
        )
    records = refresh_instance_records(output, records, classes)
    return SegmentationCrop(output, records, tuple(partial))


def mask_ids_in_padding(mask: np.ndarray, bounds: CropBounds) -> tuple[int, ...]:
    """Return IDs painted into synthetic bottom/right crop padding."""
    array = np.asarray(mask)
    if array.shape != (bounds.size, bounds.size):
        return tuple(int(value) for value in np.unique(array) if value)
    invalid = np.zeros(array.shape, dtype=bool)
    invalid[bounds.valid_height :, :] = True
    invalid[:, bounds.valid_width :] = True
    return tuple(sorted(int(value) for value in np.unique(array[invalid]) if value))
