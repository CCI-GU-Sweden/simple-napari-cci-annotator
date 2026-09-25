from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from skimage.measure import label, regionprops

from ._segmentation_io import InstanceRecord, SegmentationError, refresh_instance_records


@dataclass(frozen=True)
class ComponentCleanup:
    component_count: int
    removed_components: int
    removed_pixels: int


@dataclass(frozen=True)
class PredictedInstance:
    mask: np.ndarray
    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float


@dataclass(frozen=True)
class ComposedInstances:
    mask: np.ndarray
    instances: dict[int, InstanceRecord]
    cleanup: tuple[ComponentCleanup, ...]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SmallInstanceRemoval:
    removed_instance_ids: tuple[int, ...]
    removed_pixels: int


@dataclass(frozen=True)
class InstanceCompaction:
    old_to_new: dict[int, int]
    removed_empty_ids: tuple[int, ...]


def keep_largest_component_by_bbox(
    binary_mask: np.ndarray,
) -> tuple[np.ndarray, ComponentCleanup]:
    """Keep the 4-connected component with the largest bounding-box area.

    This deliberately mirrors the pinned segmentation plugin. Pixel area is
    only a deterministic tie-breaker; it is not the primary selection rule.
    """
    binary = np.asarray(binary_mask, dtype=bool)
    if binary.ndim != 2:
        raise SegmentationError("A predicted instance mask must be two-dimensional.")
    components = label(binary, connectivity=1)
    regions = regionprops(components)
    if not regions:
        return np.zeros(binary.shape, dtype=bool), ComponentCleanup(0, 0, 0)
    selected = max(
        regions,
        key=lambda region: (
            (region.bbox[2] - region.bbox[0])
            * (region.bbox[3] - region.bbox[1]),
            region.area,
            -region.label,
        ),
    )
    kept = components == selected.label
    return kept, ComponentCleanup(
        component_count=len(regions),
        removed_components=len(regions) - 1,
        removed_pixels=int(binary.sum() - kept.sum()),
    )


def compose_predictions(
    predictions: Sequence[PredictedInstance],
    image_shape: tuple[int, int],
    classes: Mapping[int, str],
) -> ComposedInstances:
    """Compose cleaned predictions using per-pixel confidence ownership."""
    height, width = (int(image_shape[0]), int(image_shape[1]))
    output = np.zeros((height, width), dtype=np.uint32)
    ownership = np.full((height, width), -np.inf, dtype=np.float32)
    cleaned: list[tuple[PredictedInstance, np.ndarray]] = []
    cleanup: list[ComponentCleanup] = []
    for prediction in predictions:
        if prediction.class_id not in classes:
            raise SegmentationError(
                f"Prediction uses unknown class ID {prediction.class_id}."
            )
        mask = np.asarray(prediction.mask, dtype=bool)
        if mask.shape != (height, width):
            raise SegmentationError(
                f"Prediction mask shape {mask.shape} does not match {(height, width)}."
            )
        kept, report = keep_largest_component_by_bbox(mask)
        cleanup.append(report)
        cleaned.append((prediction, kept))

    # Lower-confidence instances are painted first. Stable input order wins
    # exact ties because ownership is updated only by a strictly greater score.
    order = sorted(range(len(cleaned)), key=lambda index: cleaned[index][0].confidence)
    temporary_ids: dict[int, int] = {}
    for original_index in order:
        prediction, binary = cleaned[original_index]
        writable = binary & (prediction.confidence > ownership)
        if not np.any(writable):
            continue
        temporary_id = original_index + 1
        output[writable] = temporary_id
        ownership[writable] = prediction.confidence
        temporary_ids[original_index] = temporary_id

    # Remove fully occluded IDs and compact in original model order so IDs are
    # deterministic and independent of confidence sort order.
    compact = np.zeros_like(output)
    raw_records: dict[int, dict] = {}
    next_id = 1
    for original_index, (prediction, _) in enumerate(cleaned):
        temporary_id = temporary_ids.get(original_index)
        if temporary_id is None or not np.any(output == temporary_id):
            continue
        compact[output == temporary_id] = next_id
        raw_records[next_id] = {
            "class_id": prediction.class_id,
            "confidence": prediction.confidence,
            "source": "prediction",
            "status": "predicted",
            "lineage": [original_index + 1],
        }
        next_id += 1
    records = refresh_instance_records(compact, raw_records, classes)
    return ComposedInstances(compact, records, tuple(cleanup))


def disconnected_instance_ids(mask: np.ndarray) -> tuple[int, ...]:
    array = np.asarray(mask)
    return tuple(
        int(instance_id)
        for instance_id in np.unique(array)
        if instance_id
        and label(array == instance_id, connectivity=1).max(initial=0) > 1
    )


def remove_small_instances(
    mask: np.ndarray,
    instances: Mapping[int, InstanceRecord],
    *,
    minimum_area: int = 5,
) -> tuple[np.ndarray, dict[int, InstanceRecord], SmallInstanceRemoval]:
    """Explicitly remove instances smaller than ``minimum_area`` pixels."""
    if minimum_area < 1:
        raise SegmentationError("Minimum instance area must be at least one pixel.")
    array = np.asarray(mask)
    if array.ndim != 2:
        raise SegmentationError("The instance mask must be two-dimensional.")
    if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
        raise SegmentationError("Mask IDs must be non-negative integers.")
    array = array.astype(np.uint32, copy=False)
    present, counts = np.unique(array[array > 0], return_counts=True)
    missing = sorted(int(value) for value in present if int(value) not in instances)
    if missing:
        raise SegmentationError(f"Instance IDs missing metadata: {missing}")
    removed = tuple(
        int(instance_id)
        for instance_id, count in zip(present, counts, strict=True)
        if int(count) < minimum_area
    )
    output = array.copy()
    if removed:
        output[np.isin(output, removed)] = 0
    records = {
        int(instance_id): record
        for instance_id, record in instances.items()
        if int(instance_id) not in removed
    }
    return output, records, SmallInstanceRemoval(
        removed_instance_ids=removed,
        removed_pixels=sum(
            int(count)
            for instance_id, count in zip(present, counts, strict=True)
            if int(instance_id) in removed
        ),
    )


def compact_instance_ids(
    mask: np.ndarray,
    instances: Mapping[int, InstanceRecord],
    classes: Mapping[int, str],
) -> tuple[np.ndarray, dict[int, InstanceRecord], InstanceCompaction]:
    """Drop empty metadata and deterministically renumber present IDs to 1..N."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise SegmentationError("The instance mask must be two-dimensional.")
    if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
        raise SegmentationError("Mask IDs must be non-negative integers.")
    array = array.astype(np.uint32, copy=False)
    present = tuple(sorted(int(value) for value in np.unique(array) if value))
    missing = sorted(set(present) - {int(value) for value in instances})
    if missing:
        raise SegmentationError(f"Instance IDs missing metadata: {missing}")
    mapping = {old_id: new_id for new_id, old_id in enumerate(present, start=1)}
    output = np.zeros_like(array)
    raw_records: dict[int, InstanceRecord] = {}
    for old_id, new_id in mapping.items():
        output[array == old_id] = new_id
        record = instances[old_id]
        raw_records[new_id] = InstanceRecord(
            **{
                **record.__dict__,
                "instance_id": new_id,
            }
        )
    records = refresh_instance_records(output, raw_records, classes)
    return output, records, InstanceCompaction(
        old_to_new=mapping,
        removed_empty_ids=tuple(
            sorted({int(value) for value in instances} - set(present))
        ),
    )


def split_instance(
    mask: np.ndarray,
    instance_id: int,
    instances: Mapping[int, InstanceRecord],
    classes: Mapping[int, str],
) -> tuple[np.ndarray, dict[int, InstanceRecord]]:
    """Split one disconnected ID into deterministic 4-connected instance IDs."""
    output = np.asarray(mask, dtype=np.uint32).copy()
    components = label(output == int(instance_id), connectivity=1)
    count = int(components.max(initial=0))
    if count <= 1:
        return output, refresh_instance_records(output, instances, classes)
    original = instances.get(int(instance_id))
    if original is None:
        raise SegmentationError(f"Instance {instance_id} has no metadata.")
    next_id = int(output.max(initial=0)) + 1
    raw = dict(instances)
    for component in range(2, count + 1):
        output[components == component] = next_id
        raw[next_id] = InstanceRecord(
            instance_id=next_id,
            class_id=original.class_id,
            class_name=original.class_name,
            confidence=original.confidence,
            source="manual",
            status="corrected",
            bbox=(0, 0, 0, 0),
            area=0,
            lineage=(*original.lineage, original.instance_id),
        )
        next_id += 1
    return output, refresh_instance_records(output, raw, classes)
