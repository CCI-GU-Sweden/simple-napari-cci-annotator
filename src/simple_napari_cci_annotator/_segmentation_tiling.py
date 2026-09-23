from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import dask.array as da
import numpy as np
from skimage.segmentation import clear_border

from ._instance_mask import ComposedInstances, ComponentCleanup
from ._segmentation_io import InstanceRecord, SegmentationError, refresh_instance_records
from ._tiled_inference import InferenceCancelled, InferenceError, InferenceSettings


@dataclass
class _TileRegistry:
    total: int
    progress: Callable[[int, int, str], None] | None = None
    records: dict[int, InstanceRecord] = field(default_factory=dict)
    cleanup: dict[int, tuple[ComponentCleanup, ...]] = field(default_factory=dict)
    completed: set[int] = field(default_factory=set)
    lock: Lock = field(default_factory=Lock)

    def store(
        self,
        tile_id: int,
        records: Mapping[int, InstanceRecord],
        cleanup: tuple[ComponentCleanup, ...],
    ) -> None:
        with self.lock:
            self.records.update(records)
            self.cleanup[tile_id] = cleanup
            self.completed.add(tile_id)
            completed = len(self.completed)
        if self.progress is not None:
            self.progress(completed, self.total, f"Predicted tile {completed}/{self.total}")


class _UnionFind:
    def __init__(self, values):
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        canonical = min(first_root, second_root)
        other = max(first_root, second_root)
        self.parent[other] = canonical


class TiledSegmentationEngine:
    """Dask halo inference and class-aware one-pixel seam fusion."""

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(
        self,
        image: np.ndarray,
        settings: InferenceSettings,
        classes: Mapping[int, str],
        *,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        clear_border_instances: bool = False,
    ) -> ComposedInstances:
        settings.validate()
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise InferenceError("Tiled segmentation requires an RGB uint8 image.")
        core_size = settings.tile_size - 2 * settings.overlap
        if core_size <= 0:
            raise InferenceError(
                "Segmentation tile overlap must be smaller than half the tile size."
            )
        original_height, original_width = array.shape[:2]
        padded, padding = _pad_to_core_grid(array, core_size)
        rows = padded.shape[0] // core_size
        columns = padded.shape[1] // core_size
        tile_count = rows * columns
        if tile_count * settings.max_detections >= np.iinfo(np.uint32).max:
            raise InferenceError("The tile grid exceeds uint32 instance-ID capacity.")
        registry = _TileRegistry(tile_count, progress)

        source = da.from_array(
            padded,
            chunks=(core_size, core_size, 3),
            asarray=False,
        )

        def predict_block(block, block_id=None):
            if cancelled is not None and cancelled():
                raise InferenceCancelled("Inference cancelled by the user.")
            location = tuple(int(value) for value in (block_id or (0, 0, 0)))
            row, column = location[:2]
            tile_id = row * columns + column
            result = self.predictor.predict_image(
                np.asarray(block), settings, dict(classes)
            )
            base_id = tile_id * settings.max_detections
            labels = np.zeros(result.mask.shape, dtype=np.uint32)
            records: dict[int, InstanceRecord] = {}
            for local_id, record in sorted(result.instances.items()):
                if local_id > settings.max_detections:
                    raise InferenceError(
                        f"Tile {tile_id} returned more than max detections."
                    )
                global_id = base_id + int(local_id)
                labels[result.mask == local_id] = global_id
                records[global_id] = InstanceRecord(
                    instance_id=global_id,
                    class_id=record.class_id,
                    class_name=record.class_name,
                    confidence=record.confidence,
                    source="prediction",
                    status="predicted",
                    bbox=(0, 0, 0, 0),
                    area=0,
                    lineage=(global_id,),
                )
            registry.store(tile_id, records, result.cleanup)
            if cancelled is not None and cancelled():
                raise InferenceCancelled("Inference cancelled by the user.")
            return labels[..., np.newaxis]

        mapped = da.map_overlap(
            predict_block,
            source,
            depth={0: settings.overlap, 1: settings.overlap, 2: 0},
            boundary={0: "reflect", 1: "reflect", 2: "none"},
            trim=True,
            chunks=(core_size, core_size, 1),
            dtype=np.uint32,
            meta=np.empty((0, 0, 0), dtype=np.uint32),
            allow_rechunk=False,
        )
        try:
            temporary_mask = np.asarray(
                mapped.compute(scheduler="threads")[:, :, 0], dtype=np.uint32
            )
        except InferenceCancelled:
            raise
        except Exception as exc:
            if isinstance(exc, InferenceError):
                raise
            raise InferenceError(f"Dask segmentation inference failed: {exc}") from exc
        if cancelled is not None and cancelled():
            raise InferenceCancelled("Inference cancelled by the user.")
        if progress is not None:
            progress(tile_count, tile_count, "Merging tile seams")

        present_ids = {int(value) for value in np.unique(temporary_mask) if value}
        missing_metadata = sorted(present_ids - set(registry.records))
        if missing_metadata:
            raise SegmentationError(
                f"Tiled prediction produced IDs without metadata: {missing_metadata}."
            )
        equivalences, conflicts, ambiguous = _seam_equivalences(
            temporary_mask,
            core_size,
            registry.records,
        )
        fused, records, temporary_to_final = _canonical_relabel(
            temporary_mask,
            registry.records,
            classes,
            equivalences,
        )
        fused = fused[:original_height, :original_width]
        cleared_ids: list[int] = []
        if clear_border_instances and np.any(fused):
            before = {int(value) for value in np.unique(fused) if value}
            fused = np.asarray(clear_border(fused), dtype=np.uint32)
            after = {int(value) for value in np.unique(fused) if value}
            cleared_ids = sorted(before - after)
            records = {key: value for key, value in records.items() if key in after}
        records = refresh_instance_records(fused, records, classes)

        cleanup = tuple(
            item
            for tile_id in sorted(registry.cleanup)
            for item in registry.cleanup[tile_id]
        )
        provenance: dict[str, Any] = {
            "mode": "dask_tiled",
            "tile_size": settings.tile_size,
            "overlap": settings.overlap,
            "core_size": core_size,
            "grid": [rows, columns],
            "padding": {"bottom": padding[0], "right": padding[1]},
            "temporary_to_final": {
                str(key): value for key, value in sorted(temporary_to_final.items())
            },
            "equivalence_pairs": [list(pair) for pair in equivalences],
            "class_conflicts": conflicts,
            "ambiguous_seam_ids": sorted(ambiguous),
            "cleared_border_ids": cleared_ids,
            "component_removals": [
                {
                    "tile_id": tile_id,
                    "removed_components": sum(
                        item.removed_components for item in registry.cleanup[tile_id]
                    ),
                    "removed_pixels": sum(
                        item.removed_pixels for item in registry.cleanup[tile_id]
                    ),
                }
                for tile_id in sorted(registry.cleanup)
            ],
        }
        return ComposedInstances(fused, records, cleanup, provenance)


def _pad_to_core_grid(
    image: np.ndarray, core_size: int
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    pad_bottom = (-height) % core_size
    pad_right = (-width) % core_size
    if not pad_bottom and not pad_right:
        return np.ascontiguousarray(image), (0, 0)
    mode = "reflect" if height > 1 and width > 1 else "edge"
    padded = np.pad(
        image,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode=mode,
    )
    return np.ascontiguousarray(padded), (pad_bottom, pad_right)


def _seam_equivalences(
    mask: np.ndarray,
    core_size: int,
    records: Mapping[int, InstanceRecord],
) -> tuple[tuple[tuple[int, int], ...], list[dict[str, Any]], set[int]]:
    pairs: set[tuple[int, int]] = set()
    conflicts: list[dict[str, Any]] = []
    neighbours: dict[int, set[int]] = defaultdict(set)

    def inspect(first, second, orientation: str, coordinate: int):
        active = (first != 0) & (second != 0) & (first != second)
        for left, right in zip(first[active], second[active], strict=True):
            first_id, second_id = sorted((int(left), int(right)))
            first_record = records.get(first_id)
            second_record = records.get(second_id)
            if first_record is None or second_record is None:
                continue
            if first_record.class_id == second_record.class_id:
                pairs.add((first_id, second_id))
                neighbours[first_id].add(second_id)
                neighbours[second_id].add(first_id)
            else:
                conflicts.append(
                    {
                        "ids": [first_id, second_id],
                        "classes": [first_record.class_id, second_record.class_id],
                        "orientation": orientation,
                        "coordinate": coordinate,
                    }
                )

    for y in range(core_size, mask.shape[0], core_size):
        inspect(mask[y - 1, :], mask[y, :], "horizontal", y)
    for x in range(core_size, mask.shape[1], core_size):
        inspect(mask[:, x - 1], mask[:, x], "vertical", x)
    ambiguous = {instance_id for instance_id, values in neighbours.items() if len(values) > 1}
    unique_conflicts = {
        (
            tuple(value["ids"]),
            tuple(value["classes"]),
            value["orientation"],
            value["coordinate"],
        ): value
        for value in conflicts
    }
    return tuple(sorted(pairs)), list(unique_conflicts.values()), ambiguous


def _canonical_relabel(
    mask: np.ndarray,
    records: Mapping[int, InstanceRecord],
    classes: Mapping[int, str],
    equivalences: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, dict[int, InstanceRecord], dict[int, int]]:
    present = sorted(int(value) for value in np.unique(mask) if value)
    union_find = _UnionFind(present)
    for first, second in equivalences:
        if first in union_find.parent and second in union_find.parent:
            union_find.union(first, second)
    groups: dict[int, list[int]] = defaultdict(list)
    for temporary_id in present:
        groups[union_find.find(temporary_id)].append(temporary_id)
    ordered_groups = sorted(groups.values(), key=lambda values: min(values))
    output = np.zeros(mask.shape, dtype=np.uint32)
    raw_records: dict[int, dict[str, Any]] = {}
    temporary_to_final: dict[int, int] = {}
    for final_id, members in enumerate(ordered_groups, start=1):
        member_records = [records[value] for value in members]
        class_ids = {record.class_id for record in member_records}
        if len(class_ids) != 1:
            raise SegmentationError(
                f"A fused instance crosses classes: temporary IDs {members}."
            )
        for temporary_id in members:
            output[mask == temporary_id] = final_id
            temporary_to_final[temporary_id] = final_id
        confidences = [
            record.confidence
            for record in member_records
            if record.confidence is not None
        ]
        class_id = next(iter(class_ids))
        raw_records[final_id] = {
            "class_id": class_id,
            "class_name": classes[class_id],
            "confidence": max(confidences) if confidences else None,
            "source": "prediction",
            "status": "predicted",
            "lineage": members,
        }
    return (
        output,
        refresh_instance_records(output, raw_records, classes),
        temporary_to_final,
    )
