from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from ._annotation_io import (
    IMAGE_EXTENSIONS,
    AnnotationIO,
    BoundingBox,
    LabelValidationError,
)
from ._image_adapter import ImageConversionError, ImageProcessingSettings
from ._project_store import ProjectStore
from ._tiled_inference import Tile, create_tile_plan


class DatasetBuildError(RuntimeError):
    """Raised when the canonical annotation pool cannot form a dataset."""


class DatasetBuildCancelled(DatasetBuildError):
    """Raised when snapshot creation is cancelled between samples."""


@dataclass(frozen=True)
class DatasetBuildSettings:
    validation_fraction: float = 0.2
    seed: int = 42
    tile_size: int = 1024
    overlap: int = 205
    negative_tile_ratio: float = 1.0
    group_field: str = ""
    min_retained_area: float = 0.90
    max_clip_pixels: float = 10.0
    max_clip_fraction: float = 0.10

    def validate(self) -> None:
        if not 0 <= self.validation_fraction < 1:
            raise DatasetBuildError("Validation fraction must be in [0, 1).")
        if self.seed < 0:
            raise DatasetBuildError("Split seed must be non-negative.")
        if self.tile_size < 64:
            raise DatasetBuildError("Training tile size must be at least 64 pixels.")
        if not 0 <= self.overlap < self.tile_size:
            raise DatasetBuildError("Training overlap must be smaller than tile size.")
        if self.negative_tile_ratio < 0:
            raise DatasetBuildError("Negative-tile ratio cannot be negative.")
        if not 0 < self.min_retained_area <= 1:
            raise DatasetBuildError("Minimum retained bbox area must be in (0, 1].")
        if self.max_clip_pixels < 0 or not 0 <= self.max_clip_fraction <= 1:
            raise DatasetBuildError("Clipping tolerances cannot be negative.")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    boxes: tuple[BoundingBox, ...]
    group: str
    image_sha256: str
    label_sha256: str

    @property
    def is_positive(self) -> bool:
        return bool(self.boxes)


@dataclass(frozen=True)
class TileSelection:
    tile: Tile
    boxes: tuple[tuple[int, float, float, float, float], ...]
    rejected_box_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class DatasetPreview:
    samples: tuple[SampleRecord, ...]
    assignments: dict[str, str]
    validation_mode: str
    source_counts: dict[str, int]
    tile_counts: dict[str, int]
    box_counts: dict[str, int]
    positive_count: int
    negative_count: int
    total_boxes: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset_root: Path
    dataset_yaml: Path
    split_manifest: Path
    tile_manifest: Path
    image_count: int
    label_count: int
    rejected_box_count: int


class DatasetBuilder:
    """Validate canonical pairs and create immutable tiled YOLO snapshots."""

    def __init__(self, project: ProjectStore):
        self.project = project
        self.annotation_io = AnnotationIO(project)

    def preview(
        self,
        settings: DatasetBuildSettings,
        *,
        train_only: bool = False,
        regenerate: bool = False,
        persist: bool = True,
    ) -> DatasetPreview:
        settings.validate()
        samples, warnings, errors = self._load_samples(settings.group_field)
        patch_contract = self.project.config.training_patch
        if patch_contract.get("locked"):
            patch_size = int(patch_contract["size"])
            if settings.tile_size != patch_size:
                errors.append(
                    f"Training tile size must match the locked project patch size "
                    f"of {patch_size}."
                )
        assignments: dict[str, str] = {}
        if samples and not errors:
            try:
                assignments = self._assign_splits(
                    samples,
                    settings,
                    train_only=train_only,
                    regenerate=regenerate,
                )
            except DatasetBuildError as exc:
                errors.append(str(exc))

        group_count = len({sample.group for sample in samples})
        if not samples:
            errors.append("No valid image/label pairs are available for retraining.")
        elif group_count < 2 and not train_only:
            errors.append(
                "An independent validation split requires at least two source groups. "
                "Enable 'Train without independent validation' for an exploratory run."
            )
        if settings.group_field:
            warnings.append(
                f"Grouping uses audit metadata field {settings.group_field!r}; "
                "the user is responsible for its biological meaning."
            )
        else:
            warnings.append(
                "Grouping uses original source image paths when available. The plugin "
                "cannot detect patient/well/acquisition leakage without group metadata."
            )
        stored_split = self.project.config.dataset_split
        if (
            stored_split.get("assignments")
            and not regenerate
            and not train_only
            and (
                stored_split.get("seed") != settings.seed
                or stored_split.get("validation_fraction")
                != settings.validation_fraction
            )
        ):
            warnings.append(
                "Existing sample assignments remain stable even though split settings "
                "changed. Use Regenerate Split to reassign existing samples."
            )

        source_groups: dict[str, set[str]] = {"train": set(), "val": set()}
        tile_counts = {"train": 0, "val": 0}
        box_counts = {"train": 0, "val": 0}
        for sample in samples:
            split = assignments.get(sample.sample_id)
            if split is None:
                continue
            source_groups[split].add(sample.group)
            selections, rejected = self._plan_sample_tiles(sample, settings)
            tile_counts[split] += len(selections)
            box_counts[split] += sum(len(selection.boxes) for selection in selections)
            if rejected:
                warnings.append(
                    f"{sample.sample_id}: {rejected} bbox(es) cannot meet the configured "
                    "tile clipping tolerance and will be excluded."
                )

        source_counts = {
            split: len(groups) for split, groups in source_groups.items()
        }
        validation_mode = "none" if train_only else "independent"
        if not train_only and source_counts["val"] == 0 and samples:
            errors.append("The split has no validation source group.")
        if source_counts["train"] == 0 and samples:
            errors.append("The split has no training source group.")
        if source_counts["train"] and tile_counts["train"] == 0:
            errors.append(
                "No usable training tiles remain after bbox clipping validation."
            )
        if (
            validation_mode == "independent"
            and source_counts["val"]
            and tile_counts["val"] == 0
        ):
            errors.append(
                "No usable validation tiles remain after bbox clipping validation."
            )
        if persist and assignments and not train_only and not errors:
            stored = (
                {}
                if regenerate
                else dict(self.project.config.dataset_split.get("assignments", {}))
            )
            stored.update(assignments)
            self.project.update_dataset_split(
                stored,
                seed=settings.seed,
                validation_fraction=settings.validation_fraction,
            )

        return DatasetPreview(
            samples=samples,
            assignments=assignments,
            validation_mode=validation_mode,
            source_counts=source_counts,
            tile_counts=tile_counts,
            box_counts=box_counts,
            positive_count=sum(sample.is_positive for sample in samples),
            negative_count=sum(not sample.is_positive for sample in samples),
            total_boxes=sum(len(sample.boxes) for sample in samples),
            warnings=tuple(dict.fromkeys(warnings)),
            errors=tuple(dict.fromkeys(errors)),
        )

    def create_snapshot(
        self,
        run_root: Path,
        preview: DatasetPreview,
        settings: DatasetBuildSettings,
        *,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> DatasetSnapshot:
        if not preview.is_valid:
            raise DatasetBuildError("Cannot build an invalid dataset preview.")
        dataset_root = Path(run_root) / "dataset"
        for split in ("train", "val"):
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=False)
            (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=False)

        split_manifest = dataset_root / "split_manifest.csv"
        tile_manifest = dataset_root / "tile_manifest.csv"
        split_rows: list[dict[str, Any]] = []
        tile_rows: list[dict[str, Any]] = []
        rejected_total = 0
        image_count = 0
        total = len(preview.samples)
        for index, sample in enumerate(preview.samples, start=1):
            if cancelled is not None and cancelled():
                raise DatasetBuildCancelled("Dataset preparation cancelled.")
            split = preview.assignments[sample.sample_id]
            if progress is not None:
                progress(index - 1, total, f"Preparing {sample.sample_id}")
            selections, rejected = self._plan_sample_tiles(sample, settings)
            rejected_total += rejected
            with Image.open(sample.image_path) as image_file:
                image = np.asarray(image_file.convert("RGB"))
            for selection in selections:
                tile = selection.tile
                tile_name = (
                    f"{sample.sample_id}__x{tile.x0:06d}_y{tile.y0:06d}"
                )
                image_path = dataset_root / "images" / split / f"{tile_name}.png"
                label_path = dataset_root / "labels" / split / f"{tile_name}.txt"
                tile_image = _constant_padded_tile(image, tile, settings.tile_size)
                Image.fromarray(tile_image, mode="RGB").save(image_path)
                label_text = _format_tile_boxes(selection.boxes, settings.tile_size)
                label_path.write_text(label_text, encoding="utf-8", newline="\n")
                image_count += 1
                tile_rows.append(
                    {
                        "tile": tile_name,
                        "split": split,
                        "source_sample": sample.sample_id,
                        "source_group": sample.group,
                        "tile_id": tile.index,
                        "x0": tile.x0,
                        "y0": tile.y0,
                        "valid_width": tile.width,
                        "valid_height": tile.height,
                        "pad_right": settings.tile_size - tile.width,
                        "pad_bottom": settings.tile_size - tile.height,
                        "box_count": len(selection.boxes),
                        "image_sha256": _sha256(image_path),
                        "label_sha256": _sha256(label_path),
                    }
                )
            split_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "group": sample.group,
                    "split": split,
                    "image": str(sample.image_path.relative_to(self.project.paths.root)),
                    "label": str(sample.label_path.relative_to(self.project.paths.root)),
                    "width": sample.width,
                    "height": sample.height,
                    "box_count": len(sample.boxes),
                    "image_sha256": sample.image_sha256,
                    "label_sha256": sample.label_sha256,
                }
            )

        _write_csv(split_manifest, split_rows)
        _write_csv(tile_manifest, tile_rows)
        dataset_yaml = dataset_root / "dataset.yaml"
        dataset_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(dataset_root.resolve()),
                    "train": "images/train",
                    # Ultralytics constructs a val loader even with val=False.
                    # Reusing train only satisfies that loader contract; run.yaml
                    # remains validation_mode:none and validation is disabled.
                    "val": (
                        "images/train"
                        if preview.validation_mode == "none"
                        else "images/val"
                    ),
                    "names": dict(sorted(self.project.config.classes.items())),
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
            newline="\n",
        )
        if progress is not None:
            progress(total, total, f"Prepared {image_count} training tiles")
        return DatasetSnapshot(
            dataset_root=dataset_root,
            dataset_yaml=dataset_yaml,
            split_manifest=split_manifest,
            tile_manifest=tile_manifest,
            image_count=image_count,
            label_count=image_count,
            rejected_box_count=rejected_total,
        )

    def _load_samples(
        self, group_field: str
    ) -> tuple[tuple[SampleRecord, ...], list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []
        images_by_stem: dict[str, list[Path]] = {}
        for path in self.project.paths.images.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images_by_stem.setdefault(path.stem, []).append(path)
        labels_by_stem = {
            path.stem: path
            for path in self.project.paths.labels.glob("*.txt")
            if path.is_file()
        }
        for stem, paths in images_by_stem.items():
            if len(paths) > 1:
                errors.append(f"Duplicate canonical image stem {stem!r}.")
        for stem in sorted(set(images_by_stem) - set(labels_by_stem)):
            errors.append(f"Missing label for image sample {stem!r}.")
        for stem in sorted(set(labels_by_stem) - set(images_by_stem)):
            errors.append(f"Missing image for label sample {stem!r}.")

        audit = self._latest_audit_events()
        samples: list[SampleRecord] = []
        hashes: dict[str, str] = {}
        for stem in sorted(set(images_by_stem).intersection(labels_by_stem)):
            if len(images_by_stem[stem]) != 1:
                continue
            image_path = images_by_stem[stem][0]
            label_path = labels_by_stem[stem]
            try:
                with Image.open(image_path) as image_file:
                    width, height = image_file.size
                    image_mode = image_file.mode
                    image_file.verify()
                patch_contract = self.project.config.training_patch
                if patch_contract.get("locked"):
                    patch_size = int(patch_contract["size"])
                    if (width, height) != (patch_size, patch_size):
                        errors.append(
                            f"Invalid sample {stem!r}: image is {width}×{height}; "
                            f"expected {patch_size}×{patch_size}."
                        )
                        continue
                boxes = self.annotation_io.parse_yolo_text(
                    label_path.read_text(encoding="utf-8"), source=str(label_path)
                )
            except (OSError, ValueError, LabelValidationError) as exc:
                errors.append(f"Invalid sample {stem!r}: {exc}")
                continue
            event = audit.get(stem, {})
            processing_error = self._processing_provenance_error(
                stem, event, image_mode
            )
            if processing_error is not None:
                errors.append(processing_error)
                continue
            padding_errors = _crop_padding_errors(boxes, event)
            if padding_errors:
                errors.extend(
                    f"Invalid sample {stem!r}: {item}"
                    for item in padding_errors
                )
                continue
            image_hash = _sha256(image_path)
            if image_hash in hashes:
                warnings.append(
                    f"Samples {hashes[image_hash]!r} and {stem!r} have identical image content."
                )
            else:
                hashes[image_hash] = stem
            group = _sample_group(stem, event, group_field)
            if group_field and _lookup_field(event, group_field) in {None, ""}:
                warnings.append(
                    f"{stem}: group field {group_field!r} is missing; using source image identity."
                )
            samples.append(
                SampleRecord(
                    sample_id=stem,
                    image_path=image_path,
                    label_path=label_path,
                    width=width,
                    height=height,
                    boxes=boxes,
                    group=group,
                    image_sha256=image_hash,
                    label_sha256=_sha256(label_path),
                )
            )
        return tuple(samples), warnings, errors

    def _processing_provenance_error(
        self, stem: str, event: dict[str, Any], image_mode: str
    ) -> str | None:
        expected = self.project.config.image_processing
        if not expected.get("locked"):
            return None
        if image_mode != "RGB":
            return (
                f"Invalid sample {stem!r}: canonical image mode is "
                f"{image_mode!r}; expected normalized RGB uint8 pixels."
            )
        conversion = event.get("conversion", {})
        actual = conversion.get("settings") if isinstance(conversion, dict) else None
        try:
            normalized_expected = ImageProcessingSettings.from_mapping(
                expected
            ).to_mapping()
            normalized_actual = ImageProcessingSettings.from_mapping(
                actual
            ).to_mapping()
        except (ImageConversionError, TypeError):
            normalized_expected = expected
            normalized_actual = actual
        if normalized_actual != normalized_expected:
            return (
                f"Invalid sample {stem!r}: audit conversion settings do not "
                "match the locked project image-processing settings. Re-save "
                "the sample from its source image before retraining."
            )
        return None

    def _latest_audit_events(self) -> dict[str, dict[str, Any]]:
        events: dict[str, dict[str, Any]] = {}
        try:
            lines = self.project.paths.audit.read_text(encoding="utf-8").splitlines()
        except OSError:
            return events
        for line in lines:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            sample_id = event.get("sample_id")
            if isinstance(sample_id, str):
                previous = events.get(sample_id, {})
                merged = {**previous, **event}
                previous_metadata = previous.get("metadata", {})
                event_metadata = event.get("metadata", {})
                if isinstance(previous_metadata, dict) or isinstance(
                    event_metadata, dict
                ):
                    metadata = (
                        dict(previous_metadata)
                        if isinstance(previous_metadata, dict)
                        else {}
                    )
                    if isinstance(event_metadata, dict):
                        metadata.update(event_metadata)
                    merged["metadata"] = metadata
                events[sample_id] = merged
        return events

    def _assign_splits(
        self,
        samples: tuple[SampleRecord, ...],
        settings: DatasetBuildSettings,
        *,
        train_only: bool,
        regenerate: bool,
    ) -> dict[str, str]:
        if train_only:
            return {sample.sample_id: "train" for sample in samples}
        groups: dict[str, list[SampleRecord]] = {}
        for sample in samples:
            groups.setdefault(sample.group, []).append(sample)
        if len(groups) < 2:
            return {}
        existing = (
            {}
            if regenerate
            else self.project.config.dataset_split.get("assignments", {})
        )
        group_splits: dict[str, str] = {}
        for group, members in groups.items():
            assigned = {existing.get(member.sample_id) for member in members}
            assigned.discard(None)
            if len(assigned) > 1:
                raise DatasetBuildError(
                    f"Stored split assignments leak source group {group!r} across train/val. "
                    "Use Regenerate split to repair it."
                )
            if assigned:
                group_splits[group] = assigned.pop()

        target_val = max(1, round(len(groups) * settings.validation_fraction))
        target_val = min(target_val, len(groups) - 1)
        new_groups = _stratified_group_order(
            {
                group: groups[group]
                for group in set(groups) - set(group_splits)
            },
            settings.seed,
        )
        needed_val = max(
            0,
            target_val
            - sum(split == "val" for split in group_splits.values()),
        )
        for index, group in enumerate(new_groups):
            group_splits[group] = "val" if index < needed_val else "train"
        return {
            sample.sample_id: group_splits[sample.group] for sample in samples
        }

    def _plan_sample_tiles(
        self, sample: SampleRecord, settings: DatasetBuildSettings
    ) -> tuple[tuple[TileSelection, ...], int]:
        plan = create_tile_plan(
            sample.height,
            sample.width,
            settings.tile_size,
            settings.overlap,
        )
        assignments: dict[int, list[tuple[int, float, float, float, float]]] = {
            tile.index: [] for tile in plan.tiles
        }
        contaminated_tiles: set[int] = set()
        rejected = 0
        for box in sample.boxes:
            absolute = _absolute_box(box, sample.width, sample.height)
            selected = _choose_tile_for_box(absolute, plan.tiles, settings)
            if selected is None:
                rejected += 1
                x1, y1, x2, y2 = absolute
                contaminated_tiles.update(
                    tile.index
                    for tile in plan.tiles
                    if min(x2, tile.x1) > max(x1, tile.x0)
                    and min(y2, tile.y1) > max(y1, tile.y0)
                )
                continue
            tile, clipped = selected
            assignments[tile.index].append((box.class_id, *clipped))
        for tile_index in contaminated_tiles:
            rejected += len(assignments[tile_index])
            assignments[tile_index] = []
        positive = [
            tile
            for tile in plan.tiles
            if assignments[tile.index] and tile.index not in contaminated_tiles
        ]
        negative = [
            tile
            for tile in plan.tiles
            if not assignments[tile.index] and tile.index not in contaminated_tiles
        ]
        limit = math.ceil(len(positive) * settings.negative_tile_ratio)
        if not positive and negative:
            limit = max(1, limit)
        negative = sorted(
            negative,
            key=lambda tile: _seeded_key(
                settings.seed, f"{sample.sample_id}:{tile.index}"
            ),
        )[:limit]
        selected_tiles = sorted(positive + negative, key=lambda tile: tile.index)
        return (
            tuple(
                TileSelection(
                    tile=tile,
                    boxes=tuple(assignments[tile.index]),
                )
                for tile in selected_tiles
            ),
            rejected,
        )


def _crop_padding_errors(
    boxes: tuple[BoundingBox, ...], event: dict[str, Any]
) -> tuple[str, ...]:
    conversion = event.get("conversion", {})
    crop = conversion.get("training_crop") if isinstance(conversion, dict) else None
    if not isinstance(crop, dict):
        return ()
    try:
        size = int(crop["size"])
        valid_height = int(crop["valid_height"])
        valid_width = int(crop["valid_width"])
    except (KeyError, TypeError, ValueError):
        return ("training-crop audit metadata is invalid.",)
    errors: list[str] = []
    for index, box in enumerate(boxes, start=1):
        x2 = (box.x_center + box.width / 2.0) * size
        y2 = (box.y_center + box.height / 2.0) * size
        if x2 > valid_width + 1e-6 or y2 > valid_height + 1e-6:
            errors.append(
                f"box {index} extends into synthetic training-crop padding."
            )
    return tuple(errors)


def _sample_group(stem: str, event: dict[str, Any], group_field: str) -> str:
    if group_field:
        value = _lookup_field(event, group_field)
        if value not in {None, ""}:
            return f"metadata:{value}"
    source_path = event.get("source_path")
    if isinstance(source_path, str) and source_path:
        return f"source:{Path(source_path).resolve()}"
    return f"sample:{stem}"


def _lookup_field(value: dict[str, Any], dotted_key: str) -> Any:
    current: Any = value
    for key in (part for part in dotted_key.split(".") if part):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _absolute_box(
    box: BoundingBox, width: int, height: int
) -> tuple[float, float, float, float]:
    x1 = (box.x_center - box.width / 2) * width
    y1 = (box.y_center - box.height / 2) * height
    x2 = (box.x_center + box.width / 2) * width
    y2 = (box.y_center + box.height / 2) * height
    return x1, y1, x2, y2


def _choose_tile_for_box(
    box: tuple[float, float, float, float],
    tiles: tuple[Tile, ...],
    settings: DatasetBuildSettings,
) -> tuple[Tile, tuple[float, float, float, float]] | None:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    complete = [
        tile
        for tile in tiles
        if tile.x0 <= x1 and tile.y0 <= y1 and tile.x1 >= x2 and tile.y1 >= y2
    ]
    if complete:
        tile = max(
            complete,
            key=lambda candidate: (
                candidate.owns(center_x, center_y),
                min(
                    x1 - candidate.x0,
                    y1 - candidate.y0,
                    candidate.x1 - x2,
                    candidate.y1 - y2,
                ),
                -candidate.index,
            ),
        )
        return tile, (x1 - tile.x0, y1 - tile.y0, x2 - tile.x0, y2 - tile.y0)

    original_width = x2 - x1
    original_height = y2 - y1
    original_area = original_width * original_height
    candidates: list[
        tuple[float, int, Tile, tuple[float, float, float, float]]
    ] = []
    for tile in tiles:
        clipped = (
            max(x1, tile.x0),
            max(y1, tile.y0),
            min(x2, tile.x1),
            min(y2, tile.y1),
        )
        clipped_width = max(0.0, clipped[2] - clipped[0])
        clipped_height = max(0.0, clipped[3] - clipped[1])
        retained = clipped_width * clipped_height / original_area
        loss_x = original_width - clipped_width
        loss_y = original_height - clipped_height
        allowed_x = min(
            settings.max_clip_pixels,
            settings.max_clip_fraction * original_width,
        )
        allowed_y = min(
            settings.max_clip_pixels,
            settings.max_clip_fraction * original_height,
        )
        if (
            retained >= settings.min_retained_area
            and loss_x <= allowed_x + 1e-6
            and loss_y <= allowed_y + 1e-6
        ):
            local = (
                clipped[0] - tile.x0,
                clipped[1] - tile.y0,
                clipped[2] - tile.x0,
                clipped[3] - tile.y0,
            )
            candidates.append((retained, -tile.index, tile, local))
    if not candidates:
        return None
    _, _, tile, local = max(candidates, key=lambda item: (item[0], item[1]))
    return tile, local


def _constant_padded_tile(image: np.ndarray, tile: Tile, tile_size: int) -> np.ndarray:
    result = np.full((tile_size, tile_size, 3), 114, dtype=np.uint8)
    cropped = np.asarray(image[tile.y0 : tile.y1, tile.x0 : tile.x1], dtype=np.uint8)
    result[: tile.height, : tile.width] = cropped
    return result


def _format_tile_boxes(
    boxes: tuple[tuple[int, float, float, float, float], ...], tile_size: int
) -> str:
    lines: list[str] = []
    for class_id, x1, y1, x2, y2 in boxes:
        x_center = ((x1 + x2) / 2) / tile_size
        y_center = ((y1 + y2) / 2) / tile_size
        width = (x2 - x1) / tile_size
        height = (y2 - y1) / tile_size
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seeded_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _stratified_group_order(
    groups: dict[str, list[SampleRecord]], seed: int
) -> list[str]:
    """Interleave deterministic positive/negative and box-count strata."""
    buckets: dict[tuple[bool, int], list[str]] = {}
    for group, samples in groups.items():
        box_count = sum(len(sample.boxes) for sample in samples)
        box_bin = 0 if box_count == 0 else 1 if box_count <= 5 else 2
        buckets.setdefault((box_count > 0, box_bin), []).append(group)
    for values in buckets.values():
        values.sort(key=lambda group: _seeded_key(seed, group))
    keys = sorted(
        buckets,
        key=lambda key: _seeded_key(seed, f"stratum:{key[0]}:{key[1]}"),
    )
    ordered: list[str] = []
    while any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    return ordered


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
