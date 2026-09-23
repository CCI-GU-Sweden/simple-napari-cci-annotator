from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import yaml
from PIL import Image
from skimage.draw import polygon
from skimage.measure import approximate_polygon, find_contours

from ._dataset_builder import (
    DatasetBuildCancelled,
    DatasetBuildError,
    DatasetBuildSettings,
    DatasetBuilder,
    _lookup_field,
    _sample_group,
)
from ._image_adapter import ImageConversionError, ImageProcessingSettings
from ._instance_mask import disconnected_instance_ids
from ._project_store import ProjectStore
from ._segmentation_io import InstanceRecord, SegmentationError, SegmentationIO


@dataclass(frozen=True)
class SegmentationSampleRecord:
    sample_id: str
    image_path: Path
    mask_path: Path
    instances_path: Path
    width: int
    height: int
    instances: dict[int, InstanceRecord]
    group: str
    image_sha256: str
    mask_sha256: str
    instances_sha256: str

    @property
    def is_positive(self) -> bool:
        return bool(self.instances)

    @property
    def boxes(self) -> tuple[InstanceRecord, ...]:
        """Compatibility population used by the shared stratified splitter."""
        return tuple(self.instances.values())


@dataclass(frozen=True)
class SegmentationDatasetPreview:
    samples: tuple[SegmentationSampleRecord, ...]
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
class SegmentationDatasetSnapshot:
    dataset_root: Path
    dataset_yaml: Path
    split_manifest: Path
    tile_manifest: Path
    image_count: int
    label_count: int
    rejected_box_count: int


class SegmentationDatasetBuilder:
    """Validate canonical triples and derive an immutable YOLO-seg snapshot."""

    def __init__(self, project: ProjectStore, *, minimum_polygon_iou: float = 0.90):
        if project.config.task != "segment":
            raise DatasetBuildError("Segmentation dataset building requires a segment project.")
        self.project = project
        self.segmentation_io = SegmentationIO(project)
        self.minimum_polygon_iou = float(minimum_polygon_iou)

    def preview(
        self,
        settings: DatasetBuildSettings,
        *,
        train_only: bool = False,
        regenerate: bool = False,
        persist: bool = True,
    ) -> SegmentationDatasetPreview:
        settings.validate()
        samples, warnings, errors = self._load_samples(settings.group_field)
        contract = self.project.config.training_patch
        if contract.get("locked") and settings.tile_size != int(contract["size"]):
            errors.append(
                f"Training size must match the locked project patch size of {contract['size']}."
            )
        assignments: dict[str, str] = {}
        if samples and not errors:
            helper = DatasetBuilder.__new__(DatasetBuilder)
            helper.project = self.project
            try:
                assignments = helper._assign_splits(
                    samples, settings, train_only=train_only, regenerate=regenerate
                )
            except DatasetBuildError as exc:
                errors.append(str(exc))
        groups = {"train": set(), "val": set()}
        for sample in samples:
            split = assignments.get(sample.sample_id)
            if split:
                groups[split].add(sample.group)
        if not samples:
            errors.append("No valid image/mask/instance triples are available for retraining.")
        elif len({sample.group for sample in samples}) < 2 and not train_only:
            errors.append(
                "An independent validation split requires at least two source groups. "
                "Enable 'Train without independent validation' for an exploratory run."
            )
        if not train_only and samples and not groups["val"]:
            errors.append("The split has no validation source group.")
        if samples and not groups["train"]:
            errors.append("The split has no training source group.")
        if settings.group_field:
            warnings.append(
                f"Grouping uses audit metadata field {settings.group_field!r}; its biological meaning is user-defined."
            )
        else:
            warnings.append("Grouping uses original source image paths when available.")
        if persist and assignments and not train_only and not errors:
            stored = {} if regenerate else dict(
                self.project.config.dataset_split.get("assignments", {})
            )
            stored.update(assignments)
            self.project.update_dataset_split(
                stored,
                seed=settings.seed,
                validation_fraction=settings.validation_fraction,
            )
        counts = {split: len(value) for split, value in groups.items()}
        sample_counts = {
            split: sum(value == split for value in assignments.values())
            for split in ("train", "val")
        }
        instance_counts = {
            split: sum(
                len(sample.instances)
                for sample in samples
                if assignments.get(sample.sample_id) == split
            )
            for split in ("train", "val")
        }
        return SegmentationDatasetPreview(
            samples=samples,
            assignments=assignments,
            validation_mode="none" if train_only else "independent",
            source_counts=counts,
            tile_counts=sample_counts,
            box_counts=instance_counts,
            positive_count=sum(sample.is_positive for sample in samples),
            negative_count=sum(not sample.is_positive for sample in samples),
            total_boxes=sum(len(sample.instances) for sample in samples),
            warnings=tuple(dict.fromkeys(warnings)),
            errors=tuple(dict.fromkeys(errors)),
        )

    def create_snapshot(
        self,
        run_root: Path,
        preview: SegmentationDatasetPreview,
        settings: DatasetBuildSettings,
        *,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> SegmentationDatasetSnapshot:
        if not preview.is_valid:
            raise DatasetBuildError("Cannot build an invalid segmentation dataset preview.")
        root = Path(run_root) / "dataset"
        for split in ("train", "val"):
            for kind in ("images", "labels", "masks", "instances"):
                (root / kind / split).mkdir(parents=True, exist_ok=False)
        split_rows: list[dict[str, Any]] = []
        export_rows: list[dict[str, Any]] = []
        for index, sample in enumerate(preview.samples, start=1):
            if cancelled is not None and cancelled():
                raise DatasetBuildCancelled("Dataset preparation cancelled.")
            if progress is not None:
                progress(index - 1, len(preview.samples), f"Exporting {sample.sample_id}")
            split = preview.assignments[sample.sample_id]
            with Image.open(sample.image_path) as handle:
                image = np.asarray(handle.convert("RGB"))
            loaded = self.segmentation_io.load(sample.sample_id, image.shape[:2])
            lines: list[str] = []
            class_counts: dict[int, int] = {}
            ious: list[float] = []
            for instance_id, record in sorted(loaded.instances.items()):
                points, iou = mask_to_yolo_polygon(
                    loaded.mask == instance_id,
                    minimum_iou=self.minimum_polygon_iou,
                )
                coords = " ".join(f"{value:.8f}" for value in points.ravel())
                lines.append(f"{record.class_id} {coords}")
                ious.append(iou)
                class_counts[record.class_id] = class_counts.get(record.class_id, 0) + 1
            image_out = root / "images" / split / f"{sample.sample_id}.png"
            label_out = root / "labels" / split / f"{sample.sample_id}.txt"
            mask_out = root / "masks" / split / f"{sample.sample_id}.tif"
            instances_out = root / "instances" / split / f"{sample.sample_id}.json"
            Image.fromarray(image, mode="RGB").save(image_out)
            label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
            tifffile.imwrite(mask_out, loaded.mask, compression="deflate", photometric="minisblack")
            instances_out.write_bytes(sample.instances_path.read_bytes())
            split_rows.append({
                "sample_id": sample.sample_id,
                "group": sample.group,
                "split": split,
                "instance_count": len(loaded.instances),
                "image_sha256": sample.image_sha256,
                "mask_sha256": sample.mask_sha256,
                "instances_sha256": sample.instances_sha256,
            })
            export_rows.append({
                "tile": sample.sample_id,
                "split": split,
                "source_sample": sample.sample_id,
                "instance_count": len(loaded.instances),
                "class_counts": json.dumps(class_counts, sort_keys=True),
                "minimum_round_trip_iou": min(ious) if ious else "",
                "mean_round_trip_iou": float(np.mean(ious)) if ious else "",
                "discarded_components": 0,
                "image_sha256": _sha256(image_out),
                "label_sha256": _sha256(label_out),
                "mask_sha256": _sha256(mask_out),
            })
        split_manifest = root / "split_manifest.csv"
        tile_manifest = root / "segmentation_export_manifest.csv"
        _write_csv(split_manifest, split_rows)
        _write_csv(tile_manifest, export_rows)
        dataset_yaml = root / "dataset.yaml"
        dataset_yaml.write_text(yaml.safe_dump({
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/train" if preview.validation_mode == "none" else "images/val",
            "names": dict(sorted(self.project.config.classes.items())),
        }, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        if progress is not None:
            progress(len(preview.samples), len(preview.samples), f"Exported {len(preview.samples)} segmentation samples")
        return SegmentationDatasetSnapshot(
            dataset_root=root,
            dataset_yaml=dataset_yaml,
            split_manifest=split_manifest,
            tile_manifest=tile_manifest,
            image_count=len(preview.samples),
            label_count=len(preview.samples),
            rejected_box_count=0,
        )

    def _load_samples(self, group_field: str):
        warnings: list[str] = []
        errors: list[str] = []
        audit = DatasetBuilder.__new__(DatasetBuilder)
        audit.project = self.project
        events = audit._latest_audit_events()
        samples: list[SegmentationSampleRecord] = []
        for entry in self.segmentation_io.entries():
            if not entry.is_valid or entry.image_path is None or entry.mask_path is None or entry.instances_path is None:
                errors.append(f"Invalid sample {entry.sample_id!r}: {', '.join(entry.errors)}")
                continue
            try:
                with Image.open(entry.image_path) as handle:
                    width, height = handle.size
                    mode = handle.mode
                loaded = self.segmentation_io.load(entry.sample_id, (height, width))
                contract = self.project.config.training_patch
                if contract.get("locked") and (width, height) != (int(contract["size"]),) * 2:
                    raise SegmentationError(f"image is {width}×{height}; expected {contract['size']}×{contract['size']}")
                if mode != "RGB":
                    raise SegmentationError(f"canonical image mode is {mode!r}; expected RGB")
                disconnected = disconnected_instance_ids(loaded.mask)
                if disconnected:
                    raise SegmentationError(f"disconnected instance IDs {list(disconnected)}")
                event = events.get(entry.sample_id, {})
                expected = self.project.config.image_processing
                actual = loaded.conversion.get("settings")
                if expected.get("locked") and ImageProcessingSettings.from_mapping(actual).to_mapping() != ImageProcessingSettings.from_mapping(expected).to_mapping():
                    raise SegmentationError("conversion settings do not match the locked project settings")
                padding_errors = _mask_padding_errors(loaded.mask, event)
                if padding_errors:
                    raise SegmentationError("; ".join(padding_errors))
                for instance_id in loaded.instances:
                    mask_to_yolo_polygon(loaded.mask == instance_id, minimum_iou=self.minimum_polygon_iou)
            except (OSError, SegmentationError, ImageConversionError, ValueError, KeyError) as exc:
                errors.append(f"Invalid sample {entry.sample_id!r}: {exc}")
                continue
            event = events.get(entry.sample_id, {})
            group = _sample_group(entry.sample_id, event, group_field)
            if group_field and _lookup_field(event, group_field) in {None, ""}:
                warnings.append(f"{entry.sample_id}: missing group field {group_field!r}; using source identity.")
            samples.append(SegmentationSampleRecord(
                sample_id=entry.sample_id,
                image_path=entry.image_path,
                mask_path=entry.mask_path,
                instances_path=entry.instances_path,
                width=width,
                height=height,
                instances=loaded.instances,
                group=group,
                image_sha256=_sha256(entry.image_path),
                mask_sha256=_sha256(entry.mask_path),
                instances_sha256=_sha256(entry.instances_path),
            ))
        return tuple(samples), warnings, errors


def mask_to_yolo_polygon(mask: np.ndarray, *, minimum_iou: float = 0.90) -> tuple[np.ndarray, float]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not np.any(binary):
        raise DatasetBuildError("Cannot export an empty instance mask.")
    contours = find_contours(np.pad(binary, 1), 0.5)
    if len(contours) != 1:
        raise DatasetBuildError("An instance must have exactly one connected exterior contour.")
    contour = approximate_polygon(contours[0] - 1.0, tolerance=0.5)
    if len(contour) > 1 and np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    if len(contour) < 3:
        raise DatasetBuildError("Instance polygon has fewer than three vertices.")
    height, width = binary.shape
    contour[:, 0] = np.clip(contour[:, 0], 0, height - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, width - 1)
    reconstructed = np.zeros_like(binary)
    rr, cc = polygon(contour[:, 0], contour[:, 1], shape=binary.shape)
    reconstructed[rr, cc] = True
    union = np.count_nonzero(binary | reconstructed)
    iou = float(np.count_nonzero(binary & reconstructed) / union) if union else 1.0
    if iou < minimum_iou:
        raise DatasetBuildError(
            f"Mask-to-polygon round-trip IoU {iou:.3f} is below {minimum_iou:.3f}."
        )
    points = np.column_stack((contour[:, 1] / width, contour[:, 0] / height))
    return np.clip(points, 0.0, 1.0), iou


def _mask_padding_errors(mask: np.ndarray, event: dict[str, Any]) -> tuple[str, ...]:
    conversion = event.get("conversion", {})
    crop = conversion.get("training_crop") if isinstance(conversion, dict) else None
    if not isinstance(crop, dict):
        return ()
    try:
        valid_height = int(crop["valid_height"])
        valid_width = int(crop["valid_width"])
    except (KeyError, TypeError, ValueError):
        return ("training-crop audit metadata is invalid",)
    array = np.asarray(mask)
    if np.any(array[valid_height:, :]) or np.any(array[:, valid_width:]):
        return ("instance pixels extend into synthetic training-crop padding",)
    return ()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
