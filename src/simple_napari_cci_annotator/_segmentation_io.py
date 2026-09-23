from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import tifffile

from ._annotation_io import IMAGE_EXTENSIONS, AnnotationIO
from ._project_store import ProjectStore


class SegmentationError(RuntimeError):
    """Raised when an instance mask or its metadata is invalid."""


@dataclass(frozen=True)
class InstanceRecord:
    instance_id: int
    class_id: int
    class_name: str
    confidence: float | None
    source: str
    status: str
    bbox: tuple[int, int, int, int]
    area: int
    lineage: tuple[int, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "source": self.source,
            "status": self.status,
            "bbox": list(self.bbox),
            "area": self.area,
            "lineage": list(self.lineage),
        }


@dataclass(frozen=True)
class LoadedSegmentation:
    mask_path: Path
    instances_path: Path
    mask: np.ndarray
    instances: dict[int, InstanceRecord]
    conversion: dict[str, Any]


@dataclass(frozen=True)
class SegmentationSaveResult:
    image_path: Path
    mask_path: Path
    instances_path: Path
    instance_count: int
    operation: str


@dataclass(frozen=True)
class SegmentationReviewEntry:
    sample_id: str
    image_path: Path | None
    mask_path: Path | None
    instances_path: Path | None
    instance_count: int
    errors: tuple[str, ...]
    source_path: Path | None
    conversion: dict[str, Any]

    @property
    def box_count(self) -> int:
        """Compatibility count used by the shared review selector."""
        return self.instance_count

    @property
    def is_valid(self) -> bool:
        return not self.errors


class SegmentationIO:
    """Atomic storage and validation for RGB image/instance-mask triples."""

    MASK_SUFFIX = ".tif"

    def __init__(self, project: ProjectStore):
        if project.config.task != "segment":
            raise SegmentationError("Segmentation I/O requires a segment project.")
        self.project = project

    def find_mask(self, sample_id: str) -> tuple[Path, Path] | None:
        stem = AnnotationIO.safe_stem(sample_id)
        mask_path = self.project.paths.masks / f"{stem}{self.MASK_SUFFIX}"
        instances_path = self.project.paths.instances / f"{stem}.json"
        if mask_path.is_file() and instances_path.is_file():
            return mask_path, instances_path
        return None

    def load(self, sample_id: str, image_shape: Sequence[int]) -> LoadedSegmentation:
        pair = self.find_mask(sample_id)
        if pair is None:
            raise SegmentationError(f"No canonical mask found for {sample_id!r}.")
        mask_path, instances_path = pair
        try:
            mask = np.asarray(tifffile.imread(mask_path))
            payload = json.loads(instances_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SegmentationError(f"Could not load segmentation: {exc}") from exc
        if mask.dtype != np.dtype("uint32"):
            raise SegmentationError(
                f"Canonical masks must be uint32; found {mask.dtype}."
            )
        if mask.ndim != 2:
            raise SegmentationError(
                f"Canonical masks must be two-dimensional; found {mask.ndim}D."
            )
        expected_shape = tuple(int(value) for value in image_shape[:2])
        if mask.shape != expected_shape:
            raise SegmentationError(
                f"Mask shape {mask.shape} does not match image shape {expected_shape}."
            )
        records = _validated_records(
            mask, payload.get("instances"), self.project.config.classes
        )
        conversion = payload.get("conversion", {})
        return LoadedSegmentation(
            mask_path=mask_path,
            instances_path=instances_path,
            mask=mask,
            instances=records,
            conversion=dict(conversion) if isinstance(conversion, dict) else {},
        )

    def save(
        self,
        *,
        image_data: np.ndarray,
        sample_id: str,
        mask: np.ndarray,
        instances: Mapping[int, InstanceRecord | Mapping[str, Any]],
        source_path: Path | None = None,
        conversion_metadata: dict[str, Any] | None = None,
    ) -> SegmentationSaveResult:
        image = np.asarray(image_data)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise SegmentationError("Canonical segmentation images must be RGB uint8.")
        instance_mask = np.asarray(mask)
        if instance_mask.shape != image.shape[:2]:
            raise SegmentationError("Image and instance mask dimensions must match.")
        if not np.issubdtype(instance_mask.dtype, np.integer):
            raise SegmentationError("Instance masks must use an integer dtype.")
        if np.any(instance_mask < 0):
            raise SegmentationError("Instance masks cannot contain negative IDs.")
        if np.any(instance_mask > np.iinfo(np.uint32).max):
            raise SegmentationError("Instance IDs exceed uint32 storage capacity.")
        instance_mask = np.asarray(instance_mask, dtype=np.uint32)
        records = _records_for_save(
            instance_mask, instances, self.project.config.classes
        )
        patch = self.project.config.training_patch
        if patch.get("locked"):
            size = int(patch["size"])
            if image.shape[:2] != (size, size):
                raise SegmentationError(
                    f"Canonical segmentation samples must be {size}×{size}."
                )

        stem = AnnotationIO.safe_stem(sample_id)
        image_path = self.project.paths.images / f"{stem}.png"
        mask_path = self.project.paths.masks / f"{stem}{self.MASK_SUFFIX}"
        instances_path = self.project.paths.instances / f"{stem}.json"
        operation = (
            "updated"
            if image_path.exists() or mask_path.exists() or instances_path.exists()
            else "created"
        )
        payload = {
            "schema_version": 1,
            "sample_id": stem,
            "shape": list(instance_mask.shape),
            "mask_dtype": "uint32",
            "instances": {
                str(instance_id): record.to_mapping()
                for instance_id, record in sorted(records.items())
            },
            "conversion": conversion_metadata,
        }

        temporary_image = _temporary_path(image_path)
        temporary_mask = _temporary_path(mask_path)
        temporary_instances = _temporary_path(instances_path)
        try:
            Image.fromarray(image, mode="RGB").save(temporary_image, format="PNG")
            tifffile.imwrite(
                temporary_mask,
                instance_mask,
                compression="deflate",
                photometric="minisblack",
            )
            _write_json(temporary_instances, payload)
            _replace_many(
                (
                    (temporary_image, image_path),
                    (temporary_mask, mask_path),
                    (temporary_instances, instances_path),
                )
            )
            self._append_audit(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "operation": operation,
                    "task": "segment",
                    "sample_id": stem,
                    "image": str(image_path.relative_to(self.project.paths.root)),
                    "mask": str(mask_path.relative_to(self.project.paths.root)),
                    "instances": str(
                        instances_path.relative_to(self.project.paths.root)
                    ),
                    "source_path": str(source_path) if source_path else None,
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "conversion": conversion_metadata,
                    "instance_count": len(records),
                    "class_counts": {
                        str(class_id): sum(
                            record.class_id == class_id for record in records.values()
                        )
                        for class_id in self.project.config.classes
                    },
                    "image_sha256": _sha256(image_path),
                    "mask_sha256": _sha256(mask_path),
                    "instances_sha256": _sha256(instances_path),
                }
            )
        finally:
            temporary_image.unlink(missing_ok=True)
            temporary_mask.unlink(missing_ok=True)
            temporary_instances.unlink(missing_ok=True)
        return SegmentationSaveResult(
            image_path=image_path,
            mask_path=mask_path,
            instances_path=instances_path,
            instance_count=len(records),
            operation=operation,
        )

    def entries(self) -> tuple[SegmentationReviewEntry, ...]:
        image_by_stem = {
            path.stem: path
            for path in self.project.paths.images.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        masks = {path.stem: path for path in self.project.paths.masks.glob("*.tif")}
        metadata = {
            path.stem: path for path in self.project.paths.instances.glob("*.json")
        }
        audit = _latest_audit(self.project.paths.audit)
        entries: list[SegmentationReviewEntry] = []
        for stem in sorted(set(image_by_stem) | set(masks) | set(metadata)):
            errors: list[str] = []
            image_path = image_by_stem.get(stem)
            mask_path = masks.get(stem)
            instances_path = metadata.get(stem)
            if image_path is None:
                errors.append("missing image")
            if mask_path is None:
                errors.append("missing mask")
            if instances_path is None:
                errors.append("missing instance metadata")
            instance_count = 0
            conversion: dict[str, Any] = {}
            if not errors:
                try:
                    with Image.open(image_path) as image_file:
                        width, height = image_file.size
                    loaded = self.load(stem, (height, width))
                    instance_count = len(loaded.instances)
                    conversion = loaded.conversion
                except (OSError, SegmentationError) as exc:
                    errors.extend(str(exc).splitlines())
            event = audit.get(stem, {})
            raw_source = event.get("source_path")
            entries.append(
                SegmentationReviewEntry(
                    sample_id=stem,
                    image_path=image_path,
                    mask_path=mask_path,
                    instances_path=instances_path,
                    instance_count=instance_count,
                    errors=tuple(errors),
                    source_path=(
                        Path(raw_source) if isinstance(raw_source, str) else None
                    ),
                    conversion=conversion,
                )
            )
        return tuple(entries)

    def _append_audit(self, event: dict[str, Any]) -> None:
        payload = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        descriptor = os.open(self.project.paths.audit, flags)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def refresh_instance_records(
    mask: np.ndarray,
    instances: Mapping[int, InstanceRecord | Mapping[str, Any]],
    classes: Mapping[int, str],
    *,
    default_class_id: int = 0,
) -> dict[int, InstanceRecord]:
    """Reconcile editable mask IDs with metadata and recompute geometry."""
    array = np.asarray(mask, dtype=np.uint32)
    output: dict[int, InstanceRecord] = {}
    for instance_id in sorted(int(value) for value in np.unique(array) if value):
        raw = instances.get(instance_id, {})
        if isinstance(raw, InstanceRecord):
            class_id = raw.class_id
            confidence = raw.confidence
            source = raw.source
            status = raw.status
            lineage = raw.lineage
        else:
            if not isinstance(raw, Mapping):
                raise SegmentationError(
                    f"Instance {instance_id} metadata must be a mapping."
                )
            class_id = int(raw.get("class_id", default_class_id))
            confidence = raw.get("confidence")
            confidence = float(confidence) if confidence is not None else None
            source = str(raw.get("source", "manual"))
            status = str(raw.get("status", "corrected"))
            raw_lineage = raw.get("lineage", ())
            if not isinstance(raw_lineage, (list, tuple)):
                raise SegmentationError(
                    f"Instance {instance_id} lineage must be a sequence."
                )
            lineage = tuple(int(value) for value in raw_lineage)
        if confidence is not None and (
            not np.isfinite(confidence) or not 0 <= confidence <= 1
        ):
            raise SegmentationError(
                f"Instance {instance_id} confidence must be within 0..1."
            )
        if not source.strip() or not status.strip():
            raise SegmentationError(
                f"Instance {instance_id} source and status must be non-empty."
            )
        if class_id not in classes:
            raise SegmentationError(
                f"Instance {instance_id} uses unknown class ID {class_id}."
            )
        ys, xs = np.where(array == instance_id)
        output[instance_id] = InstanceRecord(
            instance_id=instance_id,
            class_id=class_id,
            class_name=classes[class_id],
            confidence=confidence,
            source=source,
            status=status,
            bbox=(
                int(ys.min()),
                int(xs.min()),
                int(ys.max()) + 1,
                int(xs.max()) + 1,
            ),
            area=int(len(ys)),
            lineage=lineage,
        )
    return output


def _records_for_save(mask, instances, classes) -> dict[int, InstanceRecord]:
    records = refresh_instance_records(mask, instances, classes)
    supplied = {int(value) for value in instances}
    present = set(records)
    orphaned = sorted(supplied - present)
    if orphaned:
        raise SegmentationError(
            f"Instance metadata has IDs absent from the mask: {orphaned}."
        )
    return records


def _validated_records(mask, raw_instances, classes) -> dict[int, InstanceRecord]:
    if not isinstance(raw_instances, dict):
        raise SegmentationError("Instance metadata must contain an instances mapping.")
    try:
        normalized = {int(key): value for key, value in raw_instances.items()}
    except (TypeError, ValueError) as exc:
        raise SegmentationError("Instance metadata IDs must be integers.") from exc
    return _records_for_save(mask, normalized, classes)


def _temporary_path(path: Path) -> Path:
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _replace_many(pairs: tuple[tuple[Path, Path], ...]) -> None:
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for _, destination in pairs:
            if destination.exists():
                backup = destination.parent / f".{destination.name}.bak.{uuid.uuid4().hex}"
                os.replace(destination, backup)
                backups[destination] = backup
        for temporary, destination in pairs:
            os.replace(temporary, destination)
            replaced.append(destination)
    except Exception:
        for destination in replaced:
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _latest_audit(path: Path) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        sample_id = event.get("sample_id")
        if isinstance(sample_id, str):
            events[sample_id] = {**events.get(sample_id, {}), **event}
    return events


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
