from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from ._project_store import ProjectStore


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class AnnotationError(RuntimeError):
    """Base exception for annotation I/O errors."""


class LabelValidationError(AnnotationError):
    """Raised when a YOLO bbox label cannot be validated."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class BoundingBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class LoadedAnnotations:
    label_path: Path
    boxes: tuple[BoundingBox, ...]
    rectangles: tuple[np.ndarray, ...]
    properties: dict[str, np.ndarray]


@dataclass(frozen=True)
class SaveResult:
    image_path: Path
    label_path: Path
    box_count: int
    operation: str


@dataclass(frozen=True)
class ProjectValidationReport:
    image_count: int
    label_count: int
    paired_count: int
    box_count: int
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


class AnnotationIO:
    """Reads, validates, and atomically stores YOLO bbox annotations."""

    def __init__(self, project: ProjectStore):
        self.project = project

    @property
    def class_names(self) -> dict[int, str]:
        return self.project.config.classes

    @staticmethod
    def safe_stem(value: str) -> str:
        path = Path(value)
        stem = (
            path.stem
            if path.suffix.lower() in IMAGE_EXTENSIONS.union({".txt"})
            else path.name
        )
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem.strip())
        return stem or "image"

    def find_label(
        self,
        image_stem: str,
        *,
        source_path: Path | None = None,
    ) -> Path | None:
        stem = self.safe_stem(image_stem)
        candidates = [self.project.paths.labels / f"{stem}.txt"]

        if source_path is not None:
            source_path = Path(source_path)
            candidates.append(source_path.parent / f"{stem}.txt")
            if source_path.parent.name.lower() == "images":
                candidates.append(
                    source_path.parent.parent / "labels" / f"{stem}.txt"
                )

        seen: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return candidate
        return None

    def load_label(
        self, label_path: Path, image_shape: Sequence[int]
    ) -> LoadedAnnotations:
        height, width = _validate_image_shape(image_shape)
        try:
            text = Path(label_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise AnnotationError(f"Could not read label file: {exc}") from exc

        boxes = self.parse_yolo_text(text, source=str(label_path))
        rectangles = tuple(_box_to_rectangle(box, height, width) for box in boxes)
        class_ids = np.asarray([box.class_id for box in boxes], dtype=int)
        properties = {
            "class_id": class_ids,
            "class_name": np.asarray(
                [self.class_names[class_id] for class_id in class_ids], dtype=object
            ),
            "confidence": np.full(len(boxes), np.nan, dtype=float),
            "source": np.full(len(boxes), "import", dtype=object),
            "tile_id": np.full(len(boxes), -1, dtype=int),
        }
        return LoadedAnnotations(
            label_path=Path(label_path),
            boxes=boxes,
            rectangles=rectangles,
            properties=properties,
        )

    def parse_yolo_text(
        self, text: str, *, source: str = "label"
    ) -> tuple[BoundingBox, ...]:
        errors: list[str] = []
        boxes: list[BoundingBox] = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            prefix = f"{source}, line {line_number}"
            if len(fields) != 5:
                errors.append(f"{prefix}: expected 5 fields, found {len(fields)}.")
                continue
            try:
                class_id = int(fields[0])
            except ValueError:
                errors.append(f"{prefix}: class ID must be an integer.")
                continue
            try:
                values = tuple(float(value) for value in fields[1:])
            except ValueError:
                errors.append(f"{prefix}: bbox coordinates must be numbers.")
                continue
            if class_id not in self.class_names:
                errors.append(
                    f"{prefix}: class ID {class_id} is not defined by the project."
                )
                continue
            if not all(math.isfinite(value) for value in values):
                errors.append(f"{prefix}: bbox coordinates must be finite.")
                continue
            x_center, y_center, box_width, box_height = values
            if not all(0.0 <= value <= 1.0 for value in values):
                errors.append(
                    f"{prefix}: normalized coordinates must be within [0, 1]."
                )
                continue
            if box_width <= 0.0 or box_height <= 0.0:
                errors.append(f"{prefix}: width and height must be greater than zero.")
                continue
            if (
                x_center - box_width / 2.0 < -1e-6
                or x_center + box_width / 2.0 > 1.0 + 1e-6
                or y_center - box_height / 2.0 < -1e-6
                or y_center + box_height / 2.0 > 1.0 + 1e-6
            ):
                errors.append(
                    f"{prefix}: bbox extends outside the normalized image bounds."
                )
                continue
            boxes.append(
                BoundingBox(
                    class_id=class_id,
                    x_center=x_center,
                    y_center=y_center,
                    width=box_width,
                    height=box_height,
                )
            )

        if errors:
            raise LabelValidationError(errors)
        return tuple(boxes)

    def save_pair(
        self,
        *,
        image_data: np.ndarray,
        image_name: str,
        rectangles: Iterable[np.ndarray],
        class_ids: Sequence[int],
        source_path: Path | None = None,
        sample_id: str | None = None,
        conversion_metadata: dict[str, Any] | None = None,
    ) -> SaveResult:
        image = np.asarray(image_data)
        height, width = _validate_image_shape(image.shape)
        patch_contract = self.project.config.training_patch
        if patch_contract.get("locked"):
            patch_size = int(patch_contract["size"])
            if (height, width) != (patch_size, patch_size):
                raise AnnotationError(
                    f"Canonical training images must be {patch_size}×{patch_size}; "
                    f"received {height}×{width}."
                )
        identity = (
            sample_id
            if sample_id is not None
            else Path(source_path).stem
            if source_path
            else image_name
        )
        stem = self.safe_stem(identity)
        rectangles = tuple(np.asarray(rectangle, dtype=float) for rectangle in rectangles)
        class_ids = tuple(int(class_id) for class_id in class_ids)
        if len(rectangles) != len(class_ids):
            raise LabelValidationError(
                ["Each rectangle must have one class_id property before saving."]
            )

        boxes = tuple(
            self._rectangle_to_box(rectangle, class_id, height, width, index)
            for index, (rectangle, class_id) in enumerate(
                zip(rectangles, class_ids, strict=True), start=1
            )
        )
        label_text = _format_boxes(boxes)
        self.parse_yolo_text(label_text, source=f"generated label for {stem}")

        existing_images = sorted(
            path
            for path in self.project.paths.images.iterdir()
            if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(existing_images) > 1:
            raise AnnotationError(
                f"Multiple canonical images use stem '{stem}'. Validate the project first."
            )
        suffix = (
            existing_images[0].suffix.lower()
            if existing_images
            else _preferred_image_suffix(source_path, image)
        )
        image_path = self.project.paths.images / f"{stem}{suffix}"
        label_path = self.project.paths.labels / f"{stem}.txt"
        operation = "updated" if image_path.exists() or label_path.exists() else "created"

        image_temporary = _temporary_path(image_path)
        label_temporary = _temporary_path(label_path)
        try:
            _write_image(image, image_temporary, suffix)
            _write_text(label_temporary, label_text)
            _replace_pair_with_rollback(
                image_temporary=image_temporary,
                image_path=image_path,
                label_temporary=label_temporary,
                label_path=label_path,
            )
            self._append_audit(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "operation": operation,
                    "sample_id": stem,
                    "image": str(image_path.relative_to(self.project.paths.root)),
                    "label": str(label_path.relative_to(self.project.paths.root)),
                    "source_path": str(source_path) if source_path is not None else None,
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "conversion": conversion_metadata,
                    "box_count": len(boxes),
                    "class_counts": {
                        str(class_id): sum(box.class_id == class_id for box in boxes)
                        for class_id in sorted(set(class_ids))
                    },
                    "image_sha256": _sha256(image_path),
                    "label_sha256": _sha256(label_path),
                }
            )
        finally:
            image_temporary.unlink(missing_ok=True)
            label_temporary.unlink(missing_ok=True)

        return SaveResult(
            image_path=image_path,
            label_path=label_path,
            box_count=len(boxes),
            operation=operation,
        )

    def validate_project(self) -> ProjectValidationReport:
        images = [
            path
            for path in self.project.paths.images.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        labels = [path for path in self.project.paths.labels.glob("*.txt") if path.is_file()]
        errors: list[str] = []

        images_by_stem: dict[str, list[Path]] = {}
        for image in images:
            images_by_stem.setdefault(image.stem, []).append(image)
        labels_by_stem = {label.stem: label for label in labels}

        for stem, matches in sorted(images_by_stem.items()):
            if len(matches) > 1:
                errors.append(f"Duplicate canonical image stem '{stem}'.")
            if stem not in labels_by_stem:
                errors.append(f"Missing label for image '{matches[0].name}'.")
        for stem, label in sorted(labels_by_stem.items()):
            if stem not in images_by_stem:
                errors.append(f"Missing image for label '{label.name}'.")

        box_count = 0
        crop_extents = self._crop_valid_extents()
        for label in labels:
            try:
                boxes = self.parse_yolo_text(
                    label.read_text(encoding="utf-8"), source=str(label)
                )
                box_count += len(boxes)
                crop_extent = crop_extents.get(label.stem)
                if crop_extent is not None:
                    size, valid_height, valid_width = crop_extent
                    for index, box in enumerate(boxes, start=1):
                        x2 = (box.x_center + box.width / 2.0) * size
                        y2 = (box.y_center + box.height / 2.0) * size
                        if x2 > valid_width + 1e-6 or y2 > valid_height + 1e-6:
                            errors.append(
                                f"{label}, box {index}: bbox extends into "
                                "synthetic training-crop padding."
                            )
            except (OSError, LabelValidationError) as exc:
                errors.extend(str(exc).splitlines())

        for image in images:
            try:
                with Image.open(image) as image_file:
                    width, height = image_file.size
                    image_file.verify()
                patch_contract = self.project.config.training_patch
                if patch_contract.get("locked"):
                    patch_size = int(patch_contract["size"])
                    if (width, height) != (patch_size, patch_size):
                        errors.append(
                            f"Image '{image.name}' is {width}×{height}; project "
                            f"training images must be {patch_size}×{patch_size}."
                        )
            except (OSError, ValueError) as exc:
                errors.append(f"Unreadable image '{image.name}': {exc}")

        paired = len(set(images_by_stem).intersection(labels_by_stem))
        return ProjectValidationReport(
            image_count=len(images),
            label_count=len(labels),
            paired_count=paired,
            box_count=box_count,
            errors=tuple(errors),
        )

    def _crop_valid_extents(self) -> dict[str, tuple[int, int, int]]:
        extents: dict[str, tuple[int, int, int]] = {}
        try:
            lines = self.project.paths.audit.read_text(encoding="utf-8").splitlines()
        except OSError:
            return extents
        for raw_line in lines:
            try:
                event = json.loads(raw_line)
                sample_id = event.get("sample_id")
                crop = event.get("conversion", {}).get("training_crop")
                if not isinstance(sample_id, str) or not isinstance(crop, dict):
                    continue
                size = int(crop["size"])
                valid_height = int(crop["valid_height"])
                valid_width = int(crop["valid_width"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            extents[sample_id] = (size, valid_height, valid_width)
        return extents

    def _rectangle_to_box(
        self,
        rectangle: np.ndarray,
        class_id: int,
        height: int,
        width: int,
        index: int,
    ) -> BoundingBox:
        if class_id not in self.class_names:
            raise LabelValidationError(
                [f"Box {index}: class ID {class_id} is not defined by the project."]
            )
        if rectangle.ndim != 2 or rectangle.shape[1] != 2 or len(rectangle) < 2:
            raise LabelValidationError(
                [f"Box {index}: expected rectangle vertices in (row, column) coordinates."]
            )
        if not np.all(np.isfinite(rectangle)):
            raise LabelValidationError([f"Box {index}: coordinates must be finite."])

        y_min = float(np.min(rectangle[:, 0]))
        y_max = float(np.max(rectangle[:, 0]))
        x_min = float(np.min(rectangle[:, 1]))
        x_max = float(np.max(rectangle[:, 1]))
        tolerance = 1e-6
        if (
            x_min < -tolerance
            or y_min < -tolerance
            or x_max > width + tolerance
            or y_max > height + tolerance
        ):
            raise LabelValidationError([f"Box {index}: rectangle extends outside the image."])
        x_min = float(np.clip(x_min, 0, width))
        x_max = float(np.clip(x_max, 0, width))
        y_min = float(np.clip(y_min, 0, height))
        y_max = float(np.clip(y_max, 0, height))
        if x_max <= x_min or y_max <= y_min:
            raise LabelValidationError([f"Box {index}: rectangle has zero width or height."])

        return BoundingBox(
            class_id=class_id,
            x_center=((x_min + x_max) / 2.0) / width,
            y_center=((y_min + y_max) / 2.0) / height,
            width=(x_max - x_min) / width,
            height=(y_max - y_min) / height,
        )

    def _append_audit(self, event: dict[str, object]) -> None:
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


def _validate_image_shape(image_shape: Sequence[int]) -> tuple[int, int]:
    if len(image_shape) < 2:
        raise AnnotationError("Image data must have at least two dimensions.")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise AnnotationError("Image height and width must be greater than zero.")
    return height, width


def _box_to_rectangle(box: BoundingBox, height: int, width: int) -> np.ndarray:
    x_min = (box.x_center - box.width / 2.0) * width
    x_max = (box.x_center + box.width / 2.0) * width
    y_min = (box.y_center - box.height / 2.0) * height
    y_max = (box.y_center + box.height / 2.0) * height
    return np.asarray(
        [[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]],
        dtype=float,
    )


def _format_boxes(boxes: Sequence[BoundingBox]) -> str:
    return "".join(
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} "
        f"{box.width:.6f} {box.height:.6f}\n"
        for box in boxes
    )


def _preferred_image_suffix(source_path: Path | None, image: np.ndarray) -> str:
    if source_path is not None and Path(source_path).suffix.lower() in IMAGE_EXTENSIONS:
        suffix = Path(source_path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".webp"} and image.dtype != np.uint8:
            return ".tif"
        return suffix
    if image.dtype == np.uint8:
        return ".png"
    return ".tif"


def _temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    return Path(name)


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _write_image(image: np.ndarray, path: Path, suffix: str) -> None:
    if image.ndim == 3 and image.shape[-1] not in {1, 3, 4}:
        raise AnnotationError(
            "Saving currently supports 2D grayscale or 1/3/4-channel images. "
            "Project channel selection will be added in the image-processing milestone."
        )
    if image.ndim not in {2, 3}:
        raise AnnotationError("Saving currently supports 2D image planes only.")
    data = image[..., 0] if image.ndim == 3 and image.shape[-1] == 1 else image
    try:
        pil_image = Image.fromarray(data)
        format_name = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".tif": "TIFF",
            ".tiff": "TIFF",
        }.get(suffix, suffix.removeprefix(".").upper())
        pil_image.save(path, format=format_name)
        with path.open("rb+") as stream:
            os.fsync(stream.fileno())
    except Exception as exc:  # noqa: BLE001 - Pillow exposes encoder-specific errors
        raise AnnotationError(f"Could not encode annotation image: {exc}") from exc


def _replace_pair_with_rollback(
    *,
    image_temporary: Path,
    image_path: Path,
    label_temporary: Path,
    label_path: Path,
) -> None:
    token = uuid.uuid4().hex
    image_backup = image_path.with_name(f".{image_path.name}.{token}.bak")
    label_backup = label_path.with_name(f".{label_path.name}.{token}.bak")
    had_image = image_path.exists()
    had_label = label_path.exists()
    try:
        if had_image:
            os.replace(image_path, image_backup)
        if had_label:
            os.replace(label_path, label_backup)
        os.replace(image_temporary, image_path)
        os.replace(label_temporary, label_path)
    except Exception:  # noqa: BLE001 - rollback must cover any filesystem failure
        if image_path.exists():
            image_path.unlink()
        if label_path.exists():
            label_path.unlink()
        if image_backup.exists():
            os.replace(image_backup, image_path)
        if label_backup.exists():
            os.replace(label_backup, label_path)
        raise
    else:
        image_backup.unlink(missing_ok=True)
        label_backup.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
