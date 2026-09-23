from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from ._annotation_io import IMAGE_EXTENSIONS, AnnotationError, AnnotationIO
from ._project_store import ProjectStore


@dataclass(frozen=True)
class AnnotationReviewEntry:
    sample_id: str
    image_path: Path | None
    label_path: Path | None
    box_count: int
    errors: tuple[str, ...]
    source_path: Path | None
    conversion: dict[str, Any]

    @property
    def is_valid(self) -> bool:
        return not self.errors


class AnnotationBrowser:
    """Enumerate and load canonical image/label pairs for review."""

    def __init__(self, project: ProjectStore):
        self.project = project
        self.annotation_io = AnnotationIO(project)

    def entries(self) -> tuple[AnnotationReviewEntry, ...]:
        images: dict[str, list[Path]] = {}
        for path in self.project.paths.images.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.setdefault(path.stem, []).append(path)
        labels = {
            path.stem: path
            for path in self.project.paths.labels.glob("*.txt")
            if path.is_file()
        }
        audit = self._latest_audit_events()
        entries: list[AnnotationReviewEntry] = []
        for sample_id in sorted(set(images).union(labels)):
            image_matches = images.get(sample_id, [])
            label_path = labels.get(sample_id)
            errors: list[str] = []
            image_path = image_matches[0] if len(image_matches) == 1 else None
            if not image_matches:
                errors.append("missing image")
            elif len(image_matches) > 1:
                errors.append("duplicate image stem")
            if label_path is None:
                errors.append("missing label")

            box_count = 0
            if image_path is not None and label_path is not None:
                try:
                    with Image.open(image_path) as image_file:
                        width, height = image_file.size
                        image_file.verify()
                    loaded = self.annotation_io.load_label(
                        label_path, (height, width)
                    )
                    box_count = len(loaded.boxes)
                except (AnnotationError, OSError, ValueError) as exc:
                    errors.extend(str(exc).splitlines())

            event = audit.get(sample_id, {})
            raw_source = event.get("source_path")
            source_path = Path(raw_source) if isinstance(raw_source, str) else None
            conversion = event.get("conversion", {})
            entries.append(
                AnnotationReviewEntry(
                    sample_id=sample_id,
                    image_path=image_path,
                    label_path=label_path,
                    box_count=box_count,
                    errors=tuple(errors),
                    source_path=source_path,
                    conversion=(
                        dict(conversion) if isinstance(conversion, dict) else {}
                    ),
                )
            )
        return tuple(entries)

    @staticmethod
    def load_rgb(entry: AnnotationReviewEntry) -> np.ndarray:
        if entry.image_path is None:
            raise OSError(f"Annotation {entry.sample_id!r} has no unique image.")
        with Image.open(entry.image_path) as image_file:
            return np.asarray(image_file.convert("RGB"), dtype=np.uint8)

    def _latest_audit_events(self) -> dict[str, dict[str, Any]]:
        events: dict[str, dict[str, Any]] = {}
        try:
            lines = self.project.paths.audit.read_text(
                encoding="utf-8"
            ).splitlines()
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
                events[sample_id] = {**previous, **event}
        return events


def invalid_rectangle_indices(
    rectangles: Iterable[np.ndarray], *, height: int, width: int
) -> tuple[int, ...]:
    """Return zero-based indices of malformed or out-of-image rectangles."""
    invalid: list[int] = []
    for index, value in enumerate(rectangles):
        rectangle = np.asarray(value, dtype=float)
        if (
            rectangle.ndim != 2
            or rectangle.shape[1:] != (2,)
            or len(rectangle) < 2
            or not np.all(np.isfinite(rectangle))
        ):
            invalid.append(index)
            continue
        y0 = float(np.min(rectangle[:, 0]))
        y1 = float(np.max(rectangle[:, 0]))
        x0 = float(np.min(rectangle[:, 1]))
        x1 = float(np.max(rectangle[:, 1]))
        if (
            x0 < -1e-6
            or y0 < -1e-6
            or x1 > width + 1e-6
            or y1 > height + 1e-6
            or x1 <= x0
            or y1 <= y0
        ):
            invalid.append(index)
    return tuple(invalid)
