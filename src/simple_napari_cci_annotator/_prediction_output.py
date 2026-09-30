"""Save full-size prediction results outside the canonical annotation pool."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import tifffile

from ._annotation_io import AnnotationIO, LabelValidationError, _format_boxes
from ._instance_mask import disconnected_instance_ids
from ._project_store import ProjectStore
from ._segmentation_io import (
    InstanceRecord,
    SegmentationError,
    _replace_many,
    _temporary_path,
    _write_json,
    refresh_instance_records,
)


class PredictionOutputError(RuntimeError):
    """Raised when a full-image output cannot be validated or written."""


@dataclass(frozen=True)
class PredictionOutputPaths:
    folder: Path
    image: Path
    annotation: Path
    metadata: Path

    @property
    def files(self) -> tuple[Path, Path, Path]:
        return self.image, self.annotation, self.metadata


class PredictionOutputWriter:
    """Write a task-aware RGB image and its editable prediction result."""

    def __init__(self, project: ProjectStore):
        self.project = project

    def paths(self, parent: Path, sample_id: str) -> PredictionOutputPaths:
        stem = AnnotationIO.safe_stem(sample_id)
        folder = Path(parent) / self.project.config.task / stem
        annotation_name = (
            "mask.tif" if self.project.config.task == "segment" else "labels.txt"
        )
        metadata_name = (
            "instances.json"
            if self.project.config.task == "segment"
            else "result.json"
        )
        return PredictionOutputPaths(
            folder=folder,
            image=folder / "image.png",
            annotation=folder / annotation_name,
            metadata=folder / metadata_name,
        )

    def save_detection(
        self,
        parent: Path,
        sample_id: str,
        image: np.ndarray,
        rectangles: Sequence[np.ndarray],
        class_ids: Sequence[int],
        *,
        confidences: Sequence[float | None] = (),
        sources: Sequence[str] = (),
        tile_ids: Sequence[int] = (),
        source_path: Path | None = None,
        conversion: Mapping[str, Any] | None = None,
        prediction: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> PredictionOutputPaths:
        if self.project.config.task != "detect":
            raise PredictionOutputError("Detection output requires a detect project.")
        rgb = _validate_rgb(image)
        count = len(rectangles)
        if len(class_ids) != count:
            raise PredictionOutputError("Each box needs one class ID.")
        for name, values in (
            ("confidence", confidences),
            ("source", sources),
            ("tile ID", tile_ids),
        ):
            if len(values) not in (0, count):
                raise PredictionOutputError(f"Each box needs one {name} value.")
        annotation_io = AnnotationIO(self.project)
        height, width = rgb.shape[:2]
        try:
            boxes = tuple(
                annotation_io._rectangle_to_box(
                    np.asarray(rectangle, dtype=float),
                    int(class_id),
                    height,
                    width,
                    index,
                )
                for index, (rectangle, class_id) in enumerate(
                    zip(rectangles, class_ids, strict=True), start=1
                )
            )
            label_text = _format_boxes(boxes)
            annotation_io.parse_yolo_text(label_text, source="prediction output")
        except LabelValidationError as exc:
            raise PredictionOutputError(str(exc)) from exc
        rows = []
        for index, box in enumerate(boxes):
            confidence = confidences[index] if len(confidences) else None
            if confidence is not None and not np.isfinite(confidence):
                confidence = None
            elif confidence is not None:
                confidence = float(confidence)
                if not 0 <= confidence <= 1:
                    raise PredictionOutputError(
                        f"Box {index + 1} has confidence outside 0..1."
                    )
            rows.append(
                {
                    "class_id": box.class_id,
                    "xywh": [box.x_center, box.y_center, box.width, box.height],
                    "confidence": confidence,
                    "source": str(sources[index]) if len(sources) else None,
                    "tile_id": int(tile_ids[index]) if len(tile_ids) else None,
                }
            )
        paths = self.paths(parent, sample_id)
        payload = self._base_payload(
            sample_id, rgb, source_path, conversion, prediction
        )
        payload["boxes"] = rows
        self._write(paths, rgb, label_text, payload, overwrite=overwrite)
        return paths

    def save_segmentation(
        self,
        parent: Path,
        sample_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        instances: Mapping[int, InstanceRecord],
        *,
        source_path: Path | None = None,
        conversion: Mapping[str, Any] | None = None,
        prediction: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> PredictionOutputPaths:
        if self.project.config.task != "segment":
            raise PredictionOutputError("Mask output requires a segment project.")
        rgb = _validate_rgb(image)
        array = np.asarray(mask)
        if array.shape != rgb.shape[:2]:
            raise PredictionOutputError("Mask and image dimensions must match.")
        if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
            raise PredictionOutputError("Mask IDs must be non-negative integers.")
        if np.any(array > np.iinfo(np.uint32).max):
            raise PredictionOutputError("Mask IDs exceed uint32 storage capacity.")
        array = np.asarray(array, dtype=np.uint32)
        present = {int(value) for value in np.unique(array) if value}
        if present != set(instances):
            raise PredictionOutputError(
                "Instance metadata IDs must exactly match the IDs in the mask."
            )
        disconnected = disconnected_instance_ids(array)
        if disconnected:
            raise PredictionOutputError(
                f"Disconnected instance IDs must be corrected: {list(disconnected)}."
            )
        try:
            records = refresh_instance_records(
                array, instances, self.project.config.classes
            )
        except SegmentationError as exc:
            raise PredictionOutputError(str(exc)) from exc
        paths = self.paths(parent, sample_id)
        payload = self._base_payload(
            sample_id, rgb, source_path, conversion, prediction
        )
        payload["mask_dtype"] = "uint32"
        payload["instances"] = {
            str(instance_id): record.to_mapping()
            for instance_id, record in sorted(records.items())
        }
        self._write(paths, rgb, array, payload, overwrite=overwrite)
        return paths

    def _base_payload(
        self,
        sample_id: str,
        image: np.ndarray,
        source_path: Path | None,
        conversion: Mapping[str, Any] | None,
        prediction: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": self.project.config.task,
            "sample_id": AnnotationIO.safe_stem(sample_id),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "image_shape": list(image.shape),
            "classes": {
                str(class_id): name
                for class_id, name in sorted(self.project.config.classes.items())
            },
            "source_path": str(source_path) if source_path else None,
            "conversion": _json_ready(conversion),
            "prediction": _json_ready(prediction),
        }

    @staticmethod
    def _write(
        paths: PredictionOutputPaths,
        image: np.ndarray,
        annotation: str | np.ndarray,
        metadata: dict[str, Any],
        *,
        overwrite: bool,
    ) -> None:
        if not overwrite and any(path.exists() for path in paths.files):
            raise PredictionOutputError(
                f"Output already exists in {paths.folder}. Confirm overwrite first."
            )
        paths.folder.mkdir(parents=True, exist_ok=True)
        temporary = tuple(_temporary_path(path) for path in paths.files)
        try:
            Image.fromarray(image, mode="RGB").save(temporary[0], format="PNG")
            if isinstance(annotation, str):
                with temporary[1].open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(annotation)
            else:
                tifffile.imwrite(
                    temporary[1],
                    annotation,
                    compression="deflate",
                    photometric="minisblack",
                )
            _write_json(temporary[2], metadata)
            _replace_many(tuple(zip(temporary, paths.files, strict=True)))
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise PredictionOutputError("Output images must be RGB uint8 planes.")
    if rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        raise PredictionOutputError("Output images must have positive dimensions.")
    return rgb


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_ready(item) for item in sorted(value, key=repr)]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PredictionOutputError(
            "Prediction metadata contains an unsupported value: "
            f"{type(value).__name__}."
        ) from exc
    return value
