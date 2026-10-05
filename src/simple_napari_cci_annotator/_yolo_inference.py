from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._tiled_inference import InferenceError, InferenceSettings, RawDetection


class YoloDetectionModel:
    """Small, lazy Ultralytics adapter used by tiled inference."""

    def __init__(self, model_path: str | Path):
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise InferenceError(f"Model file does not exist: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise InferenceError(
                "Ultralytics is required for YOLO prediction."
            ) from exc
        try:
            self.model = YOLO(str(path))
        except Exception as exc:  # third-party model loaders raise varied exceptions
            raise InferenceError(f"Could not load YOLO model: {exc}") from exc
        task = str(getattr(self.model, "task", "detect") or "detect")
        if task != "detect":
            raise InferenceError(
                f"This phase supports YOLO detection models; loaded task is {task!r}."
            )
        self.path = path
        self.task = task
        self.names = _normalise_names(getattr(self.model, "names", {}))

    def predict_tile(
        self, image: np.ndarray, settings: InferenceSettings
    ) -> tuple[RawDetection, ...]:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise InferenceError("A YOLO tile must have three RGB channels.")
        # Ultralytics treats NumPy sources as OpenCV BGR and flips them during
        # preprocessing. Our project images are explicitly RGB, so present BGR
        # here to preserve the configured red/green/blue channel semantics.
        bgr_image = np.ascontiguousarray(array[..., ::-1])
        try:
            results = self.model.predict(
                source=bgr_image,
                imgsz=settings.tile_size,
                conf=settings.confidence,
                iou=settings.model_iou,
                max_det=settings.max_detections,
                device=settings.device,
                verbose=False,
            )
        except Exception as exc:  # preserve a useful boundary around Ultralytics
            raise InferenceError(f"YOLO prediction failed: {exc}") from exc
        if not results:
            return ()
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return ()
        xyxy = _to_numpy(boxes.xyxy)
        confidence = _to_numpy(boxes.conf).reshape(-1)
        class_ids = _to_numpy(boxes.cls).reshape(-1)
        if not (len(xyxy) == len(confidence) == len(class_ids)):
            raise InferenceError("YOLO returned inconsistent box metadata.")
        return tuple(
            RawDetection(
                x1=float(coords[0]),
                y1=float(coords[1]),
                x2=float(coords[2]),
                y2=float(coords[3]),
                confidence=float(score),
                class_id=int(class_id),
            )
            for coords, score, class_id in zip(
                xyxy, confidence, class_ids, strict=True
            )
        )


def available_devices() -> tuple[tuple[str, str | int], ...]:
    """Return user-facing device choices and Ultralytics device values."""
    choices: list[tuple[str, str | int]] = [("CPU", "cpu")]
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index)
                choices.insert(index, (f"CUDA:{index} · {name}", index))
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            choices.insert(0, ("Apple MPS", "mps"))
    except (ImportError, RuntimeError):
        pass
    return tuple(choices)


def _to_numpy(value: Any) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.asarray(cpu_value)


def _normalise_names(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(key): str(name) for key, name in value.items()}
    if isinstance(value, (list, tuple)):
        return {index: str(name) for index, name in enumerate(value)}
    return {}
