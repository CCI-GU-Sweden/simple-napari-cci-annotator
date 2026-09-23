from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ._instance_mask import ComposedInstances, PredictedInstance, compose_predictions
from ._tiled_inference import InferenceError, InferenceSettings
from ._yolo_inference import _normalise_names, _to_numpy


class YoloSegmentationModel:
    """Ultralytics segmentation adapter for one source-resolution RGB image."""

    def __init__(self, model_path: str | Path):
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise InferenceError(f"Model file does not exist: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise InferenceError("Ultralytics is required for YOLO prediction.") from exc
        try:
            self.model = YOLO(str(path))
        except Exception as exc:
            raise InferenceError(f"Could not load YOLO model: {exc}") from exc
        self.task = str(getattr(self.model, "task", "") or "")
        if self.task != "segment":
            raise InferenceError(
                f"A segmentation project requires YOLO segmentation weights; "
                f"loaded task is {self.task!r}."
            )
        self.path = path
        self.names = _normalise_names(getattr(self.model, "names", {}))

    def predict_image(
        self,
        image: np.ndarray,
        settings: InferenceSettings,
        classes: dict[int, str],
    ) -> ComposedInstances:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise InferenceError("Segmentation inference requires an RGB uint8 image.")
        bgr = np.ascontiguousarray(array[..., ::-1])
        try:
            results = self.model.predict(
                source=bgr,
                imgsz=settings.tile_size,
                conf=settings.confidence,
                iou=settings.model_iou,
                max_det=settings.max_detections,
                device=settings.device,
                retina_masks=True,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceError(f"YOLO segmentation prediction failed: {exc}") from exc
        if not results:
            return compose_predictions((), array.shape[:2], classes)
        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None or len(boxes) == 0:
            return compose_predictions((), array.shape[:2], classes)
        mask_data = _to_numpy(masks.data)
        xyxy = _to_numpy(boxes.xyxy)
        confidence = _to_numpy(boxes.conf).reshape(-1)
        class_ids = _to_numpy(boxes.cls).reshape(-1)
        if not (len(mask_data) == len(xyxy) == len(confidence) == len(class_ids)):
            raise InferenceError("YOLO returned inconsistent segmentation metadata.")
        predictions = []
        height, width = array.shape[:2]
        for raw_mask, coords, score, class_id in zip(
            mask_data, xyxy, confidence, class_ids, strict=True
        ):
            binary = np.asarray(raw_mask) > 0.5
            if binary.shape != (height, width):
                binary = np.asarray(
                    Image.fromarray(binary.astype(np.uint8)).resize(
                        (width, height), resample=Image.Resampling.NEAREST
                    ),
                    dtype=bool,
                )
            predictions.append(
                PredictedInstance(
                    mask=binary,
                    bbox=tuple(float(value) for value in coords),
                    class_id=int(class_id),
                    confidence=float(score),
                )
            )
        return compose_predictions(predictions, (height, width), classes)
