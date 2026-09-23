from __future__ import annotations

from threading import Event

import numpy as np
from qtpy.QtCore import QThread, Signal

from ._segmentation_tiling import TiledSegmentationEngine


class SegmentationWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        predictor,
        image: np.ndarray,
        settings,
        classes,
        *,
        clear_border_instances: bool = False,
    ):
        super().__init__()
        self._predictor = predictor
        self._image = np.asarray(image)
        self._settings = settings
        self._classes = dict(classes)
        self._clear_border_instances = bool(clear_border_instances)
        self._cancel_event = Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        if self._cancel_event.is_set():
            self.cancelled.emit()
            return
        try:
            if any(
                size > self._settings.tile_size for size in self._image.shape[:2]
            ):
                result = TiledSegmentationEngine(self._predictor).predict(
                    self._image,
                    self._settings,
                    self._classes,
                    progress=self.progress.emit,
                    cancelled=self._cancel_event.is_set,
                    clear_border_instances=self._clear_border_instances,
                )
            else:
                self.progress.emit(0, 1, "Predicting image")
                result = self._predictor.predict_image(
                    self._image, self._settings, self._classes
                )
                self.progress.emit(1, 1, "Prediction complete")
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        if self._cancel_event.is_set():
            self.cancelled.emit()
        else:
            self.succeeded.emit(result)
