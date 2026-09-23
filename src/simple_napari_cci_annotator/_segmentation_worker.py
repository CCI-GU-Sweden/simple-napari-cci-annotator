from __future__ import annotations

from threading import Event

import numpy as np
from qtpy.QtCore import QThread, Signal


class SegmentationWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, predictor, image: np.ndarray, settings, classes):
        super().__init__()
        self._predictor = predictor
        self._image = np.asarray(image)
        self._settings = settings
        self._classes = dict(classes)
        self._cancel_event = Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        if self._cancel_event.is_set():
            self.cancelled.emit()
            return
        try:
            result = self._predictor.predict_image(
                self._image, self._settings, self._classes
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        if self._cancel_event.is_set():
            self.cancelled.emit()
        else:
            self.succeeded.emit(result)
