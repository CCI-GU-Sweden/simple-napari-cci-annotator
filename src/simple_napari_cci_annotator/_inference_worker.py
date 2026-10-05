from __future__ import annotations

from threading import Event

import numpy as np
from qtpy.QtCore import QThread, Signal

from ._tiled_inference import (
    InferenceCancelled,
    InferenceSettings,
    TiledInferenceEngine,
)


class InferenceWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, predictor, image: np.ndarray, settings: InferenceSettings):
        super().__init__()
        self._predictor = predictor
        self._image = np.asarray(image)
        self._settings = settings
        self._cancel_event = Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = TiledInferenceEngine(self._predictor).predict(
                self._image,
                self._settings,
                progress=self.progress.emit,
                cancelled=self._cancel_event.is_set,
            )
        except InferenceCancelled:
            self.cancelled.emit()
        except Exception as exc:  # QThread must deliver third-party failures to the GUI
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)
