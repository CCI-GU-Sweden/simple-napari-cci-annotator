"""Qt worker for sequential TIFF batch prediction."""

from __future__ import annotations

from threading import Event

from qtpy.QtCore import QThread, Signal

from ._batch_inference import BatchInferenceSettings, TiffBatchProcessor
from ._project_store import ProjectStore


class BatchInferenceWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, project: ProjectStore, model, settings: BatchInferenceSettings):
        super().__init__()
        self._processor = TiffBatchProcessor(project, model, settings)
        self._cancel_event = Event()

    def validate(self) -> int:
        return len(self._processor.validate())

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = self._processor.run(
                progress=self.progress.emit,
                cancelled=self._cancel_event.is_set,
            )
        except Exception as exc:  # QThread reports file/model failures to the UI
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)
