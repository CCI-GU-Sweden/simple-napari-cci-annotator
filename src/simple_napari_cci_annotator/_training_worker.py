from __future__ import annotations

from threading import Event

from qtpy.QtCore import QThread, Signal

from ._dataset_builder import DatasetPreview
from ._project_store import ProjectStore
from ._segmentation_dataset import SegmentationDatasetPreview
from ._training import TrainingCancelled, TrainingService, TrainingSettings


class TrainingWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(
        self,
        project: ProjectStore,
        settings: TrainingSettings,
        preview: DatasetPreview | SegmentationDatasetPreview,
    ):
        super().__init__()
        self._project = project
        self._settings = settings
        self._preview = preview
        self._cancel_event = Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = TrainingService(self._project).run(
                self._settings,
                self._preview,
                progress=self.progress.emit,
                cancelled=self._cancel_event.is_set,
            )
        except TrainingCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)
