"""Qt signals for the headless volume inference runner."""

from pathlib import Path
from threading import Event

from qtpy.QtCore import QThread, Signal

from ._volume_inference import VolumeInferenceRunner


class VolumeInferenceWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    cancelled = Signal(object)
    failed = Signal(str)

    def __init__(
        self, runner: VolumeInferenceRunner, run_root: Path, *, resume=False
    ):
        super().__init__()
        self._runner = runner
        self._run_root = Path(run_root)
        self._resume = resume
        self._cancel_event = Event()

    def validate(self) -> int:
        return self._runner.validate()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = self._runner.run(
                self._run_root, resume=self._resume,
                progress=self.progress.emit,
                cancelled=self._cancel_event.is_set,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            if result.status == "cancelled":
                self.cancelled.emit(result)
            else:
                self.succeeded.emit(result)
