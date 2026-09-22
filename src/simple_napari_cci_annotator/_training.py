from __future__ import annotations

import hashlib
import importlib.metadata
import os
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ._dataset_builder import (
    DatasetBuildCancelled,
    DatasetBuildSettings,
    DatasetBuilder,
    DatasetPreview,
)
from ._project_store import ProjectStore
from ._version import __version__


class TrainingError(RuntimeError):
    """Raised when a retraining run cannot complete."""


class TrainingCancelled(TrainingError):
    """Raised after a cooperative training cancellation request."""


@dataclass(frozen=True)
class TrainingSettings:
    model_path: Path
    destination: Path
    dataset: DatasetBuildSettings
    epochs: int = 100
    batch: int = -1
    patience: int = 30
    device: str | int = "cpu"
    train_only: bool = False
    deterministic: bool = True

    def validate(self) -> None:
        self.dataset.validate()
        if not Path(self.model_path).is_file():
            raise TrainingError(f"Training model does not exist: {self.model_path}")
        if self.epochs < 1:
            raise TrainingError("Epochs must be at least 1.")
        if self.batch != -1 and self.batch < 1:
            raise TrainingError("Batch must be Auto or at least 1.")
        if self.patience < 0:
            raise TrainingError("Patience cannot be negative.")


@dataclass(frozen=True)
class TrainingRun:
    run_root: Path
    best_model: Path | None
    last_model: Path | None
    status: str
    validation_mode: str


class TrainingService:
    """Create a frozen dataset run and train an Ultralytics detector."""

    def __init__(self, project: ProjectStore):
        self.project = project

    def run(
        self,
        settings: TrainingSettings,
        preview: DatasetPreview,
        *,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> TrainingRun:
        settings.validate()
        if not preview.is_valid:
            raise TrainingError("Dataset validation must pass before retraining.")
        run_root = _create_run_root(settings.destination)
        started = datetime.now(timezone.utc).isoformat()
        metadata = self._initial_metadata(settings, preview, started)
        _atomic_write_yaml(run_root / "run.yaml", metadata)
        (run_root / "README.txt").write_text(
            "Immutable CCI YOLO retraining run.\n"
            "dataset/ contains the exact derived training snapshot.\n"
            "training/ contains Ultralytics outputs. See run.yaml for provenance.\n",
            encoding="utf-8",
            newline="\n",
        )
        try:
            snapshot = DatasetBuilder(self.project).create_snapshot(
                run_root,
                preview,
                settings.dataset,
                progress=progress,
                cancelled=cancelled,
            )
            metadata["dataset_snapshot"] = {
                "image_count": snapshot.image_count,
                "label_count": snapshot.label_count,
                "rejected_box_count": snapshot.rejected_box_count,
                "dataset_yaml": str(snapshot.dataset_yaml.relative_to(run_root)),
                "split_manifest": str(snapshot.split_manifest.relative_to(run_root)),
                "tile_manifest": str(snapshot.tile_manifest.relative_to(run_root)),
            }
            metadata["status"] = "training"
            _atomic_write_yaml(run_root / "run.yaml", metadata)
            if cancelled is not None and cancelled():
                raise TrainingCancelled("Retraining cancelled before model startup.")
            model = _load_ultralytics_model(settings.model_path)
            task = str(getattr(model, "task", "detect") or "detect")
            if task != "detect":
                raise TrainingError(
                    f"Retraining currently supports detection models, not task {task!r}."
                )

            def on_train_epoch_end(trainer) -> None:
                epoch = int(getattr(trainer, "epoch", 0)) + 1
                metrics = getattr(trainer, "metrics", {}) or {}
                metric_text = _metric_summary(metrics)
                text = f"Epoch {epoch}/{settings.epochs}"
                if metric_text:
                    text += f" · {metric_text}"
                if progress is not None:
                    progress(epoch, settings.epochs, text)
                if cancelled is not None and cancelled():
                    trainer.stop = True

            def disable_validation(trainer) -> None:
                # Ultralytics validates on the final epoch even when val=False.
                # An exploratory run must not emit metrics against its train set.
                trainer.validate = lambda: ({}, 0.0)
                trainer.final_eval = lambda: None
                trainer.metrics = {}

            if hasattr(model, "add_callback"):
                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                if settings.train_only:
                    model.add_callback(
                        "on_pretrain_routine_end", disable_validation
                    )
            if progress is not None:
                progress(0, settings.epochs, "Starting Ultralytics training")
            model.train(
                data=str(snapshot.dataset_yaml),
                imgsz=settings.dataset.tile_size,
                epochs=settings.epochs,
                batch=settings.batch,
                patience=settings.patience,
                device=settings.device,
                seed=settings.dataset.seed,
                deterministic=settings.deterministic,
                val=not settings.train_only,
                project=str(run_root),
                name="training",
                exist_ok=True,
                workers=0,
                plots=not settings.train_only,
            )
            if cancelled is not None and cancelled():
                raise TrainingCancelled("Retraining cancelled by the user.")
            best = run_root / "training" / "weights" / "best.pt"
            last = run_root / "training" / "weights" / "last.pt"
            if not best.is_file() and not last.is_file():
                raise TrainingError(
                    "Ultralytics returned without producing best.pt or last.pt."
                )
            metadata["status"] = "completed"
            metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            metadata["outputs"] = {
                "best_model": str(best.relative_to(run_root)) if best.is_file() else None,
                "last_model": str(last.relative_to(run_root)) if last.is_file() else None,
            }
            _atomic_write_yaml(run_root / "run.yaml", metadata)
            return TrainingRun(
                run_root=run_root,
                best_model=best if best.is_file() else None,
                last_model=last if last.is_file() else None,
                status="completed",
                validation_mode=preview.validation_mode,
            )
        except (DatasetBuildCancelled, TrainingCancelled) as exc:
            metadata["status"] = "cancelled"
            metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            metadata["failure"] = str(exc)
            _atomic_write_yaml(run_root / "run.yaml", metadata)
            raise TrainingCancelled(str(exc)) from exc
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            metadata["failure"] = str(exc)
            (run_root / "failure.txt").write_text(
                traceback.format_exc(), encoding="utf-8", newline="\n"
            )
            _atomic_write_yaml(run_root / "run.yaml", metadata)
            if isinstance(exc, TrainingError):
                raise
            raise TrainingError(str(exc)) from exc

    def _initial_metadata(
        self,
        settings: TrainingSettings,
        preview: DatasetPreview,
        started: str,
    ) -> dict[str, Any]:
        model_path = Path(settings.model_path).resolve()
        return {
            "run_schema_version": 1,
            "status": "preparing_dataset",
            "started_at": started,
            "project_root": str(self.project.paths.root),
            "plugin_version": _package_version(),
            "packages": _runtime_versions(),
            "input_model": {
                "path": str(model_path),
                "sha256": _sha256(model_path),
            },
            "training": {
                "epochs": settings.epochs,
                "batch": settings.batch,
                "patience": settings.patience,
                "device": str(settings.device),
                "deterministic": settings.deterministic,
                "validation_mode": preview.validation_mode,
                "validation_enabled": not settings.train_only,
                "train_loader_reused_for_disabled_val_loader": settings.train_only,
            },
            "dataset_settings": asdict(settings.dataset),
            "split": {
                "assignments": dict(sorted(preview.assignments.items())),
                "source_counts": preview.source_counts,
                "tile_counts": preview.tile_counts,
                "box_counts": preview.box_counts,
                "warnings": list(preview.warnings),
            },
        }


def _create_run_root(destination: Path) -> Path:
    parent = Path(destination).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = parent / f"retrain_{timestamp}"
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = parent / f"{base.name}_{counter:02d}"
        counter += 1
    candidate.mkdir()
    return candidate


def _load_ultralytics_model(path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainingError("Ultralytics is required for retraining.") from exc
    try:
        return YOLO(str(path))
    except Exception as exc:
        raise TrainingError(f"Could not load training model: {exc}") from exc


def _metric_summary(metrics: dict[str, Any]) -> str:
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return f"{key.split('/')[-1]} {value:.3f}"
    return ""


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("ultralytics", "torch", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _package_version() -> str:
    return __version__


def _atomic_write_yaml(path: Path, value: dict[str, Any]) -> None:
    text = yaml.safe_dump(
        value,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
