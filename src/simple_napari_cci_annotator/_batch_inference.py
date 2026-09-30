"""Sequential TIFF batch prediction with reproducible run metadata."""

from __future__ import annotations

import hashlib
import itertools
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from uuid import uuid4

import numpy as np
import tifffile

from ._image_adapter import ImageAdapter, ImageProcessingSettings
from ._prediction_output import PredictionOutputWriter
from ._project_store import ProjectStore
from ._segmentation_io import _temporary_path, _write_json
from ._segmentation_tiling import TiledSegmentationEngine
from ._tiled_inference import (
    InferenceCancelled,
    InferenceSettings,
    TiledInferenceEngine,
)
from ._version import __version__


class BatchInferenceError(RuntimeError):
    """Raised when a batch cannot be started."""


@dataclass(frozen=True)
class BatchInferenceSettings:
    input_folder: Path
    output_parent: Path
    inference: InferenceSettings
    invert: bool = False
    clear_border_instances: bool = False


@dataclass(frozen=True)
class BatchInferenceRun:
    run_root: Path
    status: str
    completed: int
    failed: int
    processed_planes: int


def find_tiff_inputs(folder: Path) -> tuple[Path, ...]:
    """List only top-level TIFF files, including OME-TIFF variants."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise BatchInferenceError(f"Input folder does not exist: {root}")
    files = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
            ),
            key=lambda path: path.name.lower(),
        )
    )
    if not files:
        raise BatchInferenceError("The input folder has no TIFF or OME-TIFF files.")
    return files


class TiffBatchProcessor:
    """Run the locked project conversion and one selected model on every TIFF."""

    def __init__(self, project: ProjectStore, model, settings: BatchInferenceSettings):
        self.project = project
        self.model = model
        self.settings = settings
        self.adapter = ImageAdapter()
        self.writer = PredictionOutputWriter(project)

    def validate(self) -> tuple[Path, ...]:
        if not self.project.config.image_processing.get("locked"):
            raise BatchInferenceError(
                "Lock project image processing by saving a training annotation first."
            )
        ImageProcessingSettings.from_mapping(self.project.config.image_processing)
        self.settings.inference.validate()
        if getattr(self.model, "task", None) != self.project.config.task:
            raise BatchInferenceError("The loaded model task does not match the project.")
        model_classes = set(getattr(self.model, "names", {}))
        if model_classes and model_classes != set(self.project.config.classes):
            raise BatchInferenceError("The loaded model class IDs do not match the project.")
        model_path = Path(getattr(self.model, "path", ""))
        if not model_path.is_file():
            raise BatchInferenceError("The selected model file is unavailable.")
        if (
            self.project.config.task == "segment"
            and self.settings.inference.overlap * 2 >= self.settings.inference.tile_size
        ):
            raise BatchInferenceError(
                "Segmentation overlap must be smaller than half the tile size."
            )
        if Path(self.settings.output_parent).expanduser().resolve().is_relative_to(
            self.project.paths.annotations.resolve()
        ):
            raise BatchInferenceError(
                "Choose a batch output folder outside project annotations."
            )
        return find_tiff_inputs(self.settings.input_folder)

    def run(
        self,
        *,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> BatchInferenceRun:
        files = self.validate()
        processing = ImageProcessingSettings.from_mapping(
            self.project.config.image_processing
        )
        output_parent = Path(self.settings.output_parent).expanduser().resolve()
        output_parent.mkdir(parents=True, exist_ok=True)
        run_root = output_parent / (
            datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_")
            + uuid4().hex[:8]
        )
        run_root.mkdir()
        model_path = Path(self.model.path).resolve()
        metadata = {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "plugin_version": __version__,
            "input_folder": str(Path(self.settings.input_folder).resolve()),
            "output_folder": str(run_root),
            "project_root": str(self.project.paths.root),
            "project_config_sha256": _sha256(self.project.paths.config),
            "project_task": self.project.config.task,
            "classes": {
                str(key): value for key, value in self.project.config.classes.items()
            },
            "image_processing": self.project.config.image_processing,
            "batch_invert": self.settings.invert,
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "inference": asdict(self.settings.inference),
            "clear_border_instances": self.settings.clear_border_instances,
            "plane_policy": "all_non_spatial_planes",
            "files": [],
        }
        _save_metadata(run_root, metadata)
        stems = _unique_stems(files)
        completed = failed = 0
        for file_index, path in enumerate(files, start=1):
            if cancelled is not None and cancelled():
                metadata["status"] = "cancelled"
                break
            if progress is not None:
                progress(file_index - 1, len(files), f"Opening {path.name}")
            entry = {
                "source": path.name,
                "source_sha256": None,
                "status": "running",
                "planes": [],
                "error": None,
            }
            metadata["files"].append(entry)
            _save_metadata(run_root, metadata)
            try:
                entry["source_sha256"] = _sha256(path)
                self._process_file(
                    path,
                    stems[path],
                    processing,
                    run_root,
                    metadata,
                    entry,
                    file_index,
                    len(files),
                    progress,
                    cancelled,
                )
            except InferenceCancelled:
                entry["status"] = "cancelled"
                metadata["status"] = "cancelled"
                _save_metadata(run_root, metadata)
                break
            except Exception as exc:  # one unreadable image must not stop the batch
                entry["status"] = "failed"
                entry["error"] = str(exc)
                failed += 1
            else:
                entry["status"] = "completed"
                completed += 1
            _save_metadata(run_root, metadata)
            if progress is not None:
                progress(file_index, len(files), f"Finished {path.name}")
        processed_planes = sum(len(entry["planes"]) for entry in metadata["files"])
        if metadata["status"] != "cancelled":
            metadata["status"] = (
                "completed_with_errors" if failed else "completed"
            )
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_metadata(run_root, metadata)
        return BatchInferenceRun(
            run_root=run_root,
            status=metadata["status"],
            completed=completed,
            failed=failed,
            processed_planes=processed_planes,
        )

    def _process_file(
        self,
        path: Path,
        stem: str,
        processing: ImageProcessingSettings,
        run_root: Path,
        metadata: dict,
        entry: dict,
        file_index: int,
        file_count: int,
        progress: Callable[[int, int, str], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        with tifffile.TiffFile(path) as tiff:
            if not tiff.series:
                raise BatchInferenceError("TIFF file contains no image series.")
            for series_index, series in enumerate(tiff.series):
                if cancelled is not None and cancelled():
                    raise InferenceCancelled("Batch cancelled by the user.")
                data = series.asarray()
                axes = str(series.axes)
                if len(axes) != data.ndim:
                    raise BatchInferenceError(
                        f"TIFF axes {axes!r} do not match shape {data.shape}."
                    )
                processing.validate(data.shape)
                channel_axes = [
                    index for index, axis in enumerate(axes) if axis in {"C", "S"}
                ]
                if channel_axes and processing.channel_axis not in channel_axes:
                    raise BatchInferenceError(
                        f"Project channel axis {processing.channel_axis} does not "
                        f"match TIFF axes {axes!r}."
                    )
                layer = SimpleNamespace(
                    data=data,
                    name=path.stem,
                    metadata={"axes": axes},
                    rgb=bool(channel_axes and channel_axes[-1] == data.ndim - 1),
                )
                y_axis, x_axis = self.adapter.spatial_axes(
                    layer, processing.channel_axis
                )
                other_axes = tuple(
                    axis
                    for axis in range(data.ndim)
                    if axis not in {y_axis, x_axis, processing.channel_axis}
                )
                plane_indices = itertools.product(
                    *(range(data.shape[axis]) for axis in other_axes)
                )
                series_stem = (
                    f"{stem}__s{series_index:03d}"
                    if len(tiff.series) > 1
                    else stem
                )
                for indices in plane_indices:
                    if cancelled is not None and cancelled():
                        raise InferenceCancelled("Batch cancelled by the user.")
                    steps = [0] * data.ndim
                    for axis, index in zip(other_axes, indices, strict=True):
                        steps[axis] = index
                    viewer = SimpleNamespace(
                        dims=SimpleNamespace(current_step=tuple(steps))
                    )
                    converted = self.adapter.convert(
                        layer,
                        viewer,
                        processing,
                        base_stem=series_stem,
                        invert=self.settings.invert,
                    )
                    def tile_progress(current: int, total: int, text: str) -> None:
                        if progress is not None:
                            progress(
                                file_index - 1,
                                file_count,
                                f"{path.name} · {converted.sample_id}: {text}",
                            )
                    result = self._predict(converted.data, tile_progress, cancelled)
                    conversion = {
                        "settings": processing.to_mapping(),
                        "inverted": converted.inverted,
                        "plane_indices": converted.plane.non_spatial_indices,
                        "axis_labels": list(converted.plane.axis_labels),
                        "normalization_stats": list(converted.normalization_stats),
                    }
                    prediction = asdict(self.settings.inference)
                    prediction["model_path"] = str(self.model.path)
                    if self.project.config.task == "segment":
                        prediction["tiling"] = result.provenance
                        output = self.writer.save_segmentation(
                            run_root,
                            converted.sample_id,
                            converted.data,
                            result.mask,
                            result.instances,
                            source_path=path,
                            conversion=conversion,
                            prediction=prediction,
                        )
                        count = len(result.instances)
                    else:
                        output = self.writer.save_detection(
                            run_root,
                            converted.sample_id,
                            converted.data,
                            tuple(item.as_napari_rectangle() for item in result),
                            tuple(item.class_id for item in result),
                            confidences=tuple(item.confidence for item in result),
                            sources=("prediction",) * len(result),
                            tile_ids=tuple(item.tile_id for item in result),
                            source_path=path,
                            conversion=conversion,
                            prediction=prediction,
                        )
                        count = len(result)
                    entry["planes"].append(
                        {
                            "sample_id": converted.sample_id,
                            "series_index": series_index,
                            "plane_indices": converted.plane.non_spatial_indices,
                            "shape": list(converted.data.shape),
                            "object_count": count,
                            "output": {
                                "image": str(output.image.relative_to(run_root)),
                                "annotation": str(
                                    output.annotation.relative_to(run_root)
                                ),
                                "metadata": str(output.metadata.relative_to(run_root)),
                            },
                            "sha256": {
                                "image": _sha256(output.image),
                                "annotation": _sha256(output.annotation),
                                "metadata": _sha256(output.metadata),
                            },
                        }
                    )
                    _save_metadata(run_root, metadata)

    def _predict(self, image, progress, cancelled):
        if self.project.config.task == "detect":
            return TiledInferenceEngine(self.model).predict(
                image,
                self.settings.inference,
                progress=progress,
                cancelled=cancelled,
            )
        if any(size > self.settings.inference.tile_size for size in image.shape[:2]):
            return TiledSegmentationEngine(self.model).predict(
                image,
                self.settings.inference,
                self.project.config.classes,
                progress=progress,
                cancelled=cancelled,
                clear_border_instances=self.settings.clear_border_instances,
            )
        if cancelled is not None and cancelled():
            raise InferenceCancelled("Batch cancelled by the user.")
        return self.model.predict_image(
            image, self.settings.inference, self.project.config.classes
        )


def _unique_stems(paths: tuple[Path, ...]) -> dict[Path, str]:
    used: set[str] = set()
    output: dict[Path, str] = {}
    for path in paths:
        base = path.stem
        candidate = base
        index = 2
        while candidate.lower() in used:
            candidate = f"{base}__{path.suffix[1:].lower()}_{index}"
            index += 1
        used.add(candidate.lower())
        output[path] = candidate
    return output


def _save_metadata(run_root: Path, metadata: dict) -> None:
    destination = run_root / "analysis_metadata.json"
    temporary = _temporary_path(destination)
    try:
        _write_json(temporary, metadata)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
