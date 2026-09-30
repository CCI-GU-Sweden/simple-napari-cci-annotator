"""Headless project, training, and TIFF prediction pipeline.

This module does not import napari or Qt. It uses the same project contracts,
model adapters, tiling engines, and output writer as the plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ._batch_inference import (
    BatchInferenceRun,
    BatchInferenceSettings,
    TiffBatchProcessor,
)
from ._dataset_builder import DatasetBuilder
from ._image_adapter import ImageProcessingSettings
from ._project_store import ProjectStore
from ._segmentation_dataset import SegmentationDatasetBuilder
from ._tiled_inference import InferenceSettings
from ._training import TrainingError, TrainingRun, TrainingService, TrainingSettings
from ._yolo_inference import YoloDetectionModel
from ._yolo_segmentation import YoloSegmentationModel


class ProjectPipeline:
    """A synchronous, UI-free pipeline bound to one initialized project."""

    def __init__(self, project: ProjectStore):
        self.project = project

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        task: str = "detect",
        classes: dict[int, str] | None = None,
        name: str | None = None,
        image_processing: ImageProcessingSettings | None = None,
    ) -> ProjectPipeline:
        project = ProjectStore.initialize(
            Path(root), task=task, classes=classes, name=name
        )
        pipeline = cls(project)
        if image_processing is not None:
            pipeline.lock_image_processing(image_processing)
        return pipeline

    @classmethod
    def open(cls, root: str | Path) -> ProjectPipeline:
        return cls(ProjectStore.load(Path(root)))

    def lock_image_processing(self, settings: ImageProcessingSettings) -> None:
        settings.validate()
        self.project.lock_image_processing(settings.to_mapping())

    def train(
        self,
        settings: TrainingSettings,
        *,
        regenerate_split: bool = False,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> TrainingRun:
        """Validate existing canonical annotations, then run training."""
        builder = (
            SegmentationDatasetBuilder(self.project)
            if self.project.config.task == "segment"
            else DatasetBuilder(self.project)
        )
        preview = builder.preview(
            settings.dataset,
            train_only=settings.train_only,
            regenerate=regenerate_split,
        )
        if not preview.is_valid:
            raise TrainingError("Training dataset is invalid: " + "; ".join(preview.errors))
        return TrainingService(self.project).run(
            settings, preview, progress=progress, cancelled=cancelled
        )

    def predict_one(
        self,
        image_path: str | Path,
        model: str | Path | object,
        *,
        output_folder: str | Path | None = None,
        inference: InferenceSettings | None = None,
        invert: bool = False,
        clear_border_instances: bool = False,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> BatchInferenceRun:
        """Predict one TIFF file, including every nonspatial plane and series."""
        path = Path(image_path).expanduser().resolve()
        return self._predict(
            path.parent,
            model,
            output_folder=output_folder,
            inference=inference,
            invert=invert,
            clear_border_instances=clear_border_instances,
            source_files=(path,),
            progress=progress,
            cancelled=cancelled,
        )

    def predict_batch(
        self,
        input_folder: str | Path,
        model: str | Path | object,
        *,
        output_folder: str | Path | None = None,
        inference: InferenceSettings | None = None,
        invert: bool = False,
        clear_border_instances: bool = False,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> BatchInferenceRun:
        """Predict every top-level TIFF file in a folder."""
        return self._predict(
            Path(input_folder).expanduser().resolve(),
            model,
            output_folder=output_folder,
            inference=inference,
            invert=invert,
            clear_border_instances=clear_border_instances,
            source_files=None,
            progress=progress,
            cancelled=cancelled,
        )

    def _predict(
        self,
        input_folder: Path,
        model: str | Path | object,
        *,
        output_folder: str | Path | None,
        inference: InferenceSettings | None,
        invert: bool,
        clear_border_instances: bool,
        source_files: tuple[Path, ...] | None,
        progress: Callable[[int, int, str], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> BatchInferenceRun:
        predictor = self._load_model(model)
        output = (
            Path(output_folder).expanduser().resolve()
            if output_folder is not None
            else input_folder / "Prediction"
        )
        settings = BatchInferenceSettings(
            input_folder=input_folder,
            output_parent=output,
            inference=inference or InferenceSettings(),
            invert=invert,
            clear_border_instances=clear_border_instances,
            source_files=source_files,
        )
        return TiffBatchProcessor(self.project, predictor, settings).run(
            progress=progress, cancelled=cancelled
        )

    def _load_model(self, model: str | Path | object):
        if not isinstance(model, (str, Path)):
            return model
        adapter = (
            YoloSegmentationModel
            if self.project.config.task == "segment"
            else YoloDetectionModel
        )
        return adapter(model)
