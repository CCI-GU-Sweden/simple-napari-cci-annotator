from ._annotation_io import AnnotationIO
from ._dataset_builder import DatasetBuildSettings, DatasetBuilder
from ._image_adapter import ImageAdapter, ImageProcessingSettings
from ._project_store import ProjectStore
from ._segmentation_io import InstanceRecord, SegmentationIO
from ._tiled_inference import (
    Detection,
    InferenceSettings,
    TiledInferenceEngine,
    create_tile_plan,
)
from ._training import TrainingService, TrainingSettings
from ._version import __version__
from .api import ProjectPipeline

__all__ = [
    "AnnotationIO",
    "DatasetBuildSettings",
    "DatasetBuilder",
    "Detection",
    "ImageAdapter",
    "ImageProcessingSettings",
    "InferenceSettings",
    "InstanceRecord",
    "ProjectStore",
    "ProjectPipeline",
    "SegmentationIO",
    "SimpleCciAnnotatorQWidget",
    "TiledInferenceEngine",
    "TrainingService",
    "TrainingSettings",
    "create_tile_plan",
]


def __getattr__(name: str):
    if name == "SimpleCciAnnotatorQWidget":
        from ._widget import SimpleCciAnnotatorQWidget

        return SimpleCciAnnotatorQWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
