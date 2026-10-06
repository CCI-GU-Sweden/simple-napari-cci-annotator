"""The public pipeline reuses headless project and prediction services."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from simple_napari_cci_annotator.api import ProjectPipeline
from simple_napari_cci_annotator._dataset_builder import DatasetBuildSettings
from simple_napari_cci_annotator._image_adapter import ImageProcessingSettings
from simple_napari_cci_annotator._training import TrainingError, TrainingSettings
from simple_napari_cci_annotator._tiled_inference import InferenceSettings


class FakeModel:
    task = "detect"
    names = {0: "Cell"}

    def __init__(self, path):
        self.path = path
        path.write_bytes(b"fake model")

    def predict_tile(self, image, settings):
        return ()


def test_pipeline_project_one_image_and_batch(tmp_path):
    processing = ImageProcessingSettings(None, None, None, None, "min_max")
    pipeline = ProjectPipeline.create(
        tmp_path / "project",
        classes={0: "Cell"},
        image_processing=processing,
    )
    reopened = ProjectPipeline.open(tmp_path / "project")
    assert reopened.project.config.image_processing == processing.to_mapping()
    source = tmp_path / "input"
    source.mkdir()
    for name in ("first.tif", "second.ome.tiff"):
        tifffile.imwrite(
            source / name,
            np.arange(64 * 64, dtype=np.uint16).reshape(64, 64),
        )
    model = FakeModel(tmp_path / "model.pt")
    inference = InferenceSettings(tile_size=64, overlap=8)

    one = reopened.predict_one(source / "first.tif", model, inference=inference)
    assert (one.completed, one.failed, one.processed_planes) == (1, 0, 1)
    assert one.run_root.parent == source / "Prediction"
    one_metadata = json.loads((one.run_root / "analysis_metadata.json").read_text())
    assert [entry["source"] for entry in one_metadata["files"]] == ["first.tif"]

    batch = reopened.predict_batch(source, model, inference=inference)
    assert (batch.completed, batch.failed, batch.processed_planes) == (2, 0, 2)
    batch_metadata = json.loads((batch.run_root / "analysis_metadata.json").read_text())
    assert [entry["source"] for entry in batch_metadata["files"]] == [
        "first.tif", "second.ome.tiff"
    ]
    assert batch.run_root != one.run_root


def test_pipeline_train_reports_invalid_dataset_without_starting_model(tmp_path):
    pipeline = ProjectPipeline.create(tmp_path / "project")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"unused")
    settings = TrainingSettings(
        model_path=model_path,
        destination=tmp_path / "training",
        dataset=DatasetBuildSettings(),
    )
    with pytest.raises(TrainingError, match="No valid image/label pairs"):
        pipeline.train(settings)
    assert not (tmp_path / "training").exists()


def test_pipeline_predict_one_png(tmp_path):
    settings = ImageProcessingSettings(None, None, None, None, "min_max")
    pipeline = ProjectPipeline.create(
        tmp_path / "project", image_processing=settings
    )
    source = tmp_path / "input"
    source.mkdir()
    Image.fromarray(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)).save(
        source / "field.png"
    )
    model = FakeModel(tmp_path / "model.pt")

    result = pipeline.predict_one(
        source / "field.png", model, inference=InferenceSettings(tile_size=64, overlap=8)
    )

    assert (result.completed, result.failed, result.processed_planes) == (1, 0, 1)
    metadata = json.loads((result.run_root / "analysis_metadata.json").read_text())
    assert metadata["files"][0]["source"] == "field.png"
