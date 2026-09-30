"""Full-size prediction outputs stay separate from training annotations."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image
import pytest
import tifffile

from simple_napari_cci_annotator._prediction_output import (
    PredictionOutputError,
    PredictionOutputWriter,
)
from simple_napari_cci_annotator._project_store import ProjectStore
from simple_napari_cci_annotator._segmentation_io import InstanceRecord


def test_detection_output_round_trip_and_explicit_overwrite(tmp_path):
    project = ProjectStore.initialize(tmp_path / "project", task="detect")
    writer = PredictionOutputWriter(project)
    image = np.zeros((48, 80, 3), dtype=np.uint8)
    rectangle = np.asarray([[10, 20], [10, 40], [30, 40], [30, 20]])
    parent = tmp_path / "results"

    paths = writer.save_detection(
        parent,
        "field",
        image,
        (rectangle,),
        (0,),
        confidences=np.asarray([0.8]),
        sources=("prediction",),
        tile_ids=(2,),
        prediction={"tile_size": 1024},
    )

    assert paths.folder == parent / "detect" / "field"
    with Image.open(paths.image) as saved:
        assert saved.size == (80, 48)
    assert paths.annotation.read_text(encoding="utf-8") == (
        "0 0.375000 0.416667 0.250000 0.416667\n"
    )
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["boxes"][0]["confidence"] == 0.8
    assert metadata["boxes"][0]["tile_id"] == 2
    assert metadata["prediction"] == {"tile_size": 1024}
    assert not list(project.paths.images.iterdir())

    with pytest.raises(PredictionOutputError, match="Confirm overwrite"):
        writer.save_detection(parent, "field", image, (), ())
    writer.save_detection(parent, "field", image, (), (), overwrite=True)
    assert paths.annotation.read_text(encoding="utf-8") == ""


def test_segmentation_output_round_trip_and_rejects_bad_ids(tmp_path):
    project = ProjectStore.initialize(
        tmp_path / "project", task="segment", classes={0: "Cell"}
    )
    writer = PredictionOutputWriter(project)
    image = np.zeros((48, 80, 3), dtype=np.uint8)
    mask = np.zeros((48, 80), dtype=np.uint32)
    mask[10:20, 20:30] = 7
    instances = {
        7: InstanceRecord(7, 0, "Cell", 0.9, "prediction", "predicted", (0, 0, 0, 0), 0)
    }

    paths = writer.save_segmentation(
        tmp_path / "results", "field", image, mask, instances
    )

    assert paths.folder == tmp_path / "results" / "segment" / "field"
    np.testing.assert_array_equal(tifffile.imread(paths.annotation), mask)
    assert tifffile.imread(paths.annotation).dtype == np.uint32
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["instances"]["7"]["class_id"] == 0
    assert metadata["instances"]["7"]["area"] == 100
    assert not list(project.paths.images.iterdir())

    with pytest.raises(PredictionOutputError, match="exactly match"):
        writer.save_segmentation(tmp_path / "other", "field", image, mask, {})
    mask[30, 30] = 7
    with pytest.raises(PredictionOutputError, match="Disconnected"):
        writer.save_segmentation(tmp_path / "other", "field", image, mask, instances)
