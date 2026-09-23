from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from simple_napari_cci_annotator._instance_mask import (
    PredictedInstance,
    compose_predictions,
    disconnected_instance_ids,
    keep_largest_component_by_bbox,
    split_instance,
)
from simple_napari_cci_annotator._project_store import ClassMapError, ProjectStore
from simple_napari_cci_annotator._segmentation_io import (
    InstanceRecord,
    SegmentationError,
    SegmentationIO,
)
from simple_napari_cci_annotator._tiled_inference import InferenceError, InferenceSettings
from simple_napari_cci_annotator._yolo_segmentation import YoloSegmentationModel


def _record(instance_id: int, class_id: int = 0) -> InstanceRecord:
    return InstanceRecord(
        instance_id=instance_id,
        class_id=class_id,
        class_name=("Cell" if class_id == 0 else "Debris"),
        confidence=None,
        source="manual",
        status="corrected",
        bbox=(0, 0, 0, 0),
        area=0,
    )


def test_segment_project_contract_and_uint32_round_trip(tmp_path):
    project = ProjectStore.initialize(
        tmp_path / "segment", task="segment", classes={0: "Cell", 1: "Debris"}
    )
    assert project.config.task == "segment"
    assert project.paths.masks.is_dir()
    assert project.paths.instances.is_dir()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[1:3, 1:3] = 1
    mask[3:7, 4:7] = 2
    io = SegmentationIO(project)
    result = io.save(
        image_data=image,
        sample_id="sample",
        mask=mask,
        instances={1: _record(1), 2: _record(2, 1)},
        conversion_metadata={"settings": {"normalization": "min_max"}},
    )

    assert tifffile.imread(result.mask_path).dtype == np.uint32
    loaded = io.load("sample", image.shape[:2])
    np.testing.assert_array_equal(loaded.mask, mask)
    assert loaded.instances[2].class_id == 1
    assert loaded.instances[2].bbox == (3, 4, 7, 7)
    assert loaded.instances[2].area == 12
    payload = json.loads(result.instances_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1

    with pytest.raises(ClassMapError, match="used by saved annotations"):
        project.update_classes({0: "Cell"})


def test_segmentation_io_rejects_metadata_drift_and_wrong_dtype(tmp_path):
    project = ProjectStore.initialize(tmp_path / "segment", task="segment")
    io = SegmentationIO(project)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint32)
    mask[1, 1] = 1
    with pytest.raises(SegmentationError, match="absent from the mask"):
        io.save(
            image_data=image,
            sample_id="bad",
            mask=mask,
            instances={1: _record(1), 2: _record(2)},
        )

    io.save(
        image_data=image,
        sample_id="sample",
        mask=mask,
        instances={1: _record(1)},
    )
    tifffile.imwrite(project.paths.masks / "sample.tif", mask.astype(np.uint16))
    with pytest.raises(SegmentationError, match="must be uint32"):
        io.load("sample", image.shape[:2])

    oversized = np.zeros((4, 4), dtype=np.uint64)
    oversized[0, 0] = np.iinfo(np.uint32).max + 1
    with pytest.raises(SegmentationError, match="exceed uint32"):
        io.save(
            image_data=image,
            sample_id="oversized",
            mask=oversized,
            instances={},
        )


def test_largest_component_uses_bbox_area_not_pixel_area():
    mask = np.zeros((12, 20), dtype=bool)
    mask[1, 1:7] = True  # 6 pixels, bbox area 6
    mask[4, 10:13] = True
    mask[5:7, 12] = True  # 5 pixels, bbox area 9

    kept, report = keep_largest_component_by_bbox(mask)

    assert kept[4:7, 10:13].sum() == 5
    assert kept[1, 1:7].sum() == 0
    assert report.removed_components == 1
    assert report.removed_pixels == 6


def test_confidence_owns_overlap_and_ids_stay_in_model_order():
    first = np.zeros((6, 6), dtype=bool)
    first[1:5, 1:4] = True
    second = np.zeros((6, 6), dtype=bool)
    second[2:5, 2:5] = True
    result = compose_predictions(
        (
            PredictedInstance(first, (1, 1, 4, 5), 0, 0.4),
            PredictedInstance(second, (2, 2, 5, 5), 1, 0.9),
        ),
        (6, 6),
        {0: "Cell", 1: "Debris"},
    )

    assert result.mask[2, 2] == 2
    assert result.mask[1, 1] == 1
    assert result.instances[1].class_id == 0
    assert result.instances[2].class_id == 1


def test_manual_disconnected_instance_is_reported_and_explicitly_split():
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[1:3, 1:3] = 1
    mask[5:7, 5:7] = 1
    assert disconnected_instance_ids(mask) == (1,)

    split, records = split_instance(mask, 1, {1: _record(1)}, {0: "Cell"})
    assert set(np.unique(split)) == {0, 1, 2}
    assert records[1].class_id == records[2].class_id == 0
    assert disconnected_instance_ids(split) == ()


def test_segmentation_adapter_rejects_detection_weights(tmp_path, monkeypatch):
    model_path = tmp_path / "detect.pt"
    model_path.write_bytes(b"weights")

    class FakeYolo:
        task = "detect"
        names = {0: "Cell"}

        def __init__(self, path):
            self.path = path

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYolo))
    with pytest.raises(InferenceError, match="requires YOLO segmentation"):
        YoloSegmentationModel(model_path)


def test_segmentation_adapter_restores_source_resolution(tmp_path, monkeypatch):
    model_path = tmp_path / "segment.pt"
    model_path.write_bytes(b"weights")

    class FakeBoxes:
        def __init__(self):
            self.xyxy = np.asarray([[0, 0, 8, 4]], dtype=float)
            self.conf = np.asarray([0.8])
            self.cls = np.asarray([0])

        def __len__(self):
            return 1

    class FakeYolo:
        task = "segment"
        names = {0: "Cell"}

        def __init__(self, path):
            self.path = path

        def predict(self, **kwargs):
            assert kwargs["retina_masks"] is True
            low_resolution = np.asarray([[[1, 1], [0, 0]]], dtype=float)
            return [
                SimpleNamespace(
                    boxes=FakeBoxes(), masks=SimpleNamespace(data=low_resolution)
                )
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYolo))
    adapter = YoloSegmentationModel(model_path)
    result = adapter.predict_image(
        np.zeros((4, 8, 3), dtype=np.uint8),
        InferenceSettings(tile_size=8, overlap=0),
        {0: "Cell"},
    )
    assert result.mask.shape == (4, 8)
    assert result.mask.dtype == np.uint32
    assert len(result.instances) == 1
