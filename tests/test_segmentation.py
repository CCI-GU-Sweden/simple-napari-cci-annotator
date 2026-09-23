from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from simple_napari_cci_annotator._instance_mask import (
    ComposedInstances,
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
from simple_napari_cci_annotator._segmentation_tiling import (
    TiledSegmentationEngine,
    _canonical_relabel,
    _seam_equivalences,
)
from simple_napari_cci_annotator._tiled_inference import (
    InferenceCancelled,
    InferenceError,
    InferenceSettings,
)
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
    assert result.provenance == {"mode": "direct"}


class _FullTilePredictor:
    def predict_image(self, image, settings, classes):
        height, width = image.shape[:2]
        mask = np.ones((height, width), dtype=np.uint32)
        return ComposedInstances(
            mask=mask,
            instances={
                1: InstanceRecord(
                    instance_id=1,
                    class_id=0,
                    class_name=classes[0],
                    confidence=0.8,
                    source="prediction",
                    status="predicted",
                    bbox=(0, 0, height, width),
                    area=height * width,
                )
            },
            cleanup=(),
        )


def test_dask_tiling_pads_merges_and_relabels_deterministically():
    image = np.zeros((100, 130, 3), dtype=np.uint8)
    settings = InferenceSettings(
        tile_size=64, overlap=16, max_detections=10
    )
    engine = TiledSegmentationEngine(_FullTilePredictor())

    first = engine.predict(image, settings, {0: "Cell"})
    second = engine.predict(image, settings, {0: "Cell"})

    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.mask.shape == image.shape[:2]
    assert set(np.unique(first.mask)) == {1}
    assert len(first.instances) == 1
    assert first.provenance["mode"] == "dask_tiled"
    assert first.provenance["grid"] == [4, 5]
    assert first.provenance["padding"] == {"bottom": 28, "right": 30}
    assert first.provenance["equivalence_pairs"]


def test_seam_fusion_is_class_aware_and_reports_ambiguity():
    mask = np.asarray(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 4, 5, 5],
            [3, 4, 5, 5],
        ],
        dtype=np.uint32,
    )
    records = {
        1: _record(1, 0),
        2: _record(2, 1),
        3: _record(3, 0),
        4: _record(4, 0),
        5: _record(5, 1),
    }

    pairs, conflicts, ambiguous = _seam_equivalences(mask, 2, records)

    assert pairs == ((1, 3), (1, 4), (2, 5))
    assert any(item["ids"] == [1, 2] for item in conflicts)
    assert 1 in ambiguous
    fused, fused_records, mapping = _canonical_relabel(
        mask, records, {0: "Cell", 1: "Debris"}, pairs
    )
    assert mapping[1] == mapping[3] == mapping[4]
    assert mapping[2] == mapping[5]
    assert len(fused_records) == 2
    assert set(np.unique(fused)) == {1, 2}


def test_tiled_segmentation_rejects_nonpositive_core():
    with pytest.raises(InferenceError, match="smaller than half"):
        TiledSegmentationEngine(_FullTilePredictor()).predict(
            np.zeros((100, 100, 3), dtype=np.uint8),
            InferenceSettings(tile_size=64, overlap=32),
            {0: "Cell"},
        )


def test_tiled_segmentation_can_clear_source_border_instances():
    result = TiledSegmentationEngine(_FullTilePredictor()).predict(
        np.zeros((80, 90, 3), dtype=np.uint8),
        InferenceSettings(tile_size=64, overlap=16, max_detections=10),
        {0: "Cell"},
        clear_border_instances=True,
    )

    assert not np.any(result.mask)
    assert result.instances == {}
    assert result.provenance["cleared_border_ids"] == [1]


def test_tiled_segmentation_honours_cancellation():
    with pytest.raises(InferenceCancelled):
        TiledSegmentationEngine(_FullTilePredictor()).predict(
            np.zeros((80, 90, 3), dtype=np.uint8),
            InferenceSettings(tile_size=64, overlap=16, max_detections=10),
            {0: "Cell"},
            cancelled=lambda: True,
        )
