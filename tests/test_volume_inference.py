"""Stored slice inference, component preservation, and resumable commits."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simple_napari_cci_annotator._instance_mask import (
    PredictedInstance, compose_predictions,
    disconnected_instance_ids, split_prediction_components,
)
from simple_napari_cci_annotator._image_adapter import ImageProcessingSettings
from simple_napari_cci_annotator._segmentation_tiling import TiledSegmentationEngine
from simple_napari_cci_annotator._tiled_inference import InferenceSettings
from simple_napari_cci_annotator._volume_adapter import VolumeAdapter
from simple_napari_cci_annotator._volume_inference import (
    VolumeInferenceError, VolumeInferenceRunner, VolumeInferenceSettings,
)
from simple_napari_cci_annotator._volume_store import VolumeMaskStore, VolumeStorageError
from simple_napari_cci_annotator import _volume_store as store_module


CLASSES = {0: "Cell"}
INFERENCE = InferenceSettings(tile_size=64, overlap=8, max_detections=10)


class Predictor:
    task = "segment"
    names = CLASSES

    def __init__(self, path, *, empty=False):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_bytes(b"test model")
        self.calls = []
        self.empty = empty

    def predict_image(self, image, settings, classes, *, component_policy="largest"):
        self.calls.append(image.copy())
        mask = np.ones(image.shape[:2], dtype=bool)
        predictions = [] if self.empty else [PredictedInstance(mask, (0, 0, mask.shape[1], mask.shape[0]), 0, 0.8)]
        result = compose_predictions(predictions, mask.shape, classes, component_policy=component_policy)
        result.provenance["component_policy"] = component_policy
        return result


def volume(shape=(3, 8, 9), *, z_range=None, data=None):
    if data is None:
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    layer = SimpleNamespace(data=data, name="volume", metadata={"axes": "ZYX"})
    return VolumeAdapter().prepare(
        layer, ImageProcessingSettings(None, None, None, None, "min_max"),
        z_range=z_range,
    )


def runner(tmp_path, *, source=None, predictor=None, settings=None):
    return VolumeInferenceRunner(
        source or volume(), predictor or Predictor(tmp_path / "model.pt"), CLASSES,
        settings or VolumeInferenceSettings(INFERENCE),
    )


@pytest.mark.parametrize("shape", [(3, 8, 9), (2, 80, 90)])
@pytest.mark.parametrize("policy", ["largest", "preserve"])
def test_stored_predictions_match_existing_plane_engines(tmp_path, shape, policy):
    source = volume(shape)
    predictor = Predictor(tmp_path / "model.pt")
    settings = VolumeInferenceSettings(INFERENCE, component_policy=policy)
    messages = []
    result = runner(tmp_path, source=source, predictor=predictor, settings=settings).run(
        tmp_path / "run", progress=lambda *args: messages.append(args),
    )
    assert result.status == "completed"
    assert result.completed_slices == shape[0]
    with VolumeMaskStore.open(result.run_root) as store:
        assert store.complete
        assert store.manifest["status"] == "completed"
        for z in source.selection.z_indices:
            image = source.convert_slice(z).image
            if max(shape[1:]) > 64:
                expected = TiledSegmentationEngine(predictor).predict(
                    image.data, INFERENCE, CLASSES, component_policy=policy,
                    num_workers=1,
                )
            else:
                expected = predictor.predict_image(image.data, INFERENCE, CLASSES, component_policy=policy)
            expected = split_prediction_components(expected, CLASSES)
            saved, metadata = store.read_slice(z)
            np.testing.assert_array_equal(saved, expected.mask)
            assert saved.shape == shape[1:]
            assert metadata["instances"]["1"]["area"] == shape[1] * shape[2]
            assert metadata["conversion"]["normalization_stats"] == list(image.normalization_stats)
            assert metadata["cleanup"]
            # Border predictions survive by default, including tile seams.
            assert np.all(saved == 1)
        lazy = store.mask_array(xy_chunk_size=4)
        display = store.mask_array(unique_ids=True, xy_chunk_size=4)
    # Lazy views read durable output after the writer/store has been closed.
    assert lazy.chunks[0] == (1,) * shape[0]
    assert lazy.dtype == np.dtype("uint32")
    np.testing.assert_array_equal(lazy[0].compute(), np.ones(shape[1:], dtype=np.uint32))
    assert set(np.unique(display.compute())) == set(range(1, shape[0] + 1))
    assert messages[-1] == (shape[0], shape[0], "Slice inference complete")


def test_completed_empty_slices_are_distinct_from_missing_slices(tmp_path):
    predictor = Predictor(tmp_path / "model.pt", empty=True)
    result = runner(tmp_path, predictor=predictor).run(tmp_path / "run")
    with VolumeMaskStore.open(result.run_root) as store:
        assert store.completed_count == 3
        for z in range(3):
            mask, record = store.read_slice(z)
            assert not np.any(mask)
            assert record["instances"] == {}
        assert not np.any(store.mask_array(unique_ids=True).compute())


def test_selected_z_range_uses_global_z_in_records_and_local_storage(tmp_path):
    result = runner(tmp_path, source=volume((5, 8, 9), z_range=(2, 4))).run(tmp_path / "run")
    with VolumeMaskStore.open(result.run_root) as store:
        assert [entry["z_index"] for entry in store.manifest["slices"]] == [2, 3]
        assert store.mask_array().shape == (2, 8, 9)
        assert store.read_slice(2)[1]["conversion"]["plane_indices"] == [2, None, None]
        with pytest.raises(VolumeStorageError):
            store.read_slice(0)


def test_cancel_between_slices_then_resume_skips_committed_predictions(tmp_path):
    pipeline = runner(tmp_path)
    cancel = [False]

    def progress(completed, total, text):
        if text.startswith("Stored"):
            cancel[0] = True

    result = pipeline.run(tmp_path / "run", progress=progress, cancelled=lambda: cancel[0])
    assert result.status == "cancelled"
    assert result.completed_slices == 1
    with VolumeMaskStore.open(result.run_root) as store:
        with pytest.raises(VolumeStorageError, match="missing slices"):
            store.mask_array()
        with pytest.raises(VolumeStorageError, match="not been committed"):
            store.read_slice(1)
        partial = store.mask_array(allow_partial=True).compute()
        assert np.all(partial[0] == 1)
        assert not np.any(partial[1:])
    resumed = pipeline.run(result.run_root, resume=True)
    assert resumed.status == "completed"
    assert len(pipeline.predictor.calls) == 3


def test_cancel_during_prediction_discards_uncommitted_plane(tmp_path):
    pipeline = runner(tmp_path)
    call = pipeline.predictor.predict_image
    cancel = [False]

    def predict(*args, **kwargs):
        result = call(*args, **kwargs)
        cancel[0] = True
        return result

    pipeline.predictor.predict_image = predict
    result = pipeline.run(tmp_path / "run", cancelled=lambda: cancel[0])
    assert result.status == "cancelled"
    with VolumeMaskStore.open(result.run_root) as store:
        assert store.completed_count == 0


def test_cancel_during_tiled_prediction_preserves_no_partial_tile_mask(tmp_path):
    pipeline = runner(tmp_path, source=volume((1, 80, 90)))
    call = pipeline.predictor.predict_image
    cancel = [False]

    def predict(*args, **kwargs):
        result = call(*args, **kwargs)
        cancel[0] = True
        return result

    pipeline.predictor.predict_image = predict
    result = pipeline.run(tmp_path / "run", cancelled=lambda: cancel[0])
    assert result.status == "cancelled"
    assert len(pipeline.predictor.calls) == 1
    with VolumeMaskStore.open(result.run_root) as store:
        assert store.completed_count == 0
        assert not np.any(store.mask_array(allow_partial=True).compute())
    pipeline.predictor.predict_image = call
    assert pipeline.run(result.run_root, resume=True).status == "completed"


def test_pre_cancel_does_not_create_output(tmp_path):
    result = runner(tmp_path).run(tmp_path / "run", cancelled=lambda: True)
    assert result.status == "cancelled"
    assert result.run_root is None
    assert not (tmp_path / "run").exists()


def test_failure_retains_completed_slices_and_records_error(tmp_path):
    pipeline = runner(tmp_path)
    predict = pipeline.predictor.predict_image

    def fail_second(*args, **kwargs):
        if pipeline.predictor.calls:
            raise RuntimeError("model failure")
        return predict(*args, **kwargs)

    pipeline.predictor.predict_image = fail_second
    with pytest.raises(RuntimeError, match="model failure"):
        pipeline.run(tmp_path / "run")
    with VolumeMaskStore.open(tmp_path / "run") as store:
        assert store.completed_count == 1
        assert store.manifest["status"] == "failed"
        assert store.manifest["error"] == "model failure"
    pipeline.predictor.predict_image = predict
    assert pipeline.run(tmp_path / "run", resume=True).completed_slices == 3


@pytest.mark.parametrize("change", ["source", "model", "settings", "classes"])
def test_resume_rejects_changed_contract_without_mutating_existing_run(tmp_path, change):
    pipeline = runner(tmp_path)
    pipeline.run(tmp_path / "run")
    manifest_before = (tmp_path / "run" / "run.json").read_bytes()
    if change == "source":
        pipeline.volume.data[0, 0, 0] += 1
    elif change == "model":
        pipeline.predictor.path.write_bytes(b"changed weights")
    elif change == "settings":
        pipeline.settings = replace(pipeline.settings, component_policy="largest")
    else:
        pipeline.classes = {0: "Renamed cell"}
    with pytest.raises(VolumeInferenceError, match="Cannot resume"):
        pipeline.run(tmp_path / "run", resume=True)
    assert (tmp_path / "run" / "run.json").read_bytes() == manifest_before


@pytest.mark.parametrize("damage", ["metadata_missing", "metadata_corrupt", "mask_corrupt"])
def test_corrupt_committed_slice_is_recomputed_on_resume(tmp_path, damage):
    pipeline = runner(tmp_path)
    pipeline.run(tmp_path / "run")
    path = tmp_path / "run" / "slices" / "z000001.json"
    if damage == "metadata_missing":
        path.unlink()
    elif damage == "metadata_corrupt":
        path.write_text("{}")
    else:
        mask = np.load(tmp_path / "run" / "slice_masks.npy", mmap_mode="r+")
        mask[1, 0, 0] = 10
        mask.flush()
        mask._mmap.close()
    with pytest.raises(VolumeStorageError):
        VolumeMaskStore.open(tmp_path / "run")
    result = pipeline.run(tmp_path / "run", resume=True)
    assert result.completed_slices == 3
    assert len(pipeline.predictor.calls) == 4
    with VolumeMaskStore.open(result.run_root) as store:
        assert np.all(store.read_slice(1)[0] == 1)


def test_interrupted_manifest_commit_is_not_counted_as_a_completed_slice(tmp_path, monkeypatch):
    pipeline = runner(tmp_path)
    original = store_module._atomic_json
    failed = [False]

    def interrupt(path, value):
        if path.name == "run.json" and value["slices"][0]["complete"] and not failed[0]:
            failed[0] = True
            raise OSError("interrupted commit")
        return original(path, value)

    monkeypatch.setattr(store_module, "_atomic_json", interrupt)
    with pytest.raises(OSError, match="interrupted commit"):
        pipeline.run(tmp_path / "run")
    with VolumeMaskStore.open(tmp_path / "run") as store:
        assert store.completed_count == 0
        assert not np.any(store.mask_array(allow_partial=True).compute())
    assert pipeline.run(tmp_path / "run", resume=True).completed_slices == 3
    assert len(pipeline.predictor.calls) == 4


def test_memory_limit_and_model_compatibility_fail_before_output(tmp_path):
    pipeline = runner(tmp_path, settings=VolumeInferenceSettings(INFERENCE, memory_budget_bytes=1))
    with pytest.raises(VolumeInferenceError, match="exceeds"):
        pipeline.run(tmp_path / "run")
    assert not (tmp_path / "run").exists()
    pipeline.settings = VolumeInferenceSettings(INFERENCE)
    pipeline.predictor.task = "detect"
    with pytest.raises(VolumeInferenceError, match="segmentation model"):
        pipeline.validate()
    pipeline.predictor.task = "segment"
    pipeline.classes = {1: "Other"}
    with pytest.raises(VolumeInferenceError, match="class IDs"):
        pipeline.validate()


def test_existing_run_is_not_overwritten_and_writer_lock_is_exclusive(tmp_path):
    pipeline = runner(tmp_path)
    pipeline.run(tmp_path / "run")
    with pytest.raises(FileExistsError):
        pipeline.run(tmp_path / "run")
    with VolumeMaskStore.open(tmp_path / "run", writable=True):
        with pytest.raises(VolumeStorageError, match="active writer"):
            VolumeMaskStore.open(tmp_path / "run", writable=True)
    with VolumeMaskStore.open(tmp_path / "run") as store:
        assert store.complete
        with pytest.raises(VolumeStorageError, match="read-only"):
            store.set_status("running")


def test_preservation_keeps_branches_and_splits_after_confidence_ownership():
    mask = np.zeros((12, 12), dtype=bool)
    mask[1:4, 1:4] = True
    mask[8:10, 8:10] = True
    prediction = PredictedInstance(mask, (1, 1, 10, 10), 0, 0.7)
    old = compose_predictions([prediction], mask.shape, CLASSES)
    preserved = compose_predictions([prediction], mask.shape, CLASSES, component_policy="preserve")
    assert np.count_nonzero(old.mask) == 9
    assert np.count_nonzero(preserved.mask) == 13
    assert preserved.cleanup[0].removed_pixels == 0
    nodes = split_prediction_components(preserved, CLASSES)
    assert np.count_nonzero(nodes.mask) == 13
    assert len(nodes.instances) == 2
    assert disconnected_instance_ids(nodes.mask) == ()
    assert nodes.instances[1].lineage == nodes.instances[2].lineage
    assert nodes.provenance["slice_components"] == {
        "1": {"source_instance_id": 1, "component_id": 1},
        "2": {"source_instance_id": 1, "component_id": 2},
    }
    # A higher confidence crossing instance can split an originally connected
    # prediction; extraction must happen after confidence ownership.
    large = np.zeros((12, 12), dtype=bool)
    large[1:10, 1:10] = True
    crossing = np.zeros_like(large)
    crossing[:, 5] = True
    composed = compose_predictions([
        PredictedInstance(large, (1, 1, 10, 10), 0, 0.7),
        PredictedInstance(crossing, (5, 0, 6, 12), 0, 0.9),
    ], large.shape, CLASSES, component_policy="preserve")
    nodes = split_prediction_components(composed, CLASSES)
    assert len(nodes.instances) == 3
    np.testing.assert_array_equal(nodes.mask > 0, composed.mask > 0)
    assert nodes.instances[1].confidence == nodes.instances[2].confidence == 0.7


def test_yolo_adapter_preserves_components_only_when_requested(tmp_path, monkeypatch):
    from simple_napari_cci_annotator._yolo_segmentation import YoloSegmentationModel

    mask = np.zeros((1, 12, 12), dtype=float)
    mask[0, 1:4, 1:4] = 1
    mask[0, 8:10, 8:10] = 1

    class Boxes:
        xyxy = np.asarray([[1, 1, 10, 10]], dtype=float)
        conf = np.asarray([0.8])
        cls = np.asarray([0])

        def __len__(self):
            return 1

    class FakeYolo:
        task = "segment"
        names = CLASSES

        def __init__(self, path):
            pass

        def predict(self, **kwargs):
            return [SimpleNamespace(boxes=Boxes(), masks=SimpleNamespace(data=mask))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYolo))
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"weights")
    adapter = YoloSegmentationModel(model_path)
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    old = adapter.predict_image(image, INFERENCE, CLASSES)
    kept = adapter.predict_image(image, INFERENCE, CLASSES, component_policy="preserve")
    assert np.count_nonzero(old.mask) == 9
    assert np.count_nonzero(kept.mask) == 13
    assert old.provenance == {"mode": "direct"}
    assert kept.provenance["component_policy"] == "preserve"


def test_resume_cancelled_during_verification_leaves_run_unchanged(tmp_path):
    pipeline = runner(tmp_path)
    pipeline.run(tmp_path / "run")
    manifest_before = (tmp_path / "run" / "run.json").read_bytes()
    cancel = [False]

    def progress(completed, total, text):
        if text.startswith("Verifying stored slices"):
            cancel[0] = True

    result = pipeline.run(
        tmp_path / "run", resume=True, progress=progress,
        cancelled=lambda: cancel[0],
    )
    assert result.status == "cancelled"
    assert (tmp_path / "run" / "run.json").read_bytes() == manifest_before
    assert len(pipeline.predictor.calls) == 3


def test_volume_worker_reports_cancelled_result(qtbot, tmp_path):
    from simple_napari_cci_annotator._volume_worker import VolumeInferenceWorker

    worker = VolumeInferenceWorker(runner(tmp_path), tmp_path / "run")
    signals = []
    worker.cancelled.connect(signals.append)
    worker.request_cancel()
    worker.run()
    assert len(signals) == 1
    assert signals[0].status == "cancelled"
    assert signals[0].run_root is None
