"""Batch TIFF processing keeps project conversion and records each result."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from simple_napari_cci_annotator._batch_inference import (
    BatchInferenceError,
    BatchInferenceSettings,
    TiffBatchProcessor,
    find_tiff_inputs,
)
from simple_napari_cci_annotator._image_adapter import ImageProcessingSettings
from simple_napari_cci_annotator._instance_mask import ComposedInstances
from simple_napari_cci_annotator._project_store import ProjectStore
from simple_napari_cci_annotator._segmentation_io import InstanceRecord
from simple_napari_cci_annotator._tiled_inference import InferenceSettings, RawDetection


def _project(tmp_path, task="detect"):
    project = ProjectStore.initialize(
        tmp_path / "project", task=task, classes={0: "Cell"}
    )
    processing = ImageProcessingSettings(None, None, None, None, "min_max")
    return project, processing


class _DetectionModel:
    task = "detect"
    names = {0: "Cell"}

    def __init__(self, path):
        self.path = path
        self.path.write_bytes(b"model")

    def predict_tile(self, image, settings):
        return (RawDetection(5, 6, 15, 16, 0.9, 0),)


class _SegmentationModel:
    task = "segment"
    names = {0: "Cell"}

    def __init__(self, path):
        self.path = path
        self.path.write_bytes(b"model")

    def predict_image(self, image, settings, classes):
        mask = np.zeros(image.shape[:2], dtype=np.uint32)
        mask[8:18, 12:22] = 1
        record = InstanceRecord(
            1, 0, "Cell", 0.8, "prediction", "predicted", (0, 0, 0, 0), 0
        )
        return ComposedInstances(mask, {1: record}, (), {"mode": "single"})


def _settings(source, output=None):
    return BatchInferenceSettings(
        source, output or source / "Prediction", InferenceSettings(tile_size=64, overlap=8)
    )


def test_batch_requires_locked_project_and_supported_inputs(tmp_path):
    project, processing = _project(tmp_path)
    source = tmp_path / "input"
    source.mkdir()
    (source / "ignored.jpg").write_bytes(b"not a supported input")
    model = _DetectionModel(tmp_path / "model.pt")
    processor = TiffBatchProcessor(project, model, _settings(source))
    with pytest.raises(BatchInferenceError, match="Lock project"):
        processor.validate()
    project.lock_image_processing(processing.to_mapping())
    with pytest.raises(BatchInferenceError, match="no TIFF, PNG, or BMP"):
        processor.validate()
    tifffile.imwrite(source / "A.TIFF", np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))
    tifffile.imwrite(source / "b.ome.tiff", np.zeros((64, 64), dtype=np.uint16))
    assert [path.name for path in find_tiff_inputs(source)] == ["A.TIFF", "b.ome.tiff"]


def test_batch_detects_all_ome_planes_and_writes_manifest(tmp_path):
    project, processing = _project(tmp_path)
    project.lock_image_processing(processing.to_mapping())
    source = tmp_path / "input"
    source.mkdir()
    data = np.arange(2 * 64 * 64, dtype=np.uint16).reshape(2, 64, 64)
    tifffile.imwrite(source / "volume.ome.tiff", data, ome=True, metadata={"axes": "ZYX"})
    (source / "ignored.jpg").write_bytes(b"ignored")
    model = _DetectionModel(tmp_path / "model.pt")
    result = TiffBatchProcessor(project, model, _settings(source)).run()

    assert result.status == "completed"
    assert (result.completed, result.failed, result.processed_planes) == (1, 0, 2)
    assert result.run_root.parent == source / "Prediction"
    metadata = json.loads((result.run_root / "analysis_metadata.json").read_text())
    assert metadata["image_processing"] == project.config.image_processing
    assert metadata["model_sha256"]
    assert metadata["project_config_sha256"]
    assert metadata["files"][0]["source"] == "volume.ome.tiff"
    assert metadata["files"][0]["source_sha256"]
    assert len(metadata["files"][0]["planes"]) == 2
    for plane in metadata["files"][0]["planes"]:
        assert plane["object_count"] == 1
        for relative in plane["output"].values():
            assert (result.run_root / relative).is_file()
        assert all(plane["sha256"].values())
    assert not list(project.paths.images.iterdir())


def test_batch_segmentation_uses_custom_output_and_records_failed_file(tmp_path):
    project, processing = _project(tmp_path, "segment")
    project.lock_image_processing(processing.to_mapping())
    source = tmp_path / "input"
    source.mkdir()
    tifffile.imwrite(source / "good.tif", np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))
    (source / "bad.tiff").write_bytes(b"invalid")
    model = _SegmentationModel(tmp_path / "model.pt")
    output = tmp_path / "elsewhere"
    result = TiffBatchProcessor(project, model, _settings(source, output)).run()

    assert result.status == "completed_with_errors"
    assert (result.completed, result.failed, result.processed_planes) == (1, 1, 1)
    assert result.run_root.parent == output
    metadata = json.loads((result.run_root / "analysis_metadata.json").read_text())
    assert [entry["status"] for entry in metadata["files"]] == ["failed", "completed"]
    plane = metadata["files"][1]["planes"][0]
    mask = tifffile.imread(result.run_root / plane["output"]["annotation"])
    assert mask.dtype == np.uint32
    assert int(mask.max()) == 1
    assert not list(project.paths.images.iterdir())


def test_batch_cancellation_keeps_completed_output_and_marks_manifest(tmp_path):
    project, processing = _project(tmp_path)
    project.lock_image_processing(processing.to_mapping())
    source = tmp_path / "input"
    source.mkdir()
    for name in ("a.tif", "b.tif"):
        tifffile.imwrite(source / name, np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))
    model = _DetectionModel(tmp_path / "model.pt")
    stop = False

    def progress(current, total, message):
        nonlocal stop
        if current == 1 and message.startswith("Finished"):
            stop = True

    result = TiffBatchProcessor(project, model, _settings(source)).run(
        progress=progress, cancelled=lambda: stop
    )
    metadata = json.loads((result.run_root / "analysis_metadata.json").read_text())
    assert result.status == metadata["status"] == "cancelled"
    assert (result.completed, result.failed, result.processed_planes) == (1, 0, 1)
    assert [entry["source"] for entry in metadata["files"]] == ["a.tif"]
    assert metadata["files"][0]["status"] == "completed"
    assert metadata["finished_at"]


def test_batch_sanitized_stems_do_not_collide(tmp_path):
    project, processing = _project(tmp_path)
    project.lock_image_processing(processing.to_mapping())
    source = tmp_path / "input"
    source.mkdir()
    for name in ("a b.tif", "a@b.tif"):
        tifffile.imwrite(source / name, np.ones((64, 64), dtype=np.uint16))
    model = _DetectionModel(tmp_path / "model.pt")

    result = TiffBatchProcessor(project, model, _settings(source)).run()

    assert (result.completed, result.failed, result.processed_planes) == (2, 0, 2)
    metadata = json.loads((result.run_root / "analysis_metadata.json").read_text())
    ids = [entry["planes"][0]["sample_id"] for entry in metadata["files"]]
    assert len(set(ids)) == 2
    assert all(entry["status"] == "completed" for entry in metadata["files"])


def test_batch_direct_segmentation_clears_border_instances(tmp_path):
    project, processing = _project(tmp_path, "segment")
    project.lock_image_processing(processing.to_mapping())
    source = tmp_path / "input"
    source.mkdir()
    tifffile.imwrite(source / "field.tif", np.ones((64, 64), dtype=np.uint16))

    class BorderModel(_SegmentationModel):
        def predict_image(self, image, settings, classes):
            mask = np.zeros(image.shape[:2], dtype=np.uint32)
            mask[:4, :4] = 1
            mask[20:24, 20:24] = 2
            instances = {
                key: InstanceRecord(
                    key, 0, "Cell", 0.8, "prediction", "predicted", (0, 0, 0, 0), 0
                )
                for key in (1, 2)
            }
            return ComposedInstances(mask, instances, (), {"mode": "direct"})

    model = BorderModel(tmp_path / "model.pt")
    settings = BatchInferenceSettings(
        source, source / "Prediction", InferenceSettings(tile_size=64, overlap=8),
        clear_border_instances=True,
    )
    result = TiffBatchProcessor(project, model, settings).run()

    assert (result.completed, result.failed) == (1, 0)
    metadata = json.loads((result.run_root / "analysis_metadata.json").read_text())
    plane = metadata["files"][0]["planes"][0]
    mask = tifffile.imread(result.run_root / plane["output"]["annotation"])
    assert set(np.unique(mask)) == {0, 2}
    details = json.loads((result.run_root / plane["output"]["metadata"]).read_text())
    assert details["prediction"]["tiling"]["cleared_border_ids"] == [1]


def test_batch_processes_png_bmp_and_tiff_with_one_manifest(tmp_path):
    project, processing = _project(tmp_path)
    project.lock_image_processing(processing.to_mapping())
    source = tmp_path / "input"
    source.mkdir()
    pixels = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    Image.fromarray(pixels).save(source / "field.png")
    Image.fromarray((pixels % 256).astype(np.uint8)).save(source / "field.bmp")
    tifffile.imwrite(source / "field.tif", pixels)
    model = _DetectionModel(tmp_path / "model.pt")

    assert [path.name for path in find_tiff_inputs(source)] == [
        "field.bmp", "field.png", "field.tif"
    ]
    result = TiffBatchProcessor(project, model, _settings(source)).run()

    assert (result.completed, result.failed, result.processed_planes) == (3, 0, 3)
    metadata = json.loads((result.run_root / "analysis_metadata.json").read_text())
    ids = [entry["planes"][0]["sample_id"] for entry in metadata["files"]]
    assert len(set(ids)) == 3
    for entry in metadata["files"]:
        plane = entry["planes"][0]
        assert plane["shape"] == [64, 64, 3]
        assert (result.run_root / plane["output"]["image"]).is_file()


@pytest.mark.parametrize("channels", [3, 4])
def test_batch_color_png_uses_locked_channel_axis(tmp_path, channels):
    project, _ = _project(tmp_path)
    processing = ImageProcessingSettings(2, 0, 1, 2, "min_max")
    project.lock_image_processing(processing.to_mapping())
    source = tmp_path / "input"
    source.mkdir()
    rgb = np.zeros((64, 64, channels), dtype=np.uint8)
    rgb[..., 0] = np.arange(64, dtype=np.uint8)[:, None]
    rgb[..., 1] = np.arange(64, dtype=np.uint8)[None, :]
    Image.fromarray(rgb).save(source / "rgb.png")
    model = _DetectionModel(tmp_path / "model.pt")

    result = TiffBatchProcessor(project, model, _settings(source)).run()

    assert (result.completed, result.failed, result.processed_planes) == (1, 0, 1)
    metadata = json.loads((result.run_root / "analysis_metadata.json").read_text())
    plane = metadata["files"][0]["planes"][0]
    assert plane["plane_indices"] == {}
    assert plane["shape"] == [64, 64, 3]
