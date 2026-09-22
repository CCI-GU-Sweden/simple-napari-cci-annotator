"""Tests for project annotation, tiled inference, and retraining workflows."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from simple_napari_cci_annotator import (
    AnnotationIO,
    DatasetBuildSettings,
    DatasetBuilder,
    Detection,
    ImageAdapter,
    ImageProcessingSettings,
    InferenceSettings,
    ProjectStore,
    SimpleCciAnnotatorQWidget,
    TiledInferenceEngine,
    TrainingService,
    TrainingSettings,
    create_tile_plan,
)
from simple_napari_cci_annotator import _widget as widget_module
from simple_napari_cci_annotator import _training as training_module
from simple_napari_cci_annotator._annotation_io import (
    AnnotationError,
    LabelValidationError,
)
from simple_napari_cci_annotator._image_adapter import (
    ImageConversionError,
    normalize_to_uint8,
)
from simple_napari_cci_annotator._project_store import (
    ClassMapError,
    ImageProcessingLockedError,
    InvalidProjectError,
    ProjectConflictError,
    TrainingPatchLockedError,
)
from simple_napari_cci_annotator._training_crop import (
    TrainingCropError,
    crop_bounds_from_center,
    crop_bounds_from_rectangle,
    crop_rectangles,
    crop_sample_id,
    extract_padded_crop,
    invalid_crop_box_indices,
    validate_boxes_within_valid_crop,
)
from simple_napari_cci_annotator._tiled_inference import (
    InferenceCancelled,
    RawDetection,
    box_iou,
    class_aware_nms,
    extract_padded_tile,
)
from simple_napari_cci_annotator._training import TrainingCancelled
from simple_napari_cci_annotator._yolo_inference import YoloDetectionModel


def test_package_exports_and_version():
    import simple_napari_cci_annotator

    assert simple_napari_cci_annotator.__version__ == "0.6.2"
    assert ProjectStore is not None
    assert AnnotationIO is not None
    assert SimpleCciAnnotatorQWidget is not None


def test_initialize_and_reopen_project(tmp_path):
    root = tmp_path / "annotation-project"
    root.mkdir()

    project = ProjectStore.initialize(root)

    assert project.paths.config.is_file()
    assert project.paths.models.is_dir()
    assert project.paths.images.is_dir()
    assert project.paths.labels.is_dir()
    assert project.paths.audit.is_file()
    assert project.config.classes == {0: "LABEL"}
    assert project.config.image_processing == {
        "channels": "unset",
        "normalization": "unset",
        "locked": False,
    }
    assert project.config.training_patch == {
        "size": 1024,
        "padding_value": 114,
        "locked": False,
    }

    reopened = ProjectStore.load(root)
    assert reopened.config == project.config
    assert "schema_version: 1" in project.paths.config.read_text(encoding="utf-8")


def test_initialize_requires_empty_folder(tmp_path):
    root = tmp_path / "not-empty"
    root.mkdir()
    (root / "unrelated.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ProjectConflictError, match="empty folder"):
        ProjectStore.initialize(root)

    assert (root / "unrelated.txt").read_text(encoding="utf-8") == "keep me"


def test_project_class_map_can_grow_and_rename_but_not_orphan_labels(tmp_path):
    root = tmp_path / "multi-class-project"
    root.mkdir()
    project = ProjectStore.initialize(root, classes={0: "Cell", 1: "Debris"})

    project.update_classes({0: "Target cell", 1: "Artifact", 2: "Cluster"})
    assert ProjectStore.load(root).config.classes == {
        0: "Target cell",
        1: "Artifact",
        2: "Cluster",
    }

    (project.paths.labels / "sample.txt").write_text(
        "2 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    with pytest.raises(ClassMapError, match="used by saved annotations"):
        project.update_classes({0: "Target cell", 1: "Artifact"})
    with pytest.raises(InvalidProjectError, match="contiguous"):
        project.update_classes({0: "Target cell", 2: "Cluster"})


def test_existing_project_must_have_required_folders(tmp_path):
    root = tmp_path / "broken"
    root.mkdir()
    (root / "project.yaml").write_text(
        "schema_version: 1\n"
        "name: broken\n"
        "created_at: '2026-09-03T00:00:00+00:00'\n"
        "classes:\n  0: LABEL\n"
        "image_processing:\n"
        "  channels: unset\n"
        "  normalization: unset\n"
        "  locked: false\n",
        encoding="utf-8",
    )

    with pytest.raises(InvalidProjectError, match="missing folder"):
        ProjectStore.load(root)


@pytest.fixture
def annotation_io(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return AnnotationIO(ProjectStore.initialize(root))


def test_empty_yolo_label_is_valid(annotation_io):
    assert annotation_io.parse_yolo_text("\n") == ()


def test_yolo_load_keeps_class_properties_and_coordinates(annotation_io, tmp_path):
    label = tmp_path / "sample.txt"
    label.write_text("0 0.500000 0.250000 0.400000 0.200000\n", encoding="utf-8")

    loaded = annotation_io.load_label(label, (100, 200, 3))

    assert len(loaded.rectangles) == 1
    np.testing.assert_allclose(
        loaded.rectangles[0],
        np.asarray([[15, 60], [15, 140], [35, 140], [35, 60]]),
    )
    assert loaded.properties["class_id"].tolist() == [0]
    assert loaded.properties["class_name"].tolist() == ["LABEL"]
    assert loaded.properties["source"].tolist() == ["import"]
    assert np.isnan(loaded.properties["confidence"][0])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0 0.5 0.5 0.2\n", "expected 5 fields"),
        ("class 0.5 0.5 0.2 0.2\n", "class ID must be an integer"),
        ("1 0.5 0.5 0.2 0.2\n", "not defined"),
        ("0 1.2 0.5 0.2 0.2\n", r"within \[0, 1\]"),
        ("0 0.5 0.5 0 0.2\n", "greater than zero"),
        ("0 0.95 0.5 0.2 0.2\n", "extends outside"),
    ],
)
def test_yolo_validation_reports_bad_lines(annotation_io, text, expected):
    with pytest.raises(LabelValidationError, match=expected):
        annotation_io.parse_yolo_text(text, source="bad.txt")


def test_find_label_prefers_canonical_then_source(annotation_io, tmp_path):
    source_image = tmp_path / "images" / "sample.png"
    source_image.parent.mkdir()
    source_image.touch()
    source_label = source_image.with_suffix(".txt")
    source_label.write_text("", encoding="utf-8")

    assert annotation_io.find_label("sample", source_path=source_image) == source_label.resolve()

    canonical = annotation_io.project.paths.labels / "sample.txt"
    canonical.write_text("", encoding="utf-8")
    assert annotation_io.find_label("sample", source_path=source_image) == canonical.resolve()


def test_save_empty_label_overwrites_and_audits(annotation_io):
    image = np.zeros((32, 48, 3), dtype=np.uint8)

    first = annotation_io.save_pair(
        image_data=image,
        image_name="sample.png",
        rectangles=(),
        class_ids=(),
    )
    assert first.operation == "created"
    assert first.image_path.name == "sample.png"
    assert first.label_path.name == "sample.txt"
    assert first.label_path.read_text(encoding="utf-8") == ""

    rectangle = np.asarray([[4, 8], [4, 24], [20, 24], [20, 8]], dtype=float)
    second = annotation_io.save_pair(
        image_data=image,
        image_name="sample.png",
        rectangles=(rectangle,),
        class_ids=(0,),
    )
    assert second.operation == "updated"
    assert len(list(annotation_io.project.paths.images.iterdir())) == 1
    assert len(list(annotation_io.project.paths.labels.iterdir())) == 1
    assert second.label_path.read_text(encoding="utf-8").startswith("0 ")

    events = [
        json.loads(line)
        for line in annotation_io.project.paths.audit.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["operation"] for event in events] == ["created", "updated"]
    assert events[0]["box_count"] == 0
    assert events[1]["class_counts"] == {"0": 1}


def test_load_edit_save_close_and_reload_without_drift_or_duplicates(annotation_io):
    original_label = annotation_io.project.paths.labels / "field.txt"
    original_label.write_text(
        "0 0.500000 0.400000 0.250000 0.300000\n", encoding="utf-8"
    )
    loaded = annotation_io.load_label(original_label, (80, 120))
    edited = loaded.rectangles[0] + np.asarray([2.0, 3.0])

    saved = annotation_io.save_pair(
        image_data=np.zeros((80, 120, 3), dtype=np.uint8),
        image_name="field",
        sample_id="field",
        rectangles=(edited,),
        class_ids=loaded.properties["class_id"],
    )
    reopened_io = AnnotationIO(ProjectStore.load(annotation_io.project.paths.root))
    reloaded = reopened_io.load_label(saved.label_path, (80, 120))

    np.testing.assert_allclose(reloaded.rectangles[0], edited, atol=1e-4)
    assert len(list(reopened_io.project.paths.images.iterdir())) == 1
    assert len(list(reopened_io.project.paths.labels.iterdir())) == 1


@pytest.mark.parametrize(
    "method, lower, upper",
    [
        ("min_max", None, None),
        ("simple_max", None, None),
        ("percentile", 1.0, 99.0),
        ("z_score", -2.0, 2.0),
        ("fixed_range", 0.0, 100.0),
        ("dtype_range", None, None),
    ],
)
def test_normalization_methods_return_uint8(method, lower, upper):
    data = np.arange(100, dtype=np.uint16).reshape(10, 10)
    converted, stats = normalize_to_uint8(
        data, method=method, lower=lower, upper=upper
    )

    assert converted.shape == data.shape
    assert converted.dtype == np.uint8
    assert stats["method"] == method
    assert int(converted.min()) >= 0
    assert int(converted.max()) <= 255


def test_multidimensional_current_tz_plane_and_channel_mapping():
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    base = np.arange(30, dtype=np.uint16).reshape(5, 6)
    data[1, 2, 0] = base
    data[1, 2, 1] = base * 2
    data[1, 2, 3] = base * 3
    layer = SimpleNamespace(
        data=data,
        name="field",
        metadata={"axes": "TZCYX"},
        axis_labels=(),
        rgb=False,
    )
    viewer = SimpleNamespace(dims=SimpleNamespace(current_step=(1, 2, 0, 0, 0)))
    settings = ImageProcessingSettings(
        channel_axis=2,
        red_channel=1,
        green_channel=0,
        blue_channel=3,
        normalization="min_max",
    )

    converted = ImageAdapter().convert(layer, viewer, settings)

    assert converted.sample_id == "field__t001__z002"
    assert converted.data.shape == (5, 6, 3)
    assert converted.data.dtype == np.uint8
    assert converted.plane.non_spatial_indices == {0: 1, 1: 2}
    np.testing.assert_array_equal(converted.data[..., 0], converted.data[..., 1])
    np.testing.assert_array_equal(converted.data[..., 1], converted.data[..., 2])


def test_multiplane_sample_id_is_preserved_in_pair_names(annotation_io):
    sample_id = "field.ome__t001__z002"
    result = annotation_io.save_pair(
        image_data=np.zeros((12, 16, 3), dtype=np.uint8),
        image_name=sample_id,
        sample_id=sample_id,
        rectangles=(),
        class_ids=(),
    )

    assert result.image_path.stem == sample_id
    assert result.label_path.stem == sample_id


def test_project_image_processing_settings_lock(tmp_path):
    root = tmp_path / "locked-project"
    project = ProjectStore.initialize(root)
    settings = ImageProcessingSettings(
        channel_axis=2,
        red_channel=0,
        green_channel=1,
        blue_channel=2,
        normalization="percentile",
        lower=1.0,
        upper=99.0,
    )
    project.lock_image_processing(settings.to_mapping())

    reopened = ProjectStore.load(root)
    assert reopened.config.image_processing == settings.to_mapping()
    reopened.lock_image_processing(settings.to_mapping())

    changed = ImageProcessingSettings(
        channel_axis=2,
        red_channel=1,
        green_channel=0,
        blue_channel=2,
        normalization="percentile",
        lower=1.0,
        upper=99.0,
    )
    with pytest.raises(ImageProcessingLockedError):
        reopened.lock_image_processing(changed.to_mapping())


def test_project_training_patch_contract_locks(tmp_path):
    root = tmp_path / "patch-project"
    project = ProjectStore.initialize(root)

    project.lock_training_patch(512, padding_value=114)
    reopened = ProjectStore.load(root)

    assert reopened.config.training_patch == {
        "size": 512,
        "padding_value": 114,
        "locked": True,
    }
    reopened.lock_training_patch(512, padding_value=114)
    with pytest.raises(TrainingPatchLockedError):
        reopened.lock_training_patch(1024, padding_value=114)

    annotation_io = AnnotationIO(reopened)
    with pytest.raises(AnnotationError, match="must be 512×512"):
        annotation_io.save_pair(
            image_data=np.zeros((256, 512, 3), dtype=np.uint8),
            image_name="wrong-size",
            rectangles=(),
            class_ids=(),
        )


def test_training_crop_is_fixed_size_padded_and_coordinate_stable():
    image = np.zeros((300, 900, 3), dtype=np.uint8)
    image[:, :, 0] = 7
    bounds = crop_bounds_from_center(150, 700, 512, 300, 900)

    assert (bounds.y0, bounds.x0) == (0, 388)
    assert (bounds.valid_height, bounds.valid_width) == (300, 512)
    crop = extract_padded_crop(image, bounds, padding_value=114)
    assert crop.shape == (512, 512, 3)
    assert np.all(crop[:300, :, 0] == 7)
    assert np.all(crop[300:, :, :] == 114)

    rectangle = np.asarray(
        [[100, 500], [100, 600], [200, 600], [200, 500]], dtype=float
    )
    result = crop_rectangles(
        (rectangle,),
        {"class_id": np.asarray([1]), "source": np.asarray(["prediction"])},
        bounds,
    )
    np.testing.assert_allclose(
        result.rectangles[0],
        np.asarray([[100, 112], [100, 212], [200, 212], [200, 112]]),
    )
    assert result.properties["class_id"].tolist() == [1]
    assert crop_sample_id("field__z002", bounds).endswith(
        "__crop_y000000_x000388_s512"
    )


def test_training_crop_resnaps_clips_partial_boxes_and_blocks_padding():
    bounds = crop_bounds_from_rectangle(
        np.asarray([[10, 10], [10, 300], [200, 300], [200, 10]]),
        512,
        300,
        900,
    )
    assert bounds.as_rectangle()[2].tolist() == [512.0, 512.0]
    crossing = np.asarray(
        [[100, 490], [100, 540], [160, 540], [160, 490]], dtype=float
    )
    result = crop_rectangles(
        (crossing,), {"class_id": np.asarray([0])}, bounds
    )
    assert result.discarded_indices == ()
    assert result.clipped_count == 1
    np.testing.assert_allclose(
        result.rectangles[0],
        np.asarray([[100, 490], [100, 512], [160, 512], [160, 490]]),
    )

    tiny_remnant = np.asarray(
        [[100, 511], [100, 540], [160, 540], [160, 511]], dtype=float
    )
    discarded = crop_rectangles(
        (tiny_remnant,), {"class_id": np.asarray([0])}, bounds
    )
    assert discarded.discarded_indices == (0,)
    assert discarded.rectangles == ()

    in_padding = np.asarray(
        [[280, 10], [280, 20], [320, 20], [320, 10]], dtype=float
    )
    assert invalid_crop_box_indices((in_padding,), bounds) == (0,)
    with pytest.raises(TrainingCropError, match="padded pixels"):
        validate_boxes_within_valid_crop((in_padding,), bounds)


def test_project_and_dataset_validation_reject_bbox_in_crop_padding(tmp_path):
    root = tmp_path / "padding-project"
    project = ProjectStore.initialize(root)
    annotation_io = AnnotationIO(project)
    saved = annotation_io.save_pair(
        image_data=np.full((512, 512, 3), 114, dtype=np.uint8),
        image_name="crop",
        sample_id="crop",
        rectangles=(),
        class_ids=(),
        conversion_metadata={
            "training_crop": {
                "size": 512,
                "valid_height": 300,
                "valid_width": 512,
            }
        },
    )
    project.lock_training_patch(512)
    saved.label_path.write_text("0 0.5 0.8 0.1 0.1\n", encoding="utf-8")

    report = annotation_io.validate_project()
    preview = DatasetBuilder(project).preview(
        DatasetBuildSettings(tile_size=512, overlap=100),
        train_only=True,
        persist=False,
    )

    assert any("padding" in error for error in report.errors)
    assert any("padding" in error for error in preview.errors)


def test_invalid_percentile_settings_are_rejected():
    settings = ImageProcessingSettings(
        channel_axis=0,
        red_channel=0,
        green_channel=None,
        blue_channel=None,
        normalization="percentile",
        lower=99.0,
        upper=1.0,
    )
    with pytest.raises(ImageConversionError):
        settings.validate((4, 32, 32))


def test_project_validation_finds_pairs_and_missing_labels(annotation_io):
    image = np.zeros((16, 16), dtype=np.uint8)
    annotation_io.save_pair(
        image_data=image,
        image_name="paired",
        rectangles=(),
        class_ids=(),
    )
    (annotation_io.project.paths.images / "orphan.png").write_bytes(b"not decoded here")

    report = annotation_io.validate_project()

    assert report.image_count == 2
    assert report.label_count == 1
    assert report.paired_count == 1
    assert not report.is_valid
    assert any("Missing label" in error for error in report.errors)


def test_tile_plan_covers_edges_and_assigns_unique_ownership():
    plan = create_tile_plan(1800, 2500, tile_size=1024, overlap=200)

    assert plan.tiles[0].y0 == 0
    assert plan.tiles[0].x0 == 0
    assert max(tile.y1 for tile in plan.tiles) == 1800
    assert max(tile.x1 for tile in plan.tiles) == 2500
    assert len({(tile.row, tile.column) for tile in plan.tiles}) == len(plan.tiles)
    for y, x in ((0, 0), (900, 1000), (1799, 2499)):
        owners = [tile for tile in plan.tiles if tile.owns(x, y)]
        assert len(owners) == 1


def test_small_image_tile_is_reflect_padded_without_changing_source():
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    plan = create_tile_plan(5, 7, tile_size=16, overlap=4)

    padded = extract_padded_tile(image, plan.tiles[0], 16)

    assert padded.shape == (16, 16, 3)
    np.testing.assert_array_equal(padded[:5, :7], image)


def test_detection_centered_in_padding_is_discarded():
    class PaddingPredictor:
        def predict_tile(self, image, settings):
            return (
                RawDetection(1, 1, 5, 5, 0.8, 0),
                RawDetection(10, 10, 15, 15, 0.9, 0),
            )

    result = TiledInferenceEngine(PaddingPredictor()).predict(
        np.zeros((8, 8, 3), dtype=np.uint8),
        InferenceSettings(tile_size=64, overlap=0),
    )

    assert len(result) == 1
    np.testing.assert_allclose(
        [result[0].x1, result[0].y1, result[0].x2, result[0].y2],
        [1, 1, 5, 5],
    )


def _detection(x1, y1, x2, y2, score, class_id=0, tile_id=0, owned=True):
    return Detection(x1, y1, x2, y2, score, class_id, tile_id, owned)


def test_merge_is_class_aware_and_prefers_owner_before_confidence():
    owner = _detection(10, 10, 30, 30, 0.70, tile_id=1, owned=True)
    seam_duplicate = _detection(11, 11, 31, 31, 0.95, tile_id=0, owned=False)
    other_class = _detection(11, 11, 31, 31, 0.90, class_id=1, tile_id=0)

    merged = class_aware_nms((seam_duplicate, owner, other_class), 0.5)

    assert owner in merged
    assert seam_duplicate not in merged
    assert other_class in merged
    assert box_iou(owner, seam_duplicate) > 0.5


class _FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict_tile(self, image, settings):
        self.calls += 1
        # The same global object appears in the horizontal overlap of two tiles.
        if self.calls == 1:
            return (RawDetection(75, 20, 95, 40, 0.8, 0),)
        return (RawDetection(15, 20, 35, 40, 0.9, 0),)


def test_tiled_engine_maps_and_merges_overlap_predictions():
    predictor = _FakePredictor()
    settings = InferenceSettings(
        tile_size=100, overlap=40, merge_iou=0.5, device="cpu"
    )

    detections = TiledInferenceEngine(predictor).predict(
        np.zeros((80, 160, 3), dtype=np.uint8), settings
    )

    assert predictor.calls == 2
    assert len(detections) == 1
    assert detections[0].owned
    np.testing.assert_allclose(
        [detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2],
        [75, 20, 95, 40],
    )


def test_tiled_engine_cancels_between_tiles():
    settings = InferenceSettings(tile_size=100, overlap=40)
    checks = iter((False, True))

    with pytest.raises(InferenceCancelled):
        TiledInferenceEngine(_FakePredictor()).predict(
            np.zeros((80, 160, 3), dtype=np.uint8),
            settings,
            cancelled=lambda: next(checks),
        )


def test_yolo_adapter_preserves_rgb_semantics_for_numpy_sources():
    class FakeUltralyticsModel:
        def predict(self, **kwargs):
            self.source = kwargs["source"]
            return []

    adapter = YoloDetectionModel.__new__(YoloDetectionModel)
    adapter.model = FakeUltralyticsModel()
    rgb = np.asarray([[[10, 20, 30]]], dtype=np.uint8)

    assert adapter.predict_tile(rgb, InferenceSettings(tile_size=64)) == ()
    np.testing.assert_array_equal(
        adapter.model.source, np.asarray([[[30, 20, 10]]], dtype=np.uint8)
    )


def _save_training_sample(
    annotation_io,
    sample_id,
    *,
    source_path,
    positive=True,
    shape=(80, 120, 3),
    rectangle=None,
):
    if positive:
        rectangle = rectangle if rectangle is not None else np.asarray(
            [[20, 30], [20, 60], [45, 60], [45, 30]], dtype=float
        )
        rectangles = (rectangle,)
        class_ids = (0,)
    else:
        rectangles = ()
        class_ids = ()
    return annotation_io.save_pair(
        image_data=np.zeros(shape, dtype=np.uint8),
        image_name=f"{sample_id}.png",
        sample_id=sample_id,
        source_path=Path(source_path),
        rectangles=rectangles,
        class_ids=class_ids,
    )


def test_project_split_config_is_backward_compatible_and_persistent(tmp_path):
    project = ProjectStore.initialize(tmp_path / "project")

    assert project.config.dataset_split == {
        "seed": 42,
        "validation_fraction": 0.2,
        "assignments": {},
    }
    project.update_dataset_split(
        {"one": "train", "two": "val"}, seed=7, validation_fraction=0.25
    )

    reopened = ProjectStore.load(project.paths.root)
    assert reopened.config.dataset_split["seed"] == 7
    assert reopened.config.dataset_split["validation_fraction"] == 0.25
    assert reopened.config.dataset_split["assignments"] == {
        "one": "train",
        "two": "val",
    }


def test_grouped_split_is_stable_and_never_leaks_a_source(tmp_path):
    project = ProjectStore.initialize(tmp_path / "project")
    annotation_io = AnnotationIO(project)
    _save_training_sample(
        annotation_io, "field_a_z0", source_path=tmp_path / "source_a.png"
    )
    _save_training_sample(
        annotation_io, "field_a_z1", source_path=tmp_path / "source_a.png"
    )
    _save_training_sample(
        annotation_io,
        "field_b",
        source_path=tmp_path / "source_b.png",
        positive=False,
    )
    settings = DatasetBuildSettings(tile_size=64, overlap=16, seed=11)

    first = DatasetBuilder(project).preview(settings)

    assert first.is_valid
    assert first.assignments["field_a_z0"] == first.assignments["field_a_z1"]
    assert set(first.assignments.values()) == {"train", "val"}
    original = dict(first.assignments)

    _save_training_sample(
        annotation_io, "field_c", source_path=tmp_path / "source_c.png"
    )
    second = DatasetBuilder(project).preview(settings)

    assert second.is_valid
    assert all(second.assignments[key] == value for key, value in original.items())


def test_appended_audit_metadata_merges_and_survives_later_saves(tmp_path):
    project = ProjectStore.initialize(tmp_path / "project")
    annotation_io = AnnotationIO(project)
    _save_training_sample(
        annotation_io, "field_a", source_path=tmp_path / "source_a.png"
    )
    with project.paths.audit.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "operation": "metadata",
                    "sample_id": "field_a",
                    "metadata": {"patient": "P001", "well": "A01"},
                }
            )
            + "\n"
        )
        stream.write(
            json.dumps(
                {
                    "operation": "metadata",
                    "sample_id": "field_a",
                    "metadata": {"well": "A02"},
                }
            )
            + "\n"
        )
    _save_training_sample(
        annotation_io, "field_a", source_path=tmp_path / "source_a.png"
    )

    preview = DatasetBuilder(project).preview(
        DatasetBuildSettings(tile_size=64, overlap=16, group_field="metadata.patient"),
        train_only=True,
    )
    events = DatasetBuilder(project)._latest_audit_events()

    assert preview.samples[0].group == "metadata:P001"
    assert events["field_a"]["metadata"] == {
        "patient": "P001",
        "well": "A02",
    }


def test_one_source_requires_explicit_train_only_mode(tmp_path):
    project = ProjectStore.initialize(tmp_path / "project")
    annotation_io = AnnotationIO(project)
    _save_training_sample(
        annotation_io, "only", source_path=tmp_path / "only.png"
    )
    builder = DatasetBuilder(project)
    settings = DatasetBuildSettings(tile_size=64, overlap=16)

    blocked = builder.preview(settings)
    exploratory = builder.preview(settings, train_only=True)

    assert not blocked.is_valid
    assert any("at least two source groups" in error for error in blocked.errors)
    assert exploratory.is_valid
    assert exploratory.validation_mode == "none"
    assert exploratory.assignments == {"only": "train"}

    run_root = tmp_path / "train-only-run"
    run_root.mkdir()
    snapshot = builder.create_snapshot(run_root, exploratory, settings)
    assert "val: images/train" in snapshot.dataset_yaml.read_text(encoding="utf-8")


def test_snapshot_has_matching_pairs_manifests_and_no_group_leakage(tmp_path):
    project = ProjectStore.initialize(tmp_path / "project")
    annotation_io = AnnotationIO(project)
    _save_training_sample(
        annotation_io, "one", source_path=tmp_path / "one.png"
    )
    _save_training_sample(
        annotation_io,
        "two",
        source_path=tmp_path / "two.png",
        positive=False,
    )
    settings = DatasetBuildSettings(
        tile_size=64, overlap=16, negative_tile_ratio=0.5, seed=5
    )
    builder = DatasetBuilder(project)
    preview = builder.preview(settings)
    run_root = tmp_path / "run"
    run_root.mkdir()

    snapshot = builder.create_snapshot(run_root, preview, settings)

    assert snapshot.dataset_yaml.is_file()
    assert snapshot.split_manifest.is_file()
    assert snapshot.tile_manifest.is_file()
    for split in ("train", "val"):
        image_stems = {
            path.stem for path in (snapshot.dataset_root / "images" / split).glob("*.png")
        }
        label_stems = {
            path.stem for path in (snapshot.dataset_root / "labels" / split).glob("*.txt")
        }
        assert image_stems == label_stems
    manifest = snapshot.split_manifest.read_text(encoding="utf-8")
    assert "image_sha256" in manifest
    assert "label_sha256" in manifest


def test_large_bbox_rejected_when_tile_clipping_is_too_severe(tmp_path):
    project = ProjectStore.initialize(tmp_path / "project")
    annotation_io = AnnotationIO(project)
    huge = np.asarray([[10, 10], [10, 190], [50, 190], [50, 10]], dtype=float)
    _save_training_sample(
        annotation_io,
        "huge",
        source_path=tmp_path / "huge.png",
        shape=(64, 200, 3),
        rectangle=huge,
    )
    settings = DatasetBuildSettings(tile_size=64, overlap=0)

    preview = DatasetBuilder(project).preview(settings, train_only=True)

    assert not preview.is_valid
    assert preview.box_counts["train"] == 0
    assert any("cannot meet" in warning for warning in preview.warnings)
    assert any("No usable training tiles" in error for error in preview.errors)


def test_training_service_creates_timestamped_run_and_provenance(tmp_path):
    project = ProjectStore.initialize(tmp_path / "project")
    annotation_io = AnnotationIO(project)
    _save_training_sample(
        annotation_io, "one", source_path=tmp_path / "one.png"
    )
    _save_training_sample(
        annotation_io, "two", source_path=tmp_path / "two.png", positive=False
    )
    dataset_settings = DatasetBuildSettings(tile_size=64, overlap=16, seed=3)
    preview = DatasetBuilder(project).preview(dataset_settings)
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"model")

    class FakeModel:
        task = "detect"

        def add_callback(self, name, callback):
            self.callback = callback

        def train(self, **kwargs):
            weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")
            self.callback(SimpleNamespace(epoch=0, metrics={}))

    destination = tmp_path / "outputs"
    settings = TrainingSettings(
        model_path=base_model,
        destination=destination,
        dataset=dataset_settings,
        epochs=1,
        batch=1,
        patience=0,
    )
    with patch.object(training_module, "_load_ultralytics_model", return_value=FakeModel()):
        result = TrainingService(project).run(settings, preview)

    assert result.status == "completed"
    assert result.run_root.parent == destination.resolve()
    assert result.run_root.name.startswith("retrain_")
    assert result.best_model.is_file()
    run_yaml = result.run_root / "run.yaml"
    text = run_yaml.read_text(encoding="utf-8")
    assert "status: completed" in text
    assert "plugin_version: 0.6.2" in text
    assert "sha256:" in text
    assert (result.run_root / "dataset" / "tile_manifest.csv").is_file()


def test_training_cancellation_marks_partial_run(tmp_path):
    project = ProjectStore.initialize(tmp_path / "project")
    annotation_io = AnnotationIO(project)
    _save_training_sample(
        annotation_io, "only", source_path=tmp_path / "only.png"
    )
    dataset_settings = DatasetBuildSettings(tile_size=64, overlap=0)
    preview = DatasetBuilder(project).preview(dataset_settings, train_only=True)
    model = tmp_path / "base.pt"
    model.write_bytes(b"model")
    destination = tmp_path / "outputs"
    settings = TrainingSettings(
        model_path=model,
        destination=destination,
        dataset=dataset_settings,
        train_only=True,
    )

    with pytest.raises(TrainingCancelled):
        TrainingService(project).run(
            settings, preview, cancelled=lambda: True
        )

    runs = list(destination.glob("retrain_*"))
    assert len(runs) == 1
    assert "status: cancelled" in (runs[0] / "run.yaml").read_text(encoding="utf-8")


class _Signal:
    def connect(self, callback):
        self.callback = callback


class _Image:
    def __init__(self, data, name="sample", path=None, metadata=None, **kwargs):
        self.data = data
        self.name = name
        self.metadata = metadata or {}
        self.source = SimpleNamespace(path=str(path) if path else None)
        self.rgb = data.ndim == 3 and data.shape[-1] in {3, 4}
        self.axis_labels = ()
        self.visible = True


class _Shapes:
    def __init__(self, data, name, properties=None, **kwargs):
        self.data = data
        self.name = name
        self.properties = properties or {}
        self.metadata = {}
        self.current_properties = {}
        self.current_edge_color = None
        self.edge_color = kwargs.get("edge_color")
        self.face_color = kwargs.get("face_color")
        self.selected_data = set()
        self.events = SimpleNamespace(data=_Signal())
        self.visible = True
        self.mode = "select"


class _Layers(list):
    def __init__(self):
        super().__init__()
        self.selection = SimpleNamespace(
            active=None,
            events=SimpleNamespace(active=_Signal()),
        )
        self.events = SimpleNamespace(inserted=_Signal(), removed=_Signal())


class _Viewer:
    def __init__(self):
        self.layers = _Layers()
        self.dims = SimpleNamespace(
            current_step=(0, 0, 0),
            events=SimpleNamespace(current_step=_Signal()),
        )

    def add_shapes(self, data, *, name, properties=None, **kwargs):
        layer = _Shapes(data, name, properties, **kwargs)
        self.layers.append(layer)
        return layer

    def add_image(self, data, *, name, metadata=None, **kwargs):
        layer = _Image(data, name=name, metadata=metadata, **kwargs)
        self.layers.append(layer)
        return layer


def test_widget_starts_with_disabled_model_controls(qtbot):
    widget = SimpleCciAnnotatorQWidget(_Viewer())
    qtbot.addWidget(widget)

    assert widget._project is None
    assert widget._new_project_button.text() == "New Project"
    assert widget._model is None
    assert not widget._choose_model_button.isEnabled()
    assert not widget._predict_button.isEnabled()
    assert not widget._save_annotation_button.isEnabled()


def test_widget_sections_collapse_and_scroll(qtbot):
    widget = SimpleCciAnnotatorQWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.resize(420, 300)
    widget.show()

    assert widget._project_section.content.isVisible()
    assert not widget._inference_section.content.isVisible()
    assert not widget._retrain_section.content.isVisible()

    widget._inference_section.toggle_button.setChecked(True)
    widget._retrain_section.toggle_button.setChecked(True)
    qtbot.wait(10)

    assert widget._inference_section.content.isVisible()
    assert widget._retrain_section.content.isVisible()
    assert widget._scroll_area.verticalScrollBar().maximum() > 0


def test_widget_new_project_and_automatic_bbox_loading(tmp_path, qtbot):
    project_root = tmp_path / "project"
    project_root.mkdir()
    source_image = tmp_path / "source" / "field.png"
    source_image.parent.mkdir()
    source_image.touch()
    source_image.with_suffix(".txt").write_text(
        "0 0.5 0.5 0.25 0.5\n", encoding="utf-8"
    )

    viewer = _Viewer()
    image_layer = _Image(
        np.zeros((100, 200, 3), dtype=np.uint8),
        name="field",
        path=source_image,
    )
    viewer.layers.append(image_layer)
    viewer.layers.selection.active = image_layer
    widget = SimpleCciAnnotatorQWidget(viewer)
    qtbot.addWidget(widget)

    with (
        patch.object(
            widget_module.QFileDialog,
            "getExistingDirectory",
            return_value=str(project_root),
        ),
        patch.object(widget, "_show_info"),
    ):
        widget._on_new_project()

    shapes = next(layer for layer in viewer.layers if isinstance(layer, _Shapes))
    assert widget._project is not None
    assert len(shapes.data) == 1
    assert shapes.properties["class_id"].tolist() == [0]
    assert shapes.properties["class_name"].tolist() == ["LABEL"]
    assert shapes.metadata["cci_label_path"] == str(source_image.with_suffix(".txt").resolve())
    assert widget._save_annotation_button.text() == "Import Converted Image + BBoxes"


def test_widget_assigns_selected_boxes_to_current_project_class(tmp_path, qtbot):
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectStore.initialize(root, classes={0: "Cell", 1: "Debris"})
    viewer = _Viewer()
    widget = SimpleCciAnnotatorQWidget(viewer)
    qtbot.addWidget(widget)
    widget._set_project(project)
    shapes = viewer.add_shapes(
        [np.asarray([[10, 10], [10, 20], [20, 20], [20, 10]])],
        name=widget.ANNOTATION_LAYER_NAME,
        properties={
            "class_id": np.asarray([0]),
            "class_name": np.asarray(["Cell"], dtype=object),
            "confidence": np.asarray([0.9]),
            "source": np.asarray(["prediction"], dtype=object),
            "tile_id": np.asarray([3]),
        },
    )
    shapes.selected_data = {0}
    widget._class_combo.setCurrentIndex(widget._class_combo.findData(1))

    widget._on_apply_class_to_selected()

    assert shapes.properties["class_id"].tolist() == [1]
    assert shapes.properties["class_name"].tolist() == ["Debris"]
    assert shapes.properties["source"].tolist() == ["manual"]
    assert np.isnan(shapes.properties["confidence"][0])
    assert "1 Debris: 1" in widget._class_counts_label.text()
    assert widget._annotation_dirty


def test_widget_creates_and_saves_fixed_training_crop(tmp_path, qtbot):
    root = tmp_path / "project"
    root.mkdir()
    source_path = tmp_path / "field.png"
    source_path.touch()
    project = ProjectStore.initialize(root)
    viewer = _Viewer()
    source = _Image(np.zeros((300, 800, 3), dtype=np.uint8), path=source_path)
    viewer.layers.append(source)
    viewer.layers.selection.active = source
    widget = SimpleCciAnnotatorQWidget(viewer)
    qtbot.addWidget(widget)
    widget._set_project(project)
    assert not widget._save_annotation_button.isEnabled()
    source_boxes = widget._annotation_layer()
    source_boxes.data = [
        np.asarray([[100, 200], [100, 300], [200, 300], [200, 200]]),
        np.asarray([[100, 630], [100, 750], [200, 750], [200, 630]]),
    ]
    source_boxes.properties = {
        "class_id": np.asarray([0, 0]),
        "class_name": np.asarray(["LABEL", "LABEL"], dtype=object),
        "confidence": np.asarray([0.8, 0.7]),
        "source": np.asarray(["prediction", "prediction"], dtype=object),
        "tile_id": np.asarray([1, 2]),
    }
    widget._patch_size_combo.setCurrentIndex(
        widget._patch_size_combo.findData(512)
    )

    widget._on_select_training_crop()
    widget._on_create_training_crop()

    crop_image = widget._get_layer_by_name(widget.CROP_IMAGE_LAYER_NAME)
    crop_boxes = widget._crop_bbox_layer()
    assert crop_image.data.shape == (512, 512, 3)
    assert len(crop_boxes.data) == 2
    assert "1 clipped at crop boundaries" in widget._crop_status_label.text()
    assert widget._save_crop_button.isEnabled()

    valid_box = np.asarray(crop_boxes.data[0]).copy()
    crop_boxes.data[0] = np.asarray(
        [[280, 40], [280, 90], [320, 90], [320, 40]], dtype=float
    )
    widget._on_crop_shapes_changed()
    assert widget._crop_invalid_indices == (0,)
    assert not widget._save_crop_button.isEnabled()
    np.testing.assert_allclose(crop_boxes.face_color[0], [1, 0, 0, 0.35])
    np.testing.assert_allclose(crop_boxes.face_color[1], [0, 0, 0, 0])
    assert "zero-based indices: 0" in widget._crop_status_label.text()

    crop_boxes.data[0] = valid_box
    widget._on_crop_shapes_changed()
    assert widget._crop_invalid_indices == ()
    assert widget._save_crop_button.isEnabled()
    np.testing.assert_allclose(crop_boxes.face_color, np.zeros((2, 4)))
    assert widget._save_training_crop(show_message=False)
    assert project.config.training_patch == {
        "size": 512,
        "padding_value": 114,
        "locked": True,
    }
    assert len(list(project.paths.images.iterdir())) == 1
    assert len(list(project.paths.labels.iterdir())) == 1
