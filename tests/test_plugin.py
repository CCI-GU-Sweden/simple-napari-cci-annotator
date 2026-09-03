"""Tests for project initialization and annotation-only workflows."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from simple_napari_cci_annotator import (
    AnnotationIO,
    ProjectStore,
    SimpleCciAnnotatorQWidget,
)
from simple_napari_cci_annotator import _widget as widget_module
from simple_napari_cci_annotator._annotation_io import LabelValidationError
from simple_napari_cci_annotator._project_store import (
    InvalidProjectError,
    ProjectConflictError,
)


def test_package_exports_and_version():
    import simple_napari_cci_annotator

    assert simple_napari_cci_annotator.__version__ == "0.1.0"
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


class _Signal:
    def connect(self, callback):
        self.callback = callback


class _Image:
    def __init__(self, data, name="sample", path=None):
        self.data = data
        self.name = name
        self.metadata = {}
        self.source = SimpleNamespace(path=str(path) if path else None)


class _Shapes:
    def __init__(self, data, name, properties):
        self.data = data
        self.name = name
        self.properties = properties
        self.metadata = {}
        self.current_properties = {}


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

    def add_shapes(self, data, *, name, properties, **kwargs):
        layer = _Shapes(data, name, properties)
        self.layers.append(layer)
        return layer


def test_widget_starts_without_model_controls(qtbot):
    widget = SimpleCciAnnotatorQWidget(_Viewer())
    qtbot.addWidget(widget)

    assert widget._project is None
    assert widget._new_project_button.text() == "New Project"
    assert not hasattr(widget, "_model_path_input")
    assert not widget._save_annotation_button.isEnabled()


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
    assert widget._save_annotation_button.text() == "Import Annotation"
