"""Tests for simple-napari-cci-annotator.

Covers:
- Package/widget imports
- Pure-Python yolo_utils helpers (no GPU / YOLO model required)
- Widget instantiation (CCIYoloWrapper mocked)
- Default model bootstrap (bundled yolov8n.pt copied into empty folder)
- Training config write / read round-trip
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Import smoke tests
# ---------------------------------------------------------------------------


def test_package_importable():
    import simple_napari_cci_annotator  # noqa: F401


def test_version_string():
    import simple_napari_cci_annotator

    assert isinstance(simple_napari_cci_annotator.__version__, str)
    assert simple_napari_cci_annotator.__version__


def test_widget_class_importable():
    from simple_napari_cci_annotator import SimpleCciAnnotatorQWidget  # noqa: F401


# ---------------------------------------------------------------------------
# 2. _yolo_utils: _points_to_yolo_xywh
# ---------------------------------------------------------------------------


def test_points_to_yolo_xywh_empty():
    from simple_napari_cci_annotator._yolo_utils import _points_to_yolo_xywh

    assert _points_to_yolo_xywh([]) is None


def test_points_to_yolo_xywh_degenerate_zero_width():
    from simple_napari_cci_annotator._yolo_utils import _points_to_yolo_xywh

    # All x values equal → width == 0
    assert _points_to_yolo_xywh([(0.5, 0.2), (0.5, 0.8)]) is None


def test_points_to_yolo_xywh_degenerate_zero_height():
    from simple_napari_cci_annotator._yolo_utils import _points_to_yolo_xywh

    # All y values equal → height == 0
    assert _points_to_yolo_xywh([(0.2, 0.5), (0.8, 0.5)]) is None


def test_points_to_yolo_xywh_normal():
    from simple_napari_cci_annotator._yolo_utils import _points_to_yolo_xywh

    # Rectangle from (0.0, 0.0) to (1.0, 1.0) → center 0.5,0.5, wh 1.0,1.0
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    result = _points_to_yolo_xywh(pts)
    assert result is not None
    cx, cy, w, h = result
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert w == pytest.approx(1.0)
    assert h == pytest.approx(1.0)


def test_points_to_yolo_xywh_clamped():
    from simple_napari_cci_annotator._yolo_utils import _points_to_yolo_xywh

    # Points outside [0,1] should be clamped
    pts = [(-0.5, -0.5), (1.5, 1.5)]
    result = _points_to_yolo_xywh(pts)
    assert result is not None
    cx, cy, w, h = result
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    assert 0.0 <= w <= 1.0
    assert 0.0 <= h <= 1.0


# ---------------------------------------------------------------------------
# 3. _yolo_utils: save_vectors_to_txt
# ---------------------------------------------------------------------------


def test_save_vectors_to_txt(tmp_path):
    from simple_napari_cci_annotator._yolo_utils import save_vectors_to_txt

    vectors = [
        (0, [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]),
        (1, [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]),
    ]
    out = tmp_path / "labels.txt"
    save_vectors_to_txt(vectors, out)

    lines = out.read_text().splitlines()
    assert len(lines) == 2
    parts0 = lines[0].split()
    assert parts0[0] == "0"
    assert len(parts0) == 5  # class cx cy w h


def test_save_vectors_to_txt_skips_degenerate(tmp_path):
    from simple_napari_cci_annotator._yolo_utils import save_vectors_to_txt

    # Single point → degenerate, should produce no output line
    vectors = [(0, [(0.5, 0.5)])]
    out = tmp_path / "labels.txt"
    save_vectors_to_txt(vectors, out)
    assert out.read_text().strip() == ""


# ---------------------------------------------------------------------------
# 4. _yolo_utils: create_training_set
# ---------------------------------------------------------------------------


def test_create_training_set_structure(tmp_path):
    from simple_napari_cci_annotator._yolo_utils import create_training_set

    src_images = tmp_path / "images"
    src_labels = tmp_path / "labels"
    src_images.mkdir()
    src_labels.mkdir()
    dst = tmp_path / "dataset"

    # Create 4 dummy images + matching label files
    for i in range(4):
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img.save(src_images / f"img{i:02d}.png")
        (src_labels / f"img{i:02d}.txt").write_text(
            f"0 0.5 0.5 0.2 0.2\n"
        )

    create_training_set(
        path_to_images=src_images,
        path_to_vectors=src_labels,
        destination_path=dst,
        label_names=[(0, "LABEL")],
    )

    # Expected sub-directories
    assert (dst / "images" / "train").exists()
    assert (dst / "images" / "val").exists()
    assert (dst / "labels" / "train").exists()
    assert (dst / "labels" / "val").exists()

    # dataset.yaml should exist and reference the right keys
    yaml_text = (dst / "dataset.yaml").read_text()
    assert "train:" in yaml_text
    assert "val:" in yaml_text
    assert "LABEL" in yaml_text

    # All images should have been distributed
    train_imgs = list((dst / "images" / "train").iterdir())
    val_imgs = list((dst / "images" / "val").iterdir())
    assert len(train_imgs) + len(val_imgs) == 4


# ---------------------------------------------------------------------------
# 5. Widget: training config round-trip
# ---------------------------------------------------------------------------


def _make_widget(qtbot):
    """Return a SimpleCciAnnotatorQWidget with a mocked napari viewer."""
    from simple_napari_cci_annotator._widget import SimpleCciAnnotatorQWidget

    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))
    w = SimpleCciAnnotatorQWidget(viewer)
    qtbot.addWidget(w)
    return w


def test_create_training_config_defaults(tmp_path, qtbot):
    w = _make_widget(qtbot)
    w._create_training_config(tmp_path)

    cfg_file = tmp_path / "training_config.json"
    assert cfg_file.exists()
    cfg = json.loads(cfg_file.read_text())
    assert cfg["epochs"] == 100
    assert cfg["batch"] == 8
    assert cfg["patience"] == 30
    assert cfg["image_size"] == 640


def test_create_training_config_custom_image_size(tmp_path, qtbot):
    w = _make_widget(qtbot)
    w._create_training_config(tmp_path, image_size=1024)

    cfg = json.loads((tmp_path / "training_config.json").read_text())
    assert cfg["image_size"] == 1024


def test_load_training_config_reads_file(tmp_path, qtbot):
    w = _make_widget(qtbot)

    custom = {"image_size": 512, "batch": 4, "epochs": 50, "patience": 10}
    (tmp_path / "training_config.json").write_text(json.dumps(custom))

    loaded = w._load_training_config(tmp_path)
    assert loaded == custom


def test_load_training_config_falls_back_to_defaults(tmp_path, qtbot):
    w = _make_widget(qtbot)

    loaded = w._load_training_config(tmp_path)
    assert loaded["image_size"] == 640
    assert loaded["epochs"] == 100


# ---------------------------------------------------------------------------
# 6. Widget: instantiation (requires Qt; pytest-qt provides qtbot)
# ---------------------------------------------------------------------------


def test_widget_instantiation(qtbot):
    from simple_napari_cci_annotator._widget import SimpleCciAnnotatorQWidget

    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))

    with patch("simple_napari_cci_annotator._widget.CCIYoloWrapper"):
        w = SimpleCciAnnotatorQWidget(viewer)
        qtbot.addWidget(w)

    assert w is not None
    assert w._yolo is None
    assert w._model_path is None


# ---------------------------------------------------------------------------
# 7. Widget: default model bootstrap
# ---------------------------------------------------------------------------


def test_load_model_empty_folder_copies_bundled(tmp_path, qtbot):
    """Loading from an empty folder copies the bundled yolov8n.pt there."""
    from simple_napari_cci_annotator._widget import SimpleCciAnnotatorQWidget

    # Patch CCIYoloWrapper so no real YOLO model is loaded
    mock_wrapper = MagicMock()
    bundled_pt = (
        Path(__file__).parent.parent
        / "src"
        / "simple_napari_cci_annotator"
        / "models"
        / "yolov8n.pt"
    )
    if not bundled_pt.exists():
        pytest.skip("Bundled yolov8n.pt not present in repo")

    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))

    with patch(
        "simple_napari_cci_annotator._widget.CCIYoloWrapper",
        return_value=mock_wrapper,
    ):
        w = SimpleCciAnnotatorQWidget(viewer)
        qtbot.addWidget(w)

        model_folder = tmp_path / "my_model"
        model_folder.mkdir()

        w._model_path_input.setText(str(model_folder))

        # Suppress the info dialogs
        with patch.object(w, "_show_info"):
            w._on_load_model()

    copied_pt = model_folder / "yolov8n.pt"
    assert copied_pt.exists(), "Bundled model should have been copied to the folder"
    assert w._model_path == copied_pt


def test_load_model_empty_path_shows_error(tmp_path, qtbot):
    from simple_napari_cci_annotator._widget import SimpleCciAnnotatorQWidget

    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))

    with patch("simple_napari_cci_annotator._widget.CCIYoloWrapper"):
        w = SimpleCciAnnotatorQWidget(viewer)
        qtbot.addWidget(w)

        w._model_path_input.setText("")
        with patch.object(w, "_show_error") as mock_err:
            w._on_load_model()
            mock_err.assert_called_once()


def test_load_model_existing_pt_in_folder(tmp_path, qtbot):
    """If the folder already has a .pt file, that one is loaded (no copy)."""
    from simple_napari_cci_annotator._widget import SimpleCciAnnotatorQWidget

    model_folder = tmp_path / "my_model"
    model_folder.mkdir()
    fake_pt = model_folder / "custom.pt"
    fake_pt.write_bytes(b"fake")

    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))

    mock_wrapper = MagicMock()
    with patch(
        "simple_napari_cci_annotator._widget.CCIYoloWrapper",
        return_value=mock_wrapper,
    ):
        w = SimpleCciAnnotatorQWidget(viewer)
        qtbot.addWidget(w)

        w._model_path_input.setText(str(model_folder))
        with patch.object(w, "_show_info"):
            w._on_load_model()

    assert w._model_path == fake_pt
    # The existing .pt must not have been overwritten
    assert fake_pt.read_bytes() == b"fake"
