"""Volume section controls, asynchronous inference, and aligned lazy layers."""

import sys
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
from qtpy.QtCore import QObject, Signal

from simple_napari_cci_annotator._image_adapter import ImageProcessingSettings
from simple_napari_cci_annotator._instance_mask import PredictedInstance, compose_predictions
from simple_napari_cci_annotator._tiled_inference import InferenceSettings
from simple_napari_cci_annotator._volume_panel import VolumePanel, raw_volume_channels
from simple_napari_cci_annotator._volume_store import VolumeMaskStore
from simple_napari_cci_annotator._widget import SimpleCciAnnotatorQWidget


class Events(QObject):
    inserted = Signal()
    removed = Signal()
    active = Signal()
    current_step = Signal()
    selected_label = Signal()


class Image:
    def __init__(self, data, *, name, metadata=None, **kwargs):
        self.data = data
        self.name = name
        self.metadata = metadata or {}
        self.visible = True
        self.rgb = kwargs.pop("rgb", False)
        self.axis_labels = kwargs.pop("axis_labels", ())
        self.scale = kwargs.pop("scale", (1,) * (data.ndim - int(self.rgb)))
        self.translate = kwargs.pop("translate", (0,) * (data.ndim - int(self.rgb)))
        for key, value in kwargs.items():
            setattr(self, key, value)


class Labels(Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = Events()
        self.selected_label = 0


class Layers(list):
    def __init__(self):
        super().__init__()
        self.events = Events()
        self.selection = SimpleNamespace(active=None, events=Events())

    def append(self, item):
        super().append(item)
        self.events.inserted.emit()

    def remove(self, item):
        super().remove(item)
        self.events.removed.emit()


class Viewer:
    def __init__(self):
        self.layers = Layers()
        self.dims = SimpleNamespace(current_step=(0, 0, 0), events=Events())

    def add_image(self, data, **kwargs):
        item = Image(data, **kwargs)
        self.layers.append(item)
        return item

    def add_labels(self, data, **kwargs):
        item = Labels(data, **kwargs)
        self.layers.append(item)
        return item


class Predictor:
    task = "segment"
    names = {0: "Cell"}

    def __init__(self, path):
        self.path = path
        path.write_bytes(b"fake model")
        self.calls = 0

    def predict_image(self, image, settings, classes, *, component_policy="largest"):
        self.calls += 1
        mask = np.ones(image.shape[:2], dtype=bool)
        return compose_predictions(
            [PredictedInstance(mask, (0, 0, mask.shape[1], mask.shape[0]), 0, 0.8)],
            mask.shape, classes, component_policy=component_policy,
        )


@pytest.fixture
def panel(tmp_path, qtbot, monkeypatch):
    monkeypatch.setitem(sys.modules, "napari.utils.colormaps", SimpleNamespace(
        DirectLabelColormap=lambda **kwargs: kwargs,
    ))
    viewer = Viewer()
    viewer.add_image(
        np.arange(4 * 6 * 7, dtype=np.uint16).reshape(4, 6, 7), name="source",
        metadata={"axes": "ZYX"}, scale=(2, 0.3, 0.4), translate=(10, 20, 30),
        units=("micrometer",) * 3,
    )
    paths = SimpleNamespace(root=tmp_path / "project", annotations=tmp_path / "project" / "annotations")
    host = SimpleNamespace(
        napari_viewer=viewer, _project=SimpleNamespace(
            config=SimpleNamespace(task="segment", classes={0: "Cell"}), paths=paths,
        ),
        _model=Predictor(tmp_path / "model.pt"),
        _effective_processing_settings=lambda: ImageProcessingSettings(None, None, None, None, "min_max"),
        _inference_settings=lambda: InferenceSettings(tile_size=64, overlap=8),
        _is_source_image_layer=SimpleCciAnnotatorQWidget._is_source_image_layer,
        errors=[],
    )
    host._show_error = host.errors.append
    item = VolumePanel(host)
    qtbot.addWidget(item)
    yield item
    if item.busy:
        item.cancel()
        qtbot.waitUntil(lambda: not item.busy, timeout=15000)


def wait_idle(panel, qtbot):
    qtbot.waitUntil(lambda: not panel.busy, timeout=15000)
    assert not panel.host.errors
    assert "failed" not in panel.status.text()


def test_preview_preserves_raw_pixels_and_aligns_selected_z(panel):
    original = panel._selected_source.data.copy()
    panel.z_start.setValue(1)
    panel.z_stop.setValue(3)
    panel.preview_z.setValue(2)
    panel.preview()
    assert not panel.host.errors
    raw, preview = panel._views
    assert isinstance(raw.data, da.Array)
    np.testing.assert_array_equal(raw.data.compute(), original[1:3])
    assert raw.scale == (2, 0.3, 0.4)
    assert raw.translate == (12, 20, 30)
    assert raw.units == ("micrometer",) * 3
    assert preview.data.shape == (1, 6, 7, 3)
    assert preview.translate == (14, 20, 30)
    assert preview.metadata["source_z"] == 2
    np.testing.assert_array_equal(panel._selected_source.data, original)
    assert panel.source_combo.count() == 2
    panel._clear_views()
    assert panel._selected_source.visible


def test_predicts_and_displays_lazy_read_only_unassembled_masks(panel, qtbot):
    panel.start()
    assert panel.busy
    assert not panel.predict_button.isEnabled()
    assert panel.cancel_button.isEnabled()
    panel.viewer.dims.current_step = (2, 0, 0)
    panel.viewer.dims.events.current_step.emit()
    wait_idle(panel, qtbot)
    raw, labels = panel._views
    assert isinstance(labels.data, da.Array)
    assert not labels.editable
    assert labels.axis_labels == ("Z", "Y", "X")
    assert labels.scale == raw.scale
    assert labels.translate == raw.translate
    assert labels.units == raw.units
    assert not labels.metadata["assembled"]
    assert set(np.unique(labels.data.compute())) == {1, 2, 3, 4}
    assert "4/4 slices" in panel.status.text()
    assert "Raw volume aligned" in panel.status.text()
    assert panel.host._model.calls == 4
    labels.selected_label = 3
    labels.events.selected_label.emit()
    assert "source Z=2" in panel.details.text()
    assert "class 0 Cell" in panel.details.text()


def test_open_saved_run_restores_its_z_selection_without_model_calls(panel, qtbot):
    panel.z_start.setValue(1)
    panel.z_stop.setValue(3)
    panel.start()
    wait_idle(panel, qtbot)
    root = Path(panel.run_input.text())
    panel.z_start.setValue(0)
    panel.z_stop.setValue(4)
    panel.host._model = None
    panel.open_run(root)
    wait_idle(panel, qtbot)
    raw, labels = panel._views
    assert raw.data.shape == labels.data.shape == (2, 6, 7)
    assert raw.translate == labels.translate == (12, 20, 30)
    assert "Raw volume aligned" in panel.status.text()


def test_unmatched_source_is_not_used_as_raw_overlay(panel, qtbot):
    panel.start()
    wait_idle(panel, qtbot)
    panel._selected_source.data[0, 0, 0] += 1
    panel.open_run(Path(panel.run_input.text()))
    wait_idle(panel, qtbot)
    assert len(panel._views) == 1
    assert panel._views[0].metadata["cci_volume_role"] == "slice_predictions"
    assert "Raw overlay unavailable" in panel.status.text()


def test_source_closed_during_run_stops_safely_and_retains_saved_output(panel, qtbot):
    panel.start()
    root = Path(panel.run_input.text())
    panel.viewer.layers.remove(panel._run_source)
    qtbot.waitUntil(lambda: not panel.busy, timeout=15000)
    assert "cancelled" in panel.status.text()
    assert not panel.host.errors
    if (root / "run.json").exists():
        with VolumeMaskStore.open(root) as store:
            assert store.manifest["status"] in {"cancelled", "completed"}


def test_acquisition_axes_and_ambiguous_inputs_require_explicit_mapping(panel):
    source = panel._selected_source
    source.data = np.zeros((2, 3, 6, 7), dtype=np.uint16)
    source.metadata = {"axes": "TZYX"}
    source.scale = (1, 2, 0.3, 0.4)
    source.translate = (0, 10, 20, 30)
    source.units = ("second", "micrometer", "micrometer", "micrometer")
    panel._source_changed()
    assert set(panel.fixed_spins) == {0}
    panel.fixed_spins[0].setValue(1)
    assert panel.prepare().selection.fixed_indices == (1, None, None, None)
    source.metadata = {}
    panel._source_changed()
    assert panel.axis_combos["Z"].currentData() is None
    panel.preview()
    assert panel.host.errors
    assert "axes" in panel.host.errors[-1].lower()


def test_detection_project_keeps_volume_prediction_disabled(panel):
    panel.host._project.config.task = "detect"
    panel.update_context()
    assert not panel.predict_button.isEnabled()
    assert panel.open_button.isEnabled()


def test_raw_views_reorder_spatial_axes_and_read_one_z_chunk(panel):
    source = panel._selected_source
    original = source.data
    source.data = original.transpose(2, 0, 1)
    source.metadata = {"axes": "XZY"}
    source.scale = (0.4, 2, 0.3)
    source.translate = (30, 10, 20)
    panel._source_changed()
    raw = raw_volume_channels(panel.prepare())[0][1]
    assert raw.chunks[0] == (1, 1, 1, 1)
    np.testing.assert_array_equal(raw.compute(), original)


def test_failed_partial_run_can_be_opened_only_with_explicit_opt_in(panel, qtbot):
    predict = panel.host._model.predict_image

    def fail_second(*args, **kwargs):
        if panel.host._model.calls:
            raise RuntimeError("model failed")
        return predict(*args, **kwargs)

    panel.host._model.predict_image = fail_second
    panel.start()
    qtbot.waitUntil(lambda: not panel.busy, timeout=15000)
    root = Path(panel.run_input.text())
    panel.open_run(root)
    qtbot.waitUntil(lambda: not panel.busy, timeout=15000)
    assert "missing slices" in panel.status.text()
    panel.allow_partial.setChecked(True)
    panel.open_run(root)
    wait_idle(panel, qtbot)
    labels = panel._views[-1]
    mask = labels.data.compute()
    assert np.all(mask[0] == 1)
    assert not np.any(mask[1:])
    assert "unfinished" in panel.status.text()
    assert labels.metadata["volume_run"]["slices"][1]["complete"] is False


def test_volume_section_is_collapsible_and_locks_other_workflows(qtbot):
    widget = SimpleCciAnnotatorQWidget(Viewer())
    qtbot.addWidget(widget)
    widget.show()
    assert not widget._volume_section.content.isVisible()
    widget._volume_section.toggle_button.setChecked(True)
    assert widget._volume_section.content.isVisible()
    fake_worker = SimpleNamespace(request_cancel=lambda: None)
    widget._volume_panel.worker = fake_worker
    widget._update_action_state()
    assert not widget._new_project_button.isEnabled()
    assert not widget._tile_size_spin.isEnabled()
    event = SimpleNamespace(ignore=lambda: None)
    widget.closeEvent(event)
    assert "Close again" in widget._volume_panel.status.text()
    widget._volume_panel.worker = None
    widget._update_action_state()
