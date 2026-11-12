"""Volume controls and lazy previews for the annotator's collapsible section."""

from pathlib import Path
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import dask.array as da
import numpy as np
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from ._class_map import class_color
from ._image_adapter import ImageProcessingSettings
from ._tiled_inference import InferenceCancelled
from ._volume_adapter import VolumeAdapter, VolumeAxes, VolumeInputError
from ._volume_assembly import AssemblySettings, open_assembly, save_assembly
from ._volume_inference import (
    VolumeInferenceRunner, VolumeInferenceSettings, _source_digest,
)
from ._volume_store import VolumeMaskStore
from ._volume_worker import VolumeInferenceWorker
from ._volume_zarr import open_volume_zarr, write_volume_zarr


class _ArraySource:
    """Prevent Dask copying a NumPy/memmap source during view construction."""

    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.ndim = len(data.shape)

    def __getitem__(self, index):
        return self.data[index]


def raw_volume_channels(volume):
    """Return unmodified selected source channels as lazy canonical ZYX arrays."""
    data = volume.data
    spatial_xy = {volume.selection.axes.y, volume.selection.axes.x}
    source = data if isinstance(data, da.Array) else da.from_array(
        _ArraySource(data), chunks=tuple(
            min(512, size) if axis in spatial_xy else 1
            for axis, size in enumerate(data.shape)
        ),
        name=False, asarray=False, fancy=False,
        meta=np.empty((0,) * len(data.shape), dtype=data.dtype),
    )
    axes = volume.selection.axes
    channels = sorted({c for c in volume.processing.rgb_channels if c is not None})
    if axes.channel is None:
        channels = [None]
    output = []
    for channel in channels:
        index = list(volume.selection.fixed_indices)
        index[axes.z] = slice(volume.selection.z_start, volume.selection.z_stop)
        index[axes.y] = index[axes.x] = slice(None)
        if axes.channel is not None:
            index[axes.channel] = channel
        remaining = [i for i, item in enumerate(index) if isinstance(item, slice)]
        array = source[tuple(index)].transpose(
            tuple(remaining.index(axis) for axis in axes.spatial)
        )
        output.append((channel, array))
    return output


class _VolumeLoadWorker(QThread):
    """Verify files and optional source identity away from the GUI thread."""

    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, root, *, allow_partial=False, volume=None, source=None):
        super().__init__()
        self.root = Path(root)
        self.allow_partial = allow_partial
        self.volume = volume
        self.source = source
        self._cancel = Event()

    def request_cancel(self):
        self._cancel.set()

    def run(self):
        try:
            with VolumeMaskStore.open(
                self.root, cancelled=self._cancel.is_set,
                progress=lambda n, total: self.progress.emit(n, total, "Verifying saved masks"),
            ) as store:
                manifest = store.manifest
                array = store.mask_array(unique_ids=True, allow_partial=self.allow_partial)
                matched = False
                saved = manifest["contract"]["volume"]
                if self.source is not None:
                    try:
                        selection = saved["selection"]
                        axes = selection["axes"]
                        self.volume = VolumeAdapter().prepare(
                            self.source,
                            ImageProcessingSettings.from_mapping(saved["project_processing"]),
                            axes=VolumeAxes(axes["z"], axes["y"], axes["x"], axes["channel"]),
                            fixed_indices={i: value for i, value in enumerate(selection["fixed_indices"]) if value is not None},
                            z_range=tuple(selection["z_range"]), invert=saved["inverted"],
                        )
                    except ValueError:
                        self.volume = None
                candidate = self.volume.to_mapping() if self.volume is not None else None
                if candidate is not None:
                    candidate["source_name"] = saved["source_name"]
                if candidate == saved:
                    digest = _source_digest(
                        self.volume, cancelled=self._cancel.is_set,
                        progress=lambda z: self.progress.emit(0, 1, f"Verifying source Z={z}"),
                    )
                    matched = digest == manifest["contract"]["source_pixels_sha256"]
                colors = {0: (0, 0, 0, 0), None: (0.7, 0.7, 0.7, 1)}
                records = []
                offset = 0
                for entry in manifest["slices"]:
                    if self._cancel.is_set():
                        raise InferenceCancelled("Opening volume cancelled.")
                    if entry["complete"]:
                        metadata = store.slice_metadata(entry["z_index"])
                        for local, record in metadata["instances"].items():
                            display_id = offset + int(local)
                            colors[display_id] = class_color(record["class_id"])
                            records.append({
                                "index": display_id, "z_index": entry["z_index"],
                                "local_id": int(local), "class_id": record["class_id"],
                                "class_name": record["class_name"],
                                "confidence": record["confidence"],
                                "component": metadata["provenance"].get("slice_components", {}).get(local),
                                "lineage": record["lineage"],
                            })
                        offset += entry["instance_count"]
            self.succeeded.emit((self.root, manifest, array, colors, records, matched, self.volume))
        except InferenceCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class _AssemblyWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, path, settings=None):
        super().__init__()
        self.path = Path(path)
        self.settings = settings
        self._cancel = Event()

    def request_cancel(self):
        self._cancel.set()

    def run(self):
        try:
            path = self.path
            if self.settings is not None:
                path = save_assembly(path, self.settings, cancelled=self._cancel.is_set,
                                     progress=lambda n, total: self.progress.emit(n, total, "Linking slices"))
            result = open_assembly(path, cancelled=self._cancel.is_set,
                                   progress=lambda n, total: self.progress.emit(n, total, "Verifying slice masks"))
            self.succeeded.emit((path, result))
        except InferenceCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class _ZarrWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, volume, destination):
        super().__init__()
        self.volume = volume
        self.destination = Path(destination)
        self._cancel = Event()

    def request_cancel(self):
        self._cancel.set()

    def run(self):
        try:
            path = write_volume_zarr(
                self.volume, self.destination, cancelled=self._cancel.is_set,
                progress=lambda n, total: self.progress.emit(n, total, "Writing Zarr chunks"),
            )
            self.succeeded.emit(path)
        except InferenceCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class VolumePanel(QWidget):
    busy_changed = Signal()

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.viewer = host.napari_viewer
        self.worker = None
        self.load_worker = None
        self.assembly_worker = None
        self.zarr_worker = None
        self._zarr_source = None
        self._pending_zarr_source = None
        self._run_source = None
        self._load_volume = None
        self._load_source = None
        self._sources = []
        self._selected_source = None
        self._updating = False
        self._external_busy = False
        self._views = []
        self._hidden_sources = []
        self._displayed_run = None
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.zarr_convert_button = QPushButton("Copy Selected Volume to Zarr v3")
        self.zarr_convert_button.setToolTip("Copy the selected Z range and acquisition indices as raw chunked pixels.")
        self.zarr_convert_button.clicked.connect(self._choose_zarr_output)
        self.zarr_open_button = QPushButton("Open Plugin Zarr Volume")
        self.zarr_open_button.clicked.connect(self._choose_zarr_input)
        self.axis_combos = {name: QComboBox() for name in ("Z", "Y", "X", "C")}
        for combo in self.axis_combos.values():
            combo.currentIndexChanged.connect(self._axes_changed)
        self.fixed_widget = QWidget()
        self.fixed_form = QFormLayout(self.fixed_widget)
        self.fixed_spins = {}
        self.z_start = QSpinBox()
        self.z_stop = QSpinBox()
        self.preview_z = QSpinBox()
        self.memory_mb = QSpinBox()
        self.memory_mb.setRange(16, 65536)
        self.memory_mb.setValue(1024)
        self.chunked_inference = QCheckBox("Chunked slice inference (experimental)")
        self.chunked_inference.setToolTip(
            "Use disk-backed chunks. Supports local filters and min-max, simple-max, "
            "fixed-range, dtype-range, or unfiltered 8/16-bit percentile normalization. Source-reader chunks, "
            "model memory and object metadata are outside the array estimate."
        )
        self.invert = QCheckBox("Invert volume intensities")
        self.policy = QComboBox()
        self.policy.addItem("Preserve all components", "preserve")
        self.policy.addItem("Keep largest component", "largest")
        self.processing_label = QLabel()
        self.processing_label.setWordWrap(True)
        self.readiness_label = QLabel()
        self.readiness_label.setWordWrap(True)
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Output parent (defaults to project/Prediction)")
        self.output_button = QPushButton("Choose Output Folder")
        self.output_button.clicked.connect(self._choose_output)
        self.run_input = QLineEdit()
        self.run_input.setPlaceholderText("Saved volume run folder")
        self.open_button = QPushButton("Open Saved Run")
        self.open_button.clicked.connect(self._choose_run)
        self.allow_partial = QCheckBox("Show completed slices from an incomplete run")
        self.link_pixels = QSpinBox()
        self.link_pixels.setRange(1, 2147483647)
        self.link_pixels.setValue(1)
        self.link_fraction = QDoubleSpinBox()
        self.link_fraction.setRange(0, 1)
        self.link_fraction.setSingleStep(0.05)
        self.link_fraction.setValue(0.5)
        self.link_iou = QDoubleSpinBox()
        self.link_iou.setRange(0, 1)
        self.link_iou.setSingleStep(0.05)
        self.link_pixels.setToolTip("Minimum shared XY pixels on adjacent slices.")
        self.link_fraction.setToolTip("Shared pixels divided by the smaller slice instance area.")
        self.link_iou.setToolTip("Optional minimum intersection over union for adjacent slices.")
        self.assemble_button = QPushButton("Assemble 3D Objects")
        self.assemble_button.clicked.connect(self.assemble)
        self.open_assembly_button = QPushButton("Open Assembly")
        self.open_assembly_button.clicked.connect(self._choose_assembly)
        self.preview_button = QPushButton("Preview Slice Conversion")
        self.preview_button.clicked.connect(self.preview)
        self.predict_button = QPushButton("Predict Z Slices")
        self.predict_button.clicked.connect(self.start)
        self.resume_button = QPushButton("Resume Saved Run")
        self.resume_button.clicked.connect(lambda: self.start(resume=True))
        self.cancel_button = QPushButton("Cancel 3D")
        self.cancel_button.clicked.connect(self.cancel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.status = QLabel("3D: choose a volume and a segmentation project.")
        self.status.setWordWrap(True)
        self.details = QLabel("Slice instance: select a predicted label to inspect it.")
        self.details.setWordWrap(True)
        form = QFormLayout()
        form.addRow("Source volume", self.source_combo)
        for name, combo in self.axis_combos.items():
            form.addRow(f"{name} axis", combo)
        form.addRow(self.fixed_widget)
        form.addRow("First Z (inclusive)", self.z_start)
        form.addRow("Last Z (exclusive)", self.z_stop)
        form.addRow("Preview source Z", self.preview_z)
        form.addRow("Component cleanup", self.policy)
        form.addRow("Array memory budget (MiB)", self.memory_mb)
        form.addRow(self.chunked_inference)
        form.addRow(self.invert)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_input)
        output_row.addWidget(self.output_button)
        form.addRow("Output parent", output_row)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_input)
        run_row.addWidget(self.open_button)
        form.addRow("Saved run", run_row)
        form.addRow("Z link minimum overlap (pixels)", self.link_pixels)
        form.addRow("Z link minimum overlap fraction", self.link_fraction)
        form.addRow("Z link minimum IoU", self.link_iou)
        actions = QHBoxLayout()
        for button in (self.predict_button, self.resume_button, self.cancel_button):
            actions.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        zarr_actions = QHBoxLayout()
        zarr_actions.addWidget(self.zarr_convert_button)
        zarr_actions.addWidget(self.zarr_open_button)
        layout.addLayout(zarr_actions)
        layout.addWidget(self.processing_label)
        layout.addWidget(self.readiness_label)
        layout.addWidget(self.preview_button)
        layout.addWidget(self.allow_partial)
        layout.addLayout(actions)
        assembly_actions = QHBoxLayout()
        assembly_actions.addWidget(self.assemble_button)
        assembly_actions.addWidget(self.open_assembly_button)
        layout.addLayout(assembly_actions)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.details)
        self.run_input.textChanged.connect(lambda: self.update_context())
        for control in (
            self.source_combo, *self.axis_combos.values(), self.z_start, self.z_stop,
            self.preview_z, self.policy, self.memory_mb, self.invert, self.output_input,
            self.run_input, self.allow_partial, self.preview_button,
        ):
            control.setToolTip({
                self.source_combo: "Choose a source volume; generated previews are excluded.",
                self.z_stop: "Exclusive source Z bound. To process Z 0 through 9, use 10.",
                self.allow_partial: "Missing slices display as zeros and are marked unfinished in metadata.",
                self.memory_mb: "Estimate for working arrays; model and GPU memory are additional.",
            }.get(control, "Configure or inspect the selected 3D volume."))
        try:
            self.viewer.layers.events.inserted.connect(self.refresh_sources)
            self.viewer.layers.events.removed.connect(self.refresh_sources)
        except AttributeError:
            pass
        self.refresh_sources()

    @property
    def busy(self):
        return any(worker is not None for worker in
                   (self.worker, self.load_worker, self.assembly_worker, self.zarr_worker))

    def refresh_sources(self, event=None):
        if self.busy:
            if self.worker is not None and self._run_source not in self.viewer.layers:
                self.worker.request_cancel()
                self.status.setText("3D: source closed; stopping after the current operation.")
            if self.zarr_worker is not None and not any(self._zarr_source is item for item in self.viewer.layers):
                self.zarr_worker.request_cancel()
            return
        sources = [item for item in self.viewer.layers if self.host._is_source_image_layer(item)]
        if len(sources) == len(self._sources) and all(a is b for a, b in zip(sources, self._sources)):
            return
        selected = self._selected_source
        self._sources = sources
        self._updating = True
        self.source_combo.clear()
        self.source_combo.addItem("Select a source volume", -1)
        for index, source in enumerate(sources):
            self.source_combo.addItem(source.name, index)
        chosen = next((i for i, source in enumerate(sources) if source is selected), None)
        if chosen is None and len(sources) == 1:
            chosen = 0
        self.source_combo.setCurrentIndex(0 if chosen is None else chosen + 1)
        self._updating = False
        self._source_changed()

    def _source_changed(self, index=None):
        if self._updating or self.busy:
            return
        index = self.source_combo.currentData()
        source = self._sources[index] if index is not None and index >= 0 else None
        changed_source = source is not self._selected_source
        self._selected_source = source
        self._updating = True
        shape = () if source is None else source.data.shape
        try:
            inferred = VolumeAdapter().infer_axes(source) if source is not None else None
        except (ValueError, AttributeError):
            inferred = None
        defaults = {} if inferred is None else dict(zip(("Z", "Y", "X", "C"), (*inferred.spatial, inferred.channel)))
        for name, combo in self.axis_combos.items():
            combo.clear()
            combo.addItem("None (grayscale)" if name == "C" else "Select axis", None)
            for axis, size in enumerate(shape):
                combo.addItem(f"Axis {axis} · size {size}", axis)
            combo.setCurrentIndex(0 if defaults.get(name) is None else defaults[name] + 1)
        self._updating = False
        if changed_source:
            self.z_start.setValue(0)
        self._axes_changed()

    def _axes_changed(self, index=None):
        if self._updating or self.busy:
            return
        previous = {axis: spin.value() for axis, spin in self.fixed_spins.items()}
        while self.fixed_form.rowCount():
            self.fixed_form.removeRow(0)
        self.fixed_spins = {}
        source = self._selected_source
        if source is not None:
            shape = source.data.shape
            used = {combo.currentData() for combo in self.axis_combos.values()}
            for axis, size in enumerate(shape):
                if axis in used:
                    continue
                spin = QSpinBox()
                spin.setRange(0, size - 1)
                spin.setValue(min(previous.get(axis, 0), size - 1))
                spin.setToolTip("Fixed acquisition index; objects are never linked across this axis.")
                self.fixed_form.addRow(f"Fixed axis {axis}", spin)
                self.fixed_spins[axis] = spin
            z = self.axis_combos["Z"].currentData()
            depth = shape[z] if z is not None else 1
            self.z_start.setRange(0, depth - 1)
            self.z_stop.setRange(1, depth)
            self.z_stop.setValue(depth)
            self.preview_z.setRange(0, depth - 1)
        self.update_context()
        if source is not None:
            axes = [self.axis_combos[name].currentData() for name in ("Z", "Y", "X")]
            channel = self.axis_combos["C"].currentData()
            selected = axes + ([] if channel is None else [channel])
            if None in axes or len(set(selected)) != len(selected):
                self.status.setText("3D: select distinct Z, Y and X axes; leave C as None for grayscale.")
            else:
                self.status.setText("3D: axes selected. Preview or predict the volume.")

    def prepare(self):
        source = self._selected_source
        if source is None or not any(source is item for item in self.viewer.layers):
            raise VolumeInputError("Select an open source volume.")
        axes = VolumeAxes(*(self.axis_combos[name].currentData() for name in ("Z", "Y", "X", "C")))
        return VolumeAdapter().prepare(
            source, self.host._effective_processing_settings(), axes=axes,
            fixed_indices={axis: spin.value() for axis, spin in self.fixed_spins.items()},
            z_range=(self.z_start.value(), self.z_stop.value()),
            invert=self.invert.isChecked(),
        )

    def update_context(self, *, external_busy=None):
        if external_busy is not None:
            self._external_busy = external_busy
        enabled = not self.busy and not self._external_busy
        project = self.host._project
        segment = project is not None and project.config.task == "segment"
        ready = segment and self.host._model is not None and self._selected_source is not None
        if project is None:
            reason = "Open a segmentation project to predict Z slices."
        elif not segment:
            reason = "Z-slice prediction requires a segmentation project."
        elif self.host._model is None:
            reason = "Load a segmentation model to predict Z slices."
        elif self._selected_source is None:
            reason = "Select a source volume to predict Z slices."
        elif not enabled:
            reason = "Wait for the current operation to finish."
        else:
            reason = "Ready to predict Z slices."
        self.readiness_label.setText(reason)
        self.predict_button.setToolTip(reason)
        for control in (
            self.source_combo, *self.axis_combos.values(), self.fixed_widget,
            self.z_start, self.z_stop, self.preview_z, self.memory_mb, self.policy,
            self.chunked_inference,
            self.invert, self.output_input, self.output_button, self.run_input,
            self.allow_partial,
            self.link_pixels, self.link_fraction, self.link_iou,
        ):
            control.setEnabled(enabled)
        self.preview_button.setEnabled(enabled and segment and self._selected_source is not None)
        self.predict_button.setEnabled(enabled and ready)
        self.resume_button.setEnabled(enabled and ready and bool(self.run_input.text().strip()))
        self.open_button.setEnabled(enabled)
        self.zarr_convert_button.setEnabled(enabled and self._selected_source is not None)
        self.zarr_open_button.setEnabled(enabled)
        self.assemble_button.setEnabled(enabled and bool(self.run_input.text().strip()))
        self.open_assembly_button.setEnabled(enabled)
        self.cancel_button.setEnabled(self.busy)
        try:
            recipe = self.host._effective_processing_settings()
            self.processing_label.setText(
                f"Uses shared R/G/B channels {recipe.rgb_channels}, {recipe.filter_method} filter, "
                f"and {recipe.normalization} normalization. Model and tile settings come "
                "from Large-image tiled prediction. Border objects are retained."
            )
        except ValueError:
            self.processing_label.setText("Configure channels, filtering and normalization in the shared image controls.")

    def _choose_output(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose volume prediction output parent")
        if chosen:
            self.output_input.setText(chosen)

    def _choose_zarr_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Create Zarr v3 volume", "volume.zarr", "Zarr directory (*.zarr)")
        if path:
            if not path.lower().endswith(".zarr"):
                path += ".zarr"
            self.convert_zarr(Path(path))

    def convert_zarr(self, destination):
        if self.busy or self._external_busy:
            return
        try:
            volume = self.prepare()
            self._zarr_source = self._selected_source
            self.zarr_worker = _ZarrWorker(volume, destination)
            self.zarr_worker.progress.connect(self._progress)
            self.zarr_worker.succeeded.connect(self._zarr_written)
            self.zarr_worker.failed.connect(lambda message: self.status.setText(f"3D: Zarr conversion failed · {message}"))
            self.zarr_worker.cancelled.connect(lambda: self.status.setText("3D: Zarr conversion cancelled; incomplete output cannot be opened."))
            self.zarr_worker.finished.connect(self._zarr_finished)
            self.status.setText("3D: preparing Zarr conversion.")
            self.update_context()
            self.busy_changed.emit()
            self.zarr_worker.start()
        except Exception as exc:
            self.host._show_error(f"Could not start Zarr conversion: {exc}")

    def _zarr_written(self, path):
        try:
            self._pending_zarr_source = self._add_zarr_source(path)
            self.status.setText(f"3D: Zarr v3 volume saved and opened · {path}")
        except Exception as exc:
            self.status.setText(f"3D: Zarr saved at {path}, but could not be opened: {exc}")

    def _choose_zarr_input(self):
        path = QFileDialog.getExistingDirectory(self, "Open plugin Zarr volume")
        if path:
            try:
                source = self._add_zarr_source(path)
                self.refresh_sources()
                index = next(i for i, item in enumerate(self._sources) if item is source)
                self.source_combo.setCurrentIndex(index + 1)
                self.status.setText(f"3D: opened Zarr volume · {path}")
            except Exception as exc:
                self.host._show_error(f"Could not open Zarr volume: {exc}")

    def _add_zarr_source(self, path):
        data, info = open_volume_zarr(path)
        multichannel = info["axes"] == "ZYXC"
        scale = tuple(info["scale"]) + ((1.0,) if multichannel else ())
        translate = tuple(info["translate"]) + ((0.0,) if multichannel else ())
        kwargs = {}
        if not multichannel and all(unit is not None for unit in info["units"]):
            kwargs["units"] = tuple(info["units"])
        return self.viewer.add_image(
            data, name=f"Zarr · {info['source_name']}", rgb=False,
            axis_labels=tuple(info["axes"]), scale=scale, translate=translate,
            metadata={"axes": info["axes"], "cci_zarr_source": str(path)}, **kwargs,
        )

    def _zarr_finished(self):
        worker = self.zarr_worker
        self.zarr_worker = None
        self._zarr_source = None
        if worker is not None:
            worker.deleteLater()
        self.refresh_sources()
        if self._pending_zarr_source is not None:
            index = next((i for i, item in enumerate(self._sources)
                          if item is self._pending_zarr_source), None)
            self._pending_zarr_source = None
            if index is not None:
                self.source_combo.setCurrentIndex(index + 1)
        self.update_context()
        self.busy_changed.emit()

    def _choose_run(self):
        root = self.run_input.text().strip() or QFileDialog.getExistingDirectory(self, "Open a saved volume run")
        if root:
            self.open_run(Path(root))

    def _clear_views(self):
        for layer in self._views:
            if any(layer is item for item in self.viewer.layers):
                self.viewer.layers.remove(layer)
        self._views = []
        self.details.setText("Slice instance: select a predicted label to inspect it.")
        for source, visible in self._hidden_sources:
            if any(source is item for item in self.viewer.layers):
                source.visible = visible
        self._hidden_sources = []
        self._displayed_run = None

    def _show_raw(self, volume, source):
        units = volume.geometry.units
        kwargs = {"units": units} if all(unit is not None for unit in units) else {}
        for channel, array in raw_volume_channels(volume):
            suffix = "" if channel is None else f" · C{channel}"
            layer = self.viewer.add_image(
                array, name=f"3D raw · {volume.source_name}{suffix}",
                scale=volume.geometry.scale, translate=volume.geometry.translate,
                axis_labels=("Z", "Y", "X"),
                metadata={"cci_volume_display": True, "cci_volume_role": "raw"},
                blending="additive", **kwargs,
            )
            self._views.append(layer)
        # Canonical views avoid overlaying differently ordered acquisition axes.
        self._hidden_sources.append((source, source.visible))
        source.visible = False

    def preview(self):
        if self.busy or self._external_busy:
            return
        try:
            volume = self.prepare()
            converted = volume.convert_slice(self.preview_z.value())
            self._clear_views()
            self._show_raw(volume, self._selected_source)
            translate = list(volume.geometry.translate)
            translate[0] += (converted.z_index - volume.selection.z_start) * volume.geometry.scale[0]
            layer = self.viewer.add_image(
                converted.image.data[np.newaxis], name="3D slice RGB preview", rgb=True,
                scale=volume.geometry.scale, translate=tuple(translate),
                axis_labels=("Z", "Y", "X"),
                metadata={"cci_volume_display": True, "cci_volume_role": "preview",
                          "source_z": converted.z_index},
                **({"units": volume.geometry.units}
                   if all(unit is not None for unit in volume.geometry.units) else {}),
            )
            self._views.append(layer)
            self.status.setText(f"3D: RGB conversion preview at source Z={converted.z_index}.")
        except Exception as exc:
            self.host._show_error(f"Could not preview volume: {exc}")

    def start(self, checked=False, *, resume=False):
        if self.busy or self._external_busy:
            return
        try:
            project = self.host._project
            if project is None or project.config.task != "segment" or self.host._model is None:
                raise VolumeInputError("Open a segmentation project and compatible model first.")
            volume = self.prepare()
            runner = VolumeInferenceRunner(
                volume, self.host._model, project.config.classes,
                VolumeInferenceSettings(
                    self.host._inference_settings(), component_policy=self.policy.currentData(),
                    memory_budget_bytes=self.memory_mb.value() * 1024**2,
                    chunked=self.chunked_inference.isChecked(),
                ),
            )
            runner.validate()
            if resume:
                root = Path(self.run_input.text().strip())
                if not self.run_input.text().strip():
                    raise VolumeInputError("Choose a saved run to resume.")
            else:
                parent = Path(self.output_input.text().strip()) if self.output_input.text().strip() else project.paths.root / "Prediction"
                if parent.expanduser().resolve().is_relative_to(project.paths.annotations.resolve()):
                    raise VolumeInputError("Choose an output folder outside training annotations.")
                root = parent / f"volume_{uuid4().hex[:12]}"
            self.run_input.setText(str(root))
            self._run_source = self._selected_source
            self._load_volume = volume
            self.worker = VolumeInferenceWorker(runner, root, resume=resume)
            self.worker.progress.connect(self._progress)
            self.worker.succeeded.connect(self._predicted)
            self.worker.cancelled.connect(self._predicted)
            self.worker.failed.connect(self._failed)
            self.worker.finished.connect(self._inference_finished)
            self.status.setText("3D: preparing slice inference.")
            self.busy_changed.emit()
            self.update_context()
            self.worker.start()
        except Exception as exc:
            self.host._show_error(f"Could not start volume inference: {exc}")

    def _progress(self, completed, total, text):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(completed)
        self.status.setText(f"3D: {text}")

    def _predicted(self, result):
        if result.run_root is not None:
            self.run_input.setText(str(result.run_root))
        self.status.setText(
            f"3D: {result.status} · {result.completed_slices}/{result.total_slices} slices saved."
        )
        if result.run_root is not None and (result.status == "completed" or self.allow_partial.isChecked()):
            self._start_load(result.run_root, self._load_volume, self._run_source)

    def _inference_finished(self):
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self.refresh_sources()
        self.update_context()
        self.busy_changed.emit()

    def _failed(self, message):
        self.status.setText(f"3D: failed · {message}. Completed output remains in the saved run folder.")

    def open_run(self, root):
        if self.busy or self._external_busy:
            return
        try:
            volume = self.prepare()
        except ValueError:
            volume = None
        self.run_input.setText(str(root))
        self._start_load(root, volume, self._selected_source)

    def _start_load(self, root, volume, source):
        self._load_volume = volume
        self._load_source = source
        snapshot = None
        if source is not None and any(source is item for item in self.viewer.layers):
            fields = {"data": source.data, "name": source.name}
            for name in ("metadata", "axis_labels", "scale", "translate", "units", "rgb", "rotate", "shear", "affine"):
                if hasattr(source, name):
                    value = getattr(source, name)
                    if name == "affine":
                        value = np.asarray(getattr(value, "affine_matrix", value)).copy()
                    elif isinstance(value, np.ndarray):
                        value = value.copy()
                    elif name == "metadata":
                        value = {key: value[key] for key in ("axes", "axis_labels") if key in value}
                    fields[name] = value
            snapshot = SimpleNamespace(**fields)
        self.load_worker = _VolumeLoadWorker(
            root, allow_partial=self.allow_partial.isChecked(), volume=volume,
            source=snapshot,
        )
        self.load_worker.progress.connect(self._progress)
        self.load_worker.succeeded.connect(self._loaded)
        self.load_worker.failed.connect(self._failed)
        self.load_worker.cancelled.connect(lambda: self.status.setText("3D: opening saved run cancelled."))
        self.load_worker.finished.connect(self._load_finished)
        self.load_worker.start()
        self.update_context()
        self.busy_changed.emit()

    def _loaded(self, result):
        try:
            self._display_loaded(result)
        except Exception as exc:
            self._failed(f"Could not display saved volume: {exc}")

    def _display_loaded(self, result):
        root, manifest, array, colors, records, matched, volume = result
        self._clear_views()
        self._displayed_run = Path(root).resolve()
        source_present = self._load_source is not None and any(self._load_source is item for item in self.viewer.layers)
        if matched and source_present:
            self._show_raw(volume, self._load_source)
        geometry = manifest["contract"]["volume"]["geometry"]
        units = geometry["units"]
        kwargs = {"units": tuple(units)} if all(unit is not None for unit in units) else {}
        layer = self.viewer.add_labels(
            array, name=f"3D slice predictions · {Path(root).name}",
            scale=tuple(geometry["scale"]), translate=tuple(geometry["translate"]),
            axis_labels=("Z", "Y", "X"),
            metadata={"cci_volume_display": True, "cci_volume_role": "slice_predictions",
                      "run_root": str(root), "volume_run": manifest,
                      "slice_records": records, "assembled": False},
            **kwargs,
        )
        layer.editable = False
        by_id = {record["index"]: record for record in records}
        try:
            layer.events.selected_label.connect(
                lambda event=None: self._label_details(layer, by_id)
            )
        except AttributeError:
            pass
        try:
            from napari.utils.colormaps import DirectLabelColormap

            layer.colormap = DirectLabelColormap(color_dict=colors)
        except ImportError:
            pass
        self._views.append(layer)
        completed = sum(entry["complete"] for entry in manifest["slices"])
        total = len(manifest["slices"])
        self.progress.setRange(0, total)
        self.progress.setValue(completed)
        note = "Raw volume aligned." if matched and source_present else "Raw overlay unavailable: select the matching source volume and reopen."
        partial = " Missing slices are unfinished, shown as zeros." if completed < total else ""
        self.status.setText(f"3D: {completed}/{total} slices · {len(records)} unassembled instances. {note}{partial}")

    def _label_details(self, layer, by_id):
        record = by_id.get(layer.selected_label)
        if record is None:
            self.details.setText("Slice instance: background or unavailable label.")
            return
        self.details.setText(
            f"Slice instance {record['index']} · source Z={record['z_index']} · "
            f"local ID {record['local_id']} · class {record['class_id']} "
            f"{record['class_name']} · confidence {record['confidence']}. "
            "This instance has not been assembled across Z."
        )

    def _load_finished(self):
        worker = self.load_worker
        self.load_worker = None
        if worker is not None:
            worker.deleteLater()
        self.refresh_sources()
        self.update_context()
        self.busy_changed.emit()

    def assemble(self):
        if self.busy or self._external_busy or not self.run_input.text().strip():
            return
        settings = AssemblySettings(self.link_pixels.value(), self.link_fraction.value(),
                                    self.link_iou.value())
        self._start_assembly(Path(self.run_input.text().strip()), settings)

    def _choose_assembly(self):
        folder = QFileDialog.getExistingDirectory(self, "Open saved 3D assembly")
        if folder:
            self._start_assembly(Path(folder))

    def _start_assembly(self, path, settings=None):
        if self.busy or self._external_busy:
            return
        self.assembly_worker = _AssemblyWorker(path, settings)
        self.assembly_worker.progress.connect(self._progress)
        self.assembly_worker.succeeded.connect(self._assembly_loaded)
        self.assembly_worker.failed.connect(self._failed)
        self.assembly_worker.cancelled.connect(lambda: self.status.setText("3D: assembly cancelled."))
        self.assembly_worker.finished.connect(self._assembly_finished)
        self.status.setText("3D: preparing assembly.")
        self.update_context()
        self.busy_changed.emit()
        self.assembly_worker.start()

    def _assembly_loaded(self, result):
        try:
            folder, (root, run, graph, array) = result
            if self._displayed_run != Path(root).resolve():
                self._clear_views()
            else:
                for layer in list(self._views):
                    if layer.metadata.get("cci_volume_role") in {"slice_predictions", "assembly"}:
                        if any(layer is item for item in self.viewer.layers):
                            self.viewer.layers.remove(layer)
                        self._views.remove(layer)
            self._displayed_run = Path(root).resolve()
            self.run_input.setText(str(root))
            geometry = run["contract"]["volume"]["geometry"]
            units = geometry["units"]
            kwargs = {"units": tuple(units)} if all(unit is not None for unit in units) else {}
            layer = self.viewer.add_labels(
                array, name=f"3D objects · {Path(folder).name}",
                scale=tuple(geometry["scale"]), translate=tuple(geometry["translate"]),
                axis_labels=("Z", "Y", "X"),
                metadata={"cci_volume_display": True, "cci_volume_role": "assembly",
                          "assembly_root": str(folder), "run_root": str(root),
                          "assembly_settings": graph["settings"], "assembled": True}, **kwargs,
            )
            layer.editable = False
            objects = {}
            colors = {0: (0, 0, 0, 0), None: (0.7, 0.7, 0.7, 1)}
            for node in graph["nodes"]:
                obj = node["object_id"]
                objects.setdefault(obj, []).append(node)
                colors[obj] = class_color(node["class_id"])
            node_objects = {(node["z"], node["local_id"]): node["object_id"]
                            for node in graph["nodes"]}
            link_counts = {obj: [0, 0] for obj in objects}
            for link in graph["links"]:
                affected = {node_objects[tuple(link["source"])], node_objects[tuple(link["target"])]}
                for obj in affected:
                    link_counts[obj][0 if link["accepted"] else 1] += 1
            try:
                from napari.utils.colormaps import DirectLabelColormap
                layer.colormap = DirectLabelColormap(color_dict=colors)
            except ImportError:
                pass
            try:
                layer.events.selected_label.connect(
                    lambda event=None: self._object_details(layer, objects, link_counts)
                )
            except AttributeError:
                pass
            self._views.append(layer)
            accepted = sum(link["accepted"] for link in graph["links"])
            self.status.setText(f"3D: {len(objects)} objects · {accepted} links. Saved: {folder}")
            self.details.setText("3D object: select a label to inspect its slice membership.")
        except Exception as exc:
            self._failed(f"Could not display assembly: {exc}")

    def _object_details(self, layer, objects, link_counts):
        nodes = objects.get(layer.selected_label)
        if not nodes:
            self.details.setText("3D object: background or unavailable label.")
            return
        zs = [node["z"] for node in nodes]
        accepted, rejected = link_counts[layer.selected_label]
        self.details.setText(f"3D object {layer.selected_label} · class {nodes[0]['class_id']} · "
                             f"source Z={min(zs)}–{max(zs)} · {len(nodes)} slice instances · "
                             f"{sum(node['area'] for node in nodes)} voxels · "
                             f"{accepted} accepted and {rejected} rejected overlaps.")

    def _assembly_finished(self):
        worker = self.assembly_worker
        self.assembly_worker = None
        if worker is not None:
            worker.deleteLater()
        self.refresh_sources()
        self.update_context()
        self.busy_changed.emit()

    def cancel(self):
        for worker in (self.worker, self.load_worker, self.assembly_worker, self.zarr_worker):
            if worker is not None:
                worker.request_cancel()
        self.status.setText("3D: cancellation requested; completed slices will be retained.")
        self.cancel_button.setEnabled(False)
