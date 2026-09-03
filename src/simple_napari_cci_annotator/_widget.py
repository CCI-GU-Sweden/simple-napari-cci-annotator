from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._annotation_io import AnnotationError, AnnotationIO, LabelValidationError
from ._image_adapter import (
    NORMALIZATION_METHODS,
    ConvertedImage,
    ImageAdapter,
    ImageConversionError,
    ImageProcessingSettings,
)
from ._project_store import ProjectError, ProjectStore


class SimpleCciAnnotatorQWidget(QWidget):
    """Project-based YOLO bbox annotation UI.

    This milestone intentionally contains no model loading, inference, or
    retraining controls. It initializes projects and loads/saves bbox labels.
    """

    ANNOTATION_LAYER_NAME = "yolo_bboxes"

    def __init__(self, napari_viewer):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.setWindowTitle("Simple CCI Annotator")

        self._project: ProjectStore | None = None
        self._annotation_io: AnnotationIO | None = None
        self._image_adapter = ImageAdapter()
        self._annotation_image_layer = None
        self._converted_image: ConvertedImage | None = None
        self._current_sample_id: str | None = None
        self._annotation_dirty = False
        self._locked_processing_settings: ImageProcessingSettings | None = None
        self._updating_processing_controls = False
        self._last_normalization_method: str | None = None

        self._project_path_label = QLabel("No project selected")
        self._project_path_label.setWordWrap(True)
        self._project_status_label = QLabel(
            "Create a new project or open an initialized project."
        )
        self._project_status_label.setWordWrap(True)

        self._new_project_button = QPushButton("New Project")
        self._new_project_button.clicked.connect(self._on_new_project)
        self._open_project_button = QPushButton("Open Project")
        self._open_project_button.clicked.connect(self._on_open_project)

        self._image_status_label = QLabel("Select an image layer in napari.")
        self._image_status_label.setWordWrap(True)
        self._plane_status_label = QLabel("Current sample: unavailable")
        self._plane_status_label.setWordWrap(True)

        self._channel_axis_combo = QComboBox()
        self._channel_axis_combo.setToolTip(
            "Axis containing the source channels. Z and T remain controlled by napari."
        )
        self._channel_axis_combo.currentIndexChanged.connect(
            self._on_channel_axis_changed
        )
        self._red_channel_combo = QComboBox()
        self._green_channel_combo = QComboBox()
        self._blue_channel_combo = QComboBox()
        for combo in (
            self._red_channel_combo,
            self._green_channel_combo,
            self._blue_channel_combo,
        ):
            combo.setToolTip(
                "Choose the source channel mapped into this RGB output component."
            )
            combo.currentIndexChanged.connect(self._on_processing_value_changed)

        self._normalization_combo = QComboBox()
        self._normalization_combo.setToolTip(
            "Scaling is applied independently to each selected channel of the current "
            "2D plane. Preview it before the first save; it then locks for the project."
        )
        for method, display_name in NORMALIZATION_METHODS.items():
            self._normalization_combo.addItem(display_name, method)
        self._normalization_combo.setCurrentIndex(
            self._normalization_combo.findData("percentile")
        )
        self._normalization_combo.currentIndexChanged.connect(
            self._on_normalization_changed
        )

        self._lower_value_spin = QDoubleSpinBox()
        self._upper_value_spin = QDoubleSpinBox()
        for spin in (self._lower_value_spin, self._upper_value_spin):
            spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
            spin.setDecimals(4)
        self._lower_value_spin.setValue(1.0)
        self._upper_value_spin.setValue(99.0)
        self._lower_value_spin.valueChanged.connect(self._on_processing_value_changed)
        self._upper_value_spin.valueChanged.connect(self._on_processing_value_changed)
        self._lower_parameter_label = QLabel("Lower percentile")
        self._upper_parameter_label = QLabel("Upper percentile")
        self._processing_lock_label = QLabel("Image conversion is not locked yet.")
        self._processing_lock_label.setWordWrap(True)

        self._preview_button = QPushButton("Preview RGB Conversion")
        self._preview_button.setToolTip(
            "Create or update a napari RGB layer using the exact pixels that will be saved."
        )
        self._preview_button.clicked.connect(self._on_preview_conversion)
        self._label_status_label = QLabel(
            "Labels: unavailable until a project is open."
        )
        self._label_status_label.setWordWrap(True)

        self._reload_labels_button = QPushButton("Reload Labels")
        self._reload_labels_button.clicked.connect(self._on_reload_labels)
        self._save_annotation_button = QPushButton("Save Converted Image + BBoxes")
        self._save_annotation_button.clicked.connect(self._on_save_annotation)
        self._validate_project_button = QPushButton("Validate Project")
        self._validate_project_button.clicked.connect(self._on_validate_project)

        project_group = QGroupBox("Project")
        project_buttons = QHBoxLayout()
        project_buttons.addWidget(self._new_project_button)
        project_buttons.addWidget(self._open_project_button)
        project_layout = QVBoxLayout()
        project_layout.addWidget(self._project_path_label)
        project_layout.addWidget(self._project_status_label)
        project_layout.addLayout(project_buttons)
        project_group.setLayout(project_layout)

        annotation_group = QGroupBox("Image and YOLO bounding boxes")
        processing_form = QFormLayout()
        processing_form.addRow("Channel axis", self._channel_axis_combo)
        processing_form.addRow("Output red", self._red_channel_combo)
        processing_form.addRow("Output green", self._green_channel_combo)
        processing_form.addRow("Output blue", self._blue_channel_combo)
        processing_form.addRow("Normalization", self._normalization_combo)
        processing_form.addRow(self._lower_parameter_label, self._lower_value_spin)
        processing_form.addRow(self._upper_parameter_label, self._upper_value_spin)
        annotation_buttons = QHBoxLayout()
        annotation_buttons.addWidget(self._reload_labels_button)
        annotation_buttons.addWidget(self._save_annotation_button)
        annotation_layout = QVBoxLayout()
        annotation_layout.addWidget(self._image_status_label)
        annotation_layout.addWidget(self._plane_status_label)
        annotation_layout.addLayout(processing_form)
        annotation_layout.addWidget(self._processing_lock_label)
        annotation_layout.addWidget(self._preview_button)
        annotation_layout.addWidget(self._label_status_label)
        annotation_layout.addLayout(annotation_buttons)
        annotation_layout.addWidget(self._validate_project_button)
        annotation_group.setLayout(annotation_layout)

        layout = QVBoxLayout()
        layout.addWidget(project_group)
        layout.addWidget(annotation_group)
        layout.addStretch(1)
        self.setLayout(layout)

        self._connect_viewer_events()
        self._on_normalization_changed()
        self._update_action_state()

    def _connect_viewer_events(self) -> None:
        try:
            self.napari_viewer.layers.selection.events.active.connect(
                self._on_active_layer_changed
            )
        except (AttributeError, TypeError):
            pass
        try:
            self.napari_viewer.layers.events.inserted.connect(
                self._on_layers_changed
            )
            self.napari_viewer.layers.events.removed.connect(
                self._on_layers_changed
            )
        except (AttributeError, TypeError):
            pass
        try:
            self.napari_viewer.dims.events.current_step.connect(
                self._on_current_step_changed
            )
        except (AttributeError, TypeError):
            pass

    def _show_info(self, text: str) -> None:
        QMessageBox.information(self, "Simple CCI Annotator", text)

    def _show_error(self, text: str) -> None:
        QMessageBox.critical(self, "Simple CCI Annotator", text)

    def _on_new_project(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select an empty folder for the new CCI annotation project",
        )
        if not selected:
            return
        try:
            project = ProjectStore.initialize(Path(selected))
        except ProjectError as exc:
            self._show_error(str(exc))
            return
        except OSError as exc:  # pragma: no cover - platform/filesystem guard
            self._show_error(f"Could not initialize project: {exc}")
            return
        self._set_project(project)
        self._show_info(f"Project initialized: {project.paths.root}")

    def _on_open_project(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select an initialized CCI annotation project",
        )
        if not selected:
            return
        try:
            project = ProjectStore.load(Path(selected))
        except ProjectError as exc:
            self._show_error(str(exc))
            return
        self._set_project(project)

    def _set_project(self, project: ProjectStore) -> None:
        previous_annotation = self._annotation_layer()
        if previous_annotation is not None:
            self.napari_viewer.layers.remove(previous_annotation)
        self._annotation_image_layer = None
        self._project = project
        self._annotation_io = AnnotationIO(project)
        self._project_path_label.setText(str(project.paths.root))
        class_summary = ", ".join(
            f"{class_id}: {name}"
            for class_id, name in sorted(project.config.classes.items())
        )
        self._project_status_label.setText(
            f"Project loaded · schema {project.config.schema_version} · "
            f"classes [{class_summary}]"
        )
        self._apply_project_processing_settings()
        self._update_action_state()
        active = self._active_layer()
        if self._is_source_image_layer(active):
            self._configure_image_controls(active)
            self._load_annotations_for_image(active, force=True)

    def _active_layer(self):
        try:
            return self.napari_viewer.layers.selection.active
        except AttributeError:
            return None

    @staticmethod
    def _is_image_layer(layer) -> bool:
        return layer is not None and layer.__class__.__name__.lower().endswith("image")

    @staticmethod
    def _is_shapes_layer(layer) -> bool:
        return layer is not None and layer.__class__.__name__.lower().endswith("shapes")

    def _on_active_layer_changed(self, event=None) -> None:
        active = self._active_layer()
        if self._is_source_image_layer(active):
            self._configure_image_controls(active)
            self._load_annotations_for_image(active)
        self._update_action_state()

    def _on_current_step_changed(self, event=None) -> None:
        image_layer = self._annotation_image_layer
        if image_layer is None or self._annotation_io is None:
            return
        self._load_annotations_for_image(image_layer)

    def _on_layers_changed(self, event=None) -> None:
        if self._annotation_image_layer is not None:
            try:
                is_present = self._annotation_image_layer in self.napari_viewer.layers
            except TypeError:
                is_present = True
            if not is_present:
                self._annotation_image_layer = None
        self._update_action_state()

    def _source_path(self, image_layer) -> Path | None:
        source = getattr(image_layer, "source", None)
        raw_path = getattr(source, "path", None)
        if raw_path:
            return Path(raw_path)
        metadata = getattr(image_layer, "metadata", {}) or {}
        raw_path = metadata.get("path") or metadata.get("source_path")
        return Path(raw_path) if raw_path else None

    @classmethod
    def _is_source_image_layer(cls, layer) -> bool:
        return cls._is_image_layer(layer) and not bool(
            (getattr(layer, "metadata", {}) or {}).get("cci_rgb_preview")
        )

    def _image_stem(self, image_layer) -> str:
        assert self._annotation_io is not None
        source_path = self._source_path(image_layer)
        value = source_path.stem if source_path is not None else getattr(
            image_layer, "name", "image"
        )
        base_stem = self._annotation_io.safe_stem(str(value))
        try:
            settings = self._settings_from_ui()
            plane = self._image_adapter.plane_selection(
                image_layer, self.napari_viewer, settings
            )
            return self._image_adapter.sample_id(
                base_stem, plane, settings.channel_axis
            )
        except ImageConversionError:
            return base_stem

    def _load_annotations_for_image(self, image_layer, *, force: bool = False) -> None:
        if self._annotation_io is None or not self._is_image_layer(image_layer):
            return
        try:
            image_shape = tuple(int(size) for size in image_layer.data.shape)
        except (AttributeError, TypeError, ValueError):
            self._show_error("The selected image does not expose a valid shape.")
            return
        if len(image_shape) < 2:
            self._show_error("The selected image must have at least two dimensions.")
            return

        stem = self._image_stem(image_layer)
        existing = self._annotation_layer()
        if (
            not force
            and existing is not None
            and getattr(existing, "metadata", {}).get("cci_image_stem") == stem
        ):
            self._annotation_image_layer = image_layer
            self._update_image_status(image_layer)
            return

        if existing is not None and self._annotation_dirty:
            if not self._resolve_unsaved_changes():
                return

        source_path = self._source_path(image_layer)
        label_path = self._annotation_io.find_label(stem, source_path=source_path)
        try:
            if label_path is None:
                rectangles: tuple[np.ndarray, ...] = ()
                properties = self._empty_properties()
            else:
                settings = self._settings_from_ui()
                spatial_axes = self._image_adapter.spatial_axes(
                    image_layer, settings.channel_axis
                )
                spatial_shape = (
                    image_shape[spatial_axes[0]],
                    image_shape[spatial_axes[1]],
                )
                loaded = self._annotation_io.load_label(label_path, spatial_shape)
                rectangles = loaded.rectangles
                properties = loaded.properties
        except (AnnotationError, OSError) as exc:
            self._show_error(f"Could not load YOLO bbox labels:\n{exc}")
            return

        if existing is not None:
            self.napari_viewer.layers.remove(existing)

        shapes = self.napari_viewer.add_shapes(
            list(rectangles),
            name=self.ANNOTATION_LAYER_NAME,
            shape_type="rectangle",
            properties=properties,
            edge_width=2,
            edge_color="yellow",
            face_color="transparent",
        )
        shapes.metadata["cci_image_stem"] = stem
        shapes.metadata["cci_project_root"] = str(self._project.paths.root)
        shapes.metadata["cci_label_path"] = str(label_path) if label_path else None
        self._set_default_current_properties(shapes)
        try:
            shapes.events.data.connect(self._on_shapes_data_changed)
        except (AttributeError, TypeError):
            pass
        self._annotation_image_layer = image_layer
        self._current_sample_id = stem
        try:
            self._converted_image = self._convert_current_image(image_layer)
        except ImageConversionError:
            self._converted_image = None
        self._annotation_dirty = False
        self._update_image_status(image_layer)
        if label_path is None:
            self._label_status_label.setText(
                f"Labels: none found for {stem}; an empty editable layer was created."
            )
            self._save_annotation_button.setText("Add Annotation")
        else:
            self._label_status_label.setText(
                f"Labels: loaded {len(rectangles)} box(es) from {label_path}"
            )
            canonical = label_path.parent == self._project.paths.labels
            self._save_annotation_button.setText(
                "Update Converted Image + BBoxes"
                if canonical
                else "Import Converted Image + BBoxes"
            )
        self._update_action_state()

    @staticmethod
    def _empty_properties() -> dict[str, np.ndarray]:
        return {
            "class_id": np.asarray([], dtype=int),
            "class_name": np.asarray([], dtype=object),
            "confidence": np.asarray([], dtype=float),
            "source": np.asarray([], dtype=object),
        }

    def _set_default_current_properties(self, shapes_layer) -> None:
        if self._project is None:
            return
        class_id = min(self._project.config.classes)
        try:
            shapes_layer.current_properties = {
                "class_id": np.asarray([class_id]),
                "class_name": np.asarray(
                    [self._project.config.classes[class_id]], dtype=object
                ),
                "confidence": np.asarray([np.nan]),
                "source": np.asarray(["manual"], dtype=object),
            }
        except (AttributeError, KeyError, ValueError):
            pass

    def _annotation_layer(self):
        try:
            for layer in self.napari_viewer.layers:
                if (
                    getattr(layer, "name", None) == self.ANNOTATION_LAYER_NAME
                    and self._is_shapes_layer(layer)
                ):
                    return layer
        except TypeError:
            pass
        return None

    def _on_reload_labels(self) -> None:
        image_layer = self._image_for_annotation()
        if image_layer is None:
            self._show_error("Select an image layer first.")
            return
        self._load_annotations_for_image(image_layer, force=True)

    def _on_save_annotation(self) -> None:
        if self._annotation_io is None:
            self._show_error("Create or open a project first.")
            return
        image_layer = self._image_for_annotation()
        shapes_layer = self._annotation_layer()
        if image_layer is None or shapes_layer is None:
            self._show_error(
                "Select an image and create or load its bbox layer first."
            )
            return

        try:
            converted = self._convert_current_image(image_layer)
            result = self._save_converted_annotation(
                converted, shapes_layer=shapes_layer, show_message=True
            )
        except (AnnotationError, ImageConversionError, ProjectError, OSError) as exc:
            self._show_error(f"Could not save annotation:\n{exc}")
            return

        if result is not None:
            self._update_action_state()

    def _save_converted_annotation(
        self,
        converted: ConvertedImage,
        *,
        shapes_layer,
        show_message: bool,
    ):
        if self._annotation_io is None or self._project is None:
            raise AnnotationError("Create or open a project first.")
        rectangles = tuple(
            np.asarray(shape, dtype=float) for shape in shapes_layer.data
        )
        class_ids = self._class_ids(shapes_layer, len(rectangles))
        result = self._annotation_io.save_pair(
            image_data=converted.data,
            image_name=converted.sample_id,
            source_path=self._source_path(self._annotation_image_layer),
            sample_id=converted.sample_id,
            rectangles=rectangles,
            class_ids=class_ids,
            conversion_metadata={
                "settings": converted.settings.to_mapping(),
                "plane_indices": converted.plane.non_spatial_indices,
                "axis_labels": list(converted.plane.axis_labels),
                "normalization_stats": list(converted.normalization_stats),
            },
        )
        self._project.lock_image_processing(converted.settings.to_mapping())
        self._converted_image = converted
        self._current_sample_id = converted.sample_id
        self._annotation_dirty = False
        shapes_layer.metadata["cci_label_path"] = str(result.label_path)
        self._label_status_label.setText(
            f"Labels: {result.operation} {result.box_count} box(es) · {result.label_path}"
        )
        self._save_annotation_button.setText("Update Converted Image + BBoxes")
        if show_message:
            self._show_info(
                f"Annotation {result.operation}:\n{result.image_path.name}\n"
                f"{result.label_path.name}"
            )
        self._apply_project_processing_settings()
        return result

    def _on_shapes_data_changed(self, event=None) -> None:
        self._annotation_dirty = True
        image_layer = self._annotation_image_layer
        if self._is_source_image_layer(image_layer):
            try:
                live_sample = self._image_stem(image_layer)
                if live_sample == self._current_sample_id:
                    self._converted_image = self._convert_current_image(image_layer)
            except ImageConversionError:
                pass
        self._label_status_label.setText(
            "Labels: unsaved edits. Save before changing image or Z/T plane."
        )

    def _resolve_unsaved_changes(self) -> bool:
        if not self._annotation_dirty:
            return True
        response = QMessageBox.warning(
            self,
            "Unsaved bounding boxes",
            "The current bbox layer has unsaved edits. Save it before switching "
            "image, Z/T plane, or reloading labels?",
            QMessageBox.Save | QMessageBox.Discard,
            QMessageBox.Save,
        )
        if response == QMessageBox.Discard:
            self._annotation_dirty = False
            return True

        shapes_layer = self._annotation_layer()
        if shapes_layer is None or self._converted_image is None:
            self._show_error(
                "The previous plane could not be reconstructed safely. Return to it "
                "and save the annotation before switching."
            )
            return False
        try:
            self._save_converted_annotation(
                self._converted_image,
                shapes_layer=shapes_layer,
                show_message=False,
            )
        except (AnnotationError, ImageConversionError, ProjectError, OSError) as exc:
            self._show_error(f"Could not save the previous annotation:\n{exc}")
            return False
        return True

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if not self._annotation_dirty:
            event.accept()
            return
        response = QMessageBox.warning(
            self,
            "Unsaved bounding boxes",
            "Save the current bbox edits before closing the annotator?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if response == QMessageBox.Cancel:
            event.ignore()
            return
        if response == QMessageBox.Discard:
            self._annotation_dirty = False
            event.accept()
            return
        shapes_layer = self._annotation_layer()
        if shapes_layer is None or self._converted_image is None:
            self._show_error(
                "The current plane could not be reconstructed safely. Save it "
                "manually before closing."
            )
            event.ignore()
            return
        try:
            self._save_converted_annotation(
                self._converted_image,
                shapes_layer=shapes_layer,
                show_message=False,
            )
        except (AnnotationError, ImageConversionError, ProjectError, OSError) as exc:
            self._show_error(f"Could not save the current annotation:\n{exc}")
            event.ignore()
        else:
            event.accept()

    def _class_ids(self, shapes_layer, count: int) -> tuple[int, ...]:
        properties = getattr(shapes_layer, "properties", {}) or {}
        values = properties.get("class_id")
        if values is None or len(values) != count:
            if self._project is not None and len(self._project.config.classes) == 1:
                default_class = next(iter(self._project.config.classes))
                return (default_class,) * count
            raise LabelValidationError(
                ["Every bbox must have a valid class_id property."]
            )
        return tuple(int(value) for value in values)

    def _image_for_annotation(self):
        active = self._active_layer()
        if self._is_source_image_layer(active):
            return active
        return self._annotation_image_layer

    def _update_image_status(self, image_layer) -> None:
        shape = tuple(int(size) for size in image_layer.data.shape)
        dtype = getattr(image_layer.data, "dtype", "unknown")
        source_path = self._source_path(image_layer)
        source_text = (
            str(source_path)
            if source_path
            else getattr(image_layer, "name", "image")
        )
        self._image_status_label.setText(
            f"Image: {source_text} · shape {shape} · {dtype}"
        )

    def _apply_project_processing_settings(self) -> None:
        if self._project is None:
            self._locked_processing_settings = None
            return
        config = self._project.config.image_processing
        if config.get("locked"):
            try:
                self._locked_processing_settings = ImageProcessingSettings.from_mapping(
                    config
                )
            except ImageConversionError as exc:
                self._locked_processing_settings = None
                self._show_error(str(exc))
                return
            self._processing_lock_label.setText(
                "Image conversion is locked for this project. Create a new project "
                "to use different channels or normalization."
            )
        else:
            self._locked_processing_settings = None
            self._processing_lock_label.setText(
                "Image conversion will lock after the first converted image is saved."
            )

        image_layer = self._image_for_annotation()
        if self._is_source_image_layer(image_layer):
            self._configure_image_controls(image_layer)

    def _configure_image_controls(self, image_layer) -> None:
        try:
            shape = tuple(int(size) for size in image_layer.data.shape)
            labels = self._image_adapter.axis_labels(image_layer)
            inferred_axis = self._image_adapter.infer_channel_axis(image_layer)
            selected_axis = (
                self._locked_processing_settings.channel_axis
                if self._locked_processing_settings is not None
                else inferred_axis
            )
            if self._locked_processing_settings is not None:
                self._locked_processing_settings.validate(shape)
            spatial_axes = self._image_adapter.spatial_axes(image_layer, selected_axis)
        except (AttributeError, ImageConversionError) as exc:
            self._show_error(f"Could not inspect image axes: {exc}")
            return

        self._updating_processing_controls = True
        try:
            self._channel_axis_combo.clear()
            self._channel_axis_combo.addItem("None — grayscale image", None)
            for axis, (label, size) in enumerate(zip(labels, shape, strict=True)):
                if axis in spatial_axes:
                    continue
                self._channel_axis_combo.addItem(
                    f"Axis {axis}: {label} ({size} values)", axis
                )
            self._set_combo_data(self._channel_axis_combo, selected_axis)
            self._populate_channel_choices(image_layer)

            if self._locked_processing_settings is not None:
                settings = self._locked_processing_settings
                self._set_combo_data(self._normalization_combo, settings.normalization)
                if settings.lower is not None:
                    self._lower_value_spin.setValue(settings.lower)
                if settings.upper is not None:
                    self._upper_value_spin.setValue(settings.upper)
                self._last_normalization_method = settings.normalization
        finally:
            self._updating_processing_controls = False

        self._on_normalization_changed()
        self._set_processing_controls_enabled(
            self._project is not None and self._locked_processing_settings is None
        )
        self._update_plane_status(image_layer)

    def _populate_channel_choices(self, image_layer=None) -> None:
        if image_layer is None:
            image_layer = self._image_for_annotation()
        if not self._is_source_image_layer(image_layer):
            return
        channel_axis = self._channel_axis_combo.currentData()
        combos = (
            self._red_channel_combo,
            self._green_channel_combo,
            self._blue_channel_combo,
        )
        for combo in combos:
            combo.clear()

        if channel_axis is None:
            for combo in combos:
                combo.addItem("Grayscale", None)
                combo.setEnabled(False)
            return

        channel_count = int(image_layer.data.shape[int(channel_axis)])
        for combo in combos:
            combo.addItem("None — output zero", None)
            for channel in range(channel_count):
                combo.addItem(f"Channel {channel}", channel)

        if self._locked_processing_settings is not None:
            defaults = self._locked_processing_settings.rgb_channels
        elif channel_count == 1:
            defaults = (0, 0, 0)
        elif channel_count == 2:
            defaults = (0, 1, None)
        else:
            defaults = (0, 1, 2)
        for combo, selected in zip(combos, defaults, strict=True):
            self._set_combo_data(combo, selected)

    def _on_channel_axis_changed(self, index=None) -> None:
        if self._updating_processing_controls:
            return
        previous_settings = (
            self._converted_image.settings
            if self._converted_image is not None
            else None
        )
        if self._annotation_dirty and not self._resolve_unsaved_changes():
            if previous_settings is not None:
                self._updating_processing_controls = True
                try:
                    self._set_combo_data(
                        self._channel_axis_combo, previous_settings.channel_axis
                    )
                    self._populate_channel_choices()
                finally:
                    self._updating_processing_controls = False
            return
        self._updating_processing_controls = True
        try:
            self._populate_channel_choices()
        finally:
            self._updating_processing_controls = False
        self._converted_image = None
        image_layer = self._image_for_annotation()
        if self._is_source_image_layer(image_layer):
            self._update_plane_status(image_layer)
            self._load_annotations_for_image(image_layer)

    def _on_normalization_changed(self, index=None) -> None:
        method = self._normalization_combo.currentData()
        if method is None:
            return
        if method != self._last_normalization_method:
            defaults = {
                "percentile": (1.0, 99.0),
                "z_score": (-3.0, 3.0),
                "fixed_range": (0.0, 65535.0),
            }
            if method in defaults and not self._updating_processing_controls:
                lower, upper = defaults[method]
                self._lower_value_spin.setValue(lower)
                self._upper_value_spin.setValue(upper)
        self._last_normalization_method = str(method)

        uses_parameters = method in {"percentile", "z_score", "fixed_range"}
        labels = {
            "percentile": ("Lower percentile", "Upper percentile"),
            "z_score": ("Lower Z", "Upper Z"),
            "fixed_range": ("Input minimum", "Input maximum"),
        }
        lower_label, upper_label = labels.get(method, ("Lower", "Upper"))
        self._lower_parameter_label.setText(lower_label)
        self._upper_parameter_label.setText(upper_label)
        self._lower_value_spin.setEnabled(uses_parameters)
        self._upper_value_spin.setEnabled(uses_parameters)
        self._converted_image = None
        self._on_processing_value_changed()

    def _on_processing_value_changed(self, value=None) -> None:
        if self._updating_processing_controls:
            return
        self._converted_image = None
        image_layer = self._image_for_annotation()
        if self._is_source_image_layer(image_layer):
            self._update_plane_status(image_layer)
            if self._annotation_dirty:
                try:
                    self._converted_image = self._convert_current_image(image_layer)
                except ImageConversionError:
                    pass

    def _settings_from_ui(self) -> ImageProcessingSettings:
        if self._channel_axis_combo.count() == 0:
            raise ImageConversionError("Select an image before configuring channels.")
        method = self._normalization_combo.currentData()
        uses_parameters = method in {"percentile", "z_score", "fixed_range"}
        settings = ImageProcessingSettings(
            channel_axis=self._channel_axis_combo.currentData(),
            red_channel=self._red_channel_combo.currentData(),
            green_channel=self._green_channel_combo.currentData(),
            blue_channel=self._blue_channel_combo.currentData(),
            normalization=str(method),
            lower=self._lower_value_spin.value() if uses_parameters else None,
            upper=self._upper_value_spin.value() if uses_parameters else None,
        )
        settings.validate()
        return settings

    def _set_processing_controls_enabled(self, enabled: bool) -> None:
        self._channel_axis_combo.setEnabled(enabled)
        has_channel_axis = self._channel_axis_combo.currentData() is not None
        for combo in (
            self._red_channel_combo,
            self._green_channel_combo,
            self._blue_channel_combo,
        ):
            combo.setEnabled(enabled and has_channel_axis)
        self._normalization_combo.setEnabled(enabled)
        method = self._normalization_combo.currentData()
        uses_parameters = method in {"percentile", "z_score", "fixed_range"}
        self._lower_value_spin.setEnabled(enabled and uses_parameters)
        self._upper_value_spin.setEnabled(enabled and uses_parameters)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index < 0:
            raise ImageConversionError(
                f"Configured value {value!r} is unavailable for this image."
            )
        combo.setCurrentIndex(index)

    def _convert_current_image(self, image_layer) -> ConvertedImage:
        settings = self._settings_from_ui()
        if (
            self._project is not None
            and self._project.config.image_processing.get("locked")
            and settings.to_mapping() != self._project.config.image_processing
        ):
            raise ImageConversionError(
                "The selected image does not match the project's locked conversion settings."
            )
        source_path = self._source_path(image_layer)
        base_stem = (
            source_path.stem
            if source_path is not None
            else getattr(image_layer, "name", "image")
        )
        converted = self._image_adapter.convert(
            image_layer,
            self.napari_viewer,
            settings,
            base_stem=base_stem,
        )
        self._converted_image = converted
        self._current_sample_id = converted.sample_id
        self._plane_status_label.setText(
            f"Current sample: {converted.sample_id} · output {converted.data.shape} uint8"
        )
        return converted

    def _on_preview_conversion(self) -> None:
        image_layer = self._image_for_annotation()
        if not self._is_source_image_layer(image_layer):
            self._show_error("Select a source image layer first.")
            return
        try:
            converted = self._convert_current_image(image_layer)
        except ImageConversionError as exc:
            self._show_error(f"Could not convert image:\n{exc}")
            return

        preview = self._get_layer_by_name("cci_rgb_preview")
        if preview is None:
            preview = self.napari_viewer.add_image(
                converted.data,
                name="cci_rgb_preview",
                rgb=True,
                metadata={"cci_rgb_preview": True},
            )
        else:
            preview.data = converted.data
            preview.metadata["cci_rgb_preview"] = True
        self._label_status_label.setText(
            f"RGB preview updated for {converted.sample_id}. "
            "Saving will use exactly this conversion."
        )

    def _update_plane_status(self, image_layer) -> None:
        try:
            sample_id = self._image_stem(image_layer)
            settings = self._settings_from_ui()
            plane = self._image_adapter.plane_selection(
                image_layer, self.napari_viewer, settings
            )
            indices = ", ".join(
                f"{plane.axis_labels[axis]}={value}"
                for axis, value in plane.non_spatial_indices.items()
                if axis != settings.channel_axis
            )
            suffix = f" ({indices})" if indices else ""
            self._plane_status_label.setText(f"Current sample: {sample_id}{suffix}")
        except ImageConversionError as exc:
            self._plane_status_label.setText(f"Current sample unavailable: {exc}")

    def _get_layer_by_name(self, name: str):
        try:
            for layer in self.napari_viewer.layers:
                if getattr(layer, "name", None) == name:
                    return layer
        except TypeError:
            pass
        return None

    def _on_validate_project(self) -> None:
        if self._annotation_io is None:
            self._show_error("Create or open a project first.")
            return
        report = self._annotation_io.validate_project()
        summary = (
            f"Images: {report.image_count}\nLabels: {report.label_count}\n"
            f"Pairs: {report.paired_count}\nBoxes: {report.box_count}"
        )
        if report.is_valid:
            self._show_info("Project validation passed.\n\n" + summary)
        else:
            details = "\n".join(f"• {error}" for error in report.errors[:20])
            if len(report.errors) > 20:
                details += f"\n• … and {len(report.errors) - 20} more"
            self._show_error(
                "Project validation failed.\n\n" + summary + "\n\n" + details
            )

    def _update_action_state(self) -> None:
        has_project = self._project is not None
        image_layer = self._image_for_annotation()
        has_image = self._is_image_layer(image_layer)
        has_shapes = self._annotation_layer() is not None
        self._reload_labels_button.setEnabled(has_project and has_image)
        self._save_annotation_button.setEnabled(
            has_project and has_image and has_shapes
        )
        self._validate_project_button.setEnabled(has_project)
        self._preview_button.setEnabled(has_project and has_image)
        self._set_processing_controls_enabled(
            has_project
            and has_image
            and self._locked_processing_settings is None
        )
