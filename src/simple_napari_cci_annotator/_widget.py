from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._annotation_io import AnnotationError, AnnotationIO, LabelValidationError
from ._dataset_builder import (
    DatasetBuildError,
    DatasetBuildSettings,
    DatasetBuilder,
    DatasetPreview,
)
from ._image_adapter import (
    NORMALIZATION_METHODS,
    ConvertedImage,
    ImageAdapter,
    ImageConversionError,
    ImageProcessingSettings,
)
from ._inference_worker import InferenceWorker
from ._project_store import ProjectError, ProjectStore
from ._tiled_inference import Detection, InferenceError, InferenceSettings
from ._training import TrainingError, TrainingRun, TrainingSettings
from ._training_worker import TrainingWorker
from ._yolo_inference import YoloDetectionModel, available_devices


class SimpleCciAnnotatorQWidget(QWidget):
    """Project-based YOLO bbox annotation and tiled inference UI."""

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
        self._model: YoloDetectionModel | None = None
        self._inference_worker: InferenceWorker | None = None
        self._inference_sample_id: str | None = None
        self._inference_converted: ConvertedImage | None = None
        self._training_worker: TrainingWorker | None = None
        self._training_preview: DatasetPreview | None = None
        self._last_training_run: TrainingRun | None = None
        self._base_model_path = Path(__file__).resolve().parents[2] / "yolo26n.pt"

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

        self._model_path_label = QLabel("No detection model loaded")
        self._model_path_label.setWordWrap(True)
        self._model_status_label = QLabel("Model: unavailable")
        self._model_status_label.setWordWrap(True)
        self._choose_model_button = QPushButton("Choose Detection Model")
        self._choose_model_button.clicked.connect(self._on_choose_model)

        self._device_combo = QComboBox()
        for label, value in available_devices():
            self._device_combo.addItem(label, value)
        self._confidence_spin = self._fraction_spin(0.25)
        self._model_iou_spin = self._fraction_spin(0.45)
        self._merge_iou_spin = self._fraction_spin(0.50)
        self._tile_size_spin = QSpinBox()
        self._tile_size_spin.setRange(64, 8192)
        self._tile_size_spin.setSingleStep(64)
        self._tile_size_spin.setValue(1024)
        self._overlap_percent_spin = QDoubleSpinBox()
        self._overlap_percent_spin.setRange(0.0, 90.0)
        self._overlap_percent_spin.setDecimals(1)
        self._overlap_percent_spin.setSuffix(" %")
        self._overlap_percent_spin.setValue(20.0)
        self._max_detections_spin = QSpinBox()
        self._max_detections_spin.setRange(1, 100_000)
        self._max_detections_spin.setValue(300)
        self._predict_button = QPushButton("Predict Current RGB Plane")
        self._predict_button.clicked.connect(self._on_predict)
        self._cancel_inference_button = QPushButton("Cancel")
        self._cancel_inference_button.clicked.connect(self._on_cancel_inference)
        self._inference_progress = QProgressBar()
        self._inference_progress.setRange(0, 1)
        self._inference_progress.setValue(0)
        self._inference_status_label = QLabel("Inference: idle")
        self._inference_status_label.setWordWrap(True)

        self._training_model_combo = QComboBox()
        self._destination_input = QLineEdit()
        self._destination_input.setPlaceholderText(
            "Project root (default), or choose another output parent"
        )
        self._destination_button = QPushButton("Choose Destination")
        self._destination_button.clicked.connect(self._on_choose_destination)
        self._group_field_input = QLineEdit()
        self._group_field_input.setPlaceholderText(
            "Optional audit field, e.g. metadata.patient"
        )
        self._validation_fraction_spin = self._fraction_spin(0.20)
        self._validation_fraction_spin.setRange(0.05, 0.50)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.setValue(42)
        self._training_tile_size_spin = QSpinBox()
        self._training_tile_size_spin.setRange(64, 8192)
        self._training_tile_size_spin.setSingleStep(64)
        self._training_tile_size_spin.setValue(1024)
        self._training_overlap_spin = QDoubleSpinBox()
        self._training_overlap_spin.setRange(0.0, 90.0)
        self._training_overlap_spin.setDecimals(1)
        self._training_overlap_spin.setSuffix(" %")
        self._training_overlap_spin.setValue(20.0)
        self._negative_ratio_spin = QDoubleSpinBox()
        self._negative_ratio_spin.setRange(0.0, 10.0)
        self._negative_ratio_spin.setDecimals(2)
        self._negative_ratio_spin.setValue(1.0)
        self._negative_ratio_spin.setToolTip(
            "Maximum sampled background tiles per positive tile. At least one "
            "tile is kept for a reviewed-negative source."
        )
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 100_000)
        self._epochs_spin.setValue(100)
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(-1, 4096)
        self._batch_spin.setSpecialValueText("Auto")
        self._batch_spin.setValue(-1)
        self._patience_spin = QSpinBox()
        self._patience_spin.setRange(0, 100_000)
        self._patience_spin.setValue(30)
        self._training_device_combo = QComboBox()
        for label, value in available_devices():
            self._training_device_combo.addItem(label, value)
        self._train_only_checkbox = QCheckBox(
            "Train without independent validation (exploratory)"
        )
        self._train_only_checkbox.toggled.connect(self._on_train_only_toggled)
        self._train_only_warning = QLabel(
            "NO INDEPENDENT VALIDATION — generalization cannot be estimated."
        )
        self._train_only_warning.setStyleSheet("color: #c62828; font-weight: bold;")
        self._train_only_warning.setWordWrap(True)
        self._train_only_warning.setVisible(False)
        self._dataset_status_label = QLabel(
            "Dataset: open a project and save annotations to begin."
        )
        self._dataset_status_label.setWordWrap(True)
        self._training_status_label = QLabel("Retraining: idle")
        self._training_status_label.setWordWrap(True)
        self._training_progress = QProgressBar()
        self._training_progress.setRange(0, 1)
        self._training_progress.setValue(0)
        self._validate_dataset_button = QPushButton("Validate Dataset")
        self._validate_dataset_button.clicked.connect(self._on_validate_dataset)
        self._preview_split_button = QPushButton("Preview Split")
        self._preview_split_button.clicked.connect(self._on_preview_split)
        self._regenerate_split_button = QPushButton("Regenerate Split")
        self._regenerate_split_button.clicked.connect(self._on_regenerate_split)
        self._retrain_button = QPushButton("Retrain")
        self._retrain_button.clicked.connect(self._on_retrain)
        self._cancel_training_button = QPushButton("Cancel")
        self._cancel_training_button.clicked.connect(self._on_cancel_training)

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

        inference_group = QGroupBox("Large-image tiled prediction")
        inference_form = QFormLayout()
        inference_form.addRow("Device", self._device_combo)
        inference_form.addRow("Confidence", self._confidence_spin)
        inference_form.addRow("Model IoU", self._model_iou_spin)
        inference_form.addRow("Tile size", self._tile_size_spin)
        inference_form.addRow("Tile overlap", self._overlap_percent_spin)
        inference_form.addRow("Merge IoU", self._merge_iou_spin)
        inference_form.addRow("Max detections/tile", self._max_detections_spin)
        inference_buttons = QHBoxLayout()
        inference_buttons.addWidget(self._predict_button)
        inference_buttons.addWidget(self._cancel_inference_button)
        inference_layout = QVBoxLayout()
        inference_layout.addWidget(self._model_path_label)
        inference_layout.addWidget(self._model_status_label)
        inference_layout.addWidget(self._choose_model_button)
        inference_layout.addLayout(inference_form)
        inference_layout.addLayout(inference_buttons)
        inference_layout.addWidget(self._inference_progress)
        inference_layout.addWidget(self._inference_status_label)
        inference_group.setLayout(inference_layout)

        retrain_group = QGroupBox("Dataset building and retraining")
        retrain_form = QFormLayout()
        retrain_form.addRow("Starting model", self._training_model_combo)
        destination_row = QHBoxLayout()
        destination_row.addWidget(self._destination_input)
        destination_row.addWidget(self._destination_button)
        retrain_form.addRow("Output parent", destination_row)
        retrain_form.addRow("Group metadata", self._group_field_input)
        retrain_form.addRow("Validation fraction", self._validation_fraction_spin)
        retrain_form.addRow("Split seed", self._seed_spin)
        retrain_form.addRow("Training tile size", self._training_tile_size_spin)
        retrain_form.addRow("Tile overlap", self._training_overlap_spin)
        retrain_form.addRow("Negative tile ratio", self._negative_ratio_spin)
        retrain_form.addRow("Epochs", self._epochs_spin)
        retrain_form.addRow("Batch", self._batch_spin)
        retrain_form.addRow("Patience", self._patience_spin)
        retrain_form.addRow("Device", self._training_device_combo)
        dataset_buttons = QHBoxLayout()
        dataset_buttons.addWidget(self._validate_dataset_button)
        dataset_buttons.addWidget(self._preview_split_button)
        dataset_buttons.addWidget(self._regenerate_split_button)
        training_buttons = QHBoxLayout()
        training_buttons.addWidget(self._retrain_button)
        training_buttons.addWidget(self._cancel_training_button)
        retrain_layout = QVBoxLayout()
        retrain_layout.addLayout(retrain_form)
        retrain_layout.addWidget(self._train_only_checkbox)
        retrain_layout.addWidget(self._train_only_warning)
        retrain_layout.addWidget(self._dataset_status_label)
        retrain_layout.addLayout(dataset_buttons)
        retrain_layout.addLayout(training_buttons)
        retrain_layout.addWidget(self._training_progress)
        retrain_layout.addWidget(self._training_status_label)
        retrain_group.setLayout(retrain_layout)

        layout = QVBoxLayout()
        layout.addWidget(project_group)
        layout.addWidget(annotation_group)
        layout.addWidget(inference_group)
        layout.addWidget(retrain_group)
        layout.addStretch(1)
        self.setLayout(layout)

        self._connect_viewer_events()
        self._on_normalization_changed()
        self._update_action_state()

    @staticmethod
    def _fraction_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        return spin

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
        self._model = None
        self._model_path_label.setText("No detection model loaded")
        self._model_status_label.setText("Model: unavailable")
        self._training_preview = None
        self._dataset_status_label.setText(
            "Dataset: annotations have not been validated yet."
        )
        self._refresh_training_model_choices()
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
            "tile_id": np.asarray([], dtype=int),
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
                "tile_id": np.asarray([-1], dtype=int),
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
                "prediction": getattr(shapes_layer, "metadata", {}).get(
                    "cci_prediction"
                ),
            },
        )
        self._project.lock_image_processing(converted.settings.to_mapping())
        self._converted_image = converted
        self._current_sample_id = converted.sample_id
        self._annotation_dirty = False
        self._training_preview = None
        self._dataset_status_label.setText(
            "Dataset: annotation pool changed; validate or preview the split again."
        )
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
        if self._training_worker is not None and self._training_worker.isRunning():
            self._training_worker.request_cancel()
            self._training_status_label.setText(
                "Retraining: cancellation requested. Close again after it stops."
            )
            event.ignore()
            return
        if self._inference_worker is not None and self._inference_worker.isRunning():
            self._inference_worker.request_cancel()
            self._inference_status_label.setText(
                "Inference: cancelling after the current tile. Close again when it stops."
            )
            event.ignore()
            return
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
        if self._annotation_io is not None and self._is_source_image_layer(
            image_layer
        ):
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

    def _on_choose_model(self) -> None:
        if self._project is None:
            self._show_error("Create or open a project first.")
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select a YOLO detection model",
            str(self._project.paths.models),
            "YOLO models (*.pt *.onnx *.engine);;All files (*)",
        )
        if not selected:
            return
        self._load_prediction_model(Path(selected))

    def _load_prediction_model(self, path: Path) -> bool:
        if self._project is None:
            return False
        try:
            model = YoloDetectionModel(path)
            project_ids = set(self._project.config.classes)
            model_ids = set(model.names)
            if model_ids and model_ids != project_ids:
                raise InferenceError(
                    "Model class IDs do not match the project. "
                    f"Project IDs: {sorted(project_ids)}; model IDs: {sorted(model_ids)}."
                )
        except InferenceError as exc:
            self._show_error(str(exc))
            return False
        self._model = model
        self._model_path_label.setText(str(model.path))
        names = ", ".join(
            f"{class_id}: {name}" for class_id, name in sorted(model.names.items())
        )
        project_names = self._project.config.classes
        names_differ = bool(model.names) and any(
            model.names.get(class_id) != name
            for class_id, name in project_names.items()
        )
        suffix = " · names mapped to project names" if names_differ else ""
        self._model_status_label.setText(
            f"Model: loaded · task {model.task} · classes [{names or 'unknown'}]{suffix}"
        )
        self._refresh_training_model_choices(preferred=model.path)
        self._update_action_state()
        return True

    def _inference_settings(self) -> InferenceSettings:
        tile_size = self._tile_size_spin.value()
        overlap = round(tile_size * self._overlap_percent_spin.value() / 100.0)
        return InferenceSettings(
            tile_size=tile_size,
            overlap=min(overlap, tile_size - 1),
            confidence=self._confidence_spin.value(),
            model_iou=self._model_iou_spin.value(),
            merge_iou=self._merge_iou_spin.value(),
            max_detections=self._max_detections_spin.value(),
            device=self._device_combo.currentData(),
        )

    def _on_predict(self) -> None:
        if self._project is None or self._model is None:
            self._show_error("Open a project and choose a detection model first.")
            return
        if self._inference_worker is not None and self._inference_worker.isRunning():
            return
        image_layer = self._image_for_annotation()
        if not self._is_source_image_layer(image_layer):
            self._show_error("Select a source image layer first.")
            return
        if not self._resolve_unsaved_changes():
            return
        existing = self._annotation_layer()
        if existing is not None and len(getattr(existing, "data", ())) > 0:
            response = QMessageBox.warning(
                self,
                "Replace bounding boxes",
                "Prediction will replace the current bbox layer. Continue?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if response != QMessageBox.Yes:
                return
        try:
            converted = self._convert_current_image(image_layer)
            settings = self._inference_settings()
            settings.validate()
        except (ImageConversionError, InferenceError) as exc:
            self._show_error(f"Could not start inference:\n{exc}")
            return

        worker = InferenceWorker(self._model, converted.data, settings)
        worker.progress.connect(self._on_inference_progress)
        worker.succeeded.connect(self._on_inference_succeeded)
        worker.failed.connect(self._on_inference_failed)
        worker.cancelled.connect(self._on_inference_cancelled)
        worker.finished.connect(self._on_inference_thread_finished)
        self._inference_worker = worker
        self._inference_sample_id = converted.sample_id
        self._inference_converted = converted
        self._inference_progress.setRange(0, 1)
        self._inference_progress.setValue(0)
        self._inference_status_label.setText("Inference: preparing tiles")
        self._update_action_state()
        worker.start()

    def _on_cancel_inference(self) -> None:
        worker = self._inference_worker
        if worker is None or not worker.isRunning():
            return
        worker.request_cancel()
        self._inference_status_label.setText(
            "Inference: cancellation requested; waiting for the current tile"
        )
        self._cancel_inference_button.setEnabled(False)

    def _on_inference_progress(self, current: int, total: int, text: str) -> None:
        self._inference_progress.setRange(0, max(1, total))
        self._inference_progress.setValue(current)
        self._inference_status_label.setText(f"Inference: {text}")

    def _on_inference_succeeded(self, detections: object) -> None:
        result = tuple(detections)
        image_layer = self._annotation_image_layer
        if image_layer is None or self._inference_sample_id is None:
            self._inference_status_label.setText(
                "Inference: result discarded because the source image was closed"
            )
            return
        try:
            live_sample = self._image_stem(image_layer)
        except (AttributeError, ImageConversionError):
            live_sample = None
        if live_sample != self._inference_sample_id:
            self._inference_status_label.setText(
                "Inference: result discarded because the active Z/T plane changed"
            )
            return
        assert self._project is not None
        unknown = sorted(
            {item.class_id for item in result} - set(self._project.config.classes)
        )
        if unknown:
            self._show_error(
                f"The model returned class IDs not defined by the project: {unknown}."
            )
            return

        existing = self._annotation_layer()
        if existing is not None:
            self.napari_viewer.layers.remove(existing)
        typed_result: tuple[Detection, ...] = result
        rectangles = [item.as_napari_rectangle() for item in typed_result]
        properties = {
            "class_id": np.asarray(
                [item.class_id for item in typed_result], dtype=int
            ),
            "class_name": np.asarray(
                [
                    self._project.config.classes[item.class_id]
                    for item in typed_result
                ],
                dtype=object,
            ),
            "confidence": np.asarray(
                [item.confidence for item in typed_result], dtype=float
            ),
            "source": np.full(len(typed_result), "prediction", dtype=object),
            "tile_id": np.asarray(
                [item.tile_id for item in typed_result], dtype=int
            ),
        }
        shapes = self.napari_viewer.add_shapes(
            rectangles,
            name=self.ANNOTATION_LAYER_NAME,
            shape_type="rectangle",
            properties=properties,
            edge_width=2,
            edge_color="yellow",
            face_color="transparent",
        )
        settings = self._inference_settings()
        shapes.metadata["cci_image_stem"] = self._inference_sample_id
        shapes.metadata["cci_project_root"] = str(self._project.paths.root)
        shapes.metadata["cci_label_path"] = None
        shapes.metadata["cci_prediction"] = {
            "model_path": str(self._model.path) if self._model else None,
            "tile_size": settings.tile_size,
            "tile_overlap": settings.overlap,
            "confidence": settings.confidence,
            "model_iou": settings.model_iou,
            "merge_iou": settings.merge_iou,
            "device": str(settings.device),
        }
        self._set_default_current_properties(shapes)
        try:
            shapes.events.data.connect(self._on_shapes_data_changed)
        except (AttributeError, TypeError):
            pass
        self._converted_image = self._inference_converted
        self._current_sample_id = self._inference_sample_id
        self._annotation_dirty = True
        self._label_status_label.setText(
            f"Labels: {len(typed_result)} merged prediction(s), not yet saved."
        )
        self._save_annotation_button.setText("Save Prediction + Corrections")
        total = self._inference_progress.maximum()
        self._inference_progress.setValue(total)
        self._inference_status_label.setText(
            f"Inference: complete · {len(typed_result)} merged detection(s)"
        )

    def _on_inference_failed(self, message: str) -> None:
        self._inference_status_label.setText("Inference: failed")
        self._show_error(f"Inference failed:\n{message}")

    def _on_inference_cancelled(self) -> None:
        self._inference_status_label.setText("Inference: cancelled")

    def _on_inference_thread_finished(self) -> None:
        worker = self._inference_worker
        if worker is not None:
            worker.deleteLater()
        self._inference_worker = None
        self._inference_converted = None
        self._update_action_state()

    def _refresh_training_model_choices(self, preferred: Path | None = None) -> None:
        previous = preferred or self._training_model_combo.currentData()
        self._training_model_combo.clear()
        if self._base_model_path.is_file():
            self._training_model_combo.addItem(
                f"Base · {self._base_model_path.name}", str(self._base_model_path)
            )
        if self._model is not None and self._model.path != self._base_model_path:
            self._training_model_combo.addItem(
                f"Loaded fine-tuned · {self._model.path.name}",
                str(self._model.path),
            )
        if previous:
            index = self._training_model_combo.findData(str(previous))
            if index >= 0:
                self._training_model_combo.setCurrentIndex(index)

    def _on_choose_destination(self) -> None:
        start = self._destination_input.text().strip()
        if not start and self._project is not None:
            start = str(self._project.paths.root)
        selected = QFileDialog.getExistingDirectory(
            self, "Select retraining output parent", start
        )
        if selected:
            self._destination_input.setText(selected)

    def _on_train_only_toggled(self, checked: bool) -> None:
        self._train_only_warning.setVisible(checked)
        self._training_preview = None

    def _dataset_settings(self) -> DatasetBuildSettings:
        tile_size = self._training_tile_size_spin.value()
        overlap = round(tile_size * self._training_overlap_spin.value() / 100.0)
        return DatasetBuildSettings(
            validation_fraction=self._validation_fraction_spin.value(),
            seed=self._seed_spin.value(),
            tile_size=tile_size,
            overlap=min(overlap, tile_size - 1),
            negative_tile_ratio=self._negative_ratio_spin.value(),
            group_field=self._group_field_input.text().strip(),
        )

    def _preview_dataset(self, *, regenerate: bool = False) -> DatasetPreview | None:
        if self._project is None:
            self._show_error("Create or open a project first.")
            return None
        try:
            preview = DatasetBuilder(self._project).preview(
                self._dataset_settings(),
                train_only=self._train_only_checkbox.isChecked(),
                regenerate=regenerate,
                persist=True,
            )
        except (DatasetBuildError, OSError) as exc:
            self._show_error(f"Could not inspect the training dataset:\n{exc}")
            return None
        self._training_preview = preview
        train_sources = preview.source_counts["train"]
        val_sources = preview.source_counts["val"]
        train_tiles = preview.tile_counts["train"]
        val_tiles = preview.tile_counts["val"]
        summary = (
            f"Dataset: {len(preview.samples)} pair(s) · "
            f"{preview.positive_count} positive / {preview.negative_count} negative · "
            f"{preview.total_boxes} boxes\n"
            f"Split: train {train_sources} source(s), {train_tiles} tile(s) / "
            f"val {val_sources} source(s), {val_tiles} tile(s)"
        )
        if preview.validation_mode == "none":
            summary += " · NO INDEPENDENT VALIDATION"
        if preview.errors:
            summary += f"\nErrors: {' | '.join(preview.errors[:3])}"
        elif preview.warnings:
            summary += f"\nWarning: {preview.warnings[0]}"
        self._dataset_status_label.setText(summary)
        self._update_action_state()
        return preview

    def _on_validate_dataset(self) -> None:
        preview = self._preview_dataset()
        if preview is None:
            return
        if preview.errors:
            details = "\n".join(f"• {error}" for error in preview.errors[:20])
            self._show_error("Dataset validation failed.\n\n" + details)
            return
        warning_text = "\n".join(
            f"• {warning}" for warning in preview.warnings[:10]
        )
        message = (
            f"Dataset validation passed.\n\nPairs: {len(preview.samples)}\n"
            f"Positive: {preview.positive_count}\nNegative: {preview.negative_count}\n"
            f"Boxes: {preview.total_boxes}"
        )
        if warning_text:
            message += "\n\nWarnings:\n" + warning_text
        self._show_info(message)

    def _on_preview_split(self) -> None:
        preview = self._preview_dataset()
        if preview is None or preview.errors:
            return
        assignments = "\n".join(
            f"{sample.sample_id}: {preview.assignments[sample.sample_id]} "
            f"[{sample.group}]"
            for sample in preview.samples[:30]
        )
        if len(preview.samples) > 30:
            assignments += f"\n… and {len(preview.samples) - 30} more"
        self._show_info("Deterministic source-group split:\n\n" + assignments)

    def _on_regenerate_split(self) -> None:
        if self._train_only_checkbox.isChecked():
            self._show_error(
                "Disable train-only mode before regenerating train/validation assignments."
            )
            return
        response = QMessageBox.warning(
            self,
            "Regenerate train/validation split",
            "This replaces stable assignments for the current annotation pool. "
            "Existing retrain snapshots are not changed. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if response != QMessageBox.Yes:
            return
        preview = self._preview_dataset(regenerate=True)
        if preview is not None and preview.is_valid:
            self._show_info("The stable source-group split was regenerated.")

    def _training_settings(self) -> TrainingSettings:
        if self._project is None:
            raise DatasetBuildError("Create or open a project first.")
        model_value = self._training_model_combo.currentData()
        if not model_value:
            raise DatasetBuildError(
                "No training model is available. Add yolo26n.pt at the repository "
                "root or load a compatible fine-tuned model."
            )
        destination_text = self._destination_input.text().strip()
        destination = (
            Path(destination_text) if destination_text else self._project.paths.root
        )
        return TrainingSettings(
            model_path=Path(model_value),
            destination=destination,
            dataset=self._dataset_settings(),
            epochs=self._epochs_spin.value(),
            batch=self._batch_spin.value(),
            patience=self._patience_spin.value(),
            device=self._training_device_combo.currentData(),
            train_only=self._train_only_checkbox.isChecked(),
        )

    def _on_retrain(self) -> None:
        if self._training_worker is not None and self._training_worker.isRunning():
            return
        if not self._resolve_unsaved_changes():
            return
        preview = self._preview_dataset()
        if preview is None or not preview.is_valid:
            self._show_error("Fix the dataset validation errors before retraining.")
            return
        if self._train_only_checkbox.isChecked():
            response = QMessageBox.warning(
                self,
                "Exploratory train-only run",
                "There is NO independent validation set. Model quality and "
                "generalization cannot be estimated from this run. Start anyway?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if response != QMessageBox.Yes:
                return
        try:
            settings = self._training_settings()
            settings.validate()
        except (DatasetBuildError, TrainingError, OSError, ValueError) as exc:
            self._show_error(f"Could not start retraining:\n{exc}")
            return
        assert self._project is not None
        worker = TrainingWorker(self._project, settings, preview)
        worker.progress.connect(self._on_training_progress)
        worker.succeeded.connect(self._on_training_succeeded)
        worker.failed.connect(self._on_training_failed)
        worker.cancelled.connect(self._on_training_cancelled)
        worker.finished.connect(self._on_training_thread_finished)
        self._training_worker = worker
        self._training_progress.setRange(0, 1)
        self._training_progress.setValue(0)
        self._training_status_label.setText("Retraining: preparing dataset snapshot")
        self._update_action_state()
        worker.start()

    def _on_cancel_training(self) -> None:
        worker = self._training_worker
        if worker is None or not worker.isRunning():
            return
        worker.request_cancel()
        self._cancel_training_button.setEnabled(False)
        self._training_status_label.setText(
            "Retraining: cancellation requested; the current epoch may finish first"
        )

    def _on_training_progress(self, current: int, total: int, text: str) -> None:
        self._training_progress.setRange(0, max(1, total))
        self._training_progress.setValue(current)
        self._training_status_label.setText(f"Retraining: {text}")

    def _on_training_succeeded(self, result: object) -> None:
        run: TrainingRun = result
        self._last_training_run = run
        self._training_progress.setValue(self._training_progress.maximum())
        self._training_status_label.setText(
            f"Retraining: completed · {run.run_root}"
        )
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Retraining completed")
        dialog.setText(f"Training completed successfully.\n\n{run.run_root}")
        keep_button = dialog.addButton(
            "Keep Current Model", QMessageBox.AcceptRole
        )
        load_button = None
        load_model = run.best_model or run.last_model
        if load_model is not None:
            label = (
                "Load New Best Model"
                if run.best_model is not None
                else "Load New Last Model (Exploratory)"
            )
            load_button = dialog.addButton(
                label, QMessageBox.ActionRole
            )
        open_button = dialog.addButton("Open Run Folder", QMessageBox.ActionRole)
        dialog.exec()
        clicked = dialog.clickedButton()
        if load_button is not None and clicked is load_button:
            self._load_prediction_model(load_model)
        elif clicked is open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(run.run_root)))
        elif clicked is keep_button:
            pass

    def _on_training_failed(self, message: str) -> None:
        self._training_status_label.setText("Retraining: failed")
        self._show_error(f"Retraining failed:\n{message}")

    def _on_training_cancelled(self, message: str) -> None:
        self._training_status_label.setText(f"Retraining: cancelled · {message}")

    def _on_training_thread_finished(self) -> None:
        worker = self._training_worker
        if worker is not None:
            worker.deleteLater()
        self._training_worker = None
        self._update_action_state()

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
        inference_running = (
            self._inference_worker is not None
            and self._inference_worker.isRunning()
        )
        training_running = (
            self._training_worker is not None
            and self._training_worker.isRunning()
        )
        running = inference_running or training_running
        self._new_project_button.setEnabled(not running)
        self._open_project_button.setEnabled(not running)
        self._reload_labels_button.setEnabled(has_project and has_image and not running)
        self._save_annotation_button.setEnabled(
            has_project and has_image and has_shapes and not running
        )
        self._validate_project_button.setEnabled(has_project and not running)
        self._preview_button.setEnabled(has_project and has_image and not running)
        self._choose_model_button.setEnabled(has_project and not running)
        self._predict_button.setEnabled(
            has_project and has_image and self._model is not None and not running
        )
        self._cancel_inference_button.setEnabled(inference_running)
        for control in (
            self._device_combo,
            self._confidence_spin,
            self._model_iou_spin,
            self._merge_iou_spin,
            self._tile_size_spin,
            self._overlap_percent_spin,
            self._max_detections_spin,
        ):
            control.setEnabled(not running)
        has_training_model = self._training_model_combo.count() > 0
        self._validate_dataset_button.setEnabled(has_project and not running)
        self._preview_split_button.setEnabled(has_project and not running)
        self._regenerate_split_button.setEnabled(has_project and not running)
        self._retrain_button.setEnabled(
            has_project and has_training_model and not running
        )
        self._cancel_training_button.setEnabled(training_running)
        self._destination_button.setEnabled(has_project and not running)
        for control in (
            self._training_model_combo,
            self._destination_input,
            self._group_field_input,
            self._validation_fraction_spin,
            self._seed_spin,
            self._training_tile_size_spin,
            self._training_overlap_spin,
            self._negative_ratio_spin,
            self._epochs_spin,
            self._batch_spin,
            self._patience_spin,
            self._training_device_combo,
            self._train_only_checkbox,
        ):
            control.setEnabled(has_project and not running)
        self._set_processing_controls_enabled(
            has_project
            and has_image
            and self._locked_processing_settings is None
            and not running
        )
