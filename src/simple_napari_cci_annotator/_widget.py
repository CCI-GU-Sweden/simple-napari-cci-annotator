from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QColor, QDesktopServices
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ._annotation_io import AnnotationError, AnnotationIO, LabelValidationError
from ._annotation_browser import (
    AnnotationBrowser,
    AnnotationReviewEntry,
    invalid_rectangle_indices,
)
from ._class_editor import ClassMapDialog
from ._class_map import class_color, class_display_name
from ._dataset_builder import (
    DatasetBuildError,
    DatasetBuildSettings,
    DatasetBuilder,
    DatasetPreview,
)
from ._image_adapter import (
    IMAGE_FILTERS,
    NORMALIZATION_METHODS,
    ConvertedImage,
    ImageAdapter,
    ImageConversionError,
    ImageProcessingSettings,
    PlaneSelection,
)
from ._inference_worker import InferenceWorker
from ._instance_mask import (
    ComposedInstances,
    disconnected_instance_ids,
    keep_largest_component_by_bbox,
    split_instance,
)
from ._project_store import ProjectError, ProjectStore
from ._segmentation_io import (
    InstanceRecord,
    SegmentationError,
    SegmentationIO,
    SegmentationReviewEntry,
    refresh_instance_records,
)
from ._segmentation_worker import SegmentationWorker
from ._tiled_inference import Detection, InferenceError, InferenceSettings
from ._training import TrainingError, TrainingRun, TrainingSettings
from ._training_crop import (
    CropBounds,
    TrainingCropError,
    crop_bounds_from_center,
    crop_bounds_from_rectangle,
    crop_rectangles,
    crop_sample_id,
    extract_padded_crop,
    invalid_crop_box_indices,
    validate_boxes_within_valid_crop,
)
from ._training_worker import TrainingWorker
from ._yolo_inference import YoloDetectionModel, available_devices
from ._yolo_segmentation import YoloSegmentationModel


class CollapsibleSection(QWidget):
    """Compact titled section whose content can be collapsed in a dock widget."""

    def __init__(self, title: str, content_layout, *, expanded: bool = True):
        super().__init__()
        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        self.toggle_button.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; "
            "text-align: left; padding: 6px 2px; }"
        )
        self.content = QWidget()
        self.content.setLayout(content_layout)
        self.content.setVisible(expanded)
        self.toggle_button.toggled.connect(self._set_expanded)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)
        self.setLayout(layout)

    def _set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        self.content.setVisible(expanded)


class SimpleCciAnnotatorQWidget(QWidget):
    """Project-based YOLO bbox annotation and tiled inference UI."""

    ANNOTATION_LAYER_NAME = "yolo_bboxes"
    SEGMENTATION_LAYER_NAME = "yolo_instances"
    CROP_SELECTION_LAYER_NAME = "training_crop_selection"
    CROP_IMAGE_LAYER_NAME = "training_crop_rgb"
    CROP_BBOX_LAYER_NAME = "training_crop_bboxes"
    REVIEW_IMAGE_LAYER_NAME = "annotation_review_rgb"

    def __init__(self, napari_viewer):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.setWindowTitle("Simple CCI Annotator")

        self._project: ProjectStore | None = None
        self._annotation_io: AnnotationIO | None = None
        self._segmentation_io: SegmentationIO | None = None
        self._segment_instances: dict[int, InstanceRecord] = {}
        self._segmentation_errors: tuple[str, ...] = ()
        self._image_adapter = ImageAdapter()
        self._annotation_image_layer = None
        self._converted_image: ConvertedImage | None = None
        self._current_sample_id: str | None = None
        self._annotation_dirty = False
        self._annotation_invalid_indices: tuple[int, ...] = ()
        self._review_entries: tuple[
            AnnotationReviewEntry | SegmentationReviewEntry, ...
        ] = ()
        self._updating_review_combo = False
        self._locked_processing_settings: ImageProcessingSettings | None = None
        self._updating_processing_controls = False
        self._updating_class_controls = False
        self._updating_patch_controls = False
        self._last_normalization_method: str | None = None
        self._model: YoloDetectionModel | YoloSegmentationModel | None = None
        self._inference_worker: InferenceWorker | SegmentationWorker | None = None
        self._inference_sample_id: str | None = None
        self._inference_converted: ConvertedImage | None = None
        self._training_worker: TrainingWorker | None = None
        self._training_preview: DatasetPreview | None = None
        self._last_training_run: TrainingRun | None = None
        self._crop_bounds: CropBounds | None = None
        self._crop_source_image_layer = None
        self._crop_source_shapes_layer = None
        self._crop_source_converted: ConvertedImage | None = None
        self._crop_source_visibility: list[tuple[object, bool]] = []
        self._crop_dirty = False
        self._crop_discarded_count = 0
        self._crop_invalid_indices: tuple[int, ...] = ()
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
        self._edit_classes_button = QPushButton("Edit Classes")
        self._edit_classes_button.clicked.connect(self._on_edit_classes)
        self._new_project_task_combo = QComboBox()
        self._new_project_task_combo.addItem("Bounding-box detection", "detect")
        self._new_project_task_combo.addItem("Instance segmentation", "segment")
        self._new_project_task_combo.setToolTip(
            "Task for a new project. It cannot be changed after initialization."
        )

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

        self._filter_combo = QComboBox()
        self._filter_combo.setToolTip(
            "Optional spatial filter applied independently to every selected source "
            "channel before normalization and RGB conversion."
        )
        for method, display_name in IMAGE_FILTERS.items():
            self._filter_combo.addItem(display_name, method)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._filter_radius_spin = QSpinBox()
        self._filter_radius_spin.setRange(1, 100)
        self._filter_radius_spin.setValue(1)
        self._filter_radius_spin.setToolTip(
            "Filter radius in source pixels. Gaussian uses a radius-limited kernel; "
            "median, mean, and top-hat use a disk footprint; frequency low-pass "
            "uses larger values for stronger smoothing."
        )
        self._filter_radius_spin.valueChanged.connect(
            self._on_processing_value_changed
        )

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
            spin.setToolTip(
                "Lower/upper limits used by percentile, Z-score, or fixed-range "
                "normalization."
            )
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

        self._class_combo = QComboBox()
        self._class_combo.setToolTip(
            "Class assigned to newly drawn boxes. Use Apply Class to Selected "
            "to reclassify existing boxes."
        )
        self._class_combo.currentIndexChanged.connect(
            self._on_current_class_changed
        )
        self._apply_class_button = QPushButton("Apply Class to Selected")
        self._apply_class_button.clicked.connect(
            self._on_apply_class_to_selected
        )
        self._class_counts_label = QLabel("Class counts: unavailable")
        self._class_counts_label.setWordWrap(True)

        self._reload_labels_button = QPushButton("Reload Labels")
        self._reload_labels_button.clicked.connect(self._on_reload_labels)
        self._save_annotation_button = QPushButton("Save Converted Image + BBoxes")
        self._new_instance_button = QPushButton("New Mask Instance")
        self._new_instance_button.setToolTip(
            "Allocate a new instance ID, select it, and enter napari paint mode."
        )
        self._new_instance_button.clicked.connect(self._on_new_mask_instance)
        self._delete_instance_button = QPushButton("Delete Selected Instance")
        self._delete_instance_button.setToolTip(
            "Clear every pixel belonging to the selected instance ID."
        )
        self._delete_instance_button.clicked.connect(self._on_delete_mask_instance)
        self._split_instance_button = QPushButton("Split Disconnected Instance")
        self._split_instance_button.setToolTip(
            "Give each 4-connected component of the selected ID its own instance ID."
        )
        self._split_instance_button.clicked.connect(self._on_split_mask_instance)
        self._merge_instance_button = QPushButton("Merge Instance IDs")
        self._merge_instance_button.setToolTip(
            "Merge comma-separated instance IDs into the currently selected ID."
        )
        self._merge_instance_button.clicked.connect(self._on_merge_mask_instances)
        self._largest_instance_button = QPushButton("Keep Largest by BBox")
        self._largest_instance_button.setToolTip(
            "Explicitly discard all but the component with the largest bounding box."
        )
        self._largest_instance_button.clicked.connect(
            self._on_keep_largest_mask_component
        )
        self._instance_details_label = QLabel("Selected instance: unavailable")
        self._instance_details_label.setWordWrap(True)
        self._save_annotation_button.clicked.connect(self._on_save_annotation)
        self._validate_project_button = QPushButton("Validate Project")
        self._validate_project_button.clicked.connect(self._on_validate_project)

        self._review_sample_combo = QComboBox()
        self._review_sample_combo.setToolTip(
            "Saved project annotations. A warning prefix marks an invalid pair."
        )
        self._review_refresh_button = QPushButton("Refresh")
        self._review_refresh_button.setToolTip(
            "Rescan the project's canonical annotation image/label pairs."
        )
        self._review_refresh_button.clicked.connect(self._refresh_annotation_browser)
        self._review_previous_button = QPushButton("Previous")
        self._review_previous_button.setToolTip(
            "Load the previous saved annotation, prompting first if edits are unsaved."
        )
        self._review_previous_button.clicked.connect(
            lambda: self._navigate_annotation(-1)
        )
        self._review_load_button = QPushButton("Load Selected")
        self._review_load_button.setToolTip(
            "Open the selected canonical image and its YOLO boxes for correction."
        )
        self._review_load_button.clicked.connect(self._on_load_review_annotation)
        self._review_next_button = QPushButton("Next")
        self._review_next_button.setToolTip(
            "Load the next saved annotation, prompting first if edits are unsaved."
        )
        self._review_next_button.clicked.connect(
            lambda: self._navigate_annotation(1)
        )
        self._review_save_button = QPushButton("Save Corrections")
        self._review_save_button.setToolTip(
            "Update the loaded canonical image and YOLO bbox file using the "
            "current corrected layer."
        )
        self._review_save_button.clicked.connect(self._on_save_annotation)
        self._review_status_label = QLabel(
            "Review: open a project to browse saved annotations."
        )
        self._review_status_label.setWordWrap(True)

        self._model_path_label = QLabel("No detection model loaded")
        self._model_path_label.setWordWrap(True)
        self._model_status_label = QLabel("Model: unavailable")
        self._model_status_label.setWordWrap(True)
        self._choose_model_button = QPushButton("Choose Detection Model")
        self._choose_model_button.clicked.connect(self._on_choose_model)

        self._device_combo = QComboBox()
        self._device_combo.setToolTip(
            "Processor used for prediction. GPU is normally faster when available."
        )
        for label, value in available_devices():
            self._device_combo.addItem(label, value)
        self._confidence_spin = self._fraction_spin(0.25)
        self._confidence_spin.setToolTip(
            "Minimum model confidence retained as a candidate detection."
        )
        self._model_iou_spin = self._fraction_spin(0.45)
        self._model_iou_spin.setToolTip(
            "YOLO's per-tile NMS threshold. Lower values suppress more overlapping boxes."
        )
        self._merge_iou_spin = self._fraction_spin(0.50)
        self._merge_iou_spin.setToolTip(
            "Global NMS threshold used to merge duplicate detections from overlapping tiles."
        )
        self._tile_size_spin = QSpinBox()
        self._tile_size_spin.setRange(64, 8192)
        self._tile_size_spin.setSingleStep(64)
        self._tile_size_spin.setValue(1024)
        self._tile_size_spin.setToolTip(
            "Square prediction tile size in pixels. Larger tiles use more memory."
        )
        self._overlap_percent_spin = QDoubleSpinBox()
        self._overlap_percent_spin.setRange(0.0, 90.0)
        self._overlap_percent_spin.setDecimals(1)
        self._overlap_percent_spin.setSuffix(" %")
        self._overlap_percent_spin.setValue(20.0)
        self._overlap_percent_spin.setToolTip(
            "Overlap between prediction tiles. More overlap reduces seam failures but "
            "increases inference time."
        )
        self._max_detections_spin = QSpinBox()
        self._max_detections_spin.setRange(1, 100_000)
        self._max_detections_spin.setValue(300)
        self._max_detections_spin.setToolTip(
            "Maximum detections YOLO may return from one prediction tile."
        )
        self._predict_button = QPushButton("Predict Current RGB Plane")
        self._predict_button.clicked.connect(self._on_predict)
        self._cancel_inference_button = QPushButton("Cancel")
        self._cancel_inference_button.clicked.connect(self._on_cancel_inference)
        self._inference_progress = QProgressBar()
        self._inference_progress.setRange(0, 1)
        self._inference_progress.setValue(0)
        self._inference_status_label = QLabel("Inference: idle")
        self._inference_status_label.setWordWrap(True)

        self._patch_size_combo = QComboBox()
        self._patch_size_combo.addItem("1024 × 1024", 1024)
        self._patch_size_combo.addItem("512 × 512", 512)
        self._patch_size_combo.setToolTip(
            "Fixed size of every canonical training sample. It locks after the "
            "first sample is saved."
        )
        self._patch_size_combo.currentIndexChanged.connect(
            self._on_patch_size_changed
        )
        self._patch_contract_label = QLabel(
            "Training patch size is not locked until the first crop is saved."
        )
        self._patch_contract_label.setWordWrap(True)
        self._select_crop_button = QPushButton("Select Training Crop")
        self._select_crop_button.clicked.connect(self._on_select_training_crop)
        self._create_crop_button = QPushButton("Create / Refresh Crop")
        self._create_crop_button.clicked.connect(self._on_create_training_crop)
        self._save_crop_button = QPushButton("Add Crop + Corrections")
        self._save_crop_button.clicked.connect(self._on_save_training_crop)
        self._return_crop_button = QPushButton("Return to Source")
        self._return_crop_button.clicked.connect(self._on_return_to_source)
        self._crop_status_label = QLabel(
            "Crop: run inference, then place a crop over a failure location."
        )
        self._crop_status_label.setWordWrap(True)

        self._training_model_combo = QComboBox()
        self._training_model_combo.setToolTip(
            "Checkpoint used to initialize retraining: the naive base model or a "
            "previous fine-tuned project model."
        )
        self._destination_input = QLineEdit()
        self._destination_input.setPlaceholderText(
            "Project root (default), or choose another output parent"
        )
        self._destination_input.setToolTip(
            "Parent folder where a new immutable retrain_<date>_<time> run is created."
        )
        self._destination_button = QPushButton("Choose Destination")
        self._destination_button.clicked.connect(self._on_choose_destination)
        self._group_field_input = QLineEdit()
        self._group_field_input.setPlaceholderText(
            "Optional audit field, e.g. metadata.patient"
        )
        self._group_field_input.setToolTip(
            "Optional audit field used to keep related samples in the same split, "
            "for example metadata.patient."
        )
        self._validation_fraction_spin = self._fraction_spin(0.20)
        self._validation_fraction_spin.setRange(0.05, 0.50)
        self._validation_fraction_spin.setToolTip(
            "Fraction of independent source groups reserved for validation."
        )
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.setValue(42)
        self._seed_spin.setToolTip(
            "Seed used for deterministic train/validation assignment and training."
        )
        self._training_tile_size_spin = QSpinBox()
        self._training_tile_size_spin.setRange(64, 8192)
        self._training_tile_size_spin.setSingleStep(64)
        self._training_tile_size_spin.setValue(1024)
        self._training_tile_size_spin.setToolTip(
            "Locked project training size. Change it only before saving the first sample."
        )
        self._training_overlap_spin = QDoubleSpinBox()
        self._training_overlap_spin.setRange(0.0, 90.0)
        self._training_overlap_spin.setDecimals(1)
        self._training_overlap_spin.setSuffix(" %")
        self._training_overlap_spin.setValue(20.0)
        self._training_overlap_spin.setToolTip(
            "Overlap used when deriving training tiles from canonical images."
        )
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
        self._epochs_spin.setToolTip(
            "Maximum complete passes through the training dataset."
        )
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(-1, 4096)
        self._batch_spin.setSpecialValueText("Auto")
        self._batch_spin.setValue(-1)
        self._batch_spin.setToolTip(
            "Images per optimization step. Auto lets Ultralytics choose based on memory."
        )
        self._patience_spin = QSpinBox()
        self._patience_spin.setRange(0, 100_000)
        self._patience_spin.setValue(30)
        self._patience_spin.setToolTip(
            "Epochs without validation improvement before early stopping; 0 disables it."
        )
        self._training_device_combo = QComboBox()
        self._training_device_combo.setToolTip(
            "Processor used for retraining. GPU is normally much faster when available."
        )
        for label, value in available_devices():
            self._training_device_combo.addItem(label, value)
        self._train_only_checkbox = QCheckBox(
            "Train without independent validation (exploratory)"
        )
        self._train_only_checkbox.setToolTip(
            "Allow training with too few independent groups for validation. The result "
            "cannot measure generalization."
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

        project_buttons = QHBoxLayout()
        project_buttons.addWidget(self._new_project_button)
        project_buttons.addWidget(self._open_project_button)
        project_layout = QVBoxLayout()
        project_layout.addWidget(self._project_path_label)
        project_layout.addWidget(self._project_status_label)
        project_layout.addWidget(QLabel("Task for new project"))
        project_layout.addWidget(self._new_project_task_combo)
        project_layout.addLayout(project_buttons)
        project_layout.addWidget(self._edit_classes_button)
        self._project_section = CollapsibleSection(
            "Project", project_layout, expanded=True
        )

        processing_form = QFormLayout()
        processing_form.addRow("Channel axis", self._channel_axis_combo)
        processing_form.addRow("Output red", self._red_channel_combo)
        processing_form.addRow("Output green", self._green_channel_combo)
        processing_form.addRow("Output blue", self._blue_channel_combo)
        processing_form.addRow("Pre-filter", self._filter_combo)
        processing_form.addRow("Filter radius", self._filter_radius_spin)
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
        class_form = QFormLayout()
        class_form.addRow("Current class", self._class_combo)
        annotation_layout.addLayout(class_form)
        annotation_layout.addWidget(self._apply_class_button)
        mask_buttons = QHBoxLayout()
        mask_buttons.addWidget(self._new_instance_button)
        mask_buttons.addWidget(self._delete_instance_button)
        annotation_layout.addLayout(mask_buttons)
        mask_cleanup_buttons = QHBoxLayout()
        mask_cleanup_buttons.addWidget(self._split_instance_button)
        mask_cleanup_buttons.addWidget(self._merge_instance_button)
        mask_cleanup_buttons.addWidget(self._largest_instance_button)
        annotation_layout.addLayout(mask_cleanup_buttons)
        annotation_layout.addWidget(self._instance_details_label)
        annotation_layout.addWidget(self._class_counts_label)
        annotation_layout.addWidget(self._label_status_label)
        annotation_layout.addLayout(annotation_buttons)
        annotation_layout.addWidget(self._validate_project_button)
        self._annotation_section = CollapsibleSection(
            "Image and annotations",
            annotation_layout,
            expanded=True,
        )

        review_top = QHBoxLayout()
        review_top.addWidget(self._review_sample_combo)
        review_top.addWidget(self._review_refresh_button)
        review_buttons = QHBoxLayout()
        review_buttons.addWidget(self._review_previous_button)
        review_buttons.addWidget(self._review_load_button)
        review_buttons.addWidget(self._review_next_button)
        review_buttons.addWidget(self._review_save_button)
        review_layout = QVBoxLayout()
        review_layout.addLayout(review_top)
        review_layout.addLayout(review_buttons)
        review_layout.addWidget(self._review_status_label)
        self._review_section = CollapsibleSection(
            "Review saved annotations", review_layout, expanded=False
        )

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
        self._inference_section = CollapsibleSection(
            "Large-image tiled prediction",
            inference_layout,
            expanded=False,
        )

        crop_form = QFormLayout()
        crop_form.addRow("Training patch", self._patch_size_combo)
        crop_buttons = QHBoxLayout()
        crop_buttons.addWidget(self._select_crop_button)
        crop_buttons.addWidget(self._create_crop_button)
        crop_save_buttons = QHBoxLayout()
        crop_save_buttons.addWidget(self._save_crop_button)
        crop_save_buttons.addWidget(self._return_crop_button)
        crop_layout = QVBoxLayout()
        crop_layout.addLayout(crop_form)
        crop_layout.addWidget(self._patch_contract_label)
        crop_layout.addLayout(crop_buttons)
        crop_layout.addLayout(crop_save_buttons)
        crop_layout.addWidget(self._crop_status_label)
        self._crop_section = CollapsibleSection(
            "Movable training crop",
            crop_layout,
            expanded=True,
        )

        retrain_form = QFormLayout()
        retrain_form.addRow("Starting model", self._training_model_combo)
        destination_row = QHBoxLayout()
        destination_row.addWidget(self._destination_input)
        destination_row.addWidget(self._destination_button)
        retrain_form.addRow("Output parent", destination_row)
        retrain_form.addRow("Group metadata", self._group_field_input)
        retrain_form.addRow("Validation fraction", self._validation_fraction_spin)
        retrain_form.addRow("Split seed", self._seed_spin)
        retrain_form.addRow(
            "Training patch size (project)", self._training_tile_size_spin
        )
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
        self._retrain_section = CollapsibleSection(
            "Dataset building and retraining",
            retrain_layout,
            expanded=False,
        )

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.addWidget(self._project_section)
        content_layout.addWidget(self._annotation_section)
        content_layout.addWidget(self._review_section)
        content_layout.addWidget(self._inference_section)
        content_layout.addWidget(self._crop_section)
        content_layout.addWidget(self._retrain_section)
        content_layout.addStretch(1)
        content = QWidget()
        content.setLayout(content_layout)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QScrollArea.NoFrame)
        self._scroll_area.setWidget(content)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll_area)
        self.setLayout(layout)

        self._connect_viewer_events()
        self._on_normalization_changed()
        self._on_filter_changed()
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
            project = ProjectStore.initialize(
                Path(selected), task=str(self._new_project_task_combo.currentData())
            )
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

    def _update_project_status(self) -> None:
        if self._project is None:
            self._project_status_label.setText(
                "Create a new project or open an initialized project."
            )
            return
        class_summary = ", ".join(
            f"{class_id}: {name}"
            for class_id, name in sorted(self._project.config.classes.items())
        )
        patch = self._project.config.training_patch
        patch_status = "locked" if patch.get("locked") else "not locked"
        self._project_status_label.setText(
            f"Project loaded · schema {self._project.config.schema_version} · "
            f"task {self._project.config.task} · "
            f"classes [{class_summary}] · training patch {patch['size']} "
            f"({patch_status})"
        )

    def _on_edit_classes(self) -> None:
        if self._project is None:
            self._show_error("Create or open a project first.")
            return
        if not self._resolve_unsaved_changes():
            return

        previous_class_id = self._current_class_id()
        dialog = ClassMapDialog(self._project.config.classes, self)
        if not dialog.exec():
            return
        classes = dialog.classes()
        shapes = self._annotation_layer()
        if self._project.config.task == "segment":
            active_ids = {record.class_id for record in self._segment_instances.values()}
            removed_active = sorted(active_ids - set(classes))
            if removed_active:
                self._show_error(
                    "Cannot remove class ID(s) used by the current instance mask: "
                    + ", ".join(str(value) for value in removed_active)
                )
                return
        if shapes is not None:
            try:
                active_ids = set(self._class_ids(shapes, len(shapes.data)))
            except LabelValidationError as exc:
                self._show_error(f"Could not edit classes:\n{exc}")
                return
            removed_active = sorted(active_ids - set(classes))
            if removed_active:
                self._show_error(
                    "Cannot remove class ID(s) used by the current bbox layer: "
                    + ", ".join(str(value) for value in removed_active)
                )
                return
        try:
            self._project.update_classes(classes)
        except (ProjectError, OSError, ValueError) as exc:
            self._show_error(f"Could not update project classes:\n{exc}")
            return

        self._populate_class_selector(preferred=previous_class_id)
        if shapes is not None:
            properties = dict(getattr(shapes, "properties", {}) or {})
            class_ids = self._class_ids(shapes, len(shapes.data))
            properties["class_name"] = np.asarray(
                [classes[class_id] for class_id in class_ids], dtype=object
            )
            shapes.properties = properties
            self._apply_class_colors(shapes)
            self._set_default_current_properties(shapes)
            self._update_class_counts(shapes)
        elif self._project.config.task == "segment":
            self._segment_instances = {
                instance_id: InstanceRecord(
                    **{
                        **record.__dict__,
                        "class_name": classes[record.class_id],
                    }
                )
                for instance_id, record in self._segment_instances.items()
            }
            self._apply_instance_colors(self._segmentation_layer())
            self._update_class_counts()

        if self._model is not None and set(self._model.names) != set(classes):
            self._model = None
            self._model_path_label.setText("No detection model loaded")
            self._model_status_label.setText(
                "Model: unloaded because its class IDs no longer match the project"
            )
        elif self._model is not None:
            self._update_loaded_model_status()
        self._training_preview = None
        self._dataset_status_label.setText(
            "Dataset: class map changed; validate or preview the split again."
        )
        self._refresh_training_model_choices()
        self._update_project_status()
        self._update_action_state()

    def _set_project(self, project: ProjectStore) -> None:
        self._close_crop_session(remove_selection=True, restore_sources=True)
        previous_annotation = self._annotation_layer()
        if previous_annotation is not None:
            self.napari_viewer.layers.remove(previous_annotation)
        previous_mask = self._segmentation_layer()
        if previous_mask is not None:
            self.napari_viewer.layers.remove(previous_mask)
        self._annotation_image_layer = None
        self._annotation_invalid_indices = ()
        self._project = project
        self._annotation_io = AnnotationIO(project) if project.config.task == "detect" else None
        self._segmentation_io = (
            SegmentationIO(project) if project.config.task == "segment" else None
        )
        self._segment_instances = {}
        task_index = self._new_project_task_combo.findData(project.config.task)
        if task_index >= 0:
            self._new_project_task_combo.setCurrentIndex(task_index)
        self._model = None
        self._model_path_label.setText(f"No {project.config.task} model loaded")
        self._model_status_label.setText("Model: unavailable")
        self._training_preview = None
        self._dataset_status_label.setText(
            "Dataset: annotations have not been validated yet."
        )
        self._refresh_training_model_choices()
        self._project_path_label.setText(str(project.paths.root))
        self._populate_class_selector()
        self._update_project_status()
        self._update_class_counts()
        self._apply_project_processing_settings()
        self._apply_project_patch_settings()
        self._refresh_annotation_browser()
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

    @staticmethod
    def _is_labels_layer(layer) -> bool:
        return layer is not None and layer.__class__.__name__.lower().endswith("labels")

    def _on_active_layer_changed(self, event=None) -> None:
        active = self._active_layer()
        if self._is_annotation_review_image(active):
            self._load_annotations_for_image(active)
        elif self._is_source_image_layer(active):
            if (
                self._crop_bounds is not None
                and active is not self._crop_source_image_layer
            ):
                if not self._resolve_crop_unsaved_changes():
                    return
                self._close_crop_session(
                    remove_selection=True, restore_sources=True
                )
            self._configure_image_controls(active)
            self._load_annotations_for_image(active)
        self._update_action_state()

    def _on_current_step_changed(self, event=None) -> None:
        image_layer = self._annotation_image_layer
        if image_layer is None or (
            self._annotation_io is None and self._segmentation_io is None
        ):
            return
        if self._crop_bounds is not None:
            if not self._resolve_crop_unsaved_changes():
                return
            self._close_crop_session(
                remove_selection=True, restore_sources=True
            )
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
        metadata = getattr(image_layer, "metadata", {}) or {}
        original_source = metadata.get("cci_original_source_path")
        if isinstance(original_source, str) and original_source:
            return Path(original_source)
        source = getattr(image_layer, "source", None)
        raw_path = getattr(source, "path", None)
        if raw_path:
            return Path(raw_path)
        metadata = getattr(image_layer, "metadata", {}) or {}
        raw_path = metadata.get("path") or metadata.get("source_path")
        return Path(raw_path) if raw_path else None

    @classmethod
    def _is_source_image_layer(cls, layer) -> bool:
        metadata = getattr(layer, "metadata", {}) or {}
        return cls._is_image_layer(layer) and not bool(
            metadata.get("cci_rgb_preview")
            or metadata.get("cci_training_crop")
            or metadata.get("cci_annotation_review")
        )

    @classmethod
    def _is_annotation_review_image(cls, layer) -> bool:
        metadata = getattr(layer, "metadata", {}) or {}
        return cls._is_image_layer(layer) and bool(
            metadata.get("cci_annotation_review")
        )

    def _image_stem(self, image_layer) -> str:
        assert self._project is not None
        metadata = getattr(image_layer, "metadata", {}) or {}
        review_sample = metadata.get("cci_sample_id")
        if isinstance(review_sample, str) and review_sample:
            return review_sample
        source_path = self._source_path(image_layer)
        value = source_path.stem if source_path is not None else getattr(
            image_layer, "name", "image"
        )
        base_stem = AnnotationIO.safe_stem(str(value))
        try:
            settings = self._effective_processing_settings()
            plane = self._image_adapter.plane_selection(
                image_layer, self.napari_viewer, settings
            )
            return self._image_adapter.sample_id(
                base_stem, plane, settings.channel_axis
            )
        except ImageConversionError:
            return base_stem

    def _load_annotations_for_image(self, image_layer, *, force: bool = False) -> None:
        if self._project is not None and self._project.config.task == "segment":
            self._load_segmentation_for_image(image_layer, force=force)
            return
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
                if self._is_annotation_review_image(image_layer):
                    spatial_shape = image_shape[:2]
                else:
                    settings = self._effective_processing_settings()
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
            edge_color=[class_color(value) for value in properties["class_id"]]
            if len(rectangles)
            else class_color(self._current_class_id() or 0),
            face_color="transparent",
        )
        shapes.metadata["cci_image_stem"] = stem
        shapes.metadata["cci_project_root"] = str(self._project.paths.root)
        shapes.metadata["cci_label_path"] = str(label_path) if label_path else None
        self._set_default_current_properties(shapes)
        self._apply_class_colors(shapes)
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
        self._update_class_counts(shapes)
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
        self._refresh_annotation_validation(
            valid_message=self._label_status_label.text()
        )
        self._update_action_state()

    def _load_segmentation_for_image(self, image_layer, *, force: bool = False) -> None:
        if self._segmentation_io is None or not self._is_image_layer(image_layer):
            return
        stem = self._image_stem(image_layer)
        existing = self._segmentation_layer()
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
        try:
            converted = self._convert_current_image(image_layer)
            pair = self._segmentation_io.find_mask(stem)
            if pair is None:
                mask = np.zeros(converted.data.shape[:2], dtype=np.uint32)
                instances: dict[int, InstanceRecord] = {}
            else:
                loaded = self._segmentation_io.load(stem, converted.data.shape[:2])
                mask = loaded.mask
                instances = loaded.instances
        except (ImageConversionError, SegmentationError, OSError) as exc:
            self._show_error(f"Could not load instance segmentation:\n{exc}")
            return
        bbox_layer = self._annotation_layer()
        if bbox_layer is not None:
            self.napari_viewer.layers.remove(bbox_layer)
        if existing is not None:
            self.napari_viewer.layers.remove(existing)
        labels = self.napari_viewer.add_labels(
            mask,
            name=self.SEGMENTATION_LAYER_NAME,
            metadata={
                "cci_image_stem": stem,
                "cci_project_root": str(self._project.paths.root),
                "cci_mask_path": str(pair[0]) if pair else None,
            },
        )
        self._segment_instances = instances
        self._apply_instance_colors(labels)
        try:
            labels.events.data.connect(self._on_labels_data_changed)
            labels.events.selected_label.connect(self._on_selected_instance_changed)
        except (AttributeError, TypeError):
            pass
        self._annotation_image_layer = image_layer
        self._converted_image = converted
        self._current_sample_id = stem
        self._annotation_dirty = False
        self._update_image_status(image_layer)
        if pair is None:
            self._label_status_label.setText(
                f"Instances: none found for {stem}; an empty editable mask was created."
            )
            self._save_annotation_button.setText("Add Image + Instance Mask")
        else:
            self._label_status_label.setText(
                f"Instances: loaded {len(instances)} object(s) from {pair[0]}"
            )
            self._save_annotation_button.setText("Update Image + Instance Mask")
        self._refresh_segmentation_validation()
        self._update_class_counts()
        self._on_selected_instance_changed()
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

    def _populate_class_selector(self, *, preferred: int | None = None) -> None:
        self._updating_class_controls = True
        try:
            self._class_combo.clear()
            if self._project is None:
                return
            for class_id, name in sorted(self._project.config.classes.items()):
                self._class_combo.addItem(
                    class_display_name(class_id, name), class_id
                )
                index = self._class_combo.count() - 1
                self._class_combo.setItemData(
                    index,
                    QColor(class_color(class_id)),
                    Qt.ForegroundRole,
                )
            if preferred is not None:
                index = self._class_combo.findData(preferred)
                if index >= 0:
                    self._class_combo.setCurrentIndex(index)
        finally:
            self._updating_class_controls = False
        self._on_current_class_changed()

    def _current_class_id(self) -> int | None:
        value = self._class_combo.currentData()
        if value is not None:
            return int(value)
        if self._project is not None and self._project.config.classes:
            return min(self._project.config.classes)
        return None

    def _on_current_class_changed(self, index: int = -1) -> None:
        del index
        if self._updating_class_controls:
            return
        shapes = self._editable_bbox_layer()
        if shapes is not None:
            self._set_default_current_properties(shapes)

    def _apply_class_colors(self, shapes_layer) -> None:
        count = len(getattr(shapes_layer, "data", ()))
        try:
            class_ids = self._class_ids(shapes_layer, count)
        except LabelValidationError:
            return
        colors = [class_color(class_id) for class_id in class_ids]
        try:
            if colors:
                shapes_layer.edge_color = colors
            current_class_id = self._current_class_id()
            if current_class_id is not None:
                shapes_layer.current_edge_color = class_color(current_class_id)
        except (AttributeError, TypeError, ValueError):
            pass

    def _update_class_counts(self, shapes_layer=None) -> None:
        if self._project is None:
            self._class_counts_label.setText("Class counts: unavailable")
            return
        if self._project.config.task == "segment":
            counts = {
                class_id: sum(
                    record.class_id == class_id
                    for record in self._segment_instances.values()
                )
                for class_id in self._project.config.classes
            }
            summary = " · ".join(
                f"{class_id} {name}: {counts[class_id]}"
                for class_id, name in sorted(self._project.config.classes.items())
            )
            self._class_counts_label.setText(f"Class counts: {summary}")
            return
        shapes_layer = shapes_layer or self._editable_bbox_layer()
        class_ids: tuple[int, ...] = ()
        if shapes_layer is not None:
            try:
                class_ids = self._class_ids(
                    shapes_layer, len(getattr(shapes_layer, "data", ()))
                )
            except LabelValidationError:
                self._class_counts_label.setText(
                    "Class counts: unavailable (invalid class properties)"
                )
                return
        counts = {
            class_id: class_ids.count(class_id)
            for class_id in self._project.config.classes
        }
        summary = " · ".join(
            f"{class_id} {name}: {counts[class_id]}"
            for class_id, name in sorted(self._project.config.classes.items())
        )
        self._class_counts_label.setText(f"Class counts: {summary}")

    def _on_apply_class_to_selected(self) -> None:
        if self._project is not None and self._project.config.task == "segment":
            self._apply_class_to_selected_instance()
            return
        shapes = self._editable_bbox_layer()
        class_id = self._current_class_id()
        if shapes is None or self._project is None or class_id is None:
            self._show_error("Open a project and bbox layer first.")
            return
        selected = sorted(getattr(shapes, "selected_data", set()))
        if not selected:
            self._show_info("Select one or more boxes in the bbox layer first.")
            return
        count = len(shapes.data)
        if any(index < 0 or index >= count for index in selected):
            self._show_error("The bbox selection is no longer valid.")
            return
        try:
            class_ids = np.asarray(self._class_ids(shapes, count), dtype=int)
        except LabelValidationError as exc:
            self._show_error(str(exc))
            return
        properties = dict(getattr(shapes, "properties", {}) or {})
        class_names = np.asarray(
            [self._project.config.classes[value] for value in class_ids],
            dtype=object,
        )
        sources = self._property_values(properties, "source", count, "manual", object)
        confidences = self._property_values(
            properties, "confidence", count, np.nan, float
        )
        tile_ids = self._property_values(properties, "tile_id", count, -1, int)
        class_ids[selected] = class_id
        class_names[selected] = self._project.config.classes[class_id]
        sources[selected] = "manual"
        confidences[selected] = np.nan
        properties.update(
            {
                "class_id": class_ids,
                "class_name": class_names,
                "confidence": confidences,
                "source": sources,
                "tile_id": tile_ids,
            }
        )
        shapes.properties = properties
        self._apply_class_colors(shapes)
        self._set_default_current_properties(shapes)
        self._update_class_counts(shapes)
        if shapes is self._crop_bbox_layer():
            self._crop_dirty = True
            message = (
                f"Crop: assigned class {class_id} to {len(selected)} selected "
                "box(es); changes are not saved."
            )
            self._refresh_crop_validation(valid_message=message)
            self._update_action_state()
        else:
            self._annotation_dirty = True
            message = (
                f"Labels: assigned class {class_id} to "
                f"{len(selected)} selected box(es); changes are not saved."
            )
            self._refresh_annotation_validation(valid_message=message)
            self._update_action_state()

    @staticmethod
    def _property_values(
        properties: dict, key: str, count: int, default, dtype
    ) -> np.ndarray:
        values = np.asarray(properties.get(key, ()), dtype=dtype)
        if values.shape == (count,):
            return values.copy()
        return np.full(count, default, dtype=dtype)

    def _set_default_current_properties(self, shapes_layer) -> None:
        if self._project is None:
            return
        class_id = self._current_class_id()
        if class_id is None:
            return
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
            shapes_layer.current_edge_color = class_color(class_id)
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

    def _segmentation_layer(self):
        try:
            for layer in self.napari_viewer.layers:
                if (
                    getattr(layer, "name", None) == self.SEGMENTATION_LAYER_NAME
                    and self._is_labels_layer(layer)
                ):
                    return layer
        except TypeError:
            pass
        return None

    def _selected_instance_id(self) -> int:
        layer = self._segmentation_layer()
        try:
            return int(layer.selected_label) if layer is not None else 0
        except (AttributeError, TypeError, ValueError):
            return 0

    def _apply_instance_colors(self, labels_layer=None) -> None:
        labels_layer = labels_layer or self._segmentation_layer()
        if labels_layer is None:
            return
        colors = {0: "transparent"}
        colors.update(
            {
                instance_id: class_color(record.class_id)
                for instance_id, record in self._segment_instances.items()
            }
        )
        try:
            labels_layer.color = colors
            labels_layer.refresh()
        except (AttributeError, TypeError, ValueError):
            pass

    def _on_selected_instance_changed(self, event=None) -> None:
        del event
        instance_id = self._selected_instance_id()
        record = self._segment_instances.get(instance_id)
        if record is None:
            self._instance_details_label.setText(
                f"Selected instance: {instance_id or 'background'} · no metadata"
            )
        else:
            confidence = (
                "manual" if record.confidence is None else f"{record.confidence:.3f}"
            )
            self._instance_details_label.setText(
                f"Selected instance: {instance_id} · class {record.class_id} "
                f"{record.class_name} · area {record.area} px · confidence {confidence}"
            )
        self._update_action_state()

    def _on_new_mask_instance(self) -> None:
        layer = self._segmentation_layer()
        class_id = self._current_class_id()
        if layer is None or self._project is None or class_id is None:
            self._show_error("Open a segmentation project and image first.")
            return
        instance_id = int(np.asarray(layer.data).max(initial=0)) + 1
        self._segment_instances[instance_id] = InstanceRecord(
            instance_id=instance_id,
            class_id=class_id,
            class_name=self._project.config.classes[class_id],
            confidence=None,
            source="manual",
            status="corrected",
            bbox=(0, 0, 0, 0),
            area=0,
        )
        try:
            layer.selected_label = instance_id
            layer.mode = "paint"
        except (AttributeError, ValueError):
            pass
        self._annotation_dirty = True
        self._label_status_label.setText(
            f"Instances: allocated ID {instance_id}; paint its pixels in the Labels layer."
        )
        self._on_selected_instance_changed()

    def _on_delete_mask_instance(self) -> None:
        layer = self._segmentation_layer()
        instance_id = self._selected_instance_id()
        if layer is None or not instance_id:
            self._show_info("Select a non-background instance first.")
            return
        data = np.asarray(layer.data, dtype=np.uint32).copy()
        data[data == instance_id] = 0
        self._segment_instances.pop(instance_id, None)
        layer.data = data
        self._on_labels_data_changed()

    def _on_split_mask_instance(self) -> None:
        layer = self._segmentation_layer()
        instance_id = self._selected_instance_id()
        if layer is None or self._project is None or not instance_id:
            self._show_info("Select a non-background instance first.")
            return
        try:
            data, records = split_instance(
                layer.data,
                instance_id,
                self._segment_instances,
                self._project.config.classes,
            )
        except SegmentationError as exc:
            self._show_error(str(exc))
            return
        layer.data = data
        self._segment_instances = records
        self._on_labels_data_changed()

    def _on_merge_mask_instances(self) -> None:
        layer = self._segmentation_layer()
        target = self._selected_instance_id()
        if layer is None or self._project is None or not target:
            self._show_info("Select the target instance first.")
            return
        text, accepted = QInputDialog.getText(
            self,
            "Merge instance IDs",
            f"Comma-separated IDs to merge into {target}:",
        )
        if not accepted:
            return
        try:
            source_ids = {int(value.strip()) for value in text.split(",") if value.strip()}
        except ValueError:
            self._show_error("Instance IDs must be comma-separated integers.")
            return
        source_ids.discard(target)
        missing = sorted(source_ids - set(self._segment_instances))
        if not source_ids or missing:
            self._show_error(
                "Enter at least one existing source ID."
                + (f" Missing: {missing}" if missing else "")
            )
            return
        data = np.asarray(layer.data, dtype=np.uint32).copy()
        for source_id in source_ids:
            data[data == source_id] = target
            self._segment_instances.pop(source_id, None)
        layer.data = data
        self._on_labels_data_changed()

    def _on_keep_largest_mask_component(self) -> None:
        layer = self._segmentation_layer()
        instance_id = self._selected_instance_id()
        if layer is None or not instance_id:
            self._show_info("Select a non-background instance first.")
            return
        kept, report = keep_largest_component_by_bbox(layer.data == instance_id)
        data = np.asarray(layer.data, dtype=np.uint32).copy()
        data[data == instance_id] = 0
        data[kept] = instance_id
        layer.data = data
        self._label_status_label.setText(
            f"Instances: removed {report.removed_components} component(s), "
            f"{report.removed_pixels} pixel(s); save to keep this correction."
        )
        self._on_labels_data_changed()

    def _apply_class_to_selected_instance(self) -> None:
        instance_id = self._selected_instance_id()
        class_id = self._current_class_id()
        if self._project is None or not instance_id or class_id is None:
            self._show_info("Select a non-background instance first.")
            return
        record = self._segment_instances.get(instance_id)
        if record is None:
            self._show_error(f"Instance {instance_id} has no metadata.")
            return
        self._segment_instances[instance_id] = InstanceRecord(
            **{
                **record.__dict__,
                "class_id": class_id,
                "class_name": self._project.config.classes[class_id],
                "confidence": None,
                "source": "manual",
                "status": "corrected",
            }
        )
        self._annotation_dirty = True
        self._apply_instance_colors()
        self._update_class_counts()
        self._on_selected_instance_changed()

    def _on_labels_data_changed(self, event=None) -> None:
        del event
        self._annotation_dirty = True
        self._refresh_segmentation_validation()
        self._update_class_counts()
        self._on_selected_instance_changed()

    def _refresh_segmentation_validation(self) -> None:
        layer = self._segmentation_layer()
        if layer is None or self._project is None:
            self._segmentation_errors = ()
            return
        data = np.asarray(layer.data)
        errors: list[str] = []
        if data.ndim != 2:
            errors.append(f"mask must be 2D, found {data.ndim}D")
        elif self._converted_image is not None and data.shape != self._converted_image.data.shape[:2]:
            errors.append(
                f"mask shape {data.shape} differs from image {self._converted_image.data.shape[:2]}"
            )
        if not np.issubdtype(data.dtype, np.integer) or np.any(data < 0):
            errors.append("mask IDs must be non-negative integers")
        elif np.any(data > np.iinfo(np.uint32).max):
            errors.append("mask IDs exceed uint32 storage capacity")
        else:
            present = {int(value) for value in np.unique(data) if value}
            known = set(self._segment_instances)
            if present - known:
                errors.append(f"IDs missing metadata: {sorted(present - known)}")
            if known - present:
                errors.append(f"empty metadata IDs: {sorted(known - present)}")
            invalid_classes = sorted(
                instance_id
                for instance_id, record in self._segment_instances.items()
                if record.class_id not in self._project.config.classes
            )
            if invalid_classes:
                errors.append(f"instances with invalid classes: {invalid_classes}")
            disconnected = disconnected_instance_ids(data)
            if disconnected:
                errors.append(
                    f"disconnected instance IDs {list(disconnected)}; split or explicitly keep largest"
                )
            if not errors:
                self._segment_instances = refresh_instance_records(
                    data, self._segment_instances, self._project.config.classes
                )
        self._segmentation_errors = tuple(errors)
        if errors:
            self._label_status_label.setText("Instances invalid: " + " · ".join(errors))
        elif self._annotation_dirty:
            self._label_status_label.setText(
                "Instances: unsaved edits. Save before changing image or Z/T plane."
            )
        self._apply_instance_colors(layer)

    def _crop_bbox_layer(self):
        layer = self._get_layer_by_name(self.CROP_BBOX_LAYER_NAME)
        return layer if self._is_shapes_layer(layer) else None

    def _editable_bbox_layer(self):
        if self._crop_bounds is not None:
            crop_layer = self._crop_bbox_layer()
            if crop_layer is not None:
                return crop_layer
        return self._annotation_layer()

    def _on_reload_labels(self) -> None:
        image_layer = self._image_for_annotation()
        if image_layer is None:
            self._show_error("Select an image layer first.")
            return
        self._load_annotations_for_image(image_layer, force=True)

    def _on_save_annotation(self) -> None:
        if self._project is not None and self._project.config.task == "segment":
            self._on_save_segmentation()
            return
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

    def _on_save_segmentation(self) -> None:
        image_layer = self._image_for_annotation()
        labels_layer = self._segmentation_layer()
        if image_layer is None or labels_layer is None:
            self._show_error("Select an image and its instance Labels layer first.")
            return
        try:
            converted = self._convert_current_image(image_layer)
            result = self._save_converted_segmentation(
                converted, labels_layer=labels_layer, show_message=True
            )
        except (SegmentationError, ImageConversionError, ProjectError, OSError) as exc:
            self._show_error(f"Could not save segmentation:\n{exc}")
            return
        if result is not None:
            self._update_action_state()

    def _save_converted_segmentation(
        self, converted: ConvertedImage, *, labels_layer, show_message: bool
    ):
        if self._segmentation_io is None or self._project is None:
            raise SegmentationError("Create or open a segmentation project first.")
        patch_size = self._training_patch_size()
        if tuple(converted.data.shape[:2]) != (patch_size, patch_size):
            raise SegmentationError(
                f"Canonical training images must be {patch_size}×{patch_size}. "
                "Segmentation crop saving arrives in Phase 7D."
            )
        self._refresh_segmentation_validation()
        if self._segmentation_errors:
            raise SegmentationError("; ".join(self._segmentation_errors))
        conversion_metadata = {
            "settings": converted.settings.to_mapping(),
            "plane_indices": converted.plane.non_spatial_indices,
            "axis_labels": list(converted.plane.axis_labels),
            "normalization_stats": list(converted.normalization_stats),
            "prediction": getattr(labels_layer, "metadata", {}).get("cci_prediction"),
        }
        result = self._segmentation_io.save(
            image_data=converted.data,
            sample_id=converted.sample_id,
            mask=np.asarray(labels_layer.data),
            instances=self._segment_instances,
            source_path=self._source_path(self._annotation_image_layer),
            conversion_metadata=conversion_metadata,
        )
        self._project.lock_image_processing(converted.settings.to_mapping())
        self._project.lock_training_patch(patch_size, padding_value=114)
        self._converted_image = converted
        self._current_sample_id = converted.sample_id
        self._annotation_dirty = False
        labels_layer.metadata["cci_mask_path"] = str(result.mask_path)
        self._label_status_label.setText(
            f"Instances: {result.operation} {result.instance_count} object(s) · {result.mask_path}"
        )
        self._save_annotation_button.setText("Update Image + Instance Mask")
        if show_message:
            self._show_info(
                f"Segmentation {result.operation}:\n{result.image_path.name}\n"
                f"{result.mask_path.name}\n{result.instances_path.name}"
            )
        self._apply_project_processing_settings()
        self._apply_project_patch_settings()
        self._update_project_status()
        self._refresh_annotation_browser(preferred=converted.sample_id)
        return result

    def _save_converted_annotation(
        self,
        converted: ConvertedImage,
        *,
        shapes_layer,
        show_message: bool,
    ):
        if self._annotation_io is None or self._project is None:
            raise AnnotationError("Create or open a project first.")
        patch_size = self._training_patch_size()
        if tuple(converted.data.shape[:2]) != (patch_size, patch_size):
            raise AnnotationError(
                f"Canonical training images must be {patch_size}×{patch_size}. "
                "Use Select Training Crop for a larger or smaller source image."
            )
        rectangles = tuple(
            np.asarray(shape, dtype=float) for shape in shapes_layer.data
        )
        class_ids = self._class_ids(shapes_layer, len(rectangles))
        conversion_metadata = {
            "settings": converted.settings.to_mapping(),
            "plane_indices": converted.plane.non_spatial_indices,
            "axis_labels": list(converted.plane.axis_labels),
            "normalization_stats": list(converted.normalization_stats),
            "prediction": getattr(shapes_layer, "metadata", {}).get(
                "cci_prediction"
            ),
        }
        image_metadata = getattr(self._annotation_image_layer, "metadata", {}) or {}
        prior_conversion = image_metadata.get("cci_conversion")
        if isinstance(prior_conversion, dict):
            current_prediction = conversion_metadata["prediction"]
            conversion_metadata = dict(prior_conversion)
            conversion_metadata["settings"] = converted.settings.to_mapping()
            conversion_metadata.setdefault(
                "plane_indices", converted.plane.non_spatial_indices
            )
            conversion_metadata.setdefault(
                "axis_labels", list(converted.plane.axis_labels)
            )
            conversion_metadata.setdefault(
                "normalization_stats", list(converted.normalization_stats)
            )
            if current_prediction is not None:
                conversion_metadata["prediction"] = current_prediction
        result = self._annotation_io.save_pair(
            image_data=converted.data,
            image_name=converted.sample_id,
            source_path=self._source_path(self._annotation_image_layer),
            sample_id=converted.sample_id,
            rectangles=rectangles,
            class_ids=class_ids,
            conversion_metadata=conversion_metadata,
        )
        self._project.lock_image_processing(converted.settings.to_mapping())
        self._project.lock_training_patch(patch_size, padding_value=114)
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
        self._apply_project_patch_settings()
        self._update_project_status()
        self._refresh_annotation_browser(preferred=converted.sample_id)
        return result

    def _on_shapes_data_changed(self, event=None) -> None:
        del event
        self._annotation_dirty = True
        self._update_class_counts()
        image_layer = self._annotation_image_layer
        if self._is_source_image_layer(image_layer):
            try:
                live_sample = self._image_stem(image_layer)
                if live_sample == self._current_sample_id:
                    self._converted_image = self._convert_current_image(image_layer)
            except ImageConversionError:
                pass
        self._refresh_annotation_validation(
            valid_message=(
                "Labels: unsaved edits. Save before changing image or Z/T plane."
            )
        )
        self._update_action_state()

    def _refresh_annotation_validation(
        self, *, valid_message: str | None = None
    ) -> None:
        shapes = self._annotation_layer()
        converted = self._converted_image
        if shapes is None or converted is None:
            self._annotation_invalid_indices = ()
            return
        height, width = converted.data.shape[:2]
        invalid = invalid_rectangle_indices(
            shapes.data, height=height, width=width
        )
        self._annotation_invalid_indices = invalid
        face_colors = np.zeros((len(shapes.data), 4), dtype=float)
        if invalid:
            face_colors[list(invalid)] = (1.0, 0.0, 0.0, 0.35)
        try:
            shapes.face_color = (
                face_colors if len(shapes.data) else "transparent"
            )
        except (AttributeError, TypeError, ValueError):
            pass
        if invalid:
            indices = ", ".join(str(index) for index in invalid)
            self._label_status_label.setText(
                f"Labels invalid: {len(invalid)} red-filled box(es) are outside "
                f"the image bounds (zero-based indices: {indices}). Move, resize, "
                "or delete them before saving."
            )
        elif valid_message is not None:
            self._label_status_label.setText(valid_message)

    def _resolve_unsaved_changes(self) -> bool:
        if not self._annotation_dirty:
            return True
        if self._project is not None and self._project.config.task == "segment":
            return self._resolve_unsaved_segmentation()
        if (
            self._converted_image is not None
            and tuple(self._converted_image.data.shape[:2])
            != (self._training_patch_size(),) * 2
        ):
            response = QMessageBox.warning(
                self,
                "Discard full-size working boxes",
                "The current source is not the fixed training size and cannot be "
                "saved directly. Save useful failures as training crops first. "
                "Discard the remaining full-size bbox edits?",
                QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if response == QMessageBox.Discard:
                self._annotation_dirty = False
                return True
            return False
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

    def _resolve_unsaved_segmentation(self) -> bool:
        if (
            self._converted_image is not None
            and tuple(self._converted_image.data.shape[:2])
            != (self._training_patch_size(),) * 2
        ):
            response = QMessageBox.warning(
                self,
                "Discard full-size working mask",
                "This image is not the fixed training size. Segmentation crop saving "
                "arrives in Phase 7D. Discard the current mask edits?",
                QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if response == QMessageBox.Discard:
                self._annotation_dirty = False
                return True
            return False
        response = QMessageBox.warning(
            self,
            "Unsaved instance mask",
            "Save the current instance-mask edits before switching?",
            QMessageBox.Save | QMessageBox.Discard,
            QMessageBox.Save,
        )
        if response == QMessageBox.Discard:
            self._annotation_dirty = False
            return True
        layer = self._segmentation_layer()
        if layer is None or self._converted_image is None:
            self._show_error("The current segmentation cannot be reconstructed safely.")
            return False
        try:
            self._save_converted_segmentation(
                self._converted_image, labels_layer=layer, show_message=False
            )
        except (SegmentationError, ImageConversionError, ProjectError, OSError) as exc:
            self._show_error(f"Could not save the previous segmentation:\n{exc}")
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
        if self._crop_bounds is not None:
            if not self._resolve_crop_unsaved_changes():
                event.ignore()
                return
            self._close_crop_session(
                remove_selection=True, restore_sources=True
            )
        if not self._annotation_dirty:
            event.accept()
            return
        if self._project is not None and self._project.config.task == "segment":
            if self._resolve_unsaved_segmentation():
                event.accept()
            else:
                event.ignore()
            return
        if (
            self._converted_image is not None
            and tuple(self._converted_image.data.shape[:2])
            != (self._training_patch_size(),) * 2
        ):
            if self._resolve_unsaved_changes():
                event.accept()
            else:
                event.ignore()
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
        normalized: list[int] = []
        for index, value in enumerate(values):
            try:
                class_id = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise LabelValidationError(
                    [f"Box {index}: class_id must be an integer."]
                ) from exc
            if isinstance(value, (bool, np.bool_)) or value != class_id:
                raise LabelValidationError(
                    [f"Box {index}: class_id must be an integer."]
                )
            if (
                self._project is not None
                and class_id not in self._project.config.classes
            ):
                raise LabelValidationError(
                    [
                        f"Box {index}: class ID {class_id} is not defined by "
                        "the project."
                    ]
                )
            normalized.append(class_id)
        return tuple(normalized)

    def _image_for_annotation(self):
        active = self._active_layer()
        if self._is_source_image_layer(active) or self._is_annotation_review_image(
            active
        ):
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
                "to use different channels, filtering, or normalization."
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
                self._set_combo_data(self._filter_combo, settings.filter_method)
                self._filter_radius_spin.setValue(settings.filter_radius)
                self._set_combo_data(self._normalization_combo, settings.normalization)
                if settings.lower is not None:
                    self._lower_value_spin.setValue(settings.lower)
                if settings.upper is not None:
                    self._upper_value_spin.setValue(settings.upper)
                self._last_normalization_method = settings.normalization
        finally:
            self._updating_processing_controls = False

        self._on_normalization_changed()
        self._on_filter_changed()
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

    def _on_filter_changed(self, index=None) -> None:
        del index
        enabled = (
            self._filter_combo.currentData() != "none"
            and self._locked_processing_settings is None
        )
        self._filter_radius_spin.setEnabled(enabled)
        self._on_processing_value_changed()

    def _on_processing_value_changed(self, value=None) -> None:
        if self._updating_processing_controls:
            return
        self._converted_image = None
        image_layer = self._image_for_annotation()
        if (
            self._annotation_io is not None or self._segmentation_io is not None
        ) and self._is_source_image_layer(image_layer):
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
            filter_method=str(self._filter_combo.currentData()),
            filter_radius=self._filter_radius_spin.value(),
            lower=self._lower_value_spin.value() if uses_parameters else None,
            upper=self._upper_value_spin.value() if uses_parameters else None,
        )
        settings.validate()
        return settings

    def _effective_processing_settings(self) -> ImageProcessingSettings:
        if (
            self._project is not None
            and self._project.config.image_processing.get("locked")
        ):
            return ImageProcessingSettings.from_mapping(
                self._project.config.image_processing
            )
        return self._settings_from_ui()

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
        self._filter_combo.setEnabled(enabled)
        self._filter_radius_spin.setEnabled(
            enabled and self._filter_combo.currentData() != "none"
        )
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
        settings = self._effective_processing_settings()
        if self._is_annotation_review_image(image_layer):
            data = np.asarray(image_layer.data)
            if data.dtype != np.uint8 or data.ndim != 3 or data.shape[2] != 3:
                raise ImageConversionError(
                    "Canonical review images must already be RGB uint8 data."
                )
            metadata = getattr(image_layer, "metadata", {}) or {}
            sample_id = str(metadata.get("cci_sample_id") or image_layer.name)
            conversion = metadata.get("cci_conversion", {})
            raw_stats = (
                conversion.get("normalization_stats", ())
                if isinstance(conversion, dict)
                else ()
            )
            statistics = tuple(
                dict(value) for value in raw_stats if isinstance(value, dict)
            )
            if len(statistics) != 3:
                statistics = tuple(
                    {
                        "method": "already_normalized",
                        "lower": None,
                        "upper": None,
                    }
                    for _ in range(3)
                )
            converted = ConvertedImage(
                data=np.ascontiguousarray(data),
                sample_id=sample_id,
                plane=PlaneSelection(
                    axis_indices=(None, None, None),
                    plane_axes=(0, 1),
                    axis_labels=("Y", "X", "C"),
                ),
                settings=settings,
                normalization_stats=statistics,
            )
            self._converted_image = converted
            self._current_sample_id = sample_id
            self._plane_status_label.setText(
                f"Review sample: {sample_id} · canonical normalized RGB uint8"
            )
            return converted

        settings.validate(tuple(int(size) for size in image_layer.data.shape))
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

    def _training_patch_size(self) -> int:
        value = self._patch_size_combo.currentData()
        if value is not None:
            return int(value)
        if self._project is not None:
            return int(self._project.config.training_patch["size"])
        return 1024

    def _apply_project_patch_settings(self) -> None:
        if self._project is None:
            return
        contract = self._project.config.training_patch
        size = int(contract["size"])
        self._updating_patch_controls = True
        try:
            index = self._patch_size_combo.findData(size)
            if index >= 0:
                self._patch_size_combo.setCurrentIndex(index)
            self._training_tile_size_spin.setValue(size)
        finally:
            self._updating_patch_controls = False
        if contract.get("locked"):
            self._patch_contract_label.setText(
                f"Project contract: {size}×{size}, padding value "
                f"{contract['padding_value']} · locked"
            )
        else:
            self._patch_contract_label.setText(
                f"Proposed contract: {size}×{size}, padding value "
                f"{contract['padding_value']} · locks after first save"
            )

    def _on_patch_size_changed(self, index: int = -1) -> None:
        del index
        if self._updating_patch_controls:
            return
        size = self._training_patch_size()
        self._training_tile_size_spin.setValue(size)
        padding_value = (
            self._project.config.training_patch["padding_value"]
            if self._project is not None
            else 114
        )
        self._patch_contract_label.setText(
            f"Proposed contract: {size}×{size}, padding value "
            f"{padding_value} · locks after first save"
        )
        selection = self._get_layer_by_name(self.CROP_SELECTION_LAYER_NAME)
        if selection is not None and self._crop_bounds is None:
            self.napari_viewer.layers.remove(selection)
        self._crop_status_label.setText(
            f"Crop: patch size changed to {size}×{size}; place a new selection."
        )

    def _on_select_training_crop(self) -> None:
        if self._project is None:
            self._show_error("Create or open a project first.")
            return
        if self._crop_bounds is not None:
            if not self._resolve_crop_unsaved_changes():
                return
            self._close_crop_session(
                remove_selection=False, restore_sources=True
            )
        image_layer = self._image_for_annotation()
        if not self._is_source_image_layer(image_layer):
            self._show_error("Select a source image layer first.")
            return
        try:
            converted = self._convert_current_image(image_layer)
        except ImageConversionError as exc:
            self._show_error(f"Could not prepare training crop:\n{exc}")
            return
        height, width = converted.data.shape[:2]
        center_y, center_x = self._current_view_center(
            image_layer, height, width
        )
        bounds = crop_bounds_from_center(
            center_y,
            center_x,
            self._training_patch_size(),
            height,
            width,
        )
        selection = self._get_layer_by_name(self.CROP_SELECTION_LAYER_NAME)
        if selection is None:
            selection = self.napari_viewer.add_shapes(
                [bounds.as_rectangle()],
                name=self.CROP_SELECTION_LAYER_NAME,
                shape_type="rectangle",
                edge_width=3,
                edge_color="cyan",
                face_color="transparent",
            )
        else:
            selection.data = [bounds.as_rectangle()]
        selection.metadata["source_sample_id"] = converted.sample_id
        selection.metadata["source_image_layer_name"] = image_layer.name
        selection.metadata["patch_size"] = bounds.size
        try:
            selection.mode = "select"
        except AttributeError:
            pass
        self._crop_source_image_layer = image_layer
        self._crop_source_converted = converted
        self._crop_status_label.setText(
            f"Crop selection: {bounds.size}×{bounds.size} at "
            f"y={bounds.y0}, x={bounds.x0}. Move it, then click "
            "Create / Refresh Crop."
        )
        self._update_action_state()

    def _on_create_training_crop(self) -> None:
        if self._project is None:
            self._show_error("Create or open a project first.")
            return
        selection = self._get_layer_by_name(self.CROP_SELECTION_LAYER_NAME)
        if selection is None or len(getattr(selection, "data", ())) != 1:
            self._show_error("Create one training crop selection first.")
            return
        if self._crop_bounds is not None:
            if not self._resolve_crop_unsaved_changes():
                return
            self._close_crop_session(
                remove_selection=False, restore_sources=True
            )
        image_layer = self._crop_source_image_layer
        if not self._is_source_image_layer(image_layer):
            self._show_error("The crop source image is no longer available.")
            return
        try:
            converted = self._convert_current_image(image_layer)
            expected_sample = selection.metadata.get("source_sample_id")
            if expected_sample and converted.sample_id != expected_sample:
                raise TrainingCropError(
                    "The displayed Z/T plane changed after crop selection. "
                    "Select the crop again on the current plane."
                )
            height, width = converted.data.shape[:2]
            bounds = crop_bounds_from_rectangle(
                selection.data[0],
                self._training_patch_size(),
                height,
                width,
            )
            selection.data = [bounds.as_rectangle()]
            patch_config = self._project.config.training_patch
            crop_image = extract_padded_crop(
                converted.data,
                bounds,
                padding_value=int(patch_config["padding_value"]),
            )
            source_shapes = self._annotation_layer()
            if source_shapes is None:
                rectangles: tuple[np.ndarray, ...] = ()
                properties = self._empty_properties()
            else:
                rectangles = tuple(
                    np.asarray(value, dtype=float) for value in source_shapes.data
                )
                class_ids = self._class_ids(source_shapes, len(rectangles))
                properties = dict(
                    getattr(source_shapes, "properties", {}) or {}
                )
                properties["class_id"] = np.asarray(class_ids, dtype=int)
                properties["class_name"] = np.asarray(
                    [
                        self._project.config.classes[class_id]
                        for class_id in class_ids
                    ],
                    dtype=object,
                )
            cropped = crop_rectangles(rectangles, properties, bounds)
        except (ImageConversionError, LabelValidationError, TrainingCropError) as exc:
            self._show_error(f"Could not create training crop:\n{exc}")
            return

        crop_properties = self._complete_crop_properties(cropped.properties)
        crop_image_layer = self.napari_viewer.add_image(
            crop_image,
            name=self.CROP_IMAGE_LAYER_NAME,
            rgb=crop_image.ndim == 3,
            metadata={
                "cci_training_crop": True,
                "source_sample_id": converted.sample_id,
                "crop_bounds": bounds.to_mapping(),
            },
        )
        crop_shapes = self.napari_viewer.add_shapes(
            list(cropped.rectangles),
            name=self.CROP_BBOX_LAYER_NAME,
            shape_type="rectangle",
            properties=crop_properties,
            edge_width=2,
            edge_color=[
                class_color(value) for value in crop_properties["class_id"]
            ]
            if cropped.rectangles
            else class_color(self._current_class_id() or 0),
            face_color="transparent",
        )
        crop_shapes.metadata["source_sample_id"] = converted.sample_id
        crop_shapes.metadata["crop_bounds"] = bounds.to_mapping()
        self._set_default_current_properties(crop_shapes)
        self._apply_class_colors(crop_shapes)
        try:
            crop_shapes.events.data.connect(self._on_crop_shapes_changed)
        except (AttributeError, TypeError):
            pass
        self._crop_source_shapes_layer = self._annotation_layer()
        self._crop_source_converted = converted
        self._crop_bounds = bounds
        self._crop_discarded_count = len(cropped.discarded_indices)
        self._crop_dirty = True
        self._crop_source_visibility = []
        for layer in (image_layer, self._crop_source_shapes_layer, selection):
            if layer is None:
                continue
            visible = bool(getattr(layer, "visible", True))
            self._crop_source_visibility.append((layer, visible))
            try:
                layer.visible = False
            except AttributeError:
                pass
        try:
            self.napari_viewer.layers.selection.active = crop_shapes
        except AttributeError:
            pass
        padding = (
            f" · padding bottom {bounds.pad_bottom}, right {bounds.pad_right}"
            if bounds.pad_bottom or bounds.pad_right
            else ""
        )
        discarded = (
            f" · {self._crop_discarded_count} sub-2-pixel remnant(s) ignored"
            if self._crop_discarded_count
            else ""
        )
        ready_message = (
            f"Crop ready: {len(cropped.rectangles)} box(es), "
            f"{cropped.clipped_count} clipped at crop boundaries"
            f"{padding}{discarded}."
        )
        self._refresh_crop_validation(valid_message=ready_message)
        self._update_class_counts(crop_shapes)
        self._update_action_state()

    def _complete_crop_properties(
        self, properties: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        count = len(properties.get("class_id", ()))
        class_ids = self._property_values(
            properties,
            "class_id",
            count,
            self._current_class_id() or 0,
            int,
        )
        return {
            "class_id": class_ids,
            "class_name": np.asarray(
                [self._project.config.classes[value] for value in class_ids],
                dtype=object,
            ),
            "confidence": self._property_values(
                properties, "confidence", count, np.nan, float
            ),
            "source": self._property_values(
                properties, "source", count, "manual", object
            ),
            "tile_id": self._property_values(
                properties, "tile_id", count, -1, int
            ),
        }

    def _on_crop_shapes_changed(self, event=None) -> None:
        del event
        self._crop_dirty = True
        self._update_class_counts(self._crop_bbox_layer())
        self._refresh_crop_validation(
            valid_message=(
                "Crop: unsaved bbox edits. Add the crop or return and "
                "discard it."
            )
        )
        self._update_action_state()

    def _refresh_crop_validation(
        self, *, valid_message: str | None = None
    ) -> None:
        crop_shapes = self._crop_bbox_layer()
        if self._crop_bounds is None or crop_shapes is None:
            self._crop_invalid_indices = ()
            return
        rectangles = tuple(
            np.asarray(value, dtype=float) for value in crop_shapes.data
        )
        invalid = invalid_crop_box_indices(rectangles, self._crop_bounds)
        self._crop_invalid_indices = invalid

        face_colors = np.zeros((len(rectangles), 4), dtype=float)
        if invalid:
            face_colors[list(invalid)] = (1.0, 0.0, 0.0, 0.35)
        try:
            crop_shapes.face_color = (
                face_colors if len(rectangles) else "transparent"
            )
        except (AttributeError, TypeError, ValueError):
            pass

        if invalid:
            indices = ", ".join(str(index) for index in invalid)
            self._crop_status_label.setText(
                f"Crop invalid: {len(invalid)} red-filled box(es) are invalid "
                "or extend outside valid source pixels / into padding (zero-based "
                f"indices: {indices}). Move, resize, or delete them before "
                "saving."
            )
        elif valid_message is not None:
            self._crop_status_label.setText(valid_message)

    def _on_save_training_crop(self) -> None:
        self._save_training_crop(show_message=True)

    def _save_training_crop(self, *, show_message: bool) -> bool:
        if (
            self._project is None
            or self._annotation_io is None
            or self._crop_bounds is None
            or self._crop_source_converted is None
        ):
            self._show_error("Create a training crop first.")
            return False
        crop_image_layer = self._get_layer_by_name(self.CROP_IMAGE_LAYER_NAME)
        crop_shapes = self._crop_bbox_layer()
        if crop_image_layer is None or crop_shapes is None:
            self._show_error("The crop image and bbox layers are both required.")
            return False
        rectangles = tuple(
            np.asarray(value, dtype=float) for value in crop_shapes.data
        )
        try:
            validate_boxes_within_valid_crop(rectangles, self._crop_bounds)
            class_ids = self._class_ids(crop_shapes, len(rectangles))
            sample_id = crop_sample_id(
                self._crop_source_converted.sample_id, self._crop_bounds
            )
            result = self._annotation_io.save_pair(
                image_data=np.asarray(crop_image_layer.data),
                image_name=sample_id,
                source_path=self._source_path(self._crop_source_image_layer),
                sample_id=sample_id,
                rectangles=rectangles,
                class_ids=class_ids,
                conversion_metadata={
                    "settings": self._crop_source_converted.settings.to_mapping(),
                    "plane_indices": (
                        self._crop_source_converted.plane.non_spatial_indices
                    ),
                    "axis_labels": list(
                        self._crop_source_converted.plane.axis_labels
                    ),
                    "normalization_stats": list(
                        self._crop_source_converted.normalization_stats
                    ),
                    "training_crop": self._crop_bounds.to_mapping(),
                    "prediction": getattr(
                        self._crop_source_shapes_layer, "metadata", {}
                    ).get("cci_prediction"),
                },
            )
            self._project.lock_image_processing(
                self._crop_source_converted.settings.to_mapping()
            )
            self._project.lock_training_patch(
                self._crop_bounds.size,
                padding_value=int(
                    self._project.config.training_patch["padding_value"]
                ),
            )
        except (
            AnnotationError,
            LabelValidationError,
            ProjectError,
            TrainingCropError,
            OSError,
        ) as exc:
            self._show_error(f"Could not save training crop:\n{exc}")
            return False
        self._crop_dirty = False
        self._training_preview = None
        crop_shapes.metadata["cci_label_path"] = str(result.label_path)
        self._crop_status_label.setText(
            f"Crop {result.operation}: {result.box_count} box(es) · "
            f"{result.image_path.name}"
        )
        self._dataset_status_label.setText(
            "Dataset: annotation pool changed; validate or preview the split again."
        )
        self._apply_project_processing_settings()
        self._apply_project_patch_settings()
        self._update_project_status()
        self._update_action_state()
        if show_message:
            self._show_info(
                f"Training crop {result.operation}:\n{result.image_path.name}\n"
                f"{result.label_path.name}"
            )
        return True

    def _on_return_to_source(self) -> None:
        if not self._resolve_crop_unsaved_changes():
            return
        self._close_crop_session(remove_selection=False, restore_sources=True)
        self._crop_status_label.setText(
            "Crop: returned to source. Move the selection or choose another image."
        )
        self._update_action_state()

    def _resolve_crop_unsaved_changes(self) -> bool:
        if not self._crop_dirty:
            return True
        response = QMessageBox.warning(
            self,
            "Unsaved training crop",
            "The current training crop has not been saved. Save it before "
            "returning to the source image?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if response == QMessageBox.Cancel:
            return False
        if response == QMessageBox.Discard:
            self._crop_dirty = False
            return True
        return self._save_training_crop(show_message=False)

    def _close_crop_session(
        self, *, remove_selection: bool, restore_sources: bool
    ) -> None:
        for name in (self.CROP_IMAGE_LAYER_NAME, self.CROP_BBOX_LAYER_NAME):
            layer = self._get_layer_by_name(name)
            if layer is not None:
                self.napari_viewer.layers.remove(layer)
        if restore_sources:
            for layer, visible in self._crop_source_visibility:
                try:
                    if layer in self.napari_viewer.layers:
                        layer.visible = visible
                except (AttributeError, TypeError):
                    pass
        if remove_selection:
            selection = self._get_layer_by_name(self.CROP_SELECTION_LAYER_NAME)
            if selection is not None:
                self.napari_viewer.layers.remove(selection)
            self._crop_source_image_layer = None
            self._crop_source_converted = None
        self._crop_bounds = None
        self._crop_source_shapes_layer = None
        self._crop_source_visibility = []
        self._crop_discarded_count = 0
        self._crop_invalid_indices = ()
        self._crop_dirty = False
        self._update_class_counts()

    def _refresh_annotation_browser(self, preferred=None) -> None:
        if isinstance(preferred, bool):
            preferred = None
        previous = preferred or self._review_sample_combo.currentData()
        self._updating_review_combo = True
        try:
            self._review_sample_combo.clear()
            if self._project is None:
                self._review_entries = ()
                self._review_status_label.setText(
                    "Review: open a project to browse saved annotations."
                )
                return
            is_segment = self._project.config.task == "segment"
            self._review_entries = (
                SegmentationIO(self._project).entries()
                if is_segment
                else AnnotationBrowser(self._project).entries()
            )
            invalid_count = 0
            for entry in self._review_entries:
                if entry.errors:
                    invalid_count += 1
                prefix = "⚠ " if entry.errors else ""
                self._review_sample_combo.addItem(
                    f"{prefix}{entry.sample_id} · {entry.box_count} "
                    f"{'instance(s)' if is_segment else 'box(es)'}",
                    entry.sample_id,
                )
            if previous:
                index = self._review_sample_combo.findData(str(previous))
                if index >= 0:
                    self._review_sample_combo.setCurrentIndex(index)
            self._review_status_label.setText(
                f"Review: {len(self._review_entries)} saved pair(s) · "
                f"{invalid_count} invalid pair(s)."
            )
        except OSError as exc:
            self._review_entries = ()
            self._review_status_label.setText(f"Review: could not scan project: {exc}")
        finally:
            self._updating_review_combo = False
            self._update_action_state()

    def _selected_review_entry(
        self,
    ) -> AnnotationReviewEntry | SegmentationReviewEntry | None:
        sample_id = self._review_sample_combo.currentData()
        return next(
            (
                entry
                for entry in self._review_entries
                if entry.sample_id == sample_id
            ),
            None,
        )

    def _navigate_annotation(self, offset: int) -> None:
        count = self._review_sample_combo.count()
        if count == 0:
            return
        current = max(0, self._review_sample_combo.currentIndex())
        self._review_sample_combo.setCurrentIndex((current + offset) % count)
        self._on_load_review_annotation()

    def _on_load_review_annotation(self) -> None:
        entry = self._selected_review_entry()
        if entry is None:
            self._show_error("No saved annotation is selected.")
            return
        if entry.errors:
            details = "\n".join(f"• {error}" for error in entry.errors)
            self._review_status_label.setText(
                f"Review: {entry.sample_id} is invalid. {entry.errors[0]}"
            )
            self._show_error(
                f"Cannot load invalid annotation {entry.sample_id!r}.\n\n{details}"
            )
            return
        if not self._resolve_unsaved_changes():
            return
        if self._crop_bounds is not None:
            if not self._resolve_crop_unsaved_changes():
                return
            self._close_crop_session(
                remove_selection=True, restore_sources=True
            )
        try:
            data = AnnotationBrowser.load_rgb(entry)
        except OSError as exc:
            self._show_error(f"Could not load annotation image:\n{exc}")
            return

        previous = self._get_layer_by_name(self.REVIEW_IMAGE_LAYER_NAME)
        if previous is not None:
            self.napari_viewer.layers.remove(previous)
        metadata = {
            "cci_annotation_review": True,
            "cci_sample_id": entry.sample_id,
            "cci_canonical_path": str(entry.image_path),
            "cci_original_source_path": (
                str(entry.source_path) if entry.source_path is not None else None
            ),
            "cci_conversion": entry.conversion,
        }
        image_layer = self.napari_viewer.add_image(
            data,
            name=self.REVIEW_IMAGE_LAYER_NAME,
            rgb=True,
            metadata=metadata,
        )
        try:
            self.napari_viewer.layers.selection.active = image_layer
        except AttributeError:
            pass
        self._load_annotations_for_image(image_layer, force=True)
        self._review_status_label.setText(
            f"Review: loaded {entry.sample_id} · {entry.box_count} "
            f"{'instance(s)' if self._project and self._project.config.task == 'segment' else 'box(es)'}. "
            "Edit the annotation layer, save, then use Previous or Next."
        )
        try:
            self.napari_viewer.reset_view()
        except AttributeError:
            pass

    def _current_view_center(
        self, image_layer, height: int, width: int
    ) -> tuple[float, float]:
        try:
            center = tuple(self.napari_viewer.camera.center)
            transform = getattr(image_layer, "world_to_data", None)
            if callable(transform):
                center = tuple(transform(center))
            return float(center[-2]), float(center[-1])
        except (AttributeError, TypeError, ValueError, IndexError):
            return height / 2.0, width / 2.0

    def _on_choose_model(self) -> None:
        if self._project is None:
            self._show_error("Create or open a project first.")
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            f"Select a YOLO {self._project.config.task} model",
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
            model = (
                YoloSegmentationModel(path)
                if self._project.config.task == "segment"
                else YoloDetectionModel(path)
            )
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
        self._update_loaded_model_status()
        self._refresh_training_model_choices(preferred=model.path)
        self._update_action_state()
        return True

    def _update_loaded_model_status(self) -> None:
        if self._model is None or self._project is None:
            return
        names = ", ".join(
            f"{class_id}: {name}"
            for class_id, name in sorted(self._model.names.items())
        )
        project_names = self._project.config.classes
        names_differ = bool(self._model.names) and any(
            self._model.names.get(class_id) != name
            for class_id, name in project_names.items()
        )
        suffix = " · names mapped to project names" if names_differ else ""
        self._model_status_label.setText(
            f"Model: loaded · task {self._model.task} · "
            f"classes [{names or 'unknown'}]{suffix}"
        )

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
            self._show_error("Open a project and choose a matching YOLO model first.")
            return
        if self._inference_worker is not None and self._inference_worker.isRunning():
            return
        image_layer = self._image_for_annotation()
        if not self._is_source_image_layer(image_layer):
            self._show_error("Select a source image layer first.")
            return
        if not self._resolve_unsaved_changes():
            return
        existing = (
            self._segmentation_layer()
            if self._project.config.task == "segment"
            else self._annotation_layer()
        )
        if existing is not None and len(getattr(existing, "data", ())) > 0:
            response = QMessageBox.warning(
                self,
                "Replace annotations",
                "Prediction will replace the current annotation layer. Continue?",
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

        if self._project.config.task == "segment":
            worker = SegmentationWorker(
                self._model, converted.data, settings, self._project.config.classes
            )
        else:
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
        self._inference_status_label.setText(
            "Inference: predicting one image"
            if self._project.config.task == "segment"
            else "Inference: preparing tiles"
        )
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
        if isinstance(detections, ComposedInstances):
            self._on_segmentation_inference_succeeded(detections)
            return
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
            edge_color=[class_color(value) for value in properties["class_id"]]
            if rectangles
            else class_color(self._current_class_id() or 0),
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
        self._apply_class_colors(shapes)
        try:
            shapes.events.data.connect(self._on_shapes_data_changed)
        except (AttributeError, TypeError):
            pass
        self._converted_image = self._inference_converted
        self._current_sample_id = self._inference_sample_id
        direct_save = bool(
            self._converted_image is not None
            and tuple(self._converted_image.data.shape[:2])
            == (self._training_patch_size(),) * 2
        )
        self._annotation_dirty = direct_save
        self._update_class_counts(shapes)
        if direct_save:
            self._label_status_label.setText(
                f"Labels: {len(typed_result)} merged prediction(s), not yet saved."
            )
            self._save_annotation_button.setText("Save Prediction + Corrections")
        else:
            self._label_status_label.setText(
                f"Labels: {len(typed_result)} working prediction(s). Move a "
                "training crop over failures to save corrections."
            )
            self._save_annotation_button.setText(
                "Use Training Crop to Save Corrections"
            )
        total = self._inference_progress.maximum()
        self._inference_progress.setValue(total)
        self._inference_status_label.setText(
            f"Inference: complete · {len(typed_result)} merged detection(s)"
        )

    def _on_segmentation_inference_succeeded(
        self, result: ComposedInstances
    ) -> None:
        image_layer = self._annotation_image_layer
        if image_layer is None or self._inference_sample_id is None:
            self._inference_status_label.setText(
                "Inference: result discarded because the source image was closed"
            )
            return
        if self._image_stem(image_layer) != self._inference_sample_id:
            self._inference_status_label.setText(
                "Inference: result discarded because the active Z/T plane changed"
            )
            return
        existing = self._segmentation_layer()
        if existing is not None:
            self.napari_viewer.layers.remove(existing)
        labels = self.napari_viewer.add_labels(
            result.mask,
            name=self.SEGMENTATION_LAYER_NAME,
            metadata={
                "cci_image_stem": self._inference_sample_id,
                "cci_project_root": str(self._project.paths.root),
                "cci_mask_path": None,
                "cci_prediction": {
                    "model_path": str(self._model.path) if self._model else None,
                    "retina_masks": True,
                    "largest_component_policy": "bbox_area",
                },
            },
        )
        self._segment_instances = result.instances
        self._apply_instance_colors(labels)
        try:
            labels.events.data.connect(self._on_labels_data_changed)
            labels.events.selected_label.connect(self._on_selected_instance_changed)
        except (AttributeError, TypeError):
            pass
        self._converted_image = self._inference_converted
        self._current_sample_id = self._inference_sample_id
        direct_save = bool(
            self._converted_image is not None
            and tuple(self._converted_image.data.shape[:2])
            == (self._training_patch_size(),) * 2
        )
        self._annotation_dirty = direct_save
        removed_components = sum(item.removed_components for item in result.cleanup)
        removed_pixels = sum(item.removed_pixels for item in result.cleanup)
        self._label_status_label.setText(
            f"Instances: {len(result.instances)} prediction(s), not yet saved. "
            f"Cleanup removed {removed_components} component(s) / {removed_pixels} px."
        )
        self._save_annotation_button.setText(
            "Save Prediction + Corrections"
            if direct_save
            else "Segmentation Crops Arrive in Phase 7D"
        )
        self._inference_progress.setRange(0, 1)
        self._inference_progress.setValue(1)
        self._inference_status_label.setText(
            f"Inference: complete · {len(result.instances)} instance(s)"
        )
        self._refresh_segmentation_validation()
        self._update_class_counts()
        self._on_selected_instance_changed()

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
        added: set[Path] = set()
        if self._base_model_path.is_file():
            self._training_model_combo.addItem(
                f"Base · {self._base_model_path.name}", str(self._base_model_path)
            )
            added.add(self._base_model_path.resolve())
        if self._project is not None:
            for path in sorted(
                self._project.paths.models.glob("*.pt"), reverse=True
            ):
                resolved = path.resolve()
                if resolved in added:
                    continue
                self._training_model_combo.addItem(
                    f"Project · {path.name}", str(path)
                )
                added.add(resolved)
        if (
            self._model is not None
            and self._model.path.resolve() not in added
        ):
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
        tile_size = self._training_patch_size()
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
            + (
                f" · model copied to {run.promoted_model}"
                if run.promoted_model is not None
                else ""
            )
        )
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Retraining completed")
        dialog.setText(f"Training completed successfully.\n\n{run.run_root}")
        keep_button = dialog.addButton(
            "Keep Current Model", QMessageBox.AcceptRole
        )
        load_button = None
        load_model = run.promoted_model or run.best_model or run.last_model
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
        self._refresh_training_model_choices(preferred=run.promoted_model)

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
        is_segment = bool(has_project and self._project.config.task == "segment")
        image_layer = self._image_for_annotation()
        has_image = self._is_image_layer(image_layer)
        has_shapes = self._annotation_layer() is not None
        has_mask = self._segmentation_layer() is not None
        has_editable_shapes = self._editable_bbox_layer() is not None
        has_crop = self._crop_bounds is not None
        has_crop_selection = (
            self._get_layer_by_name(self.CROP_SELECTION_LAYER_NAME) is not None
        )
        direct_save_size = bool(
            self._converted_image is not None
            and tuple(self._converted_image.data.shape[:2])
            == (self._training_patch_size(),) * 2
        )
        inference_running = (
            self._inference_worker is not None
            and self._inference_worker.isRunning()
        )
        training_running = (
            self._training_worker is not None
            and self._training_worker.isRunning()
        )
        running = inference_running or training_running
        self._new_project_button.setEnabled(not running and not has_crop)
        self._new_project_task_combo.setEnabled(not running and not has_crop)
        self._open_project_button.setEnabled(not running and not has_crop)
        self._edit_classes_button.setEnabled(
            has_project and not running and not has_crop
        )
        self._class_combo.setEnabled(has_project and not running)
        self._apply_class_button.setEnabled(
            has_project
            and (has_mask if is_segment else has_editable_shapes)
            and not running
        )
        self._reload_labels_button.setEnabled(
            has_project and has_image and not running and not has_crop
        )
        can_save_annotation = (
            has_project
            and has_image
            and (has_mask if is_segment else has_shapes)
            and direct_save_size
            and not (
                self._segmentation_errors if is_segment else self._annotation_invalid_indices
            )
            and not running
            and not has_crop
        )
        self._save_annotation_button.setEnabled(can_save_annotation)
        self._validate_project_button.setEnabled(
            has_project and not is_segment and not running
        )
        self._preview_button.setEnabled(
            has_project and has_image and not running and not has_crop
        )
        self._choose_model_button.setEnabled(has_project and not running)
        self._predict_button.setEnabled(
            has_project
            and has_image
            and self._model is not None
            and not running
            and not has_crop
        )
        patch_locked = bool(
            self._project
            and self._project.config.training_patch.get("locked")
        )
        self._patch_size_combo.setEnabled(
            has_project and not patch_locked and not running and not has_crop
        )
        self._select_crop_button.setEnabled(
            has_project and not is_segment and has_image and not running
        )
        self._create_crop_button.setEnabled(
            has_project and not is_segment and has_crop_selection and not running
        )
        self._save_crop_button.setEnabled(
            has_crop and not self._crop_invalid_indices and not running
        )
        self._return_crop_button.setEnabled(has_crop and not running)
        has_review_entries = bool(self._review_entries)
        for control in (
            self._review_sample_combo,
            self._review_previous_button,
            self._review_load_button,
            self._review_next_button,
        ):
            control.setEnabled(
                has_project and has_review_entries and not running and not has_crop
            )
        self._review_refresh_button.setEnabled(
            has_project and not running and not has_crop
        )
        self._review_save_button.setEnabled(
            can_save_annotation
            and self._is_annotation_review_image(image_layer)
        )
        self._cancel_inference_button.setEnabled(inference_running)
        selected_instance = self._selected_instance_id()
        for control in (
            self._new_instance_button,
            self._delete_instance_button,
            self._split_instance_button,
            self._merge_instance_button,
            self._largest_instance_button,
        ):
            control.setVisible(is_segment)
        self._instance_details_label.setVisible(is_segment)
        self._new_instance_button.setEnabled(is_segment and has_mask and not running)
        for control in (
            self._delete_instance_button,
            self._split_instance_button,
            self._merge_instance_button,
            self._largest_instance_button,
        ):
            control.setEnabled(
                is_segment and has_mask and selected_instance > 0 and not running
            )
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
        self._validate_dataset_button.setEnabled(
            has_project and not is_segment and not running and not has_crop
        )
        self._preview_split_button.setEnabled(
            has_project and not is_segment and not running and not has_crop
        )
        self._regenerate_split_button.setEnabled(
            has_project and not is_segment and not running and not has_crop
        )
        self._retrain_button.setEnabled(
            has_project and not is_segment and has_training_model and not running and not has_crop
        )
        self._cancel_training_button.setEnabled(training_running)
        self._destination_button.setEnabled(
            has_project and not is_segment and not running and not has_crop
        )
        for control in (
            self._training_model_combo,
            self._destination_input,
            self._group_field_input,
            self._validation_fraction_spin,
            self._seed_spin,
            self._training_overlap_spin,
            self._negative_ratio_spin,
            self._epochs_spin,
            self._batch_spin,
            self._patience_spin,
            self._training_device_combo,
            self._train_only_checkbox,
        ):
            control.setEnabled(
                has_project and not is_segment and not running and not has_crop
            )
        self._training_tile_size_spin.setEnabled(False)
        self._set_processing_controls_enabled(
            has_project
            and has_image
            and self._locked_processing_settings is None
            and not running
            and not has_crop
        )
