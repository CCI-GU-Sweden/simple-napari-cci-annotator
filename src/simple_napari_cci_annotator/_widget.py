from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._annotation_io import AnnotationError, AnnotationIO, LabelValidationError
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
        self._annotation_image_layer = None

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
        self._label_status_label = QLabel(
            "Labels: unavailable until a project is open."
        )
        self._label_status_label.setWordWrap(True)

        self._reload_labels_button = QPushButton("Reload Labels")
        self._reload_labels_button.clicked.connect(self._on_reload_labels)
        self._save_annotation_button = QPushButton("Add Annotation")
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
        annotation_buttons = QHBoxLayout()
        annotation_buttons.addWidget(self._reload_labels_button)
        annotation_buttons.addWidget(self._save_annotation_button)
        annotation_layout = QVBoxLayout()
        annotation_layout.addWidget(self._image_status_label)
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
        self._update_action_state()
        active = self._active_layer()
        if self._is_image_layer(active):
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
        if self._is_image_layer(active):
            self._load_annotations_for_image(active)
        self._update_action_state()

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

    def _image_stem(self, image_layer) -> str:
        assert self._annotation_io is not None
        source_path = self._source_path(image_layer)
        value = source_path.stem if source_path is not None else getattr(
            image_layer, "name", "image"
        )
        return self._annotation_io.safe_stem(str(value))

    def _load_annotations_for_image(self, image_layer, *, force: bool = False) -> None:
        if self._annotation_io is None or not self._is_image_layer(image_layer):
            return
        image_data = np.asarray(image_layer.data)
        if image_data.ndim < 2:
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

        source_path = self._source_path(image_layer)
        label_path = self._annotation_io.find_label(stem, source_path=source_path)
        try:
            if label_path is None:
                rectangles: tuple[np.ndarray, ...] = ()
                properties = self._empty_properties()
            else:
                loaded = self._annotation_io.load_label(label_path, image_data.shape)
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
        self._annotation_image_layer = image_layer
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
                "Update Annotation" if canonical else "Import Annotation"
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

        rectangles = tuple(np.asarray(shape, dtype=float) for shape in shapes_layer.data)
        try:
            class_ids = self._class_ids(shapes_layer, len(rectangles))
            result = self._annotation_io.save_pair(
                image_data=np.asarray(image_layer.data),
                image_name=getattr(image_layer, "name", "image"),
                source_path=self._source_path(image_layer),
                rectangles=rectangles,
                class_ids=class_ids,
            )
        except (AnnotationError, OSError) as exc:
            self._show_error(f"Could not save annotation:\n{exc}")
            return

        shapes_layer.metadata["cci_label_path"] = str(result.label_path)
        self._label_status_label.setText(
            f"Labels: {result.operation} {result.box_count} box(es) · {result.label_path}"
        )
        self._save_annotation_button.setText("Update Annotation")
        self._show_info(
            f"Annotation {result.operation}:\n{result.image_path.name}\n"
            f"{result.label_path.name}"
        )
        self._update_action_state()

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
        if self._is_image_layer(active):
            return active
        return self._annotation_image_layer

    def _update_image_status(self, image_layer) -> None:
        data = np.asarray(image_layer.data)
        source_path = self._source_path(image_layer)
        source_text = (
            str(source_path)
            if source_path
            else getattr(image_layer, "name", "image")
        )
        self._image_status_label.setText(
            f"Image: {source_text} · shape {tuple(data.shape)} · {data.dtype}"
        )

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
