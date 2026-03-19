from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._yolo_utils import (
    CCIYoloWrapper,
    create_training_set,
    save_vectors_to_txt,
)


class SimpleCciAnnotatorQWidget(QWidget):
    """Minimal YOLO bbox annotator flow for napari.

    Workflow:
    1) Select model path (.pt) and load model
    2) Predict bboxes on the active image
    3) User edits shapes
    4) Add correction (image + edited labels)
    5) Retrain to produce a new model folder
    """

    PRED_LAYER_NAME = "yolo_bboxes"

    def __init__(self, napari_viewer):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.setWindowTitle("Simple CCI Annotator")

        self._yolo: CCIYoloWrapper | None = None
        self._model_path: Path | None = None
        self._destination_path: Path | None = None

        self._model_path_input = QLineEdit()
        self._model_path_input.setPlaceholderText("Path to YOLO model (.pt) or model folder")

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._on_browse_model)

        load_button = QPushButton("Load model")
        load_button.clicked.connect(self._on_load_model)

        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self._on_predict)

        self.destination_path_input = QLineEdit()
        self.destination_path_input.setPlaceholderText("Path to save retrained model")

        browse_destination_button = QPushButton("Browse")
        browse_destination_button.clicked.connect(self._on_browse_destination)

        add_correction_button = QPushButton("Add correction")
        add_correction_button.clicked.connect(self._on_add_correction)

        retrain_button = QPushButton("Retrain")
        retrain_button.clicked.connect(self._on_retrain)

        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model"))
        row_model.addWidget(self._model_path_input)
        row_model.addWidget(browse_button)

        row_actions = QHBoxLayout()
        row_actions.addWidget(load_button)
        row_actions.addWidget(predict_button)

        row_destination = QHBoxLayout()
        row_destination.addWidget(QLabel("Destination"))
        row_destination.addWidget(self.destination_path_input)
        row_destination.addWidget(browse_destination_button)

        row_train = QHBoxLayout()
        row_train.addWidget(add_correction_button)
        row_train.addWidget(retrain_button)
        #row_train.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(row_model)
        layout.addLayout(row_actions)
        layout.addLayout(row_destination)
        layout.addLayout(row_train)
        layout.addStretch(1)
        self.setLayout(layout)

    def _show_info(self, text: str) -> None:
        QMessageBox.information(self, "Simple CCI Annotator", text)

    def _show_error(self, text: str) -> None:
        QMessageBox.critical(self, "Simple CCI Annotator", text)

    def _on_browse_model(self) -> None:
        model_dir = QFileDialog.getExistingDirectory(
            self,
            "Select model folder (.pt will be loaded or yolov8n.pt will be copied)"
        )
        if model_dir:
            self._model_path_input.setText(model_dir)

    def _on_browse_destination(self) -> None:
        destination_dir = QFileDialog.getExistingDirectory(self, "Select destination for retrained model")
        if destination_dir:
            self.destination_path_input.setText(destination_dir)

    def _on_load_model(self) -> None:
        model_input = self._model_path_input.text().strip()
        if not model_input:
            self._show_error("Model path cannot be empty. Select a .pt file or a folder.")
            return

        model_path_input = Path(model_input)
        if not model_path_input.exists():
            self._show_error("Model path does not exist.")
            return

        model_path: Path
        copied_default_model = False

        if model_path_input.is_file():
            if model_path_input.suffix.lower() != ".pt":
                self._show_error("Select a valid .pt model file or a folder.")
                return
            model_path = model_path_input
        elif model_path_input.is_dir():
            pt_files = sorted(model_path_input.glob("*.pt"))
            if pt_files:
                model_path = pt_files[0]
            else:
                default_model_source = Path(__file__).parent / "models" / "yolov8n.pt"
                if not default_model_source.exists():
                    self._show_error("No .pt found in selected folder, and bundled yolov8n.pt is missing.")
                    return

                model_path = model_path_input / "yolov8n.pt"
                shutil.copy2(default_model_source, model_path)
                copied_default_model = True
        else:
            self._show_error("Select a valid .pt model file or a folder.")
            return

        try:
            self._yolo = CCIYoloWrapper(str(model_path))
            self._model_path = model_path
        except Exception as exc:  # pragma: no cover - GUI runtime guard
            self._show_error(f"Could not load model: {exc}")
            return

        if copied_default_model:
            self._show_info(f"No .pt model was found in the folder. Copied bundled model to: {model_path}")

        self._show_info(f"Model loaded: {model_path.name}")

    def _get_active_image_layer(self):
        layer = self.napari_viewer.layers.selection.active
        if layer is None:
            self._show_error("Select an image layer first.")
            return None

        if getattr(layer, "data", None) is None:
            self._show_error("Active layer has no image data.")
            return None
        return layer

    def _is_image_layer(self, layer) -> bool:
        return layer is not None and layer.__class__.__name__.lower().endswith("image")

    def _get_single_image_layer(self):
        image_layers = [layer for layer in self.napari_viewer.layers if self._is_image_layer(layer)]
        if len(image_layers) == 0:
            self._show_error("No image layer found.")
            return None
        if len(image_layers) > 1:
            self._show_error("Multiple image layers found. Keep only one image layer before adding a correction.")
            return None
        return image_layers[0]

    def _get_layer_by_name(self, name: str):
        for layer in self.napari_viewer.layers:
            if getattr(layer, "name", None) == name:
                return layer
        return None

    def _is_shapes_layer(self, layer) -> bool:
        return layer is not None and layer.__class__.__name__.lower().endswith("shapes")

    def _on_predict(self) -> None:
        if self._yolo is None:
            self._show_error("Load a model first.")
            return

        image_layer = self._get_active_image_layer()
        if image_layer is None:
            return

        image_data = np.asarray(image_layer.data)
        if image_data.ndim < 2:
            self._show_error("Unsupported image shape.")
            return

        try:
            prediction = self._yolo.predict(image_data)
            boxes = prediction[0].boxes.xyxy.cpu().numpy() if len(prediction) else np.empty((0, 4))
        except Exception as exc:  # pragma: no cover - GUI runtime guard
            self._show_error(f"Prediction failed: {exc}")
            return

        rects = []
        for x1, y1, x2, y2 in boxes:
            rects.append(np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]], dtype=float))

        existing = self._get_layer_by_name(self.PRED_LAYER_NAME)
        if existing is not None:
            self.napari_viewer.layers.remove(existing)

        self.napari_viewer.add_shapes(
            rects,
            name=self.PRED_LAYER_NAME,
            shape_type="rectangle",
            edge_width=2,
            edge_color="yellow",
            face_color="transparent",
        )
        self._show_info(f"Prediction done: {len(rects)} bbox(es). Edit them, then click Add correction.")

    def _find_shapes_layer(self):
        layer = self._get_layer_by_name(self.PRED_LAYER_NAME)
        if self._is_shapes_layer(layer):
            return layer

        selected = self.napari_viewer.layers.selection.active
        if self._is_shapes_layer(selected):
            return selected
        return None

    def _to_safe_stem(self, name: str) -> str:
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
        return stem or "image"

    def _get_model_root(self) -> Path | None:
        if self._model_path is None:
            return None
        return self._model_path.parent

    def _as_rgb_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] > 3:
            image = image[..., :3]

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _create_training_config(self, corrections_root: Path, image_size: int | None = None) -> None:
        """Create a training configuration JSON file in the corrections directory.
        Args:
            corrections_root: Path to the corrections directory
            image_size: Optional image size to use. If not provided, defaults to 640.
        """
        if image_size is None:
            image_size = 640

        config = {
            "image_size": image_size,
            "batch": 8,
            "epochs": 100,
            "patience": 30,
        }
        config_file = corrections_root / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def _load_training_config(self, corrections_root: Path) -> dict:
        """Load training configuration from JSON file, return defaults if not found."""
        config_file = corrections_root / "training_config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    return json.load(f)
            except Exception:
                pass

        # Return defaults if file doesn't exist or can't be read
        return {
            "image_size": 640,
            "batch": 8,
            "epochs": 100,
            "patience": 30,
        }

    def _on_add_correction(self) -> None:
        destination_text = self.destination_path_input.text().strip()
        self._destination_path = Path(destination_text) if destination_text else None
        model_root = self._get_model_root()
        if model_root is None:
            self._show_error("Load a model first.")
            return

        image_layer = self._get_single_image_layer()
        if image_layer is None:
            return

        shapes_layer = self._find_shapes_layer()
        if shapes_layer is None:
            self._show_error(f"No shapes layer found. Use '{self.PRED_LAYER_NAME}' or select a shapes layer.")
            return

        image_data = self._as_rgb_uint8(np.asarray(image_layer.data))
        h, w = image_data.shape[:2]
        if h <= 0 or w <= 0:
            self._show_error("Image has invalid size.")
            return

        corrections_root = model_root / "corrections"
        corrections_root.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{self._to_safe_stem(getattr(image_layer, 'name', 'image'))}_{stamp}"
        image_out = corrections_root / f"{stem}.png"
        label_out = corrections_root / f"{stem}.txt"

        vectors: list[tuple[int, list[tuple[float, float]]]] = []
        for shape in np.asarray(shapes_layer.data, dtype=float):
            pts: list[tuple[float, float]] = []
            for y, x in shape:
                xn = float(np.clip(x / w, 0.0, 1.0))
                yn = float(np.clip(y / h, 0.0, 1.0))
                pts.append((xn, yn))
            if pts:
                vectors.append((0, pts))

        try:
            Image.fromarray(image_data).save(image_out)
            save_vectors_to_txt(vectors, label_out)
            # Use the larger image dimension as the training image size
            img_size = max(h, w)
            self._create_training_config(corrections_root, image_size=img_size)
        except Exception as exc:  # pragma: no cover - GUI runtime guard
            self._show_error(f"Could not save correction: {exc}")
            return

        self._show_info(f"Saved correction to: {corrections_root}")

    def _on_retrain(self) -> None:
        if self._yolo is None or self._model_path is None:
            self._show_error("Load a model first.")
            return

        model_root = self._model_path.parent
        corrections_root = model_root / "corrections"
        source_images = corrections_root
        source_labels = corrections_root

        if not source_images.exists() or not source_labels.exists():
            self._show_error("No corrections found. Add at least one correction first.")
            return

        # Use destination path if provided, otherwise use default location
        if self._destination_path is not None:
            retrain_root = self._destination_path
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            retrain_root = model_root / f"retrained_{stamp}"

        dataset_dir = retrain_root / "dataset"
        traces_dir = retrain_root / "training_traces"

        try:
            retrain_root.mkdir(parents=True, exist_ok=True)
            create_training_set(
                path_to_images=source_images,
                path_to_vectors=source_labels,
                destination_path=dataset_dir,
                label_names=[(0, "LABEL")],
            )

            # Load training configuration
            config = self._load_training_config(corrections_root)

            self._yolo.train(
                data_set_file=dataset_dir / "dataset.yaml",
                image_size=config["image_size"],
                batch=config["batch"],
                epochs=config["epochs"],
                patience=config["patience"],
                project=traces_dir,
                name="run",
                exist_ok=True,
            )

            best_model = traces_dir / "run" / "weights" / "best.pt"
            if not best_model.exists():
                raise FileNotFoundError(f"best.pt not found at: {best_model}")

            shutil.copy2(best_model, retrain_root / "best.pt")
            shutil.rmtree(traces_dir, ignore_errors=True)
        except Exception as exc:  # pragma: no cover - GUI runtime guard
            self._show_error(f"Retrain failed: {exc}")
            return

        self._show_info(f"Retrain done. New model saved in: {retrain_root}")
