# Simple napari CCI annotator

## Installation

In a specific environment, install napari (tested with version 0.7 and 0.8)

```shell
conda create -n napari -c conda-forge napari pyqt --yes
conda activate napari
napari
```

In napari, open **Plugins → Simple CCI Annotator Plugin**. Or search for the CCI plugin.

Napari provide standalone installers for Windows, macOS, and Linux. You can install the plugin from the napari store.

Another way to install is with pip:

```bash
pip install simple-napari-cci-annotator
```

The code is on [Githb](https://github.com/CCI-GU-Sweden/simple-napari-cci-annotator).

From the repository root, install the plugin and napari into a Python 3.10–3.12 environment:

```bash
python -m pip install -e ".[all]"
napari
```

### GPU (NVIDIA CUDA) — recommended if you have an NVIDIA GPU

Check your maximum supported CUDA version first:

```shell
nvidia-smi
```

Then pick a compatible wheel index (`cu124` = CUDA 12.4, `cu126` = 12.6, etc.). Check the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the appropriate build.

Then run:

```shell
conda create -n napari-gpu -c conda-forge napari pyqt --yes
conda activate napari-gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics
pip install simple-napari-cci-annotator
```

Quick smoke test:

```shell
python -c "import numpy, scipy, torch, cv2; from ultralytics import YOLO; print(numpy.__version__, scipy.__version__, torch.__version__, cv2.__version__)"
```

## What it does

The plugin helps you run YOLO detection or instance segmentation on microscopy images, correct predictions, and save training annotations. It also supports tiled inference on large images, batch prediction of image folders, and retraining from saved annotations.

## Quick start

1. Choose **Bounding-box detection** or **Instance segmentation**, then create a project in an empty folder. Use **Open Project** to return to an existing one.
2. Open an image in napari. Set the channel mapping, optional filter, and normalization, then use **Preview RGB Conversion** to check the pixels used for prediction and training.
3. Select a compatible YOLO model and click **Predict Current RGB Plane**. Edit the resulting boxes or instance mask as needed.
4. To save training data, select a 512×512 or 1024×1024 training crop, correct it, and click **Add Crop + Corrections**. The first saved annotation locks the project's image conversion and training patch size.
5. Use **Save Current Output** to export a full-size prediction without adding it to the training annotations. For a whole folder, use **Batch image prediction** after the project settings are locked.

Inference tiles default to the selected training crop size. You can set another tile size in the prediction controls.

For a segmentation volume, expand **3D inference and assembly**, select its Z/Y/X/channel axes and Z range, then use **Predict Z Slices**. Inspect the saved per-slice masks over lazy views of the raw channels. The section supports conversion previews, cancellation, verified resume, and reopening saved runs without repeating inference. These slice instances are not yet assembled into 3D objects; see the [volume tutorial](TUTORIAL.md#predict-a-volume).

## Output

Training annotations are stored in the project folder. Detection annotations use YOLO `.txt` labels; segmentation annotations use instance-mask TIFF files and JSON metadata.

Batch prediction accepts `.tif`, `.tiff`, `.ome.tif`, `.ome.tiff`, `.png`, and `.bmp` files directly inside the selected input folder. It processes every nonspatial TIFF plane and each PNG or BMP image. Grayscale, RGB, and RGBA PNG/BMP files are supported when their channel layout matches the project's locked settings. Results go to `<input folder>/Prediction/` by default, or to an output folder you choose. Each run includes `analysis_metadata.json` with the settings and file records needed to trace its results.

## More guidance

- [GUI tutorial](TUTORIAL.md): annotation, batch prediction, and retraining steps.
- [Python API examples](API_EXAMPLES.md): project creation, training, and headless inference without napari or Qt.
- [Remaining work](docs/REMAINING_WORK.md): validation and the older plugin's retirement checklist.

This project is released under the [MIT license](LICENSE).
