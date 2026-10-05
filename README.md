# Simple napari CCI annotator

## Installation

From the repository root, install the plugin and napari into a Python 3.10–3.12 environment:

```bash
python -m pip install -e ".[all]"
napari
```

In napari, open **Plugins → CCI Annotator → Simple CCI Annotator Plugin**.

## What it does

The plugin helps you run YOLO detection or instance segmentation on microscopy images, correct predictions, and save training annotations. It also supports tiled inference on large images, batch prediction of TIFF folders, and retraining from saved annotations.

## Quick start

1. Choose **Bounding-box detection** or **Instance segmentation**, then create a project in an empty folder. Use **Open Project** to return to an existing one.
2. Open an image in napari. Set the channel mapping, optional filter, and normalization, then use **Preview RGB Conversion** to check the pixels used for prediction and training.
3. Select a compatible YOLO model and click **Predict Current RGB Plane**. Edit the resulting boxes or instance mask as needed.
4. To save training data, select a 512×512 or 1024×1024 training crop, correct it, and click **Add Crop + Corrections**. The first saved annotation locks the project's image conversion and training patch size.
5. Use **Save Current Output** to export a full-size prediction without adding it to the training annotations. For a whole folder, use **Batch TIFF prediction** after the project settings are locked.

Inference tiles default to the selected training crop size. You can set another tile size in the prediction controls.

## Output

Training annotations are stored in the project folder. Detection annotations use YOLO `.txt` labels; segmentation annotations use instance-mask TIFF files and JSON metadata.

Batch prediction accepts `.tif`, `.tiff`, `.ome.tif`, and `.ome.tiff` files directly inside the selected input folder. It processes every nonspatial plane. Results go to `<input folder>/Prediction/` by default, or to an output folder you choose. Each run includes `analysis_metadata.json` with the settings and file records needed to trace its results.

## More guidance

- [GUI tutorial](TUTORIAL.md): annotation, batch prediction, and retraining steps.
- [Python API examples](API_EXAMPLES.md): project creation, training, and headless inference without napari or Qt.
- [Remaining work](docs/REMAINING_WORK.md): validation and the older plugin's retirement checklist.

This project is released under the [MIT license](LICENSE).
