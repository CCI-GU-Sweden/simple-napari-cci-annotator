
[![License MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/CCI-GU-Sweden/simple-napari-cci-annotator/blob/main/LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://python.org)
[![tests](https://github.com/CCI-GU-Sweden/simple-napari-cci-annotator/workflows/tests/badge.svg)](https://github.com/CCI-GU-Sweden/simple-napari-cci-annotator/actions)

# Simple napari annotator

Minimal napari plugin for YOLO bbox detection + quick correction + retraining.

No dataset browser, no extra workflow logic. User provides the image in napari.

## UI flow

1. Path to model (`.pt`) or model folder
2. `Load model`
3. `Predict`
4. Optional destination folder for retrained model (browse or type)
5. Edit boxes in napari shapes layer (`yolo_bboxes`)
6. `Add correction`
7. `Retrain`

## Starting from scratch

The model input supports either:

- A `.pt` model file, or
- A folder path.

Folder behavior on `Load model`:

- If the folder already contains one or more `.pt` files, the first one is loaded.
- If the folder contains no `.pt` file, the plugin copies the bundled `yolov8n.pt` into that folder and loads it.
- Empty model input is invalid and will show an error.

If you do not have a model yet, start from a pretrained YOLOv8 nano checkpoint and use it as your initial file:

- Direct download: <https://github.com/ultralytics/assets/releases/latest/download/yolov8n.pt>
- Model overview: <https://docs.ultralytics.com/models/yolov8/>

After downloading, select that `yolov8n.pt` file in the Model field and continue with the correction/retrain loop.

## Assumptions

- Input image is already RGB 8-bit (or compatible with clipping/conversion).
- Single class (`0: LABEL`) for now.
- Exactly one image layer should be present when using `Add correction`.

## Folder behavior

Given a model path like:

`.../my_model/best.pt`

The plugin uses `.../my_model` as root.

### Add correction

Each click saves:

- Image to `my_model/corrections/<image_name>_<timestamp>.png`
- Labels to `my_model/corrections/<image_name>_<timestamp>.txt`
- Training config to `my_model/corrections/training_config.json` (created/updated)

Image layer behavior:

- If no image layer exists, `Add correction` shows an error.
- If more than one image layer exists, `Add correction` shows an error.
- If exactly one image layer exists, that image layer is used for saving correction image data.

`training_config.json` defaults:

- `image_size`: prefilled from the current image size using `max(height, width)`
- `batch`: `8`
- `epochs`: `100`
- `patience`: `30`

You can edit this file before clicking `Retrain`.

Label format is YOLO detection:

`class x_center y_center width height`

normalized to `[0, 1]`.

### Retrain

Each click creates:

- `<retrain_root>/dataset/`
  - `images/train`, `images/val`
  - `labels/train`, `labels/val`
  - `dataset.yaml`
- Trains YOLO from those corrections
- Copies best model to:
  - `<retrain_root>/best.pt`
- Deletes training traces folder after extracting `best.pt`

`<retrain_root>` resolution:

- If destination field is set, retraining outputs there.
- Otherwise it defaults to `my_model/retrained_<timestamp>`.

So the retrained folder keeps a clean dataset + final model, without run artifacts.

Warning: Data labeled will be equally divided between traning and validation (50%).
