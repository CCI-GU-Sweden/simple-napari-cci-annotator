
# Simple napari annotator

Minimal napari plugin for YOLO bbox detection + quick correction + retraining.

No dataset browser, no extra workflow logic. User provides the image in napari.

## UI flow

1. Path to model (`.pt`)
2. `Load model`
3. `Predict`
4. Optional destination folder for retrained model (browse or type)
5. Edit boxes in napari shapes layer (`yolo_bboxes`)
6. `Add correction`
7. `Retrain`

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
