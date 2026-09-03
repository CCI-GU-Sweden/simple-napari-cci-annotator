[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://python.org)

# Simple napari CCI annotator

A project-based napari plugin for reviewing and storing YOLO bounding-box annotations.

The plugin is currently being rebuilt in milestones. This milestone intentionally focuses on project and annotation management. Model loading, prediction, tiling, and retraining will return in later milestones.

## Current workflow

1. Open the plugin in napari.
2. Click **New Project** and select an empty folder, or click **Open Project** to reopen an initialized project.
3. Open/select an image layer in napari.
4. The plugin automatically looks for a same-stem YOLO bbox file and creates an editable `yolo_bboxes` Shapes layer.
5. Edit the boxes and click **Add Annotation**, **Import Annotation**, or **Update Annotation**.
6. Use **Validate Project** to check image/label pairing and label contents.

Automatic label discovery checks, in order:

1. `<project>/annotations/labels/<image_stem>.txt`
2. A `.txt` file beside the source image
3. If the source image is in an `images/` folder, the matching file in its sibling `labels/` folder

## Project structure

Selecting an empty folder with **New Project** initializes:

```text
project/
├── project.yaml
├── models/
└── annotations/
    ├── images/
    ├── labels/
    └── audit.jsonl
```

Images and labels are stored with the same stem:

```text
annotations/images/field_001.png
annotations/labels/field_001.txt
```

Saving an existing stem overwrites its canonical image/label pair. Each successful create or update is recorded in `audit.jsonl`, including checksums and box/class counts. Empty label files are valid reviewed-negative annotations.

## YOLO bbox format

Each non-empty line must contain:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates must be finite, normalized to `[0, 1]`, have positive width/height, and describe a box contained by the image. The project currently initializes with one class:

```yaml
classes:
  0: LABEL
```

Class IDs and names are kept as napari Shapes properties so the storage layer is ready for multi-class support.

## Image support in this milestone

Annotation saving currently accepts 2D grayscale or 1-, 3-, or 4-channel arrays that Pillow can encode as PNG/TIFF/JPEG/WebP/BMP. The plugin does not silently normalize or select channels yet. Project-wide channel selection and normalization are the next dedicated image-processing milestone.

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for the complete roadmap.
