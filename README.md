[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://python.org)

# Simple napari CCI annotator

A project-based napari plugin for reviewing and storing YOLO bounding-box annotations.

The plugin is currently being rebuilt in milestones. This milestone intentionally focuses on project and annotation management. Model loading, prediction, tiling, and retraining will return in later milestones.

## Current workflow

1. Open the plugin in napari.
2. Click **New Project** and select an empty folder, or click **Open Project** to reopen an initialized project.
3. Open/select an image layer in napari. RGB images and multidimensional TIFF/OME-TIFF arrays are supported.
4. For multidimensional data, select the channel axis and map up to three source channels into output red, green, and blue.
5. Select and preview the normalization used to create the RGB `uint8` training image.
6. The plugin automatically looks for a same-stem YOLO bbox file and creates an editable `yolo_bboxes` Shapes layer.
7. Edit the boxes and click **Save/Import/Update Converted Image + BBoxes**.
8. Use **Validate Project** to check image/label pairing and label contents.

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

Saving an existing stem overwrites its canonical image/label pair. Each successful create or update is recorded in `audit.jsonl`, including checksums, conversion metadata, selected Z/T indices, and box/class counts. Empty label files are valid reviewed-negative annotations.

For multidimensional sources, each selected non-channel plane has a stable sample ID. For example, `field.ome.tif` at `T=1, Z=2` becomes:

```text
annotations/images/field.ome__t001__z002.tif
annotations/labels/field.ome__t001__z002.txt
```

This prevents different Z/T planes from overwriting one another while keeping every training image and label stem identical.

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

## Image conversion

The plugin extracts the currently displayed Z/T plane using napari's current dimension positions. Spatial axes are inferred from axis metadata (`Y` and `X`) and otherwise default to the last two non-channel axes. The channel axis remains user-selectable so unusual TIFF layouts can be handled explicitly.

Each output component—red, green, and blue—can use any source channel or be left empty. Available normalization methods are:

- Min/max
- Simple max (`0..plane maximum`)
- Percentile, with configurable lower/upper percentiles
- Z-score, with configurable lower/upper Z values
- Fixed input range
- Integer data-type range

Normalization is currently calculated independently for each selected channel of each 2D plane. The converted output is always RGB `uint8` with the same Y/X dimensions as the source plane, so bbox coordinates do not change.

Channel mapping and normalization become immutable project settings after the first converted image/bbox pair is saved. Create a new project to use a different conversion. The audit log records both the configured method and the effective per-channel values used for every image.

The widget warns about unsaved bbox edits before reloading labels, changing source image/Z/T context, or closing. Choosing Save uses the cached pixels and plane identity from the annotation being edited, avoiding accidental reassignment to a newly selected plane.

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for the complete roadmap.
