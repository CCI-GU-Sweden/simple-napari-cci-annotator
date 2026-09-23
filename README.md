[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://python.org)

# Simple napari CCI annotator

A project-based napari plugin for tiled YOLO bounding-box prediction, review, and annotation storage.

The plugin is being rebuilt in milestones. Project persistence, multidimensional RGB conversion, multi-class annotation, large-image detection inference, movable failure crops, reproducible dataset building, and YOLO retraining are available.

## Current workflow

1. Open the plugin in napari.
2. Click **New Project** and select an empty folder, or click **Open Project** to reopen an initialized project.
3. Use **Edit Classes** to append or rename project classes when needed.
4. Open/select an image layer in napari. RGB images and multidimensional TIFF/OME-TIFF arrays are supported.
5. For multidimensional data, select the channel axis and map up to three source channels into output red, green, and blue.
6. Select and preview the normalization used to create the RGB `uint8` training image.
7. The plugin automatically looks for a same-stem YOLO bbox file and creates an editable `yolo_bboxes` Shapes layer.
8. Choose **Current class** before drawing a box. To reclassify boxes, select them and click **Apply Class to Selected**.
9. To run inference, choose a YOLO detection model and set device, confidence, model IoU, tile size/overlap, merge IoU, and maximum detections per tile.
10. Click **Predict Current RGB Plane**. Prediction runs outside the UI thread and can be cancelled between tiles.
11. In **Movable training crop**, choose 1024 or 512 and click **Select Training Crop**. Move the cyan square over a failure location.
12. Click **Create / Refresh Crop**, correct its local bbox layer, then click **Add Crop + Corrections**.
13. Click **Return to Source**, move the selection to another failure, and repeat. The full-size inference image is not added to the training pool.
14. Use **Validate Project** to check image/label pairing, fixed image dimensions, and label contents.
15. Use **Review saved annotations** to load previous/next project pairs, make corrections, and save the updated annotation.
16. In **Dataset building and retraining**, choose the base `yolo26n.pt`, a model in the project, or the currently loaded fine-tuned model, preview the stable split, then start retraining.
17. When training finishes, explicitly keep the current model, load the promoted model, or open the immutable run folder.

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
│   └── retrain_YYYYMMDD_HHMMSS.pt
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

Canonical training images have one project-wide size: 1024×1024 by default, or 512×512. The choice locks after the first training sample is saved. Arbitrary-size files remain valid inference sources, but cannot be saved directly into the canonical training pool; use a training crop instead. Images smaller than the chosen patch are padded without resizing.

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

Coordinates must be finite, normalized to `[0, 1]`, have positive width/height, and describe a box contained by the image. A project initializes with one class:

```yaml
classes:
  0: LABEL
```

Use **Edit Classes** to rename that class or append more classes. Class IDs are stable, contiguous YOLO indices starting at `0`; names must be non-empty and unique. Only the last class can be removed, and removal is blocked while that ID occurs in a saved label or the current bbox layer. This prevents an existing label from silently changing meaning.

Every shape stores `class_id` and `class_name`. The current-class selector sets the class and stable display color for newly drawn boxes. **Apply Class to Selected** reclassifies only the selected boxes, marks them as manual corrections, and clears stale prediction confidence. Live per-class counts are shown beside the annotation controls. Saving still writes standard five-column YOLO rows, so no custom conversion is required for retraining.

The **Review saved annotations** section lists every canonical image/label pair and provides **Previous**, **Load Selected**, and **Next** navigation. Invalid stored pairs receive a warning marker and an explanation instead of being loaded silently. During editing, malformed boxes or boxes extending outside the image receive a translucent red face while retaining their class-colored edge; saving is disabled until every red box is corrected or deleted.

## Large-image prediction and merging

Inference always uses the same converted RGB `uint8` pixels shown by **Preview RGB Conversion** and stored for training. Images are covered by deterministic square tiles (1024 pixels and 20% overlap by default). Images smaller than a tile are reflection-padded for inference; detections centered in padding are discarded.

Tile-local detections are clipped to valid pixels and translated directly into full-image coordinates. Overlap ownership regions identify which tile should represent a seam object, then a separate class-aware global NMS removes duplicates. **Model IoU** controls suppression inside each YOLO tile; **Merge IoU** controls suppression across tiles. Different classes never suppress one another.

The output `yolo_bboxes` layer preserves `class_id`, project class name, confidence, source, and tile ID. Prediction settings and the model path are attached to the layer and written into the annotation audit entry when corrections are saved. A loaded model must be a detection model and expose the same class IDs as the project; model names may differ because the project names are authoritative in the annotation UI.

## Movable fixed-size training crops

The crop workflow turns a local inference failure into one immediately retrainable sample:

1. **Select Training Crop** creates a cyan square centered on the current view. The square can be moved freely. If it is resized, **Create / Refresh Crop** snaps it back to the configured project size around its new center.
2. **Create / Refresh Crop** extracts the already-previewed RGB conversion, translates intersecting source boxes into crop coordinates, and opens `training_crop_rgb` plus `training_crop_bboxes` layers.
3. Correct the local boxes, including their classes, and click **Add Crop + Corrections**. The image and YOLO label are stored with one deterministic stem such as `field__t000__z002__crop_y001024_x002048_s1024`. Saving the same source location again updates that pair instead of creating a duplicate.
4. **Return to Source** restores the inference image and its boxes. The crop selection remains available to move to the next failure location.

No crop is resized. When the source runs out at its bottom or right edge, missing pixels are filled with RGB value `114`; the audit record stores the source bounds, valid extent, and padding. Boxes are forbidden in padded pixels. Any edited box that enters padding is shown with a translucent red face while its class-colored edge remains unchanged; **Add Crop + Corrections** stays disabled until every red box is moved, resized, or deleted. Save-time validation repeats the same check as a final safeguard.

Fully contained boxes are copied directly. Every bbox intersecting a manually reviewed crop is clipped to the crop boundary and retained as a valid YOLO annotation, even when only part of the object is visible. Boundary clipping is reported but never blocks saving. Only degenerate remnants narrower or shorter than 2 pixels are omitted with a warning; boxes entirely outside the crop are ignored. This permissive manual-crop policy is intentionally separate from the stricter clipping policy used by automatic dataset tiling.

The project stores this immutable contract in `project.yaml`:

```yaml
training_patch:
  size: 1024
  padding_value: 114
  locked: true
```

The normal **Save Converted Image + BBoxes** action remains available when the entire converted source is already exactly the configured size. Otherwise it directs the user to the crop workflow.

## Dataset building and retraining

Retraining uses the complete fixed-size canonical annotation pool. Source groups are assigned to a deterministic 80/20 train/validation split, and existing assignments remain stable as new crops are added. Crops and Z/T samples derived from the same audited source path stay in the same split. An optional audit metadata field can provide a patient, well, acquisition, or other higher-level grouping key.

### Adding audit grouping metadata

The optional **Group metadata** box accepts a dot-separated field path from `annotations/audit.jsonl`, for example `metadata.patient`, `metadata.well`, or `metadata.acquisition`. `audit.jsonl` is JSON Lines: each line is one complete JSON object. Do not wrap the records in a JSON list and do not add commas between lines.

The plugin writes annotation-save records automatically. To attach grouping information, append a metadata record using the exact canonical `sample_id` shown by the saved image/label stem. Avoid editing the file while an annotation is being saved, and keep a backup before bulk metadata edits:

```json
{"timestamp":"2026-09-22T10:30:00+02:00","operation":"metadata","sample_id":"field_001__t000__z000","metadata":{"patient":"P001","well":"A01","acquisition":"run_03"}}
{"timestamp":"2026-09-22T10:31:00+02:00","operation":"metadata","sample_id":"field_002__t000__z000","metadata":{"patient":"P001","well":"A02","acquisition":"run_03"}}
```

Then enter one grouping level in the GUI—for example `metadata.patient`. Every sample with value `P001` is treated as one indivisible source group and can only appear in train or validation, never both. Use the grouping level that represents the true independent biological unit; if several images belong to one patient/specimen/well, they should carry the same value.

Metadata records are merged per sample in append order. A later metadata record can update one field without repeating the others, and later annotation saves do not erase previously appended metadata. Keep `sample_id` spelling exact. Malformed JSON lines are ignored, so validate the dataset after editing. When **Group metadata** is left empty, the plugin groups by the original audited `source_path`, falling back to the canonical sample ID when no source path is available.

The **Starting model** control offers the repository-root `yolo26n.pt` as the naive pretrained base, every `.pt` checkpoint in the project's `models/` folder, and a compatible externally loaded model. This choice affects only the new run; successful training never silently replaces the prediction model.

Training images are generated inside a new `retrain_YYYYMMDD_HHMMSS` folder. Each canonical image already matches the locked project patch size, so it becomes one training item without resizing or further spatial subdivision. The dataset validator rejects mismatched dimensions. Reviewed-negative crops remain first-class empty-label samples.

Each run contains:

```text
retrain_YYYYMMDD_HHMMSS/
├── dataset/
│   ├── images/train/ and images/val/
│   ├── labels/train/ and labels/val/
│   ├── dataset.yaml
│   ├── split_manifest.csv
│   └── tile_manifest.csv
├── training/
│   └── weights/best.pt and last.pt (train-only runs may have only last.pt)
├── run.yaml
└── README.txt
```

`dataset.yaml` derives its complete `names` map from the project classes. `run.yaml` records the input model and checksum, package versions, dataset and training settings, split assignments, device, timestamps, warnings, outputs, and final status. Failed and cancelled runs remain on disk with their status and diagnostic information.

After a successful validated run, `training/weights/best.pt` remains in the immutable run and an atomic copy is added to `<project>/models/<run-folder-name>.pt`, for example `models/retrain_20260923_143012.pt`. The completion dialog loads this project copy when requested. Exploratory output without `best.pt` is not promoted.

With only one independent source group, validation is impossible. The plugin blocks normal retraining unless **Train without independent validation** is enabled and confirmed for that run. Such output is prominently marked exploratory and must not be used to claim generalization quality. Ultralytics requires a validation-loader path even when validation is disabled, so train-only `dataset.yaml` points that unused loader at the training images; `run.yaml` records this compatibility workaround and `validation_enabled: false`.

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

Channel mapping and normalization become immutable project settings after the first converted image/bbox pair is saved. Create a new project to use a different conversion. For inference, the locked `project.yaml` mapping is read directly and is authoritative over widget state. The audit log records both the configured method and the effective per-channel values used for every image.

Canonical training images already contain those normalized RGB `uint8` pixels, so dataset preparation tiles/copies them without applying normalization a second time. Before retraining, validation requires RGB canonical images and verifies that each sample's audit conversion matches the locked project settings. A missing or mismatched conversion record blocks the run instead of mixing normalization policies.

The widget warns about unsaved bbox edits before reloading labels, changing source image/Z/T context, or closing. Choosing Save uses the cached pixels and plane identity from the annotation being edited, avoiding accidental reassignment to a newly selected plane.

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for the complete roadmap.
