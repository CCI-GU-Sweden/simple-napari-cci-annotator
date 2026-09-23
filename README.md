[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://python.org)

# Simple napari CCI annotator

A project-based napari plugin for YOLO bounding-box and instance-segmentation prediction, review, and annotation storage.

The plugin is being rebuilt in milestones. Project persistence, multidimensional RGB conversion, multi-class annotation, large-image detection inference, movable failure crops, reproducible dataset building, and YOLO retraining are available.

## Current workflow

1. Open the plugin in napari.
2. Select **Bounding-box detection** or **Instance segmentation**, then click **New Project** and select an empty folder. A project's task is fixed, and its packaged starter checkpoint is copied into `models/` automatically. Use **Open Project** to reopen one.
3. Use **Edit Classes** to append or rename project classes when needed.
4. Open/select an image layer in napari. RGB images and multidimensional TIFF/OME-TIFF arrays are supported.
5. For multidimensional data, select the channel axis and map up to three source channels into output red, green, and blue.
6. Optionally select a spatial pre-filter and radius, then select and preview the normalization used to create the RGB `uint8` training image.
7. The plugin automatically looks for a same-stem YOLO bbox file and creates an editable `yolo_bboxes` Shapes layer.
8. Choose **Current class** before drawing a box. To reclassify boxes, select them and click **Apply Class to Selected**.
9. To run inference, choose a YOLO detection model and set device, confidence, model IoU, tile size/overlap, merge IoU, and maximum detections per tile.
10. Click **Predict Current RGB Plane**. Prediction runs outside the UI thread and can be cancelled between tiles.
11. In **Movable training crop**, choose 1024 or 512 and click **Select Training Crop**. Move the cyan square over a failure location.
12. Click **Create / Refresh Crop**, correct its local bbox layer, then click **Add Crop + Corrections**.
13. Click **Return to Source**, move the selection to another failure, and repeat. The full-size inference image is not added to the training pool.
14. Use **Validate Project** to check image/label pairing, fixed image dimensions, and label contents.
15. Use **Review saved annotations** to load previous/next project pairs, make corrections, and save the updated annotation.
16. In **Dataset building and retraining**, choose the copied starter model, another model in the project, or the currently loaded fine-tuned model, preview the stable split, then start retraining.
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
│   ├── yolo26n.pt                  # detection project, or
│   ├── yolo26n-seg.pt              # segmentation project
│   └── retrain_YYYYMMDD_HHMMSS.pt  # after successful retraining
└── annotations/
    ├── images/
    ├── labels/
    ├── masks/
    ├── instances/
    └── audit.jsonl
```

Detection projects use same-stem `images/*.png` + `labels/*.txt` pairs. Segmentation projects use same-stem triples:

```text
annotations/images/field_001.png
annotations/masks/field_001.tif
annotations/instances/field_001.json
```

Segmentation masks are lossless 2D `uint32` TIFF instance maps: `0` is background and every positive value is an instance ID. JSON metadata maps each ID to its class, confidence, review status, source, bounding box, pixel area, and merge/split lineage. Detection and segmentation annotations are never mixed in one project. Data from the retired segmentation plugin is not treated as a project and is not imported automatically; compatible YOLO segmentation checkpoints can still be selected.

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

The **Review saved annotations** section lists every canonical image/label pair and provides **Previous**, **Load Selected**, **Next**, and **Save Corrections** controls in one place. **Save Corrections** uses the same atomic image/YOLO update path as the main annotation section. Invalid stored pairs receive a warning marker and an explanation instead of being loaded silently. During editing, malformed boxes or boxes extending outside the image receive a translucent red face while retaining their class-colored edge; both save buttons are disabled until every red box is corrected or deleted.

Short tooltips on the image-processing, prediction, crop, review, split, and retraining controls explain what each parameter changes and highlight important speed, memory, and validation tradeoffs.

## Instance segmentation (Phases 7A–7E)

For a segmentation project, loading an image creates or reloads one editable `yolo_instances` napari Labels layer. Polygons are not shown. Select a project class, then use **New Mask Instance** to allocate an ID and paint it. The nearby actions apply a class, delete an instance, merge entered IDs into the selected ID, split a disconnected ID into 4-connected objects, or explicitly keep only the component with the largest bounding-box area. Per-class counts and selected-instance details remain visible.

A loaded checkpoint must report Ultralytics task `segment` and the same class IDs as the project. Prediction uses the locked RGB conversion, requests retina masks, resizes masks back with nearest-neighbor interpolation when required, removes detached prediction contamination using the original plugin's largest-bbox-component rule, and resolves pixel overlaps by confidence. That cleanup is automatic only for model output; manual disconnected masks are flagged and require an explicit split or keep-largest action.

The save and review controls atomically overwrite the same-stem image/mask/metadata triple and append an audit record. Saving is blocked for unknown IDs, missing or empty metadata, invalid classes, disconnected instances, shape mismatch, or non-integer IDs. Direct saving requires the converted source to be exactly the project's fixed 512 or 1024 training size; arbitrary-size segmentation sources use the movable crop workflow below.

Segmentation crops extract the normalized RGB image and the matching instance map together. Present IDs are deterministically compacted to `1..N`, while class metadata and source-ID lineage are retained. An object cut by the human-positioned crop boundary is allowed and marked as partial. Painting any instance into synthetic bottom/right padding is not allowed and disables saving. The standard new/delete/class/merge/split/keep-largest tools operate on the crop Labels layer, and saving writes a deterministic same-stem image/mask/JSON triple.

### Large-image segmentation tiling and fusion

Segmentation images larger than the configured tile size automatically use the Phase 7C Dask path. The core size is `tile_size − 2 × overlap`; overlap must therefore remain below half the tile size. The source is reflection-padded to a complete core grid, each core receives reflected inference halos, and model calls remain mutex-serialized for GPU safety. Temporary IDs are allocated from deterministic per-tile ranges, while confidence determines pixel ownership inside each tile.

After inference, directly adjacent IDs across every one-pixel core seam are fused only when their project classes match. A deterministic union/find pass handles transitive groups and compacts them into stable final IDs. Cross-class contacts are retained separately and reported as conflicts. One-to-many seam contacts are also recorded as ambiguous because the compatibility rule may join touching objects. Tile size, overlap, core grid, padding, temporary-to-final mapping, seam pairs, conflicts, ambiguities, component cleanup, and optional border removals are attached to prediction provenance and enter the audit record when saved.

The optional **Show segmentation core grid** control adds a cyan debug Shapes layer. **Clear instances touching image border** removes merged edge objects and should only be enabled when border objects are known contaminants.

## Large-image bbox prediction and merging

Inference always uses the same converted RGB `uint8` pixels shown by **Preview RGB Conversion** and stored for training. Images are covered by deterministic square tiles (1024 pixels and 20% overlap by default). Images smaller than a tile are reflection-padded for inference; detections centered in padding are discarded.

Tile-local detections are clipped to valid pixels and translated directly into full-image coordinates. Overlap ownership regions identify which tile should represent a seam object, then a separate class-aware global NMS removes duplicates. **Model IoU** controls suppression inside each YOLO tile; **Merge IoU** controls suppression across tiles. Different classes never suppress one another.

The output `yolo_bboxes` layer preserves `class_id`, project class name, confidence, source, and tile ID. Prediction settings and the model path are attached to the layer and written into the annotation audit entry when corrections are saved. A loaded model must be a detection model and expose the same class IDs as the project; model names may differ because the project names are authoritative in the annotation UI.

## Movable fixed-size training crops

The crop workflow turns a local inference failure into one immediately retrainable sample:

1. **Select Training Crop** creates a cyan square centered on the current view. The square can be moved freely. If it is resized, **Create / Refresh Crop** snaps it back to the configured project size around its new center.
2. **Create / Refresh Crop** extracts the already-previewed RGB conversion and the task geometry. Detection opens `training_crop_rgb` plus `training_crop_bboxes`; segmentation opens `training_crop_rgb` plus `training_crop_instances`.
3. Correct the local boxes or instance mask, including classes, and click **Add Crop + Corrections**. Detection stores an image/YOLO-label pair; segmentation stores an image/TIFF-mask/JSON-instance triple. Both use one deterministic stem such as `field__t000__z002__crop_y001024_x002048_s1024`, so saving the same location updates it instead of creating a duplicate.
4. **Return to Source** restores the inference image and its boxes. The crop selection remains available to move to the next failure location.

No crop is resized. When the source runs out at its bottom or right edge, missing pixels are filled with RGB value `114`; the audit record stores the source bounds, valid extent, and padding. Boxes are forbidden in padded pixels. Any edited box that enters padding is shown with a translucent red face while its class-colored edge remains unchanged; **Add Crop + Corrections** stays disabled until every red box is moved, resized, or deleted. Save-time validation repeats the same check as a final safeguard.

Fully contained boxes are copied directly. Every bbox intersecting a manually reviewed crop is clipped to the crop boundary and retained as a valid YOLO annotation, even when only part of the object is visible. For segmentation, cropped instances are similarly allowed to be partial, but retain explicit lineage rather than silently pretending to be complete source objects. Boundary clipping is reported but never blocks saving. Detection remnants narrower or shorter than 2 pixels are omitted with a warning; boxes entirely outside the crop are ignored. This permissive manual-crop policy is intentionally separate from automatic dataset validation.

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

The wheel contains both starter checkpoints. **New Project** copies only the task-compatible one into the project: `models/yolo26n.pt` for detection or `models/yolo26n-seg.pt` for segmentation. The **Starting model** control offers that starter, every other `.pt` checkpoint in the project's `models/` folder, and a compatible externally loaded model. Detection and segmentation weights are never substituted for each other. This choice affects only the new run; successful training never silently replaces the prediction model.

Training images are generated inside a new `retrain_YYYYMMDD_HHMMSS` folder. Each canonical image already matches the locked project patch size, so it becomes one training item without resizing or further spatial subdivision. The dataset validator rejects mismatched dimensions. Reviewed-negative crops remain first-class empty-label samples.

Each run contains:

```text
retrain_YYYYMMDD_HHMMSS/
├── dataset/
│   ├── images/train/ and images/val/
│   ├── labels/train/ and labels/val/
│   ├── masks/train/ and masks/val/       # segmentation runs
│   ├── instances/train/ and instances/val/ # segmentation runs
│   ├── dataset.yaml
│   ├── split_manifest.csv
│   └── tile_manifest.csv or segmentation_export_manifest.csv
├── training/
│   └── weights/best.pt and last.pt (train-only runs may have only last.pt)
├── run.yaml
└── README.txt
```

For segmentation, the canonical TIFF mask remains authoritative. YOLO polygon rows are generated invisibly only inside the run snapshot. Validation rejects disconnected IDs, missing/cross-class metadata, invalid or degenerate contours, polygon round-trip IoU below `0.90`, objects in synthetic padding, size/provenance mismatches, and source-group leakage. The export manifest records per-class counts, image/label/mask checksums, and minimum/mean mask→polygon→mask IoU. Empty reviewed masks produce valid empty `.txt` labels. Ultralytics must report task `segment`; a successful `best.pt` is copied to `project/models/<run-name>.pt` like detection training.

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

Before normalization, each selected 2D source-channel plane can optionally receive one project-wide spatial filter:

- Gaussian blur
- Median filter
- Mean filter
- Frequency-domain low-pass filter
- White top-hat

All filters share a radius measured in source pixels. Median, mean, and top-hat use a disk footprint. Gaussian uses a radius-limited Gaussian kernel, with `sigma = radius / 2` (and a minimum sigma of `0.5`). The explicit low-pass option uses a second-order Butterworth frequency response with reflected boundary padding; larger radii lower the cutoff and remove more fine-scale variation. Filtering is performed independently on the selected source channels before normalization and RGB stacking; it is not applied a second time during dataset preparation.

**Invert intensities for this image** is a separate per-image/per-plane toggle. It applies `255 − value` to each selected channel after normalization while keeping intentionally empty RGB outputs at zero. Unlike channel mapping, filtering, and normalization, inversion is not frozen when the project locks: it can be changed before preview, inference, or an annotation update. The chosen state is recorded in that sample's audit conversion metadata. Saved canonical review images are already converted and are never inverted a second time.

Normalization is currently calculated independently for each selected channel of each 2D plane. The converted output is always RGB `uint8` with the same Y/X dimensions as the source plane, so bbox coordinates do not change.

Channel mapping, pre-filter, radius, and normalization become immutable project settings after the first converted annotation is saved. Per-image inversion remains editable. Create a new project to change the locked conversion. For inference, the locked `project.yaml` mapping is read directly and is authoritative over widget state, while the current inversion toggle remains an explicit override. The audit log records the configured filter and normalization settings, inversion state, and effective per-channel normalization values used for every image.

Canonical training images already contain those normalized RGB `uint8` pixels, so dataset preparation tiles/copies them without applying normalization a second time. Before retraining, validation requires RGB canonical images and verifies that each sample's audit conversion matches the locked project settings. A missing or mismatched conversion record blocks the run instead of mixing normalization policies.

The widget warns about unsaved bbox edits before reloading labels, changing source image/Z/T context, or closing. Choosing Save uses the cached pixels and plane identity from the annotation being edited, avoiding accidental reassignment to a newly selected plane.

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for the complete roadmap.
