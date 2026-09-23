# YOLO Bounding-Box Annotator Improvement Plan

## 1. Goal

Turn the plugin from a single-image demo into a dependable, repeatable annotation and retraining workflow for large microscopy images.

The current scope includes multi-class YOLO object detection and a mask-first instance-segmentation workflow. The two tasks share project, conversion, tiling, device, split, and training orchestration while retaining separate annotation geometry and merge logic.

The intended workflow is:

1. Select a model or model project.
2. Load an image and, optionally, its existing YOLO label file.
3. Configure prediction settings and run whole-image or tiled inference.
4. Review and edit boxes in napari.
5. Save the image/label pair into the independently selected project's persistent annotation pool.
6. Repeat for more images without losing previous annotations.
7. Preview a deterministic train/validation split.
8. Retrain into a timestamped, reproducible run directory.
9. Inspect the result and explicitly choose whether to use the new `best.pt`.

## 2. Current implementation: useful foundations and important gaps

The existing implementation already has a small usable loop and writes five-column YOLO detection rows (`class x_center y_center width height`). It also runs training in a background thread. Those pieces can be retained behind clearer interfaces.

The main limitations observed in the current repository are:

- `_widget.py` combines UI construction, file management, image conversion, inference, annotation conversion, dataset creation, and training orchestration. This makes state and error handling difficult to extend safely.
- A model directory silently loads the alphabetically first `.pt` file. Once several retrained models exist, this can select the wrong model.
- Prediction exposes no confidence threshold, IoU/NMS threshold, device, image size, maximum detections, or tiling controls. It also discards predicted class IDs and confidence scores.
- Prediction calls Ultralytics directly on the entire array. Large images therefore rely on implicit resize/letterbox behavior and lose small-object detail.
- Prediction is synchronous in the UI thread, whereas only retraining uses a worker thread.
- The image conversion clips all non-8-bit values to `0..255`. This can destroy contrast in float and 16-bit microscopy images.
- `Add correction` requires exactly one image layer, identifies the sample from a mutable napari layer name, and always creates a timestamped duplicate. It cannot reliably update an existing sample.
- Corrections are stored in one flat directory. Image and label stems happen to match, but provenance, revision policy, validation, and dataset membership are not represented.
- A training configuration is rewritten whenever a correction is saved, and `image_size` is set to the largest source-image dimension. For large images this can request an impractical training size.
- The destination path is captured during `Add correction`, not at the moment retraining starts, so the displayed value and the value used can diverge.
- The current split alternates sorted files into train and validation, always producing a 50/50 split. It is not shuffled, configurable, stable across incremental updates, or grouped by source image.
- A future tiled dataset would leak nearly identical neighboring tiles across train and validation unless splitting happens at source-image level.
- Training artifacts are deleted after `best.pt` is copied. This removes metrics, arguments, plots, logs, and evidence needed to compare or reproduce models.
- Invalid or incomplete configuration files silently fall back to defaults. Dataset issues such as missing pairs, malformed rows, invalid class IDs, and out-of-range coordinates are not reported before training.
- Training has no progress, cancellation, elapsed time, estimated completion, or structured run summary.
- Tests cover a few helpers and basic widget paths, but not coordinate round trips, label import, incremental datasets, deterministic splitting, tiling, seam merging, device selection, or failure recovery.

## 3. Recommended project and data model

### 3.1 Related segmentation repository and consolidation strategy

There is already a useful experimental base in Simon Leclerc's public [napari-cci-yolo-segmentation repository](https://github.com/leclercsimon74/napari-cci-yolo-segmentation). The migration baseline is pinned to commit [`b0b14ca1048449008495e7e8e972e3c13479668c`](https://github.com/leclercsimon74/napari-cci-yolo-segmentation/tree/b0b14ca1048449008495e7e8e972e3c13479668c). It is not distributed on PyPI and is intended to remain on the personal GitHub account, then be discontinued once the useful work has been consolidated here. This repository should therefore be treated as a **migration source and design reference**, not as a runtime dependency, Git submodule, or separately maintained companion plugin.

The segmentation repository is MIT-licensed. If substantial code is copied or adapted, retain its copyright and MIT notice as required, preserve use
ful commit provenance in the migration commit/PR, and document the source module in comments or release notes. After parity is reached and the relevant behavior is covered here, archive the personal repository as read-only and update its README to point users to this plugin. Do not delete it: its history is useful for provenance and regression investigation.

Repository review found several concrete pieces worth carrying forward:

- `yolo_tiling_segmentation.py` uses Dask `map_overlap` to run inference on overlapped chunks, with a core/chunk size of `tile_size - 2 × overlap`, reflected padding at the image boundary, trimming back to the original dimensions, and threaded computation.
- It assigns thread-safe, globally unique instance IDs and retains YOLO confidence per instance. When predicted masks overlap, the higher-confidence instance owns the pixel.
- It reconciles instance IDs at horizontal and vertical chunk seams using an equivalence table and union/find-style grouping, then relabels equivalent objects consistently.
- It filters every predicted instance to one connected component, selecting the component with the **largest bounding-box area** rather than the greatest pixel area. This is the required parity behavior for removing detached bbox/mask contamination. It also supports optional clearing of image-border objects.
- The GUI records tiling metadata on the napari layer and can display the tile grid, which is valuable for debugging and user trust.
- Its training pipeline pairs images and masks by stem, splits source pairs deterministically at 80/20 **before** tiling, produces fixed 1024-pixel training tiles without resizing, pads edge tiles, retains a controlled sample of empty tiles, and verifies that retraining produced a segmentation checkpoint.
- Its tests already contain seam-equivalence, chunk-size, component filtering, tiling, pairing, split, and segmentation-training fixtures that can guide equivalent tests here.

Reuse should be selective rather than a wholesale copy:

- Extract a task-neutral tile-window/chunk specification, padding policy, source-coordinate transform, progress metadata, and tile-grid visualization. Both detection and segmentation can share these contracts.
- Retain Dask as an optional execution backend for lazy/out-of-core image access and large label-array assembly. First benchmark it against a bounded sequential tile iterator: the segmentation implementation protects one YOLO model with a mutex, so model calls are effectively serialized even when Dask schedules multiple tasks. GPU inference may benefit more from controlled batching than threaded calls.
- Do not use segmentation's one-pixel boundary equivalence algorithm for bounding boxes. It relies on connected pixels and instance IDs; detection needs ownership regions plus class-aware NMS/containment in source coordinates.
- For future segmentation, port and harden the equivalence/union-find approach behind a `SegmentationMerger`. Validate ambiguous seam contacts, diagonal contacts, one-to-many matches, confidence conflicts, and objects wider than the overlap before declaring parity.
- Keep the current plan's stronger project store, manifests, grouped incremental splits, normalization policy, and immutable run snapshots. The experimental repository's useful algorithms should fit these contracts rather than bringing across its GUI/file-layout coupling or hard-coded settings.
- Move or rewrite tests before moving implementation. This gives an executable definition of the behavior being preserved and makes it easier to compare results from both repositories on the same fixtures.

Suggested consolidation sequence:

1. Record the segmentation repository commit hash used as the migration baseline.
2. Inventory functions as `reuse`, `adapt`, `rewrite`, or `retire`, including their tests and dependencies.
3. Extract/implement the shared tiling contracts here and verify bbox large-image inference first.
4. Add an optional Dask backend only where it measurably improves lazy loading, memory use, or label assembly.
5. Later port segmentation-specific mask generation and seam fusion behind the common contracts.
6. Run golden-image comparisons between the old and consolidated implementations, documenting intentional changes.
7. Publish a migration note, point the personal repository to this project, and archive it once required segmentation workflows have parity.

### 3.2 Treat the selected project directory as a persistent project

The **project root can be selected independently from the model path**. For convenience, the model's parent can still be proposed as the initial default, but the UI must expose a separate Project field and must not force that relationship. The selected project owns the growing annotation pool and project configuration. Retraining produces immutable snapshots; it must not rearrange or delete the canonical annotation pool.

When the user selects an empty directory, show what will be created and initialize it as a project by creating `project.yaml`, `models/`, `annotations/images/`, and the task-specific annotation folders. Initialization also atomically copies the packaged task-compatible starter checkpoint: `yolo26n.pt` for detection or `yolo26n-seg.pt` for segmentation. Initialization must be idempotent and atomic enough that a missing/corrupt package resource or other failure rolls back instead of leaving a valid-looking partial project. A non-empty directory without `project.yaml` must not be silently adopted: offer an explicit **Initialize here** action after checking for path conflicts.

Model selection remains independent after initialization. Both starter weights are distributed as package data, but only the correct task model is copied into each new project. A selected external checkpoint may still be loaded explicitly, while successful retraining promotes `best.pt` into `project_root/models/`. Prediction and retraining require a checkpoint whose Ultralytics task and class IDs match the project.

Suggested structure:

```text
my_model_project/
├── models/
│   ├── yolo26n.pt or yolo26n-seg.pt
│   └── current.pt                  # optional explicit copy or pointer policy
├── annotations/
│   ├── images/
│   │   ├── field_001.png
│   │   └── field_002.tif
│   ├── labels/
│   │   ├── field_001.txt
│   │   └── field_002.txt
│   └── manifest.jsonl
├── project.yaml
└── retrain_20260902_143015/
    ├── dataset/
    │   ├── images/train/
    │   ├── images/val/
    │   ├── labels/train/
    │   ├── labels/val/
    │   ├── dataset.yaml
    │   └── split_manifest.csv
    ├── training/
    │   ├── weights/best.pt
    │   ├── weights/last.pt
    │   └── ... Ultralytics metrics and logs ...
    ├── run.yaml
    └── README.txt
```

For backward compatibility, a directly selected `.../best.pt` can initially propose its parent as the project root. Do not search for and select the first `.pt`; show discovered models and require an explicit choice, with a remembered `current_model` in `project.yaml`.

### 3.3 Canonical annotation pair contract

Every saved sample must have an image and label with exactly the same stem:

```text
annotations/images/field_001.png
annotations/labels/field_001.txt
```

Each label row uses standard YOLO detection format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalized relative to the saved image, not the current viewport, pyramid level, tile, or model input. Values must be finite and in `[0, 1]`; width and height must be greater than zero. An empty `.txt` file is valid and means a reviewed negative image with no objects.

The plugin should write image and label through temporary files and rename them only after both writes succeed, so a crash cannot leave a half-written pair.

### 3.4 Sample identity and repeat corrections

The stable sample ID should come from the source filename when napari provides one, with a sanitized layer name only as a fallback. Preserve the original extension when YOLO supports it and the pixel representation is suitable. Avoid timestamping every filename: timestamps belong in metadata and run folders.

When saving a sample whose stem already exists, **overwrite/update the canonical pair**. Write the new image and label atomically so both represent the same revision; do not create `field_001__2` or a timestamped duplicate. The UI should say `Update annotation` when the stem already exists and show a concise confirmation/status message so overwriting is never surprising.

Keep an append-only manifest entry for every save/update, including sample ID, original path if available, dimensions, dtype/channel conversion, content checksum, box count, class counts, model used for initial prediction, thresholds, tile settings, timestamp, and plugin version. This provides auditability without putting timestamps in canonical filenames.

The append-only manifest records that an overwrite occurred, but the old image/label content is not retained by default. Repository or external backup can provide full file history if required. Training uses only the current canonical pair so repeat edits do not accidentally overweight one image.

### 3.5 Destination and retrain run contract

Resolve paths when **Retrain** is clicked:

```text
run_parent = Destination Path if set, otherwise project_root
run_root = run_parent / "retrain_YYYYMMDD_HHMMSS"
```

Even when a custom destination is selected, `Add/Update annotation` should continue adding canonical data to `project_root/annotations`. The run receives a snapshot of that pool. If the destination already ends in a name like `retrain_*`, do not write directly into it; always create a new timestamped child to prevent accidental overwrite.

Each run should keep:

- The exact train/validation snapshot or a manifest with immutable checksums.
- The generated `dataset.yaml`.
- Effective prediction, tiling, split, augmentation, and training configuration.
- Input model path and checksum, package/plugin versions, random seed, device, start/end time, and status.
- `best.pt`, `last.pt`, metrics, curves, logs, and failure details.

Do not delete training traces by default. A separate, explicit cleanup action can remove large caches later.

## 4. Loading images and existing bounding boxes

Add a **Load labels** action with both explicit and automatic modes:

- Automatic: when the active image is `path/images/foo.png`, look first for `path/labels/foo.txt`, then `path/images/foo.txt`, and then a user-configured labels directory.
- Explicit: select one `.txt` file for the active image, or select a labels directory and match by stem.
- Project browsing: select an entry from the annotation pool and load its paired image and label together.

Before displaying imported labels:

1. Parse non-empty lines as exactly five fields for bbox mode.
2. Require an integer, non-negative class ID.
3. Require finite normalized values and positive width/height.
4. Verify that the class ID exists in the configured class map.
5. Convert YOLO `xywh` coordinates to napari rectangle vertices in `(row, column)` order using the active image dimensions.
6. Report skipped lines with filename and line number; do not silently repair ambiguous data.

Loading should replace, merge, or cancel if an editable annotation layer already contains boxes. Replace is the safe default. The layer should carry per-shape properties such as `class_id`, `class_name`, `confidence`, `source` (`prediction`, `import`, or `manual`), and optionally `tile_id`. Saving ignores confidence but preserves the class ID.

Round-trip behavior is an important invariant: loading a valid YOLO file and saving it without edits should reproduce the same boxes within a documented floating-point tolerance.

## 5. Image preparation and the 1024-pixel policy

Separate three concepts that are currently conflated:

- **Source image**: original coordinates used by napari and canonical labels.
- **Tile**: a crop from a large source image, expressed in source coordinates.
- **Model input**: a tile or small source image letterboxed to the configured YOLO `imgsz`.

Recommended default policy:

- If both source dimensions are at most `1024`, run one prediction. Let the YOLO preprocessing letterbox it to the configured inference size while preserving aspect ratio.
- If either dimension is greater than `1024`, use overlapping source-coordinate tiles of `1024 × 1024` by default.
- Edge tiles should be anchored to the image boundary where possible. If an image dimension is smaller than the tile size, pad only the model input and remember the valid, unpadded rectangle.
- Never save letterboxed or padded coordinates as canonical annotations. Map every detection back to original source-image coordinates first.

Make `1024` a configurable default, not a hidden constant. Validate it against model stride (typically use a multiple of 32), available memory, and training settings. Expose tile width/height as one square-size control initially, while the internal tile representation should support rectangular tiles later.

Microscopy data also needs an explicit intensity/channel conversion policy:

- RGB/RGBA 8-bit: use RGB and drop alpha with a visible warning or configured background behavior.
- Grayscale: repeat the selected normalized channel into RGB.
- `uint16` or float: use a declared normalization method such as dtype range, min/max, or percentile clipping; never simply clip values above 255.
- Multi-channel images: let the user map up to three channels to RGB, or choose a saved preset.
- Z/T data: initially require an explicit 2D plane/projection and record it; do not silently treat a volume as RGB.

The plugin provides a project-setup GUI for channel selection, optional Gaussian/median/mean/frequency-low-pass/white-top-hat pre-filtering with a shared radius, and image normalization, with a preview and a plain-language explanation of each option. Filtering is applied independently to selected source-channel planes before normalization and RGB stacking. The user is responsible for choosing a scientifically appropriate mapping.

The same deterministic conversion must be used when saving training images and when predicting them. Store the channel mapping, filter/radius, and normalization settings in `project.yaml` and every run manifest. These settings can be edited while the project contains no saved images, but become **immutable when the first annotation image is committed**. A separate intensity-inversion toggle is deliberately per image/plane and remains editable after this lock; store its value in each sample's audit conversion metadata and never reapply it to an already canonical review image. On every later import/save, validate compatibility and refuse data requiring a different locked conversion. To change filtering, normalization, or channel mapping, create a new project (a future explicit project-migration tool may reprocess all source images, but must never do so silently).

## 6. Tiled inference and merging

### 6.1 Tile generation

Use a deterministic grid with a configurable overlap, initially `20%` (for a 1024 tile, a stride of about 819 pixels). Always cover the complete source image. Record each tile as `(x0, y0, x1, y1)` plus its valid, non-padded extent.

The shared tile contract should be compatible with the segmentation repository's Dask convention (`core/chunk size = tile size - 2 × overlap`) even if the first bbox implementation uses a direct iterator. This makes a future Dask backend and segmentation migration possible without changing coordinates or saved metadata. Add a tile-grid napari overlay, adapted from that repository, as an optional diagnostic view.

Overlap must be large enough to include a typical object away from at least one seam. The UI can later warn when the estimated maximum object size approaches or exceeds the overlap.

Overlap greatly reduces partial objects but does not guarantee that every bbox fits inside one tile: depending on its position, an object wider than the overlap can still cross both neighboring tile boundaries. This is why training preparation still needs an explicit full-containment search and a small clipping tolerance.

### 6.2 Per-tile prediction

For every tile:

1. Extract valid pixels and pad/letterbox only for the model.
2. Run prediction with explicit `conf`, `iou`, `imgsz`, `device`, and `max_det` values.
3. Undo letterbox scaling.
4. Clip boxes to the valid tile area and discard predictions centered only in padding.
5. Translate tile-local coordinates into source-image coordinates.
6. Preserve class ID, confidence, and tile provenance.

Run tile inference in a worker with progress (`tile 12 / 48`) and cancellation. Start with a conservative batch size of one; allow tile batching after memory behavior is tested.

Execution backends should consume the same tile plan:

- **Direct/batched iterator**: recommended initial bbox backend; simple progress/cancellation and predictable GPU memory.
- **Dask local backend**: optional for lazy or out-of-core arrays and CPU-side preparation. Adapt the proven `map_overlap`/padding approach from the segmentation repository, but do not assume threaded scheduling parallelizes a single locked YOLO model.
- **Future Dask distributed backend**: out of initial scope; only consider it with explicit model-per-worker resource management.

### 6.3 Cross-tile duplicate handling

There are two different overlap problems and they should not share an ambiguous control:

- **Model NMS IoU**: passed to YOLO within each prediction call. A lower threshold suppresses more overlapping predictions.
- **Tile merge IoU**: applied in source coordinates after predictions from all tiles are collected.

Recommended first implementation:

1. Define a valid **ownership region** for each tile (roughly its non-overlapping core, with outer image borders owned by edge tiles).
2. Prefer detections whose center lies in the tile's ownership region. This prevents most seam duplicates without suppressing two genuinely adjacent objects.
3. Apply class-aware global greedy NMS to the retained source-coordinate boxes, sorted by confidence.
4. Optionally remove near-contained lower-confidence duplicates with a separate containment ratio rule.

Suggested starting defaults are confidence `0.25`, model NMS IoU `0.45`, tile overlap `20%`, and tile merge IoU `0.50`, but all must be visible and persisted. They are starting points, not domain-specific truths.

Weighted Box Fusion can be evaluated later: it may improve localization when several tiles see the same object, but it can incorrectly blend two close biological objects. Keep the merge strategy behind an interface (`nms`, later `weighted_fusion` or `soft_nms`) and validate it on seam-focused fixtures before exposing alternatives.

Important edge cases include objects larger than a tile, objects centered exactly on an ownership boundary, heavily overlapping true objects, tiny boxes, padded edge tiles, and different predictions for the same object from multiple tiles.

### 6.4 Tiled training dataset

Keep full source images and source-coordinate labels as canonical data. Generate training tiles only inside each immutable retrain snapshot. This allows tile parameters to change without corrupting or duplicating annotations.

For images over the configured training size:

- Generate deterministic overlapping tiles.
- For each object, first choose an overlapping tile that contains the complete bbox. If several do, prefer the tile whose ownership/core region contains the bbox center and then the tile with the greatest margin around the box. This makes overlap do most of the work and keeps each object exactly once.
- If no tile fully contains the bbox, choose the tile retaining the largest area. Accept tile-induced clipping only when at least `90%` of the original bbox area remains and no clipped axis loses more than `min(10 pixels, 10% of that bbox dimension)`. Make these advanced thresholds configurable, record them, and report every clipped or rejected object.
- Do not apply the 10-pixel value as a minimum object size: genuine objects smaller than 10 pixels may be valid. It is a maximum tolerated **clipping loss**. Boxes already touching the original source-image border are not penalized for pixels outside the source; only additional loss introduced by tiling is measured.
- If an object is too large to satisfy the rule even with overlap, exclude it from derived tiles with a prominent preparation warning rather than training on a severely truncated target. The user can increase tile size/overlap or explicitly accept a looser threshold.
- Include reviewed negative tiles according to a configurable ratio; do not silently flood the dataset with background.
- Name derived image and label pairs identically, for example `field_001__x000819_y000000.png` and `.txt`.
- Record source sample, tile bounds, transforms, and label decisions in `tile_manifest.csv`.

Critically, choose train/validation membership **before tiling**, at the source-image or acquisition-group level. All tiles from one source image—and ideally all fields from the same specimen/well/patient when that metadata is available—must remain in the same split.

## 7. Dataset splitting and validation

Use a configurable default split such as `80% train / 20% validation`, a fixed random seed, and group-aware assignment. The safe built-in grouping unit is the source image. Optionally allow the user to select a manifest metadata field such as patient, plate, well, acquisition, or experiment as the higher-level grouping key. Supplying correct grouping metadata is the user's responsibility; if none is supplied, state clearly that the plugin can prevent tile-level leakage but cannot detect biological/acquisition-level leakage. For single-class data, balance approximately by positive/negative status and box-count bins where dataset size permits. Multi-class stratification can be added without changing the split API.

Incremental projects need stable membership. Store assignments in the project manifest and keep existing samples in their prior split; assign only new samples while moving toward the requested ratio. Provide an explicit **Regenerate split** action when the user wants a new seeded split.

Before retraining, perform a dry-run validation and show:

- Total paired samples, reviewed positives, reviewed negatives, and total boxes.
- Train/validation source-image counts, derived tile counts, and box counts.
- Missing images or labels, duplicate stems, unreadable images, malformed rows, invalid class IDs, out-of-range boxes, zero-area boxes, and checksum duplicates.
- Warnings about too few validation images, extreme class imbalance, or possible group leakage.

For one independent source group, a meaningful validation split is impossible. Allow an explicitly acknowledged **train-only exploratory run**, but present a large warning explaining that model quality and generalization cannot be estimated. Require a dedicated `Train without independent validation` checkbox plus a confirmation at launch; do not make this a remembered preference. Mark the run metadata as `validation_mode: none`, disable validation-derived claims/metrics and any behavior that depends on a trustworthy validation set, and label its output as exploratory. Never manufacture validation by splitting tiles from the same source group.

`dataset.yaml` should use robust absolute paths or paths known to resolve relative to the YAML file, and should derive `names` from the project class map rather than hard-coding `0: LABEL`.

## 8. Proposed GUI and UX

A compact staged UI is preferable to a long row of buttons. Use collapsible sections and keep advanced values visible but unobtrusive.

```text
┌─ Project / Model ─────────────────────────────────────────┐
│ Project:  .../my_project        [Browse] [Initialize]    │
│ Model:    models/current.pt                    [Choose]   │
│ Status: Loaded · YOLO detect · 1 class · CUDA:0 (8 GB)   │
└────────────────────────────────────────────────────────────┘

┌─ Image and labels ────────────────────────────────────────┐
│ Active image: field_001.tif · 4096×3072 · uint16 · RGB   │
│ Labels: field_001.txt · 37 boxes              [Load...]   │
│                                               [New layer] │
└────────────────────────────────────────────────────────────┘

┌─ Predict ─────────────────────────────────────────────────┐
│ Device [Auto ▼]  Conf [0.25]  Model IoU [0.45]           │
│ Tile [1024]  Overlap [20%]  Merge IoU [0.50]             │
│ Max detections [300]                  [Predict] [Cancel]  │
│ 12 / 48 tiles · 00:08 elapsed                           │
└────────────────────────────────────────────────────────────┘

┌─ Annotation ──────────────────────────────────────────────┐
│ Current class: 0 — LABEL [▼] · 41 boxes · 4 edited       │
│ [Add/Update annotation]  Last saved 14:22:31             │
└────────────────────────────────────────────────────────────┘

┌─ Retrain ─────────────────────────────────────────────────┐
│ Data: 28 pairs · train 22 / val 6 · 914 boxes            │
│ Output parent: .../my_model_project            [Browse]   │
│ Epochs [100] Batch [Auto] Patience [30] Seed [42]        │
│ Image/tile size [1024] Device [Auto ▼]                   │
│ [Validate dataset] [Preview split] [Retrain] [Cancel]    │
│ Epoch 34/100 · mAP50 0.82 · 06:12 elapsed                │
└────────────────────────────────────────────────────────────┘
```

Essential live labels:

- Runtime device: `CPU`, `CUDA:0`, or `MPS`, with GPU name and memory when available.
- Loaded model path, task (`detect`), model class names, and input/default image size if discoverable.
- Active image dimensions, dtype, selected channels, normalization, and direct/tiled prediction mode.
- Number of predicted/imported/current boxes and count by class.
- Annotation pool: paired samples, positive/negative samples, missing or invalid pairs, and unsaved-edit indicator.
- Retrain preview: source counts and box counts per split, tile counts, output run path, effective batch/device.
- Worker state: current step, progress, elapsed time, cancellation state, and a concise log/details panel.

Interaction rules:

- Disable actions whose prerequisites are not met and explain why in a tooltip/status label.
- Track dirty edits on the shapes layer. Warn before prediction, label loading, image change, or plugin close would discard edits.
- Prediction should create or refresh a dedicated editable annotation layer without destroying unrelated shapes layers.
- Keep dialogs for actionable errors; send normal success and progress messages to a status panel instead of modal popups.
- Validate numeric fields immediately and show their meaning in tooltips, especially the two IoU thresholds.
- Read destination and all parameters at action time, then freeze them into the worker/run configuration.
- During empty-project setup, preview channel mapping and normalization. Lock both controls after the first annotation save and explain that a new project is required to change them.
- Show the optional split-group metadata field and an explicit warning when only source-image grouping is available.
- Gate train-only mode behind an unchecked confirmation control and keep a prominent `NO INDEPENDENT VALIDATION` banner visible throughout the run and result summary.
- On successful retraining, offer **Keep current model**, **Load new best model**, and **Open run folder**. Do not silently switch models.

## 9. Suggested internal architecture

Split responsibilities before adding more controls:

```text
Widget / view-model
    ├── ProjectStore
    │     project config, sample pairs, manifest, model lineage
    ├── AnnotationIO
    │     YOLO parse/validate/write, napari coordinate conversion
    ├── ImageAdapter
    │     dimensionality, channel mapping, intensity normalization
    ├── TilePlanner
    │     deterministic tile windows and ownership regions
    ├── InferenceService
    │     YOLO call, device/settings, coordinate restoration
    ├── DetectionMerger
    │     class-aware global NMS and containment handling
    ├── DatasetBuilder
    │     grouped split, derived tiles, YAML and validation report
    └── TrainingService
          worker lifecycle, progress, cancellation, run artifacts
```

Use small typed records for `Detection`, `TileWindow`, `SampleRecord`, `SplitAssignment`, `PredictionConfig`, and `TrainingConfig`. A detection should always state which coordinate space it occupies; this avoids subtle tile/letterbox/source-coordinate bugs.

Prefer one shared settings schema with versioning and validation over separate ad hoc JSON readers in the widget and worker. Unknown or invalid values should produce a clear error or migration message rather than silently reverting.

Keep Ultralytics-specific calls behind an adapter. This will make upgrades testable and later allow a detection model and a segmentation model to share orchestration without sharing annotation geometry code.

## 10. Multi-class readiness and later segmentation

Projects begin with one configured class for the simplest workflow, but detection is multi-class throughout:

- Store a project class map such as `{0: "LABEL"}` and load names from the model when available.
- Put `class_id` and `class_name` in napari shape properties for every box.
- Use stable class colors and a class selector for newly drawn boxes.
- Make validation and NMS class-aware.
- Write all dataset names from the project class map and reject unknown IDs.
- Avoid function signatures that accept a hard-coded class `0` or `[(0, "LABEL")]`.

For future YOLO segmentation, use [napari-cci-yolo-segmentation](https://github.com/leclercsimon74/napari-cci-yolo-segmentation) as the migration baseline rather than starting again. Define an `AnnotationGeometry` boundary: bbox uses `xywh` rows and rectangle Shapes layers; segmentation is edited and persisted as dense instance masks in napari Labels layers. YOLO polygon rows are an **export/training adapter only** and are never the primary user-facing editor. Project storage, pairing, device reporting, splitting, Dask-capable tiling provenance, workers, and run metadata can be shared. Do not mix bbox and segmentation annotations in one canonical label file or napari layer. Model task compatibility must be checked when a model is loaded.

The old repository's Dask label fusion belongs specifically on the segmentation side of this boundary: its overlapped label chunks, unique IDs, seam-equivalence graph, relabeling, and confidence metadata should become a tested `SegmentationMerger`. Bounding-box fusion remains a separate `DetectionMerger`. Both can share tile bounds and provenance, but trying to force masks and boxes through one merge algorithm would make both less reliable.

### 10.1 Proposed segmentation annotation contract

The primary segmentation representation should be a dense **instance-ID image**, not one binary mask per class and not visible polygons:

- Pixel value `0` is background; every non-zero value is a project/sample-local instance ID.
- A companion instance table maps `instance_id → class_id, class_name, confidence, source, review status, bbox, area`, plus optional tile/model provenance. Class is metadata on an instance, not encoded directly in the pixel value.
- One napari Labels layer can therefore contain all classes and all instances. Use the project class color as the base display color, with an optional deterministic brightness variation per instance so touching instances remain visible.
- Brush/fill/erase edits operate on instance IDs. The GUI needs **Current class**, **New instance**, **Select instance**, **Delete instance**, **Split disconnected components**, and **Merge selected instances** actions. Repainting an existing ID preserves its class; creating a new ID uses the selected class.
- Saving writes a same-stem normalized RGB image, lossless integer instance mask, and instance metadata sidecar atomically. The exact on-disk mask type must be selected deliberately: `uint16` PNG is compact but limited to 65,535 IDs; `uint32` TIFF or a lossless compressed array avoids that ceiling. Never use a palette/color image as the canonical mask.
- Audit events record mask checksum, instance count, per-class instance/pixel counts, conversion settings, model and tiling settings, discarded components, and all automatic repairs. Empty masks are valid reviewed negatives.

Using `skimage.measure.label` on one binary mask per class is acceptable only as an import/convenience operation. It requires distinct same-class objects to be separated by at least one background pixel (and its diagonal behavior depends on connectivity); touching objects collapse into one instance and cannot be recovered automatically. The canonical instance-ID map avoids this limitation because adjacent objects may retain different integer IDs even with no background pixel between them. A semantic class map alone is therefore insufficient for YOLO instance-segmentation retraining.

### 10.2 Mask cleanup and bbox-contamination rule

Preserve the old repository's `keep_largest_component_per_label` semantics as the default compatibility filter:

1. Threshold each YOLO instance mask.
2. Label its connected components with four-connectivity (`connectivity=1`).
3. Compute each component's enclosing bbox area `(height × width)`.
4. Retain only the component with the largest bbox area and discard the others before it enters the shared label image.
5. Record how many pixels/components were removed and expose an optional debug overlay; never silently alter a manually edited canonical mask during save.

Apply this filter to raw per-detection model output before overlap composition, and make the same check available as an explicit repair/validation action after tile merging or manual editing. Do not automatically apply it to user corrections: a biologically valid instance may genuinely have disconnected parts. Validation should instead flag multi-component instances and offer **Keep largest by bbox**, **Split into instances**, or **Keep as-is**.

The bbox-area choice is intentionally retained for parity and contamination removal, but it can prefer a sparse elongated artifact over a compact object. Record both bbox area and pixel area so a later project option can compare `largest_bbox`, `largest_pixel_area`, or model/bbox-constrained selection without changing old-project behavior.

### 10.3 Multi-class overlap policy

Extend the old single-class merger without changing its geometric strategy:

- Carry `class_id` alongside every temporary/global instance ID.
- Only create seam equivalences between IDs of the same class. Different classes may touch but must never be unioned.
- Where masks overlap in a tile or halo, the higher-confidence instance owns the pixel, matching the old implementation. Use a deterministic tie-breaker (confidence, then stable tile/order ID) and audit ties.
- After union/find, assign one canonical ID per equivalence group, preserve its class, and aggregate confidence explicitly (initial parity: maximum member confidence).
- If one equivalence group somehow contains multiple classes, treat it as a merge error, show the relevant seam/tile IDs, and keep the instances separate.
- Keep instance ID and class ID as separate concepts throughout storage, display, export, and training. A label pixel value must never be assumed to equal a YOLO class ID.

### 10.4 Invisible polygon export boundary

Ultralytics instance-segmentation training requires polygon rows, even though users edit masks. Generate those polygons only inside an immutable retrain snapshot:

- For each instance ID, retrieve its class from the instance table, apply the selected component policy, trace the exterior contour, simplify it within a configured pixel tolerance, normalize coordinates, and write one YOLO segmentation row.
- Validate at least three distinct points, finite coordinates, in-range normalized values, non-zero polygon area, class existence, and contour/bbox agreement.
- Quantify raster-mask → polygon → raster round-trip IoU and reject or warn below a configured threshold. Record simplification tolerance and round-trip metrics in the tile manifest.
- Define behavior for holes and multiple contours explicitly. YOLO's simple polygon row cannot faithfully encode every topology; initial compatibility mode keeps the largest bbox component and exterior contour, with a warning when holes or discarded islands exist.
- Never overwrite the canonical mask with its simplified polygon reconstruction. Retraining output is derived data and remains reproducible from the run snapshot.

## 11. Implementation phases

### Phase 0 — Define contracts and protect current behavior

- Implement the resolved contracts for an independently selected/initialized project root, canonical folder names, overwrite semantics, empty labels, and run naming.
- Add empty-folder initialization and safe handling for a non-empty non-project directory.
- Introduce typed configuration and records without changing the visible workflow.
- Add coordinate and file-format characterization tests around existing behavior.
- Decide migration behavior for the current flat `corrections/` directory; recommended: offer a one-time non-destructive import into `annotations/`.
- Pin the exact segmentation-repository baseline commit, create the reuse/adapt/rewrite/retire inventory, and copy its relevant test fixtures or expected outputs with attribution.

**Exit criteria:** folder and metadata contracts are documented; existing correction pairs can be discovered and migrated without data loss.

### Phase 1 — Reliable annotation persistence and label loading

**Status: completed in the annotation-only `0.2.0` milestone.**

- Create `ProjectStore` and `AnnotationIO`.
- Save same-stem image/label pairs atomically in persistent `annotations/images` and `annotations/labels` folders.
- Add atomic overwrite/update behavior, empty label support, manifest audit entries, and dataset validation.
- Add automatic/explicit YOLO bbox loading and round-trip tests.
- Preserve class properties even while the UI exposes only class `0`.
- Extract current Z/T planes from multidimensional TIFF/OME-TIFF data, map up to three selected channels to RGB, and save converted `uint8` training images.
- Provide min/max, simple-max, percentile, Z-score, fixed-range, and integer-dtype normalization with an RGB preview.
- Lock channel mapping and normalization in `project.yaml` after the first converted pair is saved.
- Protect unsaved bbox edits during image/plane changes, label reload, and widget close.

**Exit criteria:** users can load, edit, update, close, and reload annotations with no coordinate drift or unwanted duplicates.

### Phase 2 — Prediction controls and responsive inference

**Status: completed as a prerequisite of tiled inference in `0.3.0`.**

- Add confidence, model IoU, image size, maximum detections, and device controls.
- Reuse the project's locked image conversion for inference so prediction sees the same pixels as training.
- Move inference off the UI thread; add progress, cancellation, and error reporting.
- Preserve class and confidence metadata on predicted shapes.

**Exit criteria:** prediction parameters are explicit and reproducible; the UI remains responsive; non-8-bit images are not silently clipped.

### Phase 3 — Large-image tiling and merge

**Status: core workflow completed in `0.3.0`.** The initial backend is the planned direct iterator. Dask benchmarking, lazy-array optimization, and tile-grid visualization remain optional follow-up work rather than correctness requirements for the first tiled release.

- Implement/test deterministic tile planning, padding metadata, and source-coordinate transforms.
- Preserve compatibility with the segmentation repository's overlapped-core convention; consider porting its tile-grid visualization after the core workflow is validated with real datasets.
- Add overlap and ownership regions.
- Implement class-aware cross-tile NMS with a separate merge IoU control.
- Add tile progress and seam-focused visual/test fixtures.
- Benchmark the direct iterator against an optional Dask local backend on representative large NumPy, Dask, and lazy napari images before adding a second backend.

**Exit criteria:** images larger than 1024 are fully covered, boxes appear in correct global positions, padding creates no detections, and duplicates at seams are consistently resolved.

### Phase 4 — Reproducible dataset building and retraining

**Status: core workflow completed in `0.4.0`.** Dataset creation uses a direct deterministic iterator. Rich epoch-metric plots, automatic batch fallback after out-of-memory failures, and editable model lineage remain follow-up work.

- Implement group-aware deterministic 80/20 splitting and stable incremental assignments.
- Generate derived training tiles inside a timestamped run snapshot.
- Create the required `images/train`, `images/val`, `labels/train`, and `labels/val` structure plus YAML and manifests.
- Validate and preview the split before launch.
- Persist all training outputs and provenance; read destination/settings at retrain time.
- Add training progress, cancellation, and explicit model-selection after success.

**Exit criteria:** a run can be reproduced from its folder, no source leaks across splits, all image/label pairs match by stem, and repeated annotation saves grow/update the same project safely.

### Phase 5 — Quality, multi-class UI, and model lineage

**Status: multi-class UI completed in `0.5.0`; annotation navigation and automatic best-model promotion added in `0.7.0`; remaining Phase 5 quality and lineage work is still planned.** Project class maps can be appended/renamed safely, bbox classes can be selected and reassigned, class colors and counts are visible, and the existing validation, tiling, dataset YAML, inference, save/reload, and retraining paths preserve all class IDs.

- Add dataset/project browser, filters, next/previous image, autosave preference, and annotation completion/review status. **Basic saved-pair browser and previous/next correction workflow completed in `0.7.0`; filters, autosave, and explicit review states remain.**
- Add model comparison summaries and explicit `current` model lineage.
- Expose multi-class selection, coloring, per-class counts, and class-map editing. **Completed in `0.5.0`.**
- Add optional hard-negative and uncertainty-driven review queues.

`0.7.0` also makes locked project normalization directly authoritative for inference, validates canonical training-image conversion provenance before dataset construction, highlights live out-of-bounds boxes, and atomically copies a successful run's `best.pt` to `models/<retrain-folder-name>.pt`.

`0.7.1` keeps review actions together with a dedicated **Save Corrections** button and begins the parameter-help pass with concise tooltips across image conversion, inference, crops, dataset splitting, and retraining.

**Exit criteria:** multi-class detection works without changing storage or training architecture, and users can compare successive retrain runs.

### Phase 6 — Movable fixed-size bbox failure crops

**Status: completed in `0.6.0`.**

`0.6.1` made manual-crop boundary clipping permissive: partial boxes remain labeled and no longer block saving, while automatic tiling retains its conservative clipping safeguards.

`0.6.2` adds live crop validation: boxes extending into synthetic padding receive a translucent red face without changing their class-colored edge, the status identifies their zero-based indices, and **Add Crop + Corrections** remains disabled until they are corrected or removed. Save-time validation remains as a final safeguard.

- Separate arbitrary-size inference sources from canonical training samples.
- Add a movable, fixed 1024×1024 or 512×512 napari crop-selection rectangle centered on the current view.
- Snap accidental selection resizing back to the project size and clamp its movable origin to source pixels.
- Extract the locked RGB conversion without resizing and use constant value 114 only where the source image needs bottom/right padding.
- Translate class/confidence/source-aware bbox properties into local crop coordinates.
- For manually reviewed crops, clip and retain every meaningful bbox intersection regardless of retained fraction. Report boundary clipping without blocking the save; omit only remnants below 2 pixels on either axis. Keep the stricter 90%/10-pixel policy exclusively for automatic dataset tiling.
- Reject boxes drawn into synthetic padding.
- Save deterministic same-stem image/YOLO pairs using source sample ID plus crop origin and size, so saving the same crop updates it.
- Record crop bounds, valid extent, padding, source path, Z/T plane, normalization, model, and inference settings in the audit event.
- Lock `training_patch.size` and padding policy after the first canonical save, drive retraining size from that contract, and reject mismatched images during project/dataset validation.
- Keep crops from the same audited source in one train/validation group.

**Exit criteria:** a user can infer an arbitrary-size image, move a fixed crop over a failure, correct local boxes, save an exact-size YOLO pair, repeat at other locations, retrain, and restart the loop without coordinate drift or source leakage.

### Phase 7 — Segmentation adapter

**Status: in progress. Baseline pinned to segmentation repository commit `b0b14ca1048449008495e7e8e972e3c13479668c`. Phases 7A–7E now have usable implementations: project contracts, mask-first editing, Dask tiling/fusion, movable mask crops, and mask-derived YOLO segmentation retraining. Phase 7F and real-data/golden-model testing remain. Backward compatibility with projects from the personal plugin is intentionally out of scope because it did not define project storage. Existing segmentation model weights may be reused when their task and class IDs match.**

Recommended decisions to confirm before implementation:

| Question | Recommended default | Reason |
|---|---|---|
| User-facing geometry | One napari Labels instance map | Matches the original mask-first workflow; no visible polygons. |
| Multi-class representation | Instance ID pixels plus a separate instance→class table | Supports many classes and touching instances in one layer without encoding class into pixel IDs. |
| Binary class-mask conversion | Optional import/split tool using `skimage.measure.label(connectivity=1)` | Convenient when objects are separated, but touching same-class objects cannot be recovered. |
| Detached mask contamination | Keep the connected component with the largest bbox area for model output | Exact compatibility with the original repository; report every removal. |
| Manual disconnected instance | Warn and offer keep/split/largest; do not silently filter | User corrections must not be destroyed by an automatic prediction cleanup rule. |
| Initial tile fusion | Original one-pixel seam equivalence plus union/find, extended to require equal class | Preserves the known strategy while preventing cross-class fusion. |
| Training representation | Derive YOLO polygons invisibly inside each run | Keeps masks canonical and the GUI simple while remaining compatible with Ultralytics. |

#### Phase 7A — Contracts, migration inventory, and fixtures

**Implementation status: contract implemented; fixture migration remains in progress.** A project is initialized as immutable task `detect` or `segment`. Canonical segmentation storage is RGB `uint8` PNG + lossless 2D `uint32` TIFF + versioned JSON instance metadata under `annotations/images`, `annotations/masks`, and `annotations/instances`. Updates use temporary files, rollback backups, and an append-only audit event. The instance schema stores class ID/name, confidence, status, source, bbox, pixel area, and lineage. Detection/segmentation weight mismatch is rejected by separate adapters. Contract, round-trip, component-policy, overlap, split, adapter, crop, polygon-export, and mocked-training fixtures are present; pinned real-model golden fixtures still need to be migrated.

Pinned-repository inventory:

| Original part | Decision for this plugin |
|---|---|
| `yolo_tiling_segmentation.py` | **Adapt** in Phase 7C: reflected Dask halos, unique temporary IDs, confidence ownership, seam equivalence, and union/find; add class-aware fusion and deterministic final IDs. |
| `_gui.py` | **Rewrite**: retain the mask-first interaction, but use the project/task contract, one Labels layer, instance metadata, shared conversion, and current review UI. |
| `_segmentation_training.py` | **Adapt** in Phase 7E: retain relevant mask tiling/export behavior inside immutable run snapshots; replace path/config handling with the current dataset builder and audit contracts. |
| Old segmentation tests/fixtures | **Reuse as attributed golden inputs/outputs** for seam fusion and largest-bbox filtering; wrap them in the current task-aware APIs. |
| Old standalone configuration and implicit folder workflow | **Retire**: there is no backward-compatible project format to preserve. |

- Inventory the pinned repository modules and tests as `reuse`, `adapt`, `rewrite`, or `retire`. At minimum cover `yolo_tiling_segmentation.py`, `_gui.py`, `_segmentation_training.py`, and both segmentation test modules.
- Add a project task contract: initially one project is either `detect` or `segment`, fixed after the first canonical annotation. Do not put bbox `.txt` files and masks under an ambiguous shared annotation type.
- Finalize canonical segmentation paths, for example `annotations/images/`, `annotations/masks/`, and `annotations/instances/`, with same-stem triples and atomic update/rollback.
- Decide the canonical integer mask format after testing napari, Pillow/tifffile, Windows, and maximum-instance behavior. Prefer a lossless `uint32` format unless interoperability requirements justify a guarded `uint16` limit.
- Define and version the instance metadata schema, including class, confidence, review status, bbox, area, source, and lineage after merge/split.
- Copy the old seam, component-filtering, tile, and training fixtures with attribution. Add golden outputs generated by the pinned commit before changing algorithms.
- Add task-aware model inspection and reject detection weights in a segmentation project (and vice versa) before prediction or retraining.
- Ignore backward compatibility

**Exit criteria:** a versioned mask/instance contract exists; old fixtures run locally; task mismatch and unsupported mask dtype fail clearly; no GUI or model code needs to guess what a pixel value means.

#### Phase 7B — Single-image segmentation and mask-first editing

**Implementation status: first usable path complete.** The Ultralytics segmentation adapter returns source-resolution binary masks and bbox/class/confidence metadata, prediction composition uses stable instance IDs and confidence ownership, and automatic model cleanup follows largest connected-component bbox area. The GUI provides one Labels layer with new/delete/class/merge/split/keep-largest actions, counts, selected-instance details, live validation, atomic save/review/reload, and task-aware model rejection. Manual disconnected instances are reported rather than silently changed. Direct canonical save remains limited to the locked 512/1024 size; arbitrary-size sources now use the Phase 7D movable mask crop.

- Add an Ultralytics segmentation adapter returning per-instance `{mask, bbox, class_id, confidence}` records in source-image coordinates.
- Reuse the locked project RGB conversion exactly as bbox inference does. Request full-resolution/retina masks where supported and resize binary masks with nearest-neighbor only.
- Before composition, apply the pinned largest-connected-component-by-bbox filter to every predicted mask and record removed contamination.
- Compose predictions into one `uint32` instance map using confidence ownership for overlapping pixels, stable instance IDs, and a companion instance table.
- Show one editable napari Labels layer, not polygons. Provide class-aware instance selection, new/delete/merge/split actions, per-class counts, and confidence/details for the selected instance.
- Add live validation for unknown instance IDs, missing metadata, invalid classes, disconnected instances, empty metadata entries, out-of-image shape mismatch, and mask values exceeding the chosen storage type.
- Save, review, navigate, overwrite, audit, close, and reload segmentation triples without instance-ID or class drift.

**Exit criteria:** a small image can be predicted, corrected entirely as a Labels layer, saved, closed, and reloaded with identical pixels, instance IDs, classes, and metadata. No polygon is shown to the user.

#### Phase 7C — Large-image Dask tiling and old-strategy fusion

**Implementation status: baseline path implemented; pinned real-model golden comparison remains.** Large segmentation inputs are padded to a deterministic core grid and evaluated with Dask `map_overlap`, reflected halos, serialized model access, deterministic per-tile temporary ID ranges, per-tile confidence ownership, and largest-bbox component cleanup. One-pixel seams feed a same-class deterministic union/find relabeler; cross-class contacts and one-to-many ambiguity are reported rather than merged. The Labels layer stores full tiling provenance, optional source-border clearing is available, and a cyan core-grid debug overlay can be enabled from the inference UI. Deterministic synthetic parity fixtures pass; comparison against the pinned repository's real-model golden images remains before declaring the full exit criterion complete. Hardened seam metrics remain a future comparison mode rather than silently replacing parity behavior.

Port the original strategy as the first parity implementation:

1. Compute `chunk_size = tile_size - 2 × overlap` and reject non-positive cores.
2. Pad the source to a complete core grid; use reflected image halos for inference and constant-zero padding for label-equivalence passes.
3. Run per-chunk segmentation through Dask `map_overlap(..., depth=overlap, boundary="reflect", trim=True)`. Keep the model mutex initially because the old code serializes model calls safely; benchmark controlled GPU batching later.
4. Allocate globally unique temporary IDs thread-safely and store class/confidence metadata for every ID.
5. Apply largest-component-by-bbox filtering per predicted mask, then resolve overlapping pixels by confidence.
6. Scan one-pixel horizontal and vertical core seams, collect **same-class** neighboring ID pairs, group transitive equivalences with union/find, and relabel every group to one deterministic canonical ID.
7. Remap class/confidence/provenance tables to canonical IDs, trim to the original image extent, and optionally clear image-border instances.

Keep the tile-grid/debug overlay and store `tile_size`, overlap, core size, padding, temporary-to-final ID mapping, equivalence pairs, class conflicts, and component removals in layer/run provenance. Prediction and merge may remain separate debug buttons during parity work; combine them only after golden comparisons pass.

The one-pixel seam rule can incorrectly join two distinct touching objects or fail when fragments do not touch exactly after thresholding. Preserve it for baseline parity, but flag ambiguous one-to-many and many-to-one seam groups. A later hardened mode may require a minimum seam-contact length, halo IoU, bbox compatibility, or confidence agreement; it must be compared against the pinned output rather than silently replacing it.

**Exit criteria:** golden large-image fixtures match the old repository for single-class output, multi-class fragments merge only within class, IDs are deterministic after canonical relabeling, and memory remains bounded for images much larger than one tile.

#### Phase 7D — Movable mask crops and correction accumulation

**Implementation status: implemented; broad real-image testing remains.** The shared movable 512/1024 selector now crops normalized RGB and the current instance map together. Crop-local IDs are deterministically compacted while retaining class and source-ID lineage. Partial boundary objects are retained and reported; disconnected IDs or pixels painted into synthetic padding block saving. All mask editing actions target the crop layer during a crop session. Atomic same-stem triples, deterministic overwrite, review refresh, and source return are wired into the existing workflow.

- Reuse the fixed 512/1024 movable crop workflow, but crop the normalized RGB image plus the instance map and instance table.
- Translate/reindex crop-local instance IDs deterministically while retaining source instance lineage and class.
- Define boundary policy separately from bbox: a cropped mask may be intentionally partial, but disconnected slivers and instances entering synthetic padding must be highlighted. Do not infer class from pixel ID.
- Save same-stem image/mask/instance triples into the persistent annotation pool. Saving the same source/crop updates it rather than creating a timestamp duplicate.
- Extend **Review saved annotations** to task-aware bbox or mask correction while keeping one-click save and previous/next navigation.

**Exit criteria:** users can move a crop over a segmentation failure, paint/erase/split/merge instances, assign classes, save it, return to the source, and repeat without mask-coordinate or metadata drift.

#### Phase 7E — Mask-to-YOLO dataset construction and retraining

**Implementation status: first usable path implemented; real Ultralytics CPU smoke and pinned-repository golden comparison remain.** A task-aware builder validates canonical triples, uses the stable audited source-group split, preserves empty reviewed masks, and copies paired RGB/mask/instance artifacts into the immutable run. Exterior polygons are derived per instance only inside that run and must pass a `0.90` raster round-trip IoU threshold. The export manifest records per-class counts, checksums, discarded-component count, and IoU summaries. Training rejects a non-segmentation checkpoint, records task/class lineage, preserves Ultralytics outputs, and promotes `best.pt` to the project model folder. Synthetic multi-class, empty-mask, polygon, grouping, and widget crop fixtures are present.

- Split by audited source group before deriving tiles, preserving the existing stable train/validation contract and reviewed-negative masks.
- Tile canonical image/mask pairs only inside the immutable run. Keep related masks, instance metadata, and image tiles paired by stem.
- Apply the explicit component policy and invisible polygon exporter from §10.4. Store derived polygon labels, mask tiles for inspection, round-trip IoU, discarded components, and per-class counts in the run snapshot.
- Reject cross-class instance metadata errors, invalid polygons, topology loss beyond threshold, masks in synthetic padding, and train/validation source leakage before starting Ultralytics.
- Train only a segmentation checkpoint, retain all artifacts, promote `best.pt` into `project/models/<run-name>.pt`, and record the task/class map in model lineage.
- Add a CPU smoke fixture plus mocked multi-class and empty-mask runs. Compare a small retrain snapshot with the pinned repository where contracts overlap.

**Exit criteria:** mask-only user annotations produce a standard YOLO segmentation dataset without manual polygons; a retrained segmentation model can be loaded and used to restart the correction loop reproducibly.

#### Phase 7F — Consolidation and retirement of the personal plugin

- Run golden-image comparisons covering small prediction, large tiled prediction, seam fusion, largest-bbox filtering, optional border clearing, crop save, and retraining export.
- Document intentional differences, especially class-aware merging, canonical mask metadata, deterministic final IDs, and stricter project/audit behavior.
- Publish migration instructions for old `images/` + `masks/` datasets and checkpoints. Import non-destructively into a new segmentation project; never rewrite the old dataset in place.
- Update the personal repository README to point here, tag its final migration baseline, and archive it read-only after the required workflows reach parity. Keep it available for provenance and regression investigation.

**Exit criteria:** this plugin covers the old supported workflow plus multi-class instance metadata; migration is documented and tested; the personal repository can be discontinued without losing reproducibility.

## 12. Testing and acceptance strategy

### Unit tests

- YOLO `xywh ↔ xyxy ↔ napari (row, column)` conversions, including non-square images and boundaries.
- Label parser errors, empty labels, multi-class IDs, finite/range checks, and stable formatting.
- Image/label stem matching, duplicate detection, update policy, atomic-save failure recovery, and manifest entries.
- Tile coverage for odd dimensions, small images, edge anchoring, padding, ownership regions, and deterministic ordering.
- Local-to-source coordinate mapping through crop, pad, and letterbox transforms.
- Class-aware NMS, equal confidence, containment, boundary centers, and close distinct objects.
- Backend parity for tile windows and source coordinates between direct and Dask execution.
- Box clipping/retained-area rules for generated training tiles.
- Stable grouped splitting, seed reproducibility, incremental additions, positive/negative balance, and no group leakage.
- Device/config validation and config schema migration.
- Instance-map and instance-table round trips, including touching same-class and different-class instances, empty masks, large IDs, and missing/orphan metadata.
- Largest-component-by-bbox selection where bbox area and pixel area choose different components, plus explicit verification that manual masks are never filtered silently.
- Mask overlap ownership by confidence, deterministic ties, class-aware seam equivalence, transitive union/find groups, ID canonicalization, and metadata remapping.
- `skimage.label` import behavior for one-pixel gaps, direct/diagonal contacts, and configurable connectivity.
- Mask→polygon→mask round-trip IoU, simplification, holes, multiple contours, tiny/degenerate contours, and class-ID preservation.

### Integration tests

- Mock model outputs over multiple overlapping tiles and verify the final napari layer.
- Load a label file, edit boxes, save, reload, and compare geometry/class properties.
- Accumulate several corrections, update one, build a run, and verify every dataset pair and manifest checksum.
- Verify that custom destination creates `destination/retrain_<timestamp>/...` while canonical annotations remain under the model project.
- Simulate prediction/training exception and cancellation; confirm UI controls recover and partial runs are marked failed/cancelled rather than presented as successful.
- Run a tiny CPU-only training smoke test separately from normal fast tests when dependencies permit.
- Compare direct single-image segmentation and Dask-tiled output against golden masks from the pinned segmentation repository.
- Predict a multi-class large image, merge seams, edit touching instances in one Labels layer, save/reload, build a YOLO segmentation snapshot, and verify instance/class consistency at every boundary.
- Exercise ambiguous one-to-many and many-to-one seam contacts and confirm they are reported rather than silently cross-class merged.
- Import an old same-stem `images/` + `masks/` dataset non-destructively and verify that disconnected/touching-object warnings are preserved in the migration report.

### Manual validation set

Maintain several small, redistributable fixtures:

- Small RGB image below 1024.
- Wide and tall images with non-multiple-of-32 dimensions.
- 16-bit grayscale microscopy image.
- Large image with objects centered on tile seams and corners.
- Segmentation seam fixtures migrated from the personal repository, including one-to-one, one-to-many, diagonal, and ambiguous neighbor contacts.
- Two genuinely overlapping objects that must survive merge.
- Empty reviewed image.
- Multi-class labels, even before multi-class editing is exposed, to ensure the architecture does not erase class IDs.

## 13. Additional improvements worth including

- **Autosave/recovery:** keep a recoverable draft when napari or the model process fails, but require an explicit setting before automatic canonical saves.
- **Annotation status:** track `unreviewed`, `predicted`, `corrected`, and `verified`; train only on allowed statuses.
- **Negative examples:** make zero-box reviewed images first-class data and show their count.
- **Duplicate/data leakage detection:** use content hashes or perceptual hashes to warn about renamed duplicates across splits.
- **Quality checks:** flag unusually small/large boxes, boxes clipped to borders, high pairwise overlap, and sharp box-count changes after an edit.
- **Keyboard workflow:** shortcuts for predict, save/update, delete selected boxes, choose class, and move to the next project image.
- **Reproducibility:** pin/record Ultralytics and Torch versions, seed all relevant RNGs where possible, and record whether deterministic training was requested.
- **Resource safety:** estimate tile count and storage before a run; support automatic batch size only with a safe fallback after out-of-memory errors.
- **Privacy/provenance:** make storage of absolute original paths optional because run folders may be shared.
- **Export/import:** allow the canonical annotation pool to be used directly as a standard YOLO dataset without proprietary conversion.
- **Model compatibility:** verify that loaded weights are a detection model and that their class map agrees with the project before inference or retraining.
- **Metrics:** show precision, recall, mAP50, and mAP50-95 with plain-language caveats, plus per-class metrics later. Do not use training loss alone to choose a model.
- **Data evolution:** warn that repeatedly training only on newly corrected mistakes can cause forgetting; default retraining should use the complete approved annotation pool, with optional sampling weights introduced only after evaluation.

## 14. Resolved product decisions

These decisions are implementation requirements and should live in versioned project/run configuration rather than remain implicit UI behavior:

1. **Independent project root:** the project directory is selected independently from the model. Selecting an empty directory initializes the project structure and configuration. A non-empty directory requires an explicit, conflict-checked initialization action.
2. **Overwrite existing samples:** saving an existing stem atomically replaces its canonical image/label pair. The manifest records the update, but duplicate timestamped samples and built-in content history are not created.
3. **Project-wide image conversion:** project setup provides channel-selection, optional pre-filter/radius, normalization, and a per-image inversion toggle with a preview. The user is responsible for the scientific choice. Channel/filter/normalization settings lock after the first annotation image is saved; inversion deliberately remains editable and is audited per sample.
4. **Split metadata:** source image is the minimum grouping boundary. The user may nominate a metadata field for biological/acquisition grouping and is responsible for its correctness. The plugin warns when it cannot protect against higher-level leakage.
5. **Tile-clipping tolerance:** first select a tile containing the full bbox. Only when none exists may a bbox be clipped, and then it must retain at least 90% of its area while losing no more than `min(10 px, 10% of the original dimension)` on any clipped axis. The 10-pixel value limits lost content; it is not a minimum object size. Rejected objects are reported, and advanced users can explicitly change the threshold per run.
6. **Train-only mode:** retraining without an independent validation group is allowed, but only through an explicit per-run acknowledgement and a large warning. Such runs are marked exploratory and do not report validation-derived quality claims.
