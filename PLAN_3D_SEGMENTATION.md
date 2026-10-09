# Plan for 3D segmentation and bounding boxes

Add a **3D inference and assembly** collapsible section to the existing plugin layout. Apply the project's existing 2D model to every Z slice, store and display the resulting predictions over the source volume, then assemble them using configurable overlap criteria. Support segmentation projects with masks and detection projects with bounding boxes. Keep slice inference and assembly as separate actions so users can inspect predictions and rerun assembly without repeating model inference.

This is an implementation plan only. The first release should support spatial volumes with the same channel conversion as the 2D project, including branching shapes and multiple instances joining or separating between slices. Native 3D model training and time tracking are outside this scope.

## Current implementation and reuse

| Existing module | Reuse and required extension |
| --- | --- |
| `_widget.py` | Add another `CollapsibleSection` in the existing scroll area, sharing project, conversion, and model state. |
| `_image_adapter.py` | Reuse RGB channel mapping, filtering, inversion, and per-plane normalization. Add explicit volume axes and Z iteration instead of depending on the live viewer position. |
| `_yolo_segmentation.py` | Reuse the segmentation model and its prediction lock. The model still receives a source-resolution RGB plane. |
| `_yolo_inference.py` and `_tiled_inference.py` | Reuse the detection model, tile ownership, full-image box coordinates, and class-aware duplicate merging for detection projects. |
| `_segmentation_tiling.py` | Reuse halo tiling, class-aware seam fusion, and source-shape cropping for each slice. Its Dask computation currently materializes a full NumPy plane; it does not provide storage for a lazy volume. |
| `_instance_mask.py` | Reuse confidence ownership and instance bookkeeping. Add a configurable component cleanup policy because the current largest-component rule can discard valid branches. |
| `_segmentation_worker.py` and `_batch_inference.py` | Follow cancellation, progress, settings capture, and run metadata patterns. Implement a dedicated volume worker and core pipeline. |
| `_segmentation_io.py` and `_prediction_output.py` | Reuse output conventions and safe writes. Define separate volume records because current instance metadata contains 2D bounding boxes and pixel areas. |
| `api.py` | Later expose the same volume pipeline without Qt or napari. |

Preserve the existing 2D behavior and project annotation format. Store volume predictions as analysis outputs, separate from the 2D training annotation pool.

## Input contract and channel compatibility

Accept one image layer containing Z, Y, X and an optional channel axis. Support grayscale `ZYX` and multichannel arrangements such as `ZCYX`, `CZYX`, and `ZYXC`, including RGB volumes. Accept NumPy, memory-mapped, and Dask-backed sources without converting the whole volume to NumPy.

“Same dimension as 2D, with Z” means the same channel meaning and preprocessing recipe applied to each XY plane. A larger XY field of view is valid because inference is tiled. Require all slices within a volume to have the same XY shape and channel layout.

1. Read reliable axis metadata, then show explicit Z, Y, X, and channel selectors. Require confirmation through axis selection when metadata is ambiguous; do not infer Z or C from axis length alone. The current fallback interprets a three-dimensional array as `CYX`, so it must not silently classify a grayscale `ZYX` volume that way.
2. Resolve channel identity separately from the stored channel axis index. Adding Z can shift that index: a `CYX` project with C at axis 0 can use a `ZCYX` volume with C at axis 1. Record both the project conversion and its resolved volume axes without rewriting the locked project settings.
3. Validate channel availability and model/project class compatibility before starting. Reuse selected R/G/B channels, zero-filled channels, grayscale replication, filtering, inversion, and normalization exactly as in 2D. Keep normalization per slice and per channel initially; store its measured statistics for every slice.
4. If the source also has T or another acquisition axis, process one explicitly selected index and freeze it for the run. Never connect objects across time. Full time-series processing can be added separately.
5. Capture shape, axes, selected indices, scale, units, and spatial transforms. Produce masks in canonical `ZYX` order with corresponding source coordinates; map display transforms correctly rather than copying a channel-bearing transform blindly. Validate unsupported transforms before inference.

## Collapsible section and staged workflow

Enable the 3D section for either project task with a compatible model. The project task determines whether each slice produces instance masks or bounding boxes. Keep the existing sections and shared controls in place.

The 3D section should contain these controls:

- Source image, axis mapping, selected acquisition indices, and optional Z range.
- A summary of the shared channel conversion and a **Preview slice conversion** action.
- Existing inference parameters: model, device, tile size, halo overlap, confidence, model IoU, and maximum detections.
- Output directory and **Predict Z slices**, **Cancel**, and **Open saved run** actions.
- Assembly parameters and **Assemble 3D objects**, **Cancel assembly**, and **Export** actions.
- Separate inference and assembly status, completed Z count, object count, and ambiguous link count.

Use this state progression: input validated → inference running → slice predictions available → assembly running → assembled objects available. A cancelled run can contain valid completed slices; missing slices must be visibly identified and excluded from normal complete-volume assembly.

For segmentation, after slice inference show the original image volume and a **Slice predictions** Labels layer. Preserve the original pixels; the converted RGB preview is an additional layer. Enable Z scrolling and 3D inspection before assembly. Use unique display IDs across slices so repeated local IDs do not imply a 3D identity. Label these as unassembled predictions. Detection projects use the slice rectangles described in the bounding-box workflow below.

After segmentation assembly, add a separate **3D objects** Labels layer with stable object IDs. Detection assembly instead displays enclosing 3D boxes. Keep the slice predictions available for comparison. Provide class colors and object details, including contributing slice instances and fusion/fission locations. Changing viewer Z during a run must not change the input selection or cause results to be discarded.

## Slice inference and storage

Implement a volume adapter, a Qt-independent slice inference runner, a storage abstraction, and a volume worker. Process Z slices in order, extracting only the channels needed for that slice. Use the current direct path for a slice that fits one tile and the current tiled segmentation path otherwise.

Start with a disk-backed NumPy mask store, exposed as a Dask array for viewing and assembly. This uses existing dependencies and provides durable output; Dask alone is a computation abstraction, so a graph of model calls must not be the persistent result. Dask supports lazy wrapping of arrays with NumPy-style slicing, as described in its [array creation documentation](https://docs.dask.org/en/stable/array-creation.html). Benchmark wrapping and chunk reads to ensure opening the layer does not copy the volume.

Proposed run layout:

```text
Prediction/<volume_run>/
  run.json
  slice_masks.npy
  slice_instances.jsonl
  assemblies/<assembly_run>/
    objects.npy
    objects.jsonl
    links.jsonl
    assembly.json
```

Use unsigned integer labels with 0 reserved for background. Store compact local IDs per slice, identified by `(z, local_id)`. Allocate a separate deterministic display mapping and final object mapping; guard ID capacity before writing. Start with `uint32`, and reject overflow explicitly or choose a tested wider representation before allocation.

Write and flush each completed slice and its records before marking it complete in the manifest. Zero data does not distinguish a completed empty slice from an unfinished slice; completion state must be explicit. Recovery should recompute any slice whose data and metadata were not both committed. Preserve completed work after cancellation or errors and prevent loading incomplete output as a completed run.

Persist source identity, axis mapping, source transforms, model fingerprint, software version, classes, inference parameters, conversion statistics, cleanup policy, Z selection, and completion state. Resume only after validating the input and settings against the saved run; require a source content fingerprint for resumable input, or keep reopening limited to inspection when the source cannot be verified.

Memory should scale with the current plane and bounded buffers, rather than Z depth. The existing tile engine still requires a whole plane and its intermediates: measure that peak and enforce a memory budget. Extremely large XY planes will require a later extension to tile-backed intermediate storage. Serialize access to the model and avoid nested unbounded Dask parallelism.

Consider chunked Zarr storage after benchmarking the first backend if compression or XY chunk access is needed. This would require an explicit dependency and format compatibility decision. Offer streamed TIFF/BigTIFF export with axes and spatial metadata as an interchange output after the core storage path works.

## Preserve geometry before assembly

Disable per-slice border removal by default: removing XY-border instances at each Z can truncate valid 3D objects. Offer optional filtering of final objects touching selected volume faces, and distinguish a selected Z range boundary from the original acquisition boundary.

Add a 3D-specific component policy to the 2D prediction composition path. Preserve disconnected components that survive confidence ownership, and represent them as separate slice nodes sharing prediction provenance. They may join through neighboring slices into one 3D object. Keep the existing largest-component rule as the 2D default, with its current tests unchanged.

Run tile seam fusion first, then derive connected slice nodes from the final plane mask. Extend cleanup provenance so any removed geometry remains traceable. Do not apply the current 2D validation rule that every object must have one connected XY component to a completed 3D object; a connected volume can have multiple separate components in one slice.

## Assembly from overlap links

Represent each connected slice instance as a graph node keyed by `(z, local_id, component_id)` with class, confidence, area, and bounding box. Edges describe accepted connections between slices. Graph connected components define final object IDs, permitting multiple nodes from the same slice to belong to one object.

For adjacent slices, calculate exact intersections of pairs of positive labels using a sparse pair-count table. Include only same-class pairs by default. Avoid a dense matrix indexed by maximum label ID and avoid comparing every instance pair.

For masks A and B, calculate:

```text
intersection = count(A and B at the same XY coordinates)
overlap_fraction = intersection / min(area(A), area(B))
IoU = intersection / (area(A) + area(B) - intersection)
coverage_A = intersection / area(A)
coverage_B = intersection / area(B)
```

The initial acceptance rule should be `intersection >= minimum_pixels` and `overlap_fraction >= minimum_fraction`, with an optional IoU floor. Overlap relative to the smaller mask accommodates a small cross-section joining a larger one. Its tendency to accept small fragments must be evaluated alongside the absolute intersection requirement.

Expose these parameters rather than claiming a universal threshold. Tune initial defaults on representative microscopy volumes and record them in each assembly. Separate model IoU and tile halo overlap from these Z assembly parameters in both UI wording and metadata.

Keep every qualifying edge, including one-to-many and many-to-one links. Use union-find or graph components to assign deterministic IDs ordered by the earliest contributing node. Preserve edge metrics and directed slice relationships even though final object membership uses an undirected graph.

| Situation | Intended assembly behavior |
| --- | --- |
| New object or object ending | A node without a previous/next accepted link starts/ends naturally. Retain single-slice objects unless an explicit final size filter removes them. |
| One-to-one continuation | Assign one final ID across slices. |
| Fission across Z | Keep all accepted links from one node to multiple next-slice nodes; all branches belong to the connected 3D object. |
| Fusion across Z | Keep all accepted links from multiple nodes into one next-slice node; they become one connected 3D object. |
| Branches separating and rejoining | Preserve every branch and reconnection, including multiple XY components sharing a final object ID. |
| Many-to-many connections | Retain valid links and mark the junction for inspection. |
| Nearby objects or weak incidental contact | Reject insufficient overlap; report near-threshold and competing links for review. |
| Class disagreement | Keep objects separate and report the conflict. Do not silently merge classes. |
| Empty or missing prediction slice | End objects by default. An incomplete, unprocessed slice is a run error, not an empty prediction. |
| Object touching the volume boundary | Preserve it and mark boundary contact; apply optional final filtering afterward. |

Fusion and fission here describe cross-section topology in space. These links do not establish biological events over time. A merged 2D prediction can falsely join distinct objects, and overlap alone cannot resolve that ambiguity. Flag nodes with multiple predecessors/successors and allow users to accept or reject links and rerun assembly from the stored graph. Rejected links must be excluded before recomputing connected components; an alternate accepted path can still connect the same nodes and should be visible.

Add optional gap bridging and lateral tolerance only after adjacent-slice assembly is validated. Default both to disabled. Gap bridging must have an explicit maximum Z distance, use physical spacing where available, require stronger evidence, and avoid competing intermediate objects. A lateral search can propose links using a bounded dilation, but it must preserve original mask voxels. Keep inferred links visibly distinct from direct overlap links. A gap-linked object may be logically connected without voxel continuity; report that distinction and do not fill empty voxels silently.

## Assembly output and resource limits

Use two passes: stream neighboring slices to collect accepted edges and node records, then stream masks to write final object IDs. Keep only bounded image buffers in memory; graph metadata scales with instance and link counts. Add an estimate and configurable budget for that graph, with disk-backed records and a future partitioned assembly path for runs that exceed it.

Store per-object class, voxel count, physical volume when spacing is known, centroid, 3D bounding box, Z extent, confidence summary, boundary flags, contributing nodes, and junction flags. Use a voxel-weighted confidence summary identified as an aggregate of 2D predictions, rather than a calibrated 3D probability.

Give each assembly its own directory and settings, so threshold changes produce a new result without overwriting masks or previous assembly decisions. Interrupted assembly must preserve slice predictions and must not publish a partial mask as a completed result.

Display stored results lazily and keep initial layers read-only. Napari accepts integer label arrays and supports volumetric display, as described in its [Labels documentation](https://napari.org/stable/howtos/layers/labels.html). Link decisions provide the first review mechanism; arbitrary voxel painting requires a separate mutable storage and metadata reconciliation design.

## 3D bounding box workflow

Support two ways to obtain an axis-aligned 3D bounding box: derive one from an assembled segmentation object, or assemble the existing 2D detector's boxes across Z. Both reuse the volume adapter, channel conversion, run manifest, cancellation, and link review workflow. These are reconstructed volume boxes from slice predictions; the model itself remains a 2D model.

### Boxes from segmentation objects

Calculate the minimum enclosing box of each final object's occupied voxels, including all its branches. Export `(z_min, y_min, x_min, z_max, y_max, x_max)` with exclusive upper bounds. A voxel at index `(z, y, x)` occupies the index interval from that coordinate to `(z+1, y+1, x+1)`, so a single-slice object has Z extent 1. Keep mask volume and enclosing box volume as separate measurements: the box can contain background and other objects.

Store index bounds with scale, units, and transforms. For rotated or sheared data, transformed corners describe the source-aligned box in world coordinates; a world-axis-aligned envelope is a separate derived result. Make the display coordinate convention explicit and verify alignment against image voxel boundaries.

### Boxes assembled from detection slices

1. Run the current tiled detection engine on every converted XY slice. Finish existing tile duplicate merging before Z linking, and preserve class, confidence, source coordinates, and tile provenance for each retained detection.
2. Store detection records keyed by `(z, detection_id)` in `slice_detections.jsonl`. Display unassembled rectangles at their source Z with class and confidence. No dense mask volume is needed for this path.
3. Compare same-class rectangles on adjacent slices using XY intersection area, intersection divided by the smaller rectangle area, and optional XY IoU. Require positive area and a configurable minimum overlap. Use spatial indexing or a bounding-box sweep to avoid comparing every pair. Record detection thresholds separately from segmentation thresholds.
4. Apply the graph assembly and review rules, allowing appearance, endings, one-to-many, many-to-one, and many-to-many links. A split or merged group produces one enclosing box for the accepted graph component, with its junctions recorded. Keep each contributing slice box so users can inspect the shape and undo false connections.
5. For each group, take the minimum lower XY bounds and maximum upper XY bounds of its contributing detections. Set `z_min` to the first contributing source slice and `z_max` to the last contributing source slice plus 1. Preserve global source Z indices when processing a selected range. Retain single-slice groups by default.
6. Save `boxes_3d.jsonl`, optional CSV, accepted/rejected links, class, Z extent, member detection IDs, boundary flags, and a clearly named aggregate confidence. Do not report segmented voxel count or object volume from detection boxes; only report enclosing box volume.

Box overlap is weaker evidence than mask overlap because enclosing rectangles include background. Nearby objects can have overlapping boxes without touching. Use separate defaults, flag competing connections, and test false merges explicitly. Gap and lateral tolerance should follow the same opt-in policy as segmentation. Leave overlapping final boxes intact; suppressing them solely by 3D IoU could discard distinct objects.

### Display and validation of volume boxes

Show final cuboids as wireframes made from their 12 edges with a shared object ID, class color, and matching source transforms. Napari's [Shapes documentation](https://napari.org/stable/howtos/layers/shapes.html#3d-rendering) describes 3D paths but does not provide native cuboid shapes, so validate edge rendering on the supported napari versions. In the 2D view, generate a rectangle for each cuboid intersecting the current Z slice rather than depending on a 3D path to show its cross-section. Keep original slice rectangles separately available.

Initially make these displays read-only and use link decisions for corrections. Test wireframe alignment, 2D cross-sections, class colors, single-slice thickness, fractional XY bounds, anisotropic spacing, nonzero Z ranges, empty slices, junctions, overlapping independent boxes, cancelled runs, and JSON/CSV round trips. Verify segmentation-derived bounds against known voxel extents and detection-derived bounds against the original member rectangles.

## Implementation sequence and acceptance checks

Point 1 is implemented in `_volume_adapter.py`, with shared explicit-plane conversion in `_image_adapter.py`. The input records capture axes, acquisition indices, Z range, project and resolved channel settings, and canonical ZYX geometry without loading the volume. Axis-aligned scale and translation are supported; rotation, shear, and additional affine transforms are explicitly rejected for now. The collapsible section and prediction actions remain scheduled for points 2 and 3.

1. **Define volume contracts and axes.** Add the adapter and settings/result records. Validate grayscale, RGB, channel-first/channel-last layouts, moved channel axes, acquisition selection, anisotropic scale, and invalid ambiguous inputs. Confirm a selected slice produces the same converted RGB pixels as the 2D path.
2. **Implement stored slice inference.** Add storage, manifest handling, and a cancellable core runner. Verify equality with the existing per-plane prediction path under the same cleanup policy, including tiled seams, padding, empty slices, class records, and source shape. Add the preservation policy and tests for disconnected components separately.
3. **Add the collapsible section and intermediate preview.** Confirm raw volume and masks align in Z, Y, X and world coordinates. Check preview before assembly, progress updates, cancellation, closed sources, and scrolling during inference. Verify opening stored masks does not trigger model inference.
4. **Implement adjacent-slice assembly.** Test synthetic objects with known outcomes: appearance, disappearance, continuation, fusion, fission, branching, rejoining, many-to-many links, weak contacts, class conflicts, and boundary contact. Check overlap threshold boundaries, deterministic IDs, and the preserved node-to-object mapping.
5. **Add review and durable assembly output.** Test link overrides, alternate connecting paths, repeated assembly with different thresholds, cancellation, interrupted writes, reopening, and export round trips. Keep metadata synchronized with final masks.
6. **Validate performance and optional tolerance.** Measure RAM, VRAM, inference time, storage size, graph growth, and napari responsiveness on representative volumes. Confirm image memory remains bounded as Z grows. Implement opt-in gap/lateral links with dedicated false-merge cases only after the baseline works.
7. **Add 3D bounding boxes.** First derive boxes from assembled masks, then add stored detection-slice inference and rectangle overlap assembly. Validate detection grouping, enclosing bounds, display, and export with the cases above. Share the graph/review infrastructure while keeping task-specific metrics and records explicit.
8. **Expose the API and document the workflow.** Add headless inference/assembly entry points for both tasks, tutorial examples, and the difference between slice IDs and 3D object IDs. Run the existing 2D, batch, API, and plugin tests to catch regressions.

The feature is complete when a user can load a compatible volume, predict every selected Z slice using the project's channel recipe, inspect persisted masks or detections over the raw volume, assemble and review complex 3D objects or enclosing boxes, and reopen/export results without rerunning inference. Storage integrity, alignment, deterministic assembly, and existing 2D behavior must all pass validation before release.
