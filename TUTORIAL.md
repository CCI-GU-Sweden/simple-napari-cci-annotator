# GUI tutorial

This guide follows a project from an image to saved predictions and training data. Install and open the plugin as described in the [README](README.md).

## Create a project and prepare an image

1. Select **Bounding-box detection** or **Instance segmentation** and click **New Project**. Choose an empty folder. The task cannot be changed later. The plugin copies a matching starter model into the project's `models/` folder.
2. Use **Edit Classes** to set the names you need. Class IDs are stable YOLO indices starting at 0.
3. Open an image in napari and select it. For multidimensional images, choose the channel axis and map source channels to red, green, and blue. Choose a filter and normalization if needed.
4. Click **Preview RGB Conversion**. The displayed RGB plane is what the model receives and what training annotations store. Check a representative plane before saving your first annotation: the project's channel mapping, filter, and normalization lock after that save.

The plugin uses napari's current Z/T position for single-plane prediction. For a stack, browsing does not automatically convert each plane or load its annotation. Click **Load / Reload Annotation** to create an editable layer or reopen a saved annotation for the current plane; prediction also prepares the selected plane on demand. When you move away, the previous annotation is hidden and locked. Unsaved edits trigger the save/discard prompt; if you cancel, return to the original plane to continue editing or saving them. Full-size working annotations larger than the project training size still require a training crop for the training dataset.

## Predict and correct

1. Select a YOLO model whose task and class IDs match the project.
2. Adjust confidence, device, and tiling settings if needed, then click **Predict Current RGB Plane**. Prediction runs in the background and can be cancelled.
3. For detection, edit the `yolo_bboxes` Shapes layer and choose a class for new boxes. For segmentation, edit the `yolo_instances` Labels layer and use the instance controls to assign classes or correct IDs.
4. Choose **Save Current Output** to export the full-size converted image and its edited result. This export does not add an annotation to the training pool.

To bring in an existing TIFF instance mask, open it in napari as a Labels layer (or open it as an Image layer and use **Convert to Labels**). Keep the matching source image open and select its plane. Select the external Labels layer, choose the class for its instances, then click **Import Selected Labels Layer**. The plugin copies the mask into `yolo_instances` and leaves the external layer untouched. Zero is background; nonzero values identify instances. A binary foreground value of 255 becomes ID 1. Disconnected regions sharing one ID stay together on import; use **Split Disconnected Instance** if needed before saving for training. The mask must match the image plane pixel-for-pixel.

The inference tile size follows the chosen training patch size (512 or 1024 pixels) until you change the inference tile control manually.

## Save training annotations

1. Choose a **Training patch** size. Use **Select Training Crop** and move the square over a region you want to annotate.
2. Click **Create / Refresh Crop**, correct the boxes or mask in the crop, then click **Add Crop + Corrections**.
3. Click **Return to Source** to inspect another region. The crop can be moved and reused.

The first saved annotation locks the training patch size and image conversion for the project. Use **Review saved annotations** to revisit and edit saved samples. The project stores detection images with same-stem YOLO `.txt` labels, or segmentation images with same-stem mask TIFF and JSON instance metadata.

## Predict a folder

After saving an annotation and locking the project settings, expand **Batch image prediction**. Choose an input folder, select the model, and click **Predict All Images**. The plugin processes every top-level TIFF/OME-TIFF, PNG, and BMP file, including every nonspatial plane in each TIFF. It uses the locked project conversion settings. An unreadable or incompatible file is recorded as failed while the batch continues.

The default output parent is `Prediction` inside the input folder; you can choose another. Each run has its own folder containing the prediction files and `analysis_metadata.json`. Use **Cancel Batch** to stop after the current cancellable operation; completed outputs remain available.

## Predict a volume

Open a segmentation project, load a compatible model, and expand **3D inference and assembly**. Choose the source volume. Z, Y, X and channel axes are filled from declared axis metadata when possible; choose them explicitly for an ambiguous image. For a grayscale volume, leave the C axis set to **None (grayscale)**. Preview works without a loaded model, but prediction requires one. Select one fixed index for each acquisition axis such as time. Set **First Z (inclusive)** and **Last Z (exclusive)** to choose the range; a range of 0 to 10 processes slices 0 through 9.

For repeated access, **Copy Selected Volume to Zarr v3** writes the selected Z range and acquisition indices to a new `.zarr` directory in 1 × 512 × 512 spatial chunks. It copies raw values and every channel, without applying the RGB conversion or inversion. The copy runs in the background and opens as a lazy Dask source when complete. Scale and translation are preserved, and source Z start is reflected in the new translation. The new source's Z indices start at zero; the original selection is recorded in the Zarr metadata. The copy does not replace the original image. If conversion is cancelled or fails, the incomplete directory is not usable and must be removed manually before reusing its name.

Use **Open Plugin Zarr Volume** to reopen a copy made by this plugin. The output is a Zarr v3 group containing array `0` and plugin-specific metadata; it is not an OME-NGFF export. Other Zarr arrays can be opened as napari image layers with a suitable reader and selected as sources. Zarr v3 conversion requires Python 3.11 or later. Chunking limits the data requested by the copy, although the source reader may need to decode larger underlying TIFF or Dask chunks.

For large slices, enable **Chunked slice inference (experimental)** before **Predict Z Slices**. Conversion uses two passes to compute full-slice normalization bounds and then write RGB chunks; tile predictions, seam fusion, component extraction and mask writes use temporary disk-backed arrays. The mode supports no filter or local median, mean, Gaussian and white-top-hat filters, with min-max, simple-max, fixed-range or dtype-range normalization. Percentile normalization is also supported for unfiltered 8- or 16-bit integer data. Global frequency low-pass filtering, Z-score normalization and other percentile recipes require the existing full-slice mode. The selected execution mode must match when resuming a run.

Chunked inference uses additional scratch disk space under the run folder for each slice, and normally removes that scratch data after the slice finishes or is cancelled. Its array budget is an estimate, not a hard memory limit: source-reader chunks, model/GPU memory, operating-system caching, and object or seam metadata are additional. The original slice-mask storage remains a memory-mapped file. Test the experimental mode on representative data before relying on it for a large run.

Multichannel copies use ZYXC axis order. The 3D volume controls map this channel axis for inference. For 2D conversion of such a copy, the project's channel-axis setting must be 3; a project locked to a different axis cannot use that copy for 2D conversion.

The volume uses the project's shared R/G/B channels, filtering and normalization. Model, device, confidence and tile settings come from **Large-image tiled prediction**. The channel axis selector in the 3D section resolves where those channels are located in the volume. Inversion and component cleanup are volume controls; **Preserve all components** is the default. Border objects are retained.

Choose a preview Z and click **Preview Slice Conversion** to inspect the RGB input over lazy views of the original selected channels. These raw views keep the source pixel values and display the volume in ZYX order. The original image stays in napari, temporarily hidden while the canonical raw views are shown.

Click **Predict Z Slices**. Results go to a new run folder under `<project>/Prediction/` unless you choose another output parent. Progress and cancellation run in the background. Once inference finishes, the plugin displays the raw channels and a read-only **3D slice predictions** Labels layer. Scroll Z or switch napari to its 3D view to inspect it. Labels have unique display IDs across slices and class colors; select a label to see its source Z, local ID, class and confidence.

To link a complete run, set the **Z link** thresholds and click **Assemble 3D Objects**. Adjacent masks of the same class link when their shared pixels meet the minimum pixel count, the overlap fraction (intersection divided by the smaller mask area), and the optional IoU floor. All qualifying links are kept, including branches; a single-slice instance remains an object. The defaults are starting points, not validated thresholds for every sample. Review the result carefully: merged predictions or incidental overlaps can join distinct objects, while nonoverlapping sections will stay separate. There is no gap bridging or manual link editing yet.

Each assembly is saved in a new folder inside the run's `assemblies` directory. Its graph records accepted and rejected overlapping pairs and the settings used; the original slice masks are unchanged. The displayed read-only 3D object layer is lazy. Select an object to inspect its Z extent, voxel count and link counts. Use **Open Assembly** and choose an assembly folder to reopen it without a model. To retain the aligned raw overlay, first open the matching saved run. Reopen the saved run to return to the slice predictions. Assembly reads adjacent masks in XY blocks and keeps its object graph in memory; it requires all slices to be complete.

Use the saved-run field and **Open Saved Run** to reopen stored masks without running the model. When the selected source matches the saved pixels and geometry, its saved Z range and acquisition selection are used for the raw overlay. Masks can also be opened without the source or model; the status explains when the raw overlay is unavailable. Verification runs in the background.

After cancellation or a failure, completed slices remain on disk. Enable **Show completed slices from an incomplete run** before opening partial output; unfinished slices are marked in metadata and displayed as zeros. To continue inference, select the matching source and model, restore the original axes, Z range, conversion and tile settings, and click **Resume Saved Run**. Resume checks the source and model fingerprints before reusing completed slices. Closing the plugin during active 3D work requests cancellation; close it again once the worker stops.

## Retrain

In **Dataset building and retraining**, select a starting model, preview the dataset, and start training after validation passes. A normal validation split needs at least two independent source groups. **Train without independent validation** is available for exploratory runs with fewer groups. A completed validated run stores its model and run metadata in a new retraining folder; loading the new model for prediction is a separate choice.
