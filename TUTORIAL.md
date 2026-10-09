# GUI tutorial

This guide follows a project from an image to saved predictions and training data. Install and open the plugin as described in the [README](README.md).

## Create a project and prepare an image

1. Select **Bounding-box detection** or **Instance segmentation** and click **New Project**. Choose an empty folder. The task cannot be changed later. The plugin copies a matching starter model into the project's `models/` folder.
2. Use **Edit Classes** to set the names you need. Class IDs are stable YOLO indices starting at 0.
3. Open an image in napari and select it. For multidimensional images, choose the channel axis and map source channels to red, green, and blue. Choose a filter and normalization if needed.
4. Click **Preview RGB Conversion**. The displayed RGB plane is what the model receives and what training annotations store. Check a representative plane before saving your first annotation: the project's channel mapping, filter, and normalization lock after that save.

The plugin uses napari's current Z/T position for single-plane prediction. Change the displayed plane to work on another one.

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

Open a segmentation project, load a compatible model, and expand **3D inference and assembly**. Choose the source volume. Z, Y, X and channel axes are filled from declared axis metadata when possible; choose them explicitly for an ambiguous image. Select one fixed index for each acquisition axis such as time. Set **First Z (inclusive)** and **Last Z (exclusive)** to choose the range; a range of 0 to 10 processes slices 0 through 9.

The volume uses the project's shared R/G/B channels, filtering and normalization. Model, device, confidence and tile settings come from **Large-image tiled prediction**. The channel axis selector in the 3D section resolves where those channels are located in the volume. Inversion and component cleanup are volume controls; **Preserve all components** is the default. Border objects are retained.

Choose a preview Z and click **Preview Slice Conversion** to inspect the RGB input over lazy views of the original selected channels. These raw views keep the source pixel values and display the volume in ZYX order. The original image stays in napari, temporarily hidden while the canonical raw views are shown.

Click **Predict Z Slices**. Results go to a new run folder under `<project>/Prediction/` unless you choose another output parent. Progress and cancellation run in the background. Once inference finishes, the plugin displays the raw channels and a read-only **3D slice predictions** Labels layer. Scroll Z or switch napari to its 3D view to inspect it. Labels have unique display IDs across slices and class colors; select a label to see its source Z, local ID, class and confidence. These are unassembled slice instances; connecting them into 3D objects is the next implementation phase.

Use the saved-run field and **Open Saved Run** to reopen stored masks without running the model. When the selected source matches the saved pixels and geometry, its saved Z range and acquisition selection are used for the raw overlay. Masks can also be opened without the source or model; the status explains when the raw overlay is unavailable. Verification runs in the background.

After cancellation or a failure, completed slices remain on disk. Enable **Show completed slices from an incomplete run** before opening partial output; unfinished slices are marked in metadata and displayed as zeros. To continue inference, select the matching source and model, restore the original axes, Z range, conversion and tile settings, and click **Resume Saved Run**. Resume checks the source and model fingerprints before reusing completed slices. Closing the plugin during active 3D work requests cancellation; close it again once the worker stops.

## Retrain

In **Dataset building and retraining**, select a starting model, preview the dataset, and start training after validation passes. A normal validation split needs at least two independent source groups. **Train without independent validation** is available for exploratory runs with fewer groups. A completed validated run stores its model and run metadata in a new retraining folder; loading the new model for prediction is a separate choice.
