# Headless Python API examples

The API uses the same project, training, inference, and output code as the napari widget. It does not import napari or Qt. Install the package as described in the [README](README.md), then run these examples in Python.

## Create or open a project

```python
from simple_napari_cci_annotator import ImageProcessingSettings, ProjectPipeline

pipeline = ProjectPipeline.create(
    "project",
    task="detect",  # use "segment" for instance segmentation
    classes={0: "Cell"},
    image_processing=ImageProcessingSettings(
        channel_axis=None,
        red_channel=None,
        green_channel=None,
        blue_channel=None,
        normalization="min_max",
    ),
)

# On a later run:
pipeline = ProjectPipeline.open("project")
```

This channel mapping is for grayscale TIFFs. For multichannel images, set `channel_axis` and the RGB source-channel indices to match your files. Project conversion settings must be locked before prediction; `create(..., image_processing=...)` does that. An existing project can call `pipeline.lock_image_processing(settings)`.

## Predict one image or a folder

```python
from simple_napari_cci_annotator import InferenceSettings

inference = InferenceSettings(tile_size=512, overlap=102)
model = pipeline.project.paths.models / "yolo26n.pt"
one = pipeline.predict_one("input/field_001.ome.tiff", model, inference=inference)
batch = pipeline.predict_batch("input", model, inference=inference)

print(one.status, one.run_root)
print(batch.completed, batch.failed, batch.run_root)
```

`predict_one` accepts TIFF, PNG, or BMP. It processes every nonspatial plane and series in a TIFF, or the single image in a PNG/BMP. `predict_batch` processes all top-level TIFF, PNG, and BMP files in a folder. Grayscale, RGB, and RGBA PNG/BMP files must match the project's locked channel settings. The default output parent is `input/Prediction`; pass `output_folder="results"` to change it. Each run writes `analysis_metadata.json` alongside detection or segmentation results. The methods also accept `progress(current, total, message)` and `cancelled()` callbacks.

For a segmentation project, use `yolo26n-seg.pt` or another compatible segmentation model instead.

## Train from saved project annotations

```python
from pathlib import Path

from simple_napari_cci_annotator import DatasetBuildSettings, TrainingSettings

training = TrainingSettings(
    model_path=model,
    destination=Path("training_runs"),
    dataset=DatasetBuildSettings(tile_size=1024, overlap=205),
    train_only=True,  # exploratory run without an independent validation split
)
run = pipeline.train(training)
print(run.status, run.run_root, run.best_model)
```

Training uses annotations already saved in the project. The dataset tile size must match the project's locked training patch size, and normal validated training needs independent source groups. The API validates the dataset before starting Ultralytics and does not create or review annotations automatically.
