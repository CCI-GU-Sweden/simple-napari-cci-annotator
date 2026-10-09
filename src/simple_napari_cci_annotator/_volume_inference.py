"""Sequential, cancellable segmentation of a prepared volume into a store."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ._instance_mask import split_prediction_components
from ._segmentation_io import _sha256
from ._segmentation_tiling import TiledSegmentationEngine
from ._tiled_inference import InferenceCancelled, InferenceError, InferenceSettings
from ._version import __version__
from ._volume_adapter import PreparedVolume
from ._volume_store import VolumeMaskStore


class VolumeInferenceError(InferenceError):
    """A volume run cannot start or resume with the requested inputs."""


@dataclass(frozen=True)
class VolumeInferenceSettings:
    inference: InferenceSettings = InferenceSettings()
    component_policy: str = "preserve"
    memory_budget_bytes: int = 1024**3


@dataclass(frozen=True)
class VolumeInferenceRun:
    # None means cancellation occurred before storage was created/opened.
    run_root: Path | None
    status: str
    completed_slices: int
    total_slices: int


class VolumeInferenceRunner:
    """Use the existing per-plane engines, with one active tile at a time.

    Predictor implementations must honor the component_policy keyword when
    preserve is requested. The standard YOLO segmentation adapter supports it.
    The source and model must stay unchanged while this runner is active.
    """

    def __init__(
        self, volume: PreparedVolume, predictor, classes: Mapping[int, str],
        settings: VolumeInferenceSettings | None = None,
    ):
        self.volume = volume
        self.predictor = predictor
        self.classes = dict(classes)
        self.settings = settings or VolumeInferenceSettings()

    def validate(self) -> int:
        settings = self.settings
        inference = settings.inference
        inference.validate()
        if settings.component_policy not in {"largest", "preserve"}:
            raise VolumeInferenceError("Component policy must be largest or preserve.")
        if inference.overlap * 2 >= inference.tile_size:
            raise VolumeInferenceError("Segmentation overlap must be smaller than half the tile size.")
        if getattr(self.predictor, "task", None) != "segment":
            raise VolumeInferenceError("Volume slice masks require a segmentation model.")
        if (
            not self.classes
            or any(type(key) is not int or key < 0 for key in self.classes)
            or any(not isinstance(name, str) or not name.strip() for name in self.classes.values())
            or set(getattr(self.predictor, "names", {})) != set(self.classes)
        ):
            raise VolumeInferenceError("Model class IDs must match valid project classes.")
        model_path = getattr(self.predictor, "path", None)
        if model_path is None or not Path(model_path).is_file():
            raise VolumeInferenceError("The selected model file is unavailable.")
        if type(settings.memory_budget_bytes) is not int or settings.memory_budget_bytes < 1:
            raise VolumeInferenceError("Memory budget must be a positive byte count.")
        height, width = self.volume.geometry.shape[1:]
        core = inference.tile_size - 2 * inference.overlap
        tiled = max(height, width) > inference.tile_size
        rows = (height + core - 1) // core if tiled else 1
        columns = (width + core - 1) // core if tiled else 1
        if (
            height * width > np.iinfo(np.uint32).max
            or rows * columns * inference.max_detections >= np.iinfo(np.uint32).max
        ):
            raise VolumeInferenceError("Slice instance IDs exceed uint32 capacity.")
        # Conservative array estimate for conversion, cleanup, tile assembly,
        # and component extraction. Model/driver memory is outside this budget.
        area = rows * columns * core**2 if tiled else height * width
        estimate = area * 256 + inference.tile_size**2 * 64
        if estimate > settings.memory_budget_bytes:
            raise VolumeInferenceError(
                f"Estimated slice array memory ({estimate} bytes) exceeds "
                f"the budget ({settings.memory_budget_bytes} bytes)."
            )
        return estimate

    def run(
        self, run_root: Path, *, resume: bool = False,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> VolumeInferenceRun:
        self.validate()
        total = self.volume.geometry.shape[0]
        root = Path(run_root).expanduser().resolve()

        def report(completed, text):
            if progress is not None:
                progress(completed, total, text)

        store = None
        try:
            _check_cancelled(cancelled)
            report(0, "Verifying source pixels and model before inference")
            source_digest = _source_digest(
                self.volume, cancelled=cancelled,
                progress=lambda z: report(0, f"Verifying source Z={z}"),
            )
            _check_cancelled(cancelled)
            model_path = Path(self.predictor.path).resolve()
            contract = {
                "software_version": __version__,
                "task": "segment",
                "volume": self.volume.to_mapping(),
                "source_pixels_sha256": source_digest,
                "model": {
                    "path": str(model_path), "sha256": _sha256(model_path),
                    "names": {str(key): str(value) for key, value in self.predictor.names.items()},
                },
                "classes": {str(key): value for key, value in sorted(self.classes.items())},
                "inference": asdict(self.settings.inference),
                "component_policy": self.settings.component_policy,
                "border_removal": False,
                "tile_workers": 1,
                "mask_dtype": "uint32",
            }
            _check_cancelled(cancelled)
            if resume:
                store = VolumeMaskStore.open(
                    root, writable=True, recover=True, cancelled=cancelled,
                    progress=lambda current, count: report(
                        0, f"Verifying stored slices {current}/{count}"
                    ),
                )
                if store.manifest["contract"] != contract:
                    store.close()
                    store = None
                    raise VolumeInferenceError(
                        "Cannot resume: source, model, conversion, or inference settings changed."
                    )
            else:
                store = VolumeMaskStore.create(root, contract)
            store.set_status("running")
            report(store.completed_count, "Slice inference started")
            for z_index in self.volume.selection.z_indices:
                _check_cancelled(cancelled)
                if store.is_complete(z_index):
                    continue
                converted = self.volume.convert_slice(z_index)
                _check_cancelled(cancelled)
                report(store.completed_count, f"Predicting source Z={z_index}")
                image = converted.image
                if max(image.data.shape[:2]) > self.settings.inference.tile_size:
                    result = TiledSegmentationEngine(self.predictor).predict(
                        image.data, self.settings.inference, self.classes,
                        progress=lambda current, count, text: report(
                            store.completed_count,
                            f"Z={z_index} · {current}/{count} tiles · {text}",
                        ),
                        cancelled=cancelled,
                        component_policy=self.settings.component_policy,
                        num_workers=1,
                    )
                else:
                    kwargs = (
                        {"component_policy": self.settings.component_policy}
                        if self.settings.component_policy != "largest" else {}
                    )
                    result = self.predictor.predict_image(
                        image.data, self.settings.inference, self.classes, **kwargs
                    )
                _check_cancelled(cancelled)
                result = split_prediction_components(result, self.classes)
                _check_cancelled(cancelled)
                store.write_slice(z_index, result, {
                    "sample_id": image.sample_id,
                    "plane_indices": list(image.plane.axis_indices),
                    "normalization_stats": list(image.normalization_stats),
                    "inverted": image.inverted,
                })
                report(store.completed_count, f"Stored source Z={z_index}")
                # Release the previous plane before converting the next one.
                del converted, image, result
            _check_cancelled(cancelled)
            store.set_status("completed")
            report(total, "Slice inference complete")
            return VolumeInferenceRun(root, "completed", total, total)
        except InferenceCancelled:
            if store is None:
                return VolumeInferenceRun(None, "cancelled", 0, total)
            store.set_status("cancelled")
            report(store.completed_count, "Inference cancelled; completed slices retained")
            return VolumeInferenceRun(root, "cancelled", store.completed_count, total)
        except Exception as exc:
            if store is not None:
                store.set_status("failed", error=str(exc))
            raise
        finally:
            if store is not None:
                store.close()


def _check_cancelled(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise InferenceCancelled("Volume inference cancelled.")


def _source_digest(
    volume: PreparedVolume, *, cancelled=None, progress=None
) -> str:
    """Hash selected raw pixels in bounded XY blocks, independent of storage.

    Only selected acquisition indices and consumed channels are fingerprinted.
    Channel order and source shape are also recorded in the run contract.
    """
    digest = hashlib.sha256()
    axes = volume.selection.axes
    channels = sorted({
        channel for channel in volume.processing.rgb_channels
        if channel is not None
    })
    if axes.channel is None:
        channels = [None]
    for z_index in volume.selection.z_indices:
        if progress is not None:
            progress(z_index)
        plane = volume.selection.plane(z_index)
        for channel in channels:
            for y in range(0, volume.source_shape[axes.y], 512):
                for x in range(0, volume.source_shape[axes.x], 512):
                    _check_cancelled(cancelled)
                    index = list(plane.axis_indices)
                    index[axes.y] = slice(y, y + 512)
                    index[axes.x] = slice(x, x + 512)
                    if axes.channel is not None:
                        index[axes.channel] = channel
                    block = np.asarray(volume.data[tuple(index)])
                    digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()
