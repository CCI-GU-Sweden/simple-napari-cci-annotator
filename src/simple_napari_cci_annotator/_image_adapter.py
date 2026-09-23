from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import convolve, grey_opening, median_filter
from skimage.morphology import disk


class ImageConversionError(ValueError):
    """Raised when a source image cannot be converted with the chosen settings."""


NORMALIZATION_METHODS = {
    "min_max": "Min / max",
    "simple_max": "Simple max",
    "percentile": "Percentile",
    "z_score": "Z-score",
    "fixed_range": "Fixed range",
    "dtype_range": "Data type range",
}

IMAGE_FILTERS = {
    "none": "None",
    "gaussian": "Gaussian blur",
    "median": "Median filter",
    "mean": "Mean filter",
    "low_pass": "Low-pass (frequency)",
    "white_tophat": "White top-hat",
}


@dataclass(frozen=True)
class ImageProcessingSettings:
    channel_axis: int | None
    red_channel: int | None
    green_channel: int | None
    blue_channel: int | None
    normalization: str
    filter_method: str = "none"
    filter_radius: int = 1
    lower: float | None = None
    upper: float | None = None
    output_dtype: str = "uint8"
    scope: str = "per_plane_per_channel"

    @property
    def rgb_channels(self) -> tuple[int | None, int | None, int | None]:
        return self.red_channel, self.green_channel, self.blue_channel

    def to_mapping(self) -> dict[str, Any]:
        return {
            "channel_axis": self.channel_axis,
            "rgb_channels": list(self.rgb_channels),
            "filter": {
                "method": self.filter_method,
                "radius": self.filter_radius,
            },
            "normalization": {
                "method": self.normalization,
                "lower": self.lower,
                "upper": self.upper,
                "scope": self.scope,
            },
            "output_dtype": self.output_dtype,
            "locked": True,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ImageProcessingSettings:
        try:
            channels = value["rgb_channels"]
            normalization = value["normalization"]
            image_filter = value.get("filter", {"method": "none", "radius": 1})
            settings = cls(
                channel_axis=_optional_int(value.get("channel_axis")),
                red_channel=_optional_int(channels[0]),
                green_channel=_optional_int(channels[1]),
                blue_channel=_optional_int(channels[2]),
                normalization=str(normalization["method"]),
                filter_method=str(image_filter.get("method", "none")),
                filter_radius=int(image_filter.get("radius", 1)),
                lower=_optional_float(normalization.get("lower")),
                upper=_optional_float(normalization.get("upper")),
                output_dtype=str(value.get("output_dtype", "uint8")),
                scope=str(normalization.get("scope", "per_plane_per_channel")),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ImageConversionError(
                "The project's image_processing configuration is invalid."
            ) from exc
        settings.validate()
        return settings

    def validate(self, shape: Sequence[int] | None = None) -> None:
        if self.normalization not in NORMALIZATION_METHODS:
            raise ImageConversionError(
                f"Unknown normalization method: {self.normalization!r}."
            )
        if self.filter_method not in IMAGE_FILTERS:
            raise ImageConversionError(
                f"Unknown image filter: {self.filter_method!r}."
            )
        if (
            isinstance(self.filter_radius, bool)
            or not isinstance(self.filter_radius, int)
            or not 1 <= self.filter_radius <= 100
        ):
            raise ImageConversionError("Filter radius must be an integer within 1..100.")
        if self.output_dtype != "uint8":
            raise ImageConversionError("Only uint8 RGB project images are supported.")
        if self.scope != "per_plane_per_channel":
            raise ImageConversionError(
                "Only per-plane, per-channel normalization is currently supported."
            )
        if self.channel_axis is None:
            if any(channel is not None for channel in self.rgb_channels):
                raise ImageConversionError(
                    "RGB channel indices require a selected channel axis."
                )
        elif self.channel_axis < 0:
            raise ImageConversionError("Channel axis must be non-negative.")

        if all(channel is None for channel in self.rgb_channels) and self.channel_axis is not None:
            raise ImageConversionError("Select at least one source channel for R, G, or B.")
        for channel in self.rgb_channels:
            if channel is not None and channel < 0:
                raise ImageConversionError("Channel indices must be non-negative.")

        if self.normalization in {"percentile", "z_score", "fixed_range"}:
            if self.lower is None or self.upper is None or self.lower >= self.upper:
                raise ImageConversionError(
                    "This normalization requires a lower value smaller than the upper value."
                )
        if self.normalization == "percentile" and (
            self.lower < 0.0 or self.upper > 100.0
        ):
            raise ImageConversionError("Percentiles must be within 0..100.")

        if shape is not None:
            ndim = len(shape)
            if self.channel_axis is not None:
                if self.channel_axis >= ndim:
                    raise ImageConversionError(
                        f"Channel axis {self.channel_axis} does not exist for shape {tuple(shape)}."
                    )
                channel_count = int(shape[self.channel_axis])
                for channel in self.rgb_channels:
                    if channel is not None and channel >= channel_count:
                        raise ImageConversionError(
                            f"Channel {channel} does not exist on axis {self.channel_axis} "
                            f"(size {channel_count})."
                        )


@dataclass(frozen=True)
class PlaneSelection:
    axis_indices: tuple[int | None, ...]
    plane_axes: tuple[int, int]
    axis_labels: tuple[str, ...]

    @property
    def non_spatial_indices(self) -> dict[int, int]:
        return {
            axis: int(index)
            for axis, index in enumerate(self.axis_indices)
            if index is not None and axis not in self.plane_axes
        }


@dataclass(frozen=True)
class ConvertedImage:
    data: np.ndarray
    sample_id: str
    plane: PlaneSelection
    settings: ImageProcessingSettings
    normalization_stats: tuple[dict[str, float | str | None], ...]
    inverted: bool = False


class ImageAdapter:
    """Extracts the current 2D plane and converts selected channels to RGB uint8."""

    def axis_labels(self, image_layer) -> tuple[str, ...]:
        shape = _shape_of(image_layer.data)
        ndim = len(shape)

        metadata = getattr(image_layer, "metadata", {}) or {}
        raw_axes = metadata.get("axes") or metadata.get("axis_labels")
        if isinstance(raw_axes, str) and len(raw_axes) == ndim:
            return tuple(raw_axes)
        if isinstance(raw_axes, (tuple, list)) and len(raw_axes) == ndim:
            return tuple(str(axis) for axis in raw_axes)

        layer_labels = getattr(image_layer, "axis_labels", None)
        if layer_labels and len(layer_labels) == ndim:
            labels = tuple(str(label) for label in layer_labels)
            if any(label.strip() for label in labels):
                return labels

        if (
            ndim >= 3
            and shape[-1] in {3, 4}
            and bool(getattr(image_layer, "rgb", False))
        ):
            leading = tuple(f"A{axis}" for axis in range(ndim - 3))
            return leading + ("Y", "X", "C")

        defaults = {
            2: ("Y", "X"),
            3: ("C", "Y", "X"),
            4: ("Z", "C", "Y", "X"),
            5: ("T", "Z", "C", "Y", "X"),
        }
        if ndim in defaults:
            return defaults[ndim]
        return tuple(f"A{axis}" for axis in range(ndim - 2)) + ("Y", "X")

    def infer_channel_axis(self, image_layer) -> int | None:
        shape = _shape_of(image_layer.data)
        labels = self.axis_labels(image_layer)
        for axis, label in enumerate(labels):
            if label.strip().lower() in {"c", "ch", "channel", "channels"}:
                return axis

        rgb = bool(getattr(image_layer, "rgb", False))
        if len(shape) >= 3 and shape[-1] in {3, 4} and rgb:
            return len(shape) - 1
        if len(shape) == 3:
            return 0
        return None

    def spatial_axes(
        self, image_layer, channel_axis: int | None
    ) -> tuple[int, int]:
        shape = _shape_of(image_layer.data)
        labels = tuple(label.lower() for label in self.axis_labels(image_layer))
        y_axes = [axis for axis, label in enumerate(labels) if label in {"y", "row"}]
        x_axes = [axis for axis, label in enumerate(labels) if label in {"x", "col", "column"}]
        if y_axes and x_axes and y_axes[-1] != x_axes[-1]:
            return y_axes[-1], x_axes[-1]

        candidates = [axis for axis in range(len(shape)) if axis != channel_axis]
        if len(candidates) < 2:
            raise ImageConversionError("The source must contain two spatial axes.")
        return candidates[-2], candidates[-1]

    def current_steps(self, viewer, ndim: int) -> tuple[int, ...]:
        try:
            steps = tuple(int(step) for step in viewer.dims.current_step)
        except (AttributeError, TypeError, ValueError):
            return (0,) * ndim
        if len(steps) >= ndim:
            return steps[-ndim:]
        return (0,) * (ndim - len(steps)) + steps

    def plane_selection(
        self,
        image_layer,
        viewer,
        settings: ImageProcessingSettings,
    ) -> PlaneSelection:
        shape = _shape_of(image_layer.data)
        settings.validate(shape)
        y_axis, x_axis = self.spatial_axes(image_layer, settings.channel_axis)
        steps = self.current_steps(viewer, len(shape))
        indices: list[int | None] = []
        for axis, axis_size in enumerate(shape):
            if axis in {y_axis, x_axis, settings.channel_axis}:
                indices.append(None)
            else:
                indices.append(int(np.clip(steps[axis], 0, int(axis_size) - 1)))
        return PlaneSelection(
            axis_indices=tuple(indices),
            plane_axes=(y_axis, x_axis),
            axis_labels=self.axis_labels(image_layer),
        )

    def sample_id(
        self,
        base_stem: str,
        plane: PlaneSelection,
        channel_axis: int | None,
    ) -> str:
        parts = [_safe_stem(base_stem)]
        for axis, index in enumerate(plane.axis_indices):
            if index is None or axis in plane.plane_axes or axis == channel_axis:
                continue
            label = _safe_axis_label(plane.axis_labels[axis], axis)
            parts.append(f"{label}{index:03d}")
        return "__".join(parts)

    def convert(
        self,
        image_layer,
        viewer,
        settings: ImageProcessingSettings,
        *,
        base_stem: str | None = None,
        invert: bool = False,
    ) -> ConvertedImage:
        shape = _shape_of(image_layer.data)
        plane = self.plane_selection(image_layer, viewer, settings)
        channels = self._extract_channels(image_layer.data, shape, plane, settings)
        reference_shape = next(
            channel.shape for channel in channels if channel is not None
        )

        normalized: list[np.ndarray] = []
        statistics: list[dict[str, float | str | None]] = []
        for channel in channels:
            if channel is None:
                normalized.append(np.zeros(reference_shape, dtype=np.uint8))
                statistics.append({"method": "empty", "lower": None, "upper": None})
                continue
            filtered = apply_image_filter(
                channel,
                method=settings.filter_method,
                radius=settings.filter_radius,
            )
            converted, stats = normalize_to_uint8(
                filtered,
                method=settings.normalization,
                lower=settings.lower,
                upper=settings.upper,
            )
            if invert:
                converted = np.asarray(255 - converted, dtype=np.uint8)
            stats["filter_method"] = settings.filter_method
            stats["filter_radius"] = settings.filter_radius
            normalized.append(converted)
            statistics.append(stats)

        rgb = np.ascontiguousarray(np.stack(normalized, axis=-1), dtype=np.uint8)
        source_name = base_stem or getattr(image_layer, "name", "image")
        return ConvertedImage(
            data=rgb,
            sample_id=self.sample_id(source_name, plane, settings.channel_axis),
            plane=plane,
            settings=settings,
            normalization_stats=tuple(statistics),
            inverted=bool(invert),
        )

    def _extract_channels(
        self,
        data,
        shape: tuple[int, ...],
        plane: PlaneSelection,
        settings: ImageProcessingSettings,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if settings.channel_axis is None:
            image = _extract_plane(data, shape, plane, channel_axis=None, channel=None)
            return image, image, image

        extracted: list[np.ndarray | None] = []
        first_shape: tuple[int, int] | None = None
        for channel in settings.rgb_channels:
            if channel is None:
                extracted.append(None)
                continue
            plane_data = _extract_plane(
                data,
                shape,
                plane,
                channel_axis=settings.channel_axis,
                channel=channel,
            )
            first_shape = plane_data.shape
            extracted.append(plane_data)

        if first_shape is None:
            raise ImageConversionError("Select at least one source channel.")
        return tuple(extracted)  # type: ignore[return-value]


def apply_image_filter(data, *, method: str, radius: int) -> np.ndarray:
    """Filter one 2D source channel before normalization and RGB stacking."""
    values = np.asarray(data)
    if values.ndim != 2:
        raise ImageConversionError(
            f"Expected a 2D channel plane, received shape {values.shape}."
        )
    if method not in IMAGE_FILTERS:
        raise ImageConversionError(f"Unknown image filter: {method!r}.")
    if (
        isinstance(radius, bool)
        or not isinstance(radius, int)
        or not 1 <= radius <= 100
    ):
        raise ImageConversionError("Filter radius must be an integer within 1..100.")
    work = values.astype(np.float64, copy=False)
    if method == "none":
        return work

    footprint = disk(radius).astype(bool)
    if method == "median":
        return median_filter(work, footprint=footprint, mode="reflect")
    if method == "mean":
        kernel = footprint.astype(np.float64)
        kernel /= kernel.sum()
        return convolve(work, kernel, mode="reflect")
    if method == "gaussian":
        coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        sigma = max(radius / 2.0, 0.5)
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        kernel *= footprint
        kernel /= kernel.sum()
        return convolve(work, kernel, mode="reflect")
    if method == "low_pass":
        return _frequency_low_pass(work, radius)
    if method == "white_tophat":
        opened = grey_opening(work, footprint=footprint, mode="reflect")
        return np.asarray(work - opened, dtype=np.float64)
    raise ImageConversionError(f"Unknown image filter: {method!r}.")


def _frequency_low_pass(data: np.ndarray, radius: int) -> np.ndarray:
    """Apply a second-order Butterworth low-pass with reflected boundaries."""
    padding = min(max(8, radius * 2), max(data.shape))
    mode = "reflect" if min(data.shape) > 1 else "edge"
    padded = np.pad(data, padding, mode=mode)
    frequencies_y = np.fft.fftfreq(padded.shape[0])[:, np.newaxis]
    frequencies_x = np.fft.rfftfreq(padded.shape[1])[np.newaxis, :]
    radial_frequency = np.sqrt(frequencies_y**2 + frequencies_x**2)
    cutoff = 0.5 / float(radius)
    transfer = 1.0 / np.sqrt(1.0 + (radial_frequency / cutoff) ** 4)
    spectrum = np.fft.rfft2(padded)
    filtered = np.fft.irfft2(spectrum * transfer, s=padded.shape)
    return np.asarray(
        filtered[padding:-padding, padding:-padding], dtype=np.float64
    )


def normalize_to_uint8(
    data,
    *,
    method: str,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[np.ndarray, dict[str, float | str | None]]:
    values = np.asarray(data)
    if values.ndim != 2:
        raise ImageConversionError(
            f"Expected a 2D channel plane, received shape {values.shape}."
        )
    work = values.astype(np.float64, copy=False)
    finite = np.isfinite(work)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8), {
            "method": method,
            "lower": None,
            "upper": None,
        }
    finite_values = work[finite]

    if method == "min_max":
        low_value = float(np.min(finite_values))
        high_value = float(np.max(finite_values))
    elif method == "simple_max":
        low_value = 0.0
        high_value = float(np.max(finite_values))
    elif method == "percentile":
        if lower is None or upper is None or not 0 <= lower < upper <= 100:
            raise ImageConversionError(
                "Percentile normalization requires 0 ≤ lower < upper ≤ 100."
            )
        low_value, high_value = (
            float(value) for value in np.percentile(finite_values, [lower, upper])
        )
    elif method == "z_score":
        if lower is None or upper is None or lower >= upper:
            raise ImageConversionError(
                "Z-score normalization requires lower < upper."
            )
        mean = float(np.mean(finite_values))
        standard_deviation = float(np.std(finite_values))
        if standard_deviation <= 0 or not math.isfinite(standard_deviation):
            return np.zeros(values.shape, dtype=np.uint8), {
                "method": method,
                "lower": mean,
                "upper": mean,
            }
        low_value = mean + lower * standard_deviation
        high_value = mean + upper * standard_deviation
    elif method == "fixed_range":
        if lower is None or upper is None or lower >= upper:
            raise ImageConversionError("Fixed range requires lower < upper.")
        low_value = float(lower)
        high_value = float(upper)
    elif method == "dtype_range":
        if np.issubdtype(values.dtype, np.integer):
            limits = np.iinfo(values.dtype)
            low_value = float(limits.min)
            high_value = float(limits.max)
        elif np.issubdtype(values.dtype, np.bool_):
            low_value, high_value = 0.0, 1.0
        else:
            raise ImageConversionError(
                "Data type range is only defined for integer and boolean images."
            )
    else:
        raise ImageConversionError(f"Unknown normalization method: {method!r}.")

    stats: dict[str, float | str | None] = {
        "method": method,
        "lower": low_value,
        "upper": high_value,
    }
    if high_value <= low_value or not all(
        math.isfinite(value) for value in (low_value, high_value)
    ):
        return np.zeros(values.shape, dtype=np.uint8), stats

    scaled = (work - low_value) / (high_value - low_value)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8), stats


def _extract_plane(
    data,
    shape: tuple[int, ...],
    plane: PlaneSelection,
    *,
    channel_axis: int | None,
    channel: int | None,
) -> np.ndarray:
    index: list[int | slice] = []
    for axis in range(len(shape)):
        if axis in plane.plane_axes:
            index.append(slice(None))
        elif axis == channel_axis:
            if channel is None:
                raise ImageConversionError("A source channel must be selected.")
            index.append(channel)
        else:
            selected = plane.axis_indices[axis]
            if selected is None:
                raise ImageConversionError(
                    f"Axis {axis} was not resolved to a plane index."
                )
            index.append(selected)

    result = np.asarray(data[tuple(index)])
    if result.ndim != 2:
        raise ImageConversionError(
            f"The selected axes produced shape {result.shape}, not a 2D plane."
        )
    remaining_axes = [axis for axis, item in enumerate(index) if isinstance(item, slice)]
    if tuple(remaining_axes) == plane.plane_axes:
        return result
    if tuple(reversed(remaining_axes)) == plane.plane_axes:
        return result.T
    raise ImageConversionError("Could not order the selected plane as Y, X.")


def _shape_of(data) -> tuple[int, ...]:
    try:
        shape = tuple(int(size) for size in data.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ImageConversionError("The image layer does not expose a valid shape.") from exc
    if len(shape) < 2 or any(size <= 0 for size in shape):
        raise ImageConversionError(f"Invalid image shape: {shape}.")
    return shape


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    return stem or "image"


def _safe_axis_label(value: str, axis: int) -> str:
    label = re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()
    return label or f"a{axis}"


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
