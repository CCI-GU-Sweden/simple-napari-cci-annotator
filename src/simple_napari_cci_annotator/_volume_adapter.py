"""Validated volume inputs and slice conversion without Qt or napari."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ._image_adapter import (
    ConvertedImage,
    ImageAdapter,
    ImageConversionError,
    ImageProcessingSettings,
    PlaneSelection,
    _shape_of,
)


class VolumeInputError(ImageConversionError):
    """A volume cannot be processed with the requested axes or conversion."""


def _is_index(value) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, np.integer)
    )


@dataclass(frozen=True)
class VolumeAxes:
    """Source axis positions, separate from channel identity."""

    z: int
    y: int
    x: int
    channel: int | None = None

    @property
    def spatial(self) -> tuple[int, int, int]:
        return self.z, self.y, self.x

    def validate(self, shape: tuple[int, ...]) -> None:
        axes = self.spatial
        if self.channel is not None:
            axes += (self.channel,)
        if any(
            not _is_index(axis) or not 0 <= axis < len(shape) for axis in axes
        ):
            raise VolumeInputError(
                "Volume axes must be valid source axis indices."
            )
        if len(set(axes)) != len(axes):
            raise VolumeInputError("Z, Y, X, and channel axes must be distinct.")

    def to_mapping(self) -> dict[str, int | None]:
        return {
            "z": int(self.z),
            "y": int(self.y),
            "x": int(self.x),
            "channel": None if self.channel is None else int(self.channel),
        }


@dataclass(frozen=True)
class VolumeSelection:
    """Frozen acquisition indices and an exclusive upper Z bound."""

    axes: VolumeAxes
    axis_labels: tuple[str, ...]
    fixed_indices: tuple[int | None, ...]
    z_start: int
    z_stop: int

    @property
    def z_indices(self) -> range:
        return range(self.z_start, self.z_stop)

    def plane(self, z_index: int) -> PlaneSelection:
        if not _is_index(z_index) or z_index not in self.z_indices:
            raise VolumeInputError(
                "Z index is outside the selected volume range."
            )
        indices = list(self.fixed_indices)
        indices[self.axes.z] = int(z_index)
        return PlaneSelection(
            axis_indices=tuple(indices),
            plane_axes=(self.axes.y, self.axes.x),
            axis_labels=self.axis_labels,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "axes": self.axes.to_mapping(),
            "axis_labels": list(self.axis_labels),
            "fixed_indices": list(self.fixed_indices),
            "z_range": [self.z_start, self.z_stop],
        }


@dataclass(frozen=True)
class VolumeGeometry:
    """Canonical ZYX shape and transforms for the selected output volume."""

    shape: tuple[int, int, int]
    scale: tuple[float, float, float]
    translate: tuple[float, float, float]
    units: tuple[str | None, str | None, str | None]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "axes": "ZYX",
            "shape": list(self.shape),
            "scale": list(self.scale),
            "translate": list(self.translate),
            "units": list(self.units),
        }


@dataclass(frozen=True)
class ConvertedVolumeSlice:
    """Converted RGB pixels and provenance at an original source Z index."""

    z_index: int
    image: ConvertedImage


@dataclass(frozen=True)
class PreparedVolume:
    """Captured input contract. Pixels stay in the source array until sliced.

    The array is referenced, not copied; callers must keep its backing store
    open and prevent pixel edits while processing it.
    """

    data: Any
    source_name: str
    source_shape: tuple[int, ...]
    source_dtype: str
    selection: VolumeSelection
    geometry: VolumeGeometry
    project_processing: ImageProcessingSettings
    processing: ImageProcessingSettings
    invert: bool = False

    def convert_slice(self, z_index: int) -> ConvertedVolumeSlice:
        if _shape_of(self.data) != self.source_shape:
            raise VolumeInputError(
                "Source shape changed after volume preparation."
            )
        image = ImageAdapter().convert_plane(
            self.data,
            self.selection.plane(z_index),
            self.processing,
            base_stem=self.source_name,
            invert=self.invert,
        )
        return ConvertedVolumeSlice(int(z_index), image)

    def iter_slices(self) -> Iterator[ConvertedVolumeSlice]:
        for z_index in self.selection.z_indices:
            yield self.convert_slice(z_index)

    def to_mapping(self) -> dict[str, Any]:
        """Serializable input metadata, without evaluating source pixels."""
        return {
            "source_name": self.source_name,
            "source_shape": list(self.source_shape),
            "source_dtype": self.source_dtype,
            "selection": self.selection.to_mapping(),
            "geometry": self.geometry.to_mapping(),
            "project_processing": self.project_processing.to_mapping(),
            "resolved_processing": self.processing.to_mapping(),
            "inverted": self.invert,
        }


class VolumeAdapter:
    """Resolve declared volumes without guessing axes from shape."""

    def infer_axes(self, image_layer) -> VolumeAxes:
        """Read declared axes for UI defaults; ambiguous input requires selection."""
        shape = _shape_of(image_layer.data)
        axes = _axes_from_labels(_declared_labels(image_layer, len(shape)))
        axes.validate(shape)
        return axes

    def prepare(
        self,
        image_layer,
        processing: ImageProcessingSettings,
        *,
        axes: VolumeAxes | None = None,
        fixed_indices: Mapping[int, int] | None = None,
        z_range: tuple[int, int] | None = None,
        invert: bool = False,
    ) -> PreparedVolume:
        data = image_layer.data
        shape = _shape_of(data)
        labels = _declared_labels(image_layer, len(shape))
        if axes is None:
            axes = _axes_from_labels(labels)
        axes.validate(shape)
        processing.validate()
        if (axes.channel is None) != (processing.channel_axis is None):
            raise VolumeInputError(
                "Volume channel layout must match the project's grayscale "
                "or multichannel conversion."
            )
        resolved = replace(processing, channel_axis=axes.channel)
        resolved.validate(shape)

        # An explicit mapping is authoritative; preserve source labels for
        # other axes but name mapped axes consistently in slice provenance.
        named = list(labels or tuple(f"A{axis}" for axis in range(len(shape))))
        for axis, name in zip(axes.spatial, ("Z", "Y", "X"), strict=True):
            named[axis] = name
        if axes.channel is not None:
            named[axes.channel] = "C"
        indices = _fixed_indices(shape, axes, fixed_indices or {})
        start, stop = _z_range(shape[axes.z], z_range)
        selection = VolumeSelection(axes, tuple(named), indices, start, stop)
        geometry = _geometry(image_layer, shape, selection)
        try:
            dtype = np.dtype(data.dtype)
        except (AttributeError, TypeError, ValueError) as exc:
            raise VolumeInputError(
                "Source must expose a valid numeric dtype."
            ) from exc
        if dtype.kind not in "buif":
            raise VolumeInputError(
                "Volume pixels must be real numeric or boolean values."
            )
        return PreparedVolume(
            data=data,
            source_name=str(getattr(image_layer, "name", "image")),
            source_shape=shape,
            source_dtype=str(dtype),
            selection=selection,
            geometry=geometry,
            project_processing=processing,
            processing=resolved,
            invert=bool(invert),
        )


def _declared_labels(image_layer, ndim: int) -> tuple[str, ...] | None:
    metadata = getattr(image_layer, "metadata", {}) or {}
    raw = metadata.get("axes")
    if raw is None:
        raw = metadata.get("axis_labels")
    if raw is None:
        raw = getattr(image_layer, "axis_labels", None)
    if raw is None:
        return None
    if not isinstance(raw, (str, tuple, list)):
        raise VolumeInputError(
            "Axis labels must be a string or sequence of names."
        )
    if len(raw) == 0:
        return None
    if bool(getattr(image_layer, "rgb", False)) and len(raw) == ndim - 1:
        raw = tuple(raw) + ("C",)
    if len(raw) != ndim:
        raise VolumeInputError(
            "Declared axis labels must match the source dimensions."
        )
    return tuple(str(label) for label in raw)


def _axes_from_labels(labels: tuple[str, ...] | None) -> VolumeAxes:
    if labels is None:
        raise VolumeInputError(
            "Select explicit volume axes; source axes are ambiguous."
        )
    aliases = {
        "z": "Z", "depth": "Z", "y": "Y", "row": "Y",
        "x": "X", "col": "X", "column": "X",
        "c": "C", "ch": "C", "channel": "C", "channels": "C",
    }
    found: dict[str, int] = {}
    for axis, label in enumerate(labels):
        name = aliases.get(label.strip().lower())
        if name is None:
            continue
        if name in found:
            raise VolumeInputError(
                "Duplicate spatial/channel labels require explicit axes."
            )
        found[name] = axis
    if not all(name in found for name in ("Z", "Y", "X")):
        raise VolumeInputError(
            "Select explicit Z, Y, X axes; source axes are ambiguous."
        )
    return VolumeAxes(found["Z"], found["Y"], found["X"], found.get("C"))


def _fixed_indices(shape, axes, requested) -> tuple[int | None, ...]:
    unsliced = set(axes.spatial)
    if axes.channel is not None:
        unsliced.add(axes.channel)
    for axis, index in requested.items():
        if (
            not _is_index(axis)
            or not 0 <= axis < len(shape)
            or axis in unsliced
        ):
            raise VolumeInputError(
                "Fixed indices may select only acquisition axes."
            )
        if not _is_index(index) or not 0 <= index < shape[axis]:
            raise VolumeInputError(f"Invalid acquisition index for axis {axis}.")
    extra = set(range(len(shape))) - unsliced
    if extra - set(requested):
        raise VolumeInputError(
            "Select a fixed index for every acquisition axis."
        )
    return tuple(
        int(requested[axis]) if axis in extra else None
        for axis in range(len(shape))
    )


def _z_range(depth, requested) -> tuple[int, int]:
    if requested is None:
        return 0, depth
    if len(requested) != 2 or any(
        not _is_index(value) for value in requested
    ):
        raise VolumeInputError(
            "Z range must contain integer start and exclusive stop."
        )
    start, stop = requested
    if not 0 <= start < stop <= depth:
        raise VolumeInputError("Z range must be nonempty and within the source depth.")
    return int(start), int(stop)


def _geometry(image_layer, shape, selection) -> VolumeGeometry:
    # Napari omits the last RGB channel dimension from its layer transforms.
    rgb = bool(getattr(image_layer, "rgb", False))
    if rgb and (
        selection.axes.channel != len(shape) - 1 or shape[-1] not in (3, 4)
    ):
        raise VolumeInputError(
            "RGB sources require a final channel axis of size 3 or 4."
        )
    transform_ndim = len(shape) - int(rgb)
    for attribute, default in (("rotate", 0), ("shear", 0)):
        value = np.asarray(
            getattr(image_layer, attribute, default), dtype=float
        )
        identity = np.eye(transform_ndim) if value.ndim == 2 else 0
        if (
            (value.ndim == 2 and value.shape != (transform_ndim,) * 2)
            or not np.all(np.isfinite(value))
            or not np.allclose(value, identity)
        ):
            raise VolumeInputError(
                "Rotated or sheared volume transforms are not supported yet."
            )
    affine = getattr(image_layer, "affine", None)
    if affine is not None:
        matrix = np.asarray(
            getattr(affine, "affine_matrix", affine), dtype=float
        )
        if (
            matrix.shape != (transform_ndim + 1,) * 2
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix, np.eye(transform_ndim + 1))
        ):
            raise VolumeInputError(
                "Additional affine volume transforms are not supported yet."
            )
    scale = np.asarray(
        getattr(image_layer, "scale", np.ones(transform_ndim)), dtype=float
    )
    translate = np.asarray(
        getattr(image_layer, "translate", np.zeros(transform_ndim)), dtype=float
    )
    if (
        scale.shape != (transform_ndim,)
        or translate.shape != (transform_ndim,)
        or not np.all(np.isfinite(scale))
        or not np.all(np.isfinite(translate))
        or np.any(scale == 0)
    ):
        raise VolumeInputError(
            "Source scale and translation must be finite axis vectors "
            "with nonzero scale."
        )
    units = getattr(image_layer, "units", None)
    if units is None:
        units = (None,) * transform_ndim
    if len(units) != transform_ndim:
        raise VolumeInputError(
            "Source units must match the layer transform dimensions."
        )
    spatial = selection.axes.spatial
    output_translate = [float(translate[axis]) for axis in spatial]
    output_translate[0] += selection.z_start * float(scale[selection.axes.z])
    return VolumeGeometry(
        shape=(
            selection.z_stop - selection.z_start,
            shape[spatial[1]],
            shape[spatial[2]],
        ),
        scale=tuple(float(scale[axis]) for axis in spatial),
        translate=tuple(output_translate),
        units=tuple(
            None if units[axis] is None else str(units[axis]) for axis in spatial
        ),
    )
