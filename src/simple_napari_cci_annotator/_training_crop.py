from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TrainingCropError(ValueError):
    """Raised when a training crop cannot be created safely."""


@dataclass(frozen=True)
class CropBounds:
    """Fixed-size crop canvas anchored in source-image pixel coordinates."""

    y0: int
    x0: int
    size: int
    source_height: int
    source_width: int

    @property
    def y1(self) -> int:
        return self.y0 + self.size

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def valid_height(self) -> int:
        return max(0, min(self.size, self.source_height - self.y0))

    @property
    def valid_width(self) -> int:
        return max(0, min(self.size, self.source_width - self.x0))

    @property
    def pad_bottom(self) -> int:
        return self.size - self.valid_height

    @property
    def pad_right(self) -> int:
        return self.size - self.valid_width

    def as_rectangle(self) -> np.ndarray:
        return np.asarray(
            [
                [self.y0, self.x0],
                [self.y0, self.x1],
                [self.y1, self.x1],
                [self.y1, self.x0],
            ],
            dtype=float,
        )

    def to_mapping(self) -> dict[str, int]:
        return {
            "y0": self.y0,
            "y1": self.y1,
            "x0": self.x0,
            "x1": self.x1,
            "size": self.size,
            "valid_height": self.valid_height,
            "valid_width": self.valid_width,
            "pad_bottom": self.pad_bottom,
            "pad_right": self.pad_right,
        }


@dataclass(frozen=True)
class CropBoxResult:
    rectangles: tuple[np.ndarray, ...]
    properties: dict[str, np.ndarray]
    clipped_count: int
    rejected_indices: tuple[int, ...]
    ignored_count: int


def crop_bounds_from_center(
    center_y: float,
    center_x: float,
    size: int,
    source_height: int,
    source_width: int,
) -> CropBounds:
    """Create a fixed-size crop, clamping its movable origin to source pixels."""
    _validate_dimensions(size, source_height, source_width)
    maximum_y0 = max(0, source_height - size)
    maximum_x0 = max(0, source_width - size)
    y0 = max(0, min(int(round(center_y - size / 2)), maximum_y0))
    x0 = max(0, min(int(round(center_x - size / 2)), maximum_x0))
    return CropBounds(y0, x0, size, source_height, source_width)


def crop_bounds_from_rectangle(
    rectangle: np.ndarray,
    size: int,
    source_height: int,
    source_width: int,
) -> CropBounds:
    points = np.asarray(rectangle, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        raise TrainingCropError("Crop selection must be a 2D rectangle.")
    if not np.all(np.isfinite(points)):
        raise TrainingCropError("Crop selection coordinates must be finite.")
    center_y = float(np.min(points[:, 0]) + np.max(points[:, 0])) / 2.0
    center_x = float(np.min(points[:, 1]) + np.max(points[:, 1])) / 2.0
    return crop_bounds_from_center(
        center_y, center_x, size, source_height, source_width
    )


def extract_padded_crop(
    image: np.ndarray, bounds: CropBounds, *, padding_value: int = 114
) -> np.ndarray:
    """Extract source pixels into an exact square canvas without resizing."""
    array = np.asarray(image)
    if array.ndim not in {2, 3}:
        raise TrainingCropError("Converted crop source must be a 2D or RGB image.")
    if tuple(array.shape[:2]) != (bounds.source_height, bounds.source_width):
        raise TrainingCropError(
            "Crop bounds do not match the converted source image shape."
        )
    if not 0 <= int(padding_value) <= 255:
        raise TrainingCropError("Padding value must be within 0..255.")
    output_shape = (bounds.size, bounds.size, *array.shape[2:])
    output = np.full(output_shape, int(padding_value), dtype=array.dtype)
    source = array[
        bounds.y0 : bounds.y0 + bounds.valid_height,
        bounds.x0 : bounds.x0 + bounds.valid_width,
    ]
    output[: bounds.valid_height, : bounds.valid_width] = source
    return np.ascontiguousarray(output)


def crop_rectangles(
    rectangles: tuple[np.ndarray, ...],
    properties: dict[str, np.ndarray],
    bounds: CropBounds,
    *,
    min_retained_area: float = 0.90,
    max_clip_pixels: float = 10.0,
    max_clip_fraction: float = 0.10,
) -> CropBoxResult:
    """Translate boxes into a crop and reject severe artificial truncation."""
    if not 0 < min_retained_area <= 1:
        raise TrainingCropError("Minimum retained area must be within (0, 1].")
    count = len(rectangles)
    accepted_rectangles: list[np.ndarray] = []
    accepted_indices: list[int] = []
    rejected: list[int] = []
    clipped_count = 0
    ignored_count = 0
    crop_x1 = bounds.x0 + bounds.valid_width
    crop_y1 = bounds.y0 + bounds.valid_height

    for index, raw_rectangle in enumerate(rectangles):
        rectangle = np.asarray(raw_rectangle, dtype=float)
        if rectangle.ndim != 2 or rectangle.shape[1] != 2 or len(rectangle) < 2:
            raise TrainingCropError(
                f"Box {index} does not contain valid rectangle vertices."
            )
        y0 = float(np.min(rectangle[:, 0]))
        y1 = float(np.max(rectangle[:, 0]))
        x0 = float(np.min(rectangle[:, 1]))
        x1 = float(np.max(rectangle[:, 1]))
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            raise TrainingCropError(f"Box {index} has zero width or height.")
        clipped_x0 = max(x0, bounds.x0)
        clipped_y0 = max(y0, bounds.y0)
        clipped_x1 = min(x1, crop_x1)
        clipped_y1 = min(y1, crop_y1)
        if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
            ignored_count += 1
            continue
        clipped_width = clipped_x1 - clipped_x0
        clipped_height = clipped_y1 - clipped_y0
        retained = clipped_width * clipped_height / (width * height)
        loss_x = width - clipped_width
        loss_y = height - clipped_height
        allowed_x = min(max_clip_pixels, max_clip_fraction * width)
        allowed_y = min(max_clip_pixels, max_clip_fraction * height)
        was_clipped = loss_x > 1e-6 or loss_y > 1e-6
        if was_clipped and (
            retained < min_retained_area
            or loss_x > allowed_x + 1e-6
            or loss_y > allowed_y + 1e-6
        ):
            rejected.append(index)
            continue
        if was_clipped:
            clipped_count += 1
        accepted_indices.append(index)
        accepted_rectangles.append(
            np.asarray(
                [
                    [clipped_y0 - bounds.y0, clipped_x0 - bounds.x0],
                    [clipped_y0 - bounds.y0, clipped_x1 - bounds.x0],
                    [clipped_y1 - bounds.y0, clipped_x1 - bounds.x0],
                    [clipped_y1 - bounds.y0, clipped_x0 - bounds.x0],
                ],
                dtype=float,
            )
        )

    accepted_properties: dict[str, np.ndarray] = {}
    for key, raw_values in properties.items():
        values = np.asarray(raw_values)
        if values.shape[:1] == (count,):
            accepted_properties[key] = values[accepted_indices].copy()
    return CropBoxResult(
        rectangles=tuple(accepted_rectangles),
        properties=accepted_properties,
        clipped_count=clipped_count,
        rejected_indices=tuple(rejected),
        ignored_count=ignored_count,
    )


def validate_boxes_within_valid_crop(
    rectangles: tuple[np.ndarray, ...], bounds: CropBounds
) -> None:
    """Reject annotations that extend into synthetic crop padding."""
    errors: list[str] = []
    for index, raw_rectangle in enumerate(rectangles):
        rectangle = np.asarray(raw_rectangle, dtype=float)
        if rectangle.ndim != 2 or rectangle.shape[1] != 2 or len(rectangle) < 2:
            errors.append(f"Box {index} is not a valid rectangle.")
            continue
        y0 = float(np.min(rectangle[:, 0]))
        y1 = float(np.max(rectangle[:, 0]))
        x0 = float(np.min(rectangle[:, 1]))
        x1 = float(np.max(rectangle[:, 1]))
        if (
            x0 < -1e-6
            or y0 < -1e-6
            or x1 > bounds.valid_width + 1e-6
            or y1 > bounds.valid_height + 1e-6
        ):
            errors.append(
                f"Box {index} extends into padded pixels; move or resize it "
                "inside the valid crop area."
            )
    if errors:
        raise TrainingCropError("\n".join(errors))


def crop_sample_id(source_sample_id: str, bounds: CropBounds) -> str:
    return (
        f"{source_sample_id}__crop_y{bounds.y0:06d}_x{bounds.x0:06d}"
        f"_s{bounds.size}"
    )


def _validate_dimensions(size: int, height: int, width: int) -> None:
    if size not in {512, 1024}:
        raise TrainingCropError("Training crop size must be 512 or 1024.")
    if height <= 0 or width <= 0:
        raise TrainingCropError("Source image dimensions must be positive.")
