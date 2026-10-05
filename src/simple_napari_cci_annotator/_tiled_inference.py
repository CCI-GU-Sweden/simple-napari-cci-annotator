from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np


class InferenceError(RuntimeError):
    """Raised when tiled detection cannot be completed safely."""


class InferenceCancelled(InferenceError):
    """Raised after a cooperative cancellation request."""


@dataclass(frozen=True)
class Tile:
    index: int
    row: int
    column: int
    y0: int
    x0: int
    y1: int
    x1: int
    owner_y0: float
    owner_x0: float
    owner_y1: float
    owner_x1: float

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    def owns(self, x: float, y: float) -> bool:
        return (
            self.owner_x0 <= x < self.owner_x1
            and self.owner_y0 <= y < self.owner_y1
        )


@dataclass(frozen=True)
class TilePlan:
    image_height: int
    image_width: int
    tile_size: int
    overlap: int
    tiles: tuple[Tile, ...]


@dataclass(frozen=True)
class RawDetection:
    """One model detection in tile-local XYXY coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


@dataclass(frozen=True)
class Detection:
    """One detection mapped into full-image XYXY coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    tile_id: int
    owned: bool

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def as_napari_rectangle(self) -> np.ndarray:
        return np.asarray(
            [
                [self.y1, self.x1],
                [self.y1, self.x2],
                [self.y2, self.x2],
                [self.y2, self.x1],
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class InferenceSettings:
    tile_size: int = 1024
    overlap: int = 205
    confidence: float = 0.25
    model_iou: float = 0.45
    merge_iou: float = 0.50
    max_detections: int = 300
    device: str | int = "cpu"

    def validate(self) -> None:
        if self.tile_size < 64:
            raise InferenceError("Tile size must be at least 64 pixels.")
        if not 0 <= self.overlap < self.tile_size:
            raise InferenceError("Tile overlap must be smaller than the tile size.")
        for name, value in (
            ("confidence", self.confidence),
            ("model IoU", self.model_iou),
            ("merge IoU", self.merge_iou),
        ):
            if not 0 <= value <= 1:
                raise InferenceError(f"{name} must be between 0 and 1.")
        if self.max_detections < 1:
            raise InferenceError("Maximum detections must be at least 1.")


def create_tile_plan(
    image_height: int,
    image_width: int,
    tile_size: int = 1024,
    overlap: int = 205,
) -> TilePlan:
    if image_height < 1 or image_width < 1:
        raise InferenceError("The image must have positive spatial dimensions.")
    if tile_size < 1 or not 0 <= overlap < tile_size:
        raise InferenceError(
            "Tile size must be positive and overlap must be in [0, tile size)."
        )

    y_starts = _axis_starts(image_height, tile_size, overlap)
    x_starts = _axis_starts(image_width, tile_size, overlap)
    y_owners = _ownership_intervals(y_starts, image_height, tile_size)
    x_owners = _ownership_intervals(x_starts, image_width, tile_size)
    tiles: list[Tile] = []
    for row, y0 in enumerate(y_starts):
        for column, x0 in enumerate(x_starts):
            owner_y0, owner_y1 = y_owners[row]
            owner_x0, owner_x1 = x_owners[column]
            tiles.append(
                Tile(
                    index=len(tiles),
                    row=row,
                    column=column,
                    y0=y0,
                    x0=x0,
                    y1=min(y0 + tile_size, image_height),
                    x1=min(x0 + tile_size, image_width),
                    owner_y0=owner_y0,
                    owner_x0=owner_x0,
                    owner_y1=owner_y1,
                    owner_x1=owner_x1,
                )
            )
    return TilePlan(image_height, image_width, tile_size, overlap, tuple(tiles))


def extract_padded_tile(image: np.ndarray, tile: Tile, tile_size: int) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in {2, 3}:
        raise InferenceError("Inference input must be a 2D or RGB image.")
    cropped = array[tile.y0 : tile.y1, tile.x0 : tile.x1]
    pad_y = tile_size - tile.height
    pad_x = tile_size - tile.width
    if pad_y <= 0 and pad_x <= 0:
        return np.ascontiguousarray(cropped)
    pad_width = [(0, max(0, pad_y)), (0, max(0, pad_x))]
    if array.ndim == 3:
        pad_width.append((0, 0))
    mode = "reflect" if tile.height > 1 and tile.width > 1 else "edge"
    return np.ascontiguousarray(np.pad(cropped, pad_width, mode=mode))


def map_tile_detections(
    detections: Iterable[RawDetection], tile: Tile
) -> list[Detection]:
    mapped: list[Detection] = []
    for box in detections:
        values = np.asarray(
            [box.x1, box.y1, box.x2, box.y2, box.confidence], dtype=float
        )
        if not np.all(np.isfinite(values)):
            continue
        local_x1 = float(np.clip(min(box.x1, box.x2), 0, tile.width))
        local_x2 = float(np.clip(max(box.x1, box.x2), 0, tile.width))
        local_y1 = float(np.clip(min(box.y1, box.y2), 0, tile.height))
        local_y2 = float(np.clip(max(box.y1, box.y2), 0, tile.height))
        if local_x2 <= local_x1 or local_y2 <= local_y1:
            continue
        center_x = (box.x1 + box.x2) / 2
        center_y = (box.y1 + box.y2) / 2
        if not 0 <= center_x < tile.width or not 0 <= center_y < tile.height:
            continue
        global_x1 = local_x1 + tile.x0
        global_x2 = local_x2 + tile.x0
        global_y1 = local_y1 + tile.y0
        global_y2 = local_y2 + tile.y0
        global_center_x = (global_x1 + global_x2) / 2
        global_center_y = (global_y1 + global_y2) / 2
        mapped.append(
            Detection(
                x1=global_x1,
                y1=global_y1,
                x2=global_x2,
                y2=global_y2,
                confidence=float(box.confidence),
                class_id=int(box.class_id),
                tile_id=tile.index,
                owned=tile.owns(global_center_x, global_center_y),
            )
        )
    return mapped


def box_iou(first: Detection, second: Detection) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_nms(
    detections: Sequence[Detection], iou_threshold: float
) -> tuple[Detection, ...]:
    """Merge duplicates, preferring the tile that owns a detection's center."""
    if not 0 <= iou_threshold <= 1:
        raise InferenceError("Merge IoU must be between 0 and 1.")
    ordered = sorted(
        detections,
        key=lambda item: (item.owned, item.confidence),
        reverse=True,
    )
    kept: list[Detection] = []
    for candidate in ordered:
        if any(
            existing.class_id == candidate.class_id
            and box_iou(existing, candidate) > iou_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return tuple(sorted(kept, key=lambda item: item.confidence, reverse=True))


class TiledInferenceEngine:
    """Run a detector tile-by-tile and merge results in source coordinates."""

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(
        self,
        image: np.ndarray,
        settings: InferenceSettings,
        *,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[Detection, ...]:
        settings.validate()
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise InferenceError("YOLO detection expects an RGB image with shape Y×X×3.")
        plan = create_tile_plan(
            array.shape[0], array.shape[1], settings.tile_size, settings.overlap
        )
        candidates: list[Detection] = []
        total = len(plan.tiles)
        for number, tile in enumerate(plan.tiles, start=1):
            if cancelled is not None and cancelled():
                raise InferenceCancelled("Inference cancelled by the user.")
            if progress is not None:
                progress(number - 1, total, f"Predicting tile {number}/{total}")
            tile_image = extract_padded_tile(array, tile, settings.tile_size)
            raw = self.predictor.predict_tile(tile_image, settings)
            candidates.extend(map_tile_detections(raw, tile))
        if progress is not None:
            progress(total, total, f"Merging {len(candidates)} tile detections")
        if cancelled is not None and cancelled():
            raise InferenceCancelled("Inference cancelled by the user.")
        return class_aware_nms(candidates, settings.merge_iou)


def _axis_starts(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def _ownership_intervals(
    starts: Sequence[int], length: int, tile_size: int
) -> tuple[tuple[float, float], ...]:
    boundaries = [0.0]
    for left, right in zip(starts, starts[1:]):
        left_end = min(left + tile_size, length)
        boundaries.append((left_end + right) / 2)
    boundaries.append(float(length))
    return tuple(zip(boundaries, boundaries[1:]))
