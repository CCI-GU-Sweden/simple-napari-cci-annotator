"""Exact two-pass chunked RGB conversion for a subset of volume recipes.

This is a building block for a disk-backed inference engine; the existing
volume runner does not yet use it. It never approximates a full-plane recipe.
"""

import math

import numpy as np

from ._image_adapter import ImageConversionError, apply_image_filter
from ._tiled_inference import InferenceCancelled


_METHODS = {"min_max", "simple_max", "fixed_range", "dtype_range", "percentile"}


def convert_slice_to_chunks(volume, z_index, output, *, chunk_xy=512, cancelled=None, progress=None):
    """Write RGB uint8 chunks to an array-like destination and return statistics.

    Local filters and recipes with streamable exact bounds are supported. The
    destination must have the full YX3 shape and support slice assignment.
    """
    settings = volume.processing
    if settings.filter_method == "low_pass" or settings.normalization not in _METHODS:
        raise ImageConversionError(
            "Chunked conversion currently requires a local filter and min-max, "
            "simple-max, fixed-range, dtype-range, or supported percentile normalization."
        )
    dtype = np.dtype(volume.source_dtype)
    percentile = settings.normalization == "percentile"
    if percentile and (settings.filter_method != "none" or
                       (not np.issubdtype(dtype, np.integer) and not np.issubdtype(dtype, np.bool_)) or
                       dtype.itemsize > 2):
        raise ImageConversionError("Chunked percentile requires unfiltered 8- or 16-bit integer data.")
    if type(chunk_xy) is not int or chunk_xy < 1:
        raise ImageConversionError("Chunk size must be a positive integer.")
    height, width = volume.geometry.shape[1:]
    if tuple(output.shape) != (height, width, 3) or np.dtype(output.dtype) != np.dtype("uint8"):
        raise ImageConversionError("Chunk output must be a YX3 uint8 array.")
    # Validate the requested source slice before any output is written.
    plane = volume.selection.plane(z_index)
    axes = volume.selection.axes
    channels = (None, None, None) if axes.channel is None else settings.rgb_channels
    distinct = [None] if axes.channel is None else sorted({c for c in channels if c is not None})
    if not distinct:
        raise ImageConversionError("Select at least one source channel.")
    total = 2 * len(distinct) * ((height + chunk_xy - 1) // chunk_xy) * ((width + chunk_xy - 1) // chunk_xy)
    done = 0

    def blocks():
        for y in range(0, height, chunk_xy):
            for x in range(0, width, chunk_xy):
                yield slice(y, min(y + chunk_xy, height)), slice(x, min(x + chunk_xy, width))

    def read(channel, y, x):
        if cancelled is not None and cancelled():
            raise InferenceCancelled("Chunked volume conversion cancelled.")
        index = list(plane.axis_indices)
        radius = settings.filter_radius * (2 if settings.filter_method == "white_tophat" else 1)
        if settings.filter_method == "none":
            radius = 0
        expanded_y = slice(max(0, y.start - radius), min(height, y.stop + radius))
        expanded_x = slice(max(0, x.start - radius), min(width, x.stop + radius))
        index[axes.y], index[axes.x] = expanded_y, expanded_x
        if axes.channel is not None:
            index[axes.channel] = channel
        remaining = [i for i, value in enumerate(index) if isinstance(value, slice)]
        raw = np.asarray(volume.data[tuple(index)])
        raw = raw.transpose(remaining.index(axes.y), remaining.index(axes.x))
        filtered = apply_image_filter(raw, method=settings.filter_method, radius=settings.filter_radius)
        return filtered[y.start - expanded_y.start:y.stop - expanded_y.start,
                        x.start - expanded_x.start:x.stop - expanded_x.start]

    bounds = {}
    for channel in distinct:
        low, high = math.inf, -math.inf
        if percentile:
            limits = np.iinfo(dtype) if dtype.kind in "iu" else None
            minimum = int(limits.min) if limits is not None else 0
            maximum = int(limits.max) if limits is not None else 1
            histogram = np.zeros(maximum - minimum + 1, dtype=np.uint64)
        for y, x in blocks():
            raw = read(channel, y, x)
            done += 1
            if progress is not None:
                progress(done, total)
            values = raw.astype(np.float64, copy=False)
            finite = values[np.isfinite(values)]
            if finite.size:
                low = min(low, float(finite.min()))
                high = max(high, float(finite.max()))
                if percentile:
                    histogram += np.bincount(finite.astype(np.int64) - minimum,
                                             minlength=len(histogram)).astype(np.uint64)
        if not math.isfinite(low):
            bounds[channel] = (None, None)
        elif percentile:
            cumulative = np.cumsum(histogram)
            count = int(cumulative[-1])

            def quantile(percent):
                position = (count - 1) * (float(percent) / 100.0)
                lower_index = int(math.floor(position))
                fraction = position - lower_index
                a = int(np.searchsorted(cumulative, lower_index + 1)) + minimum
                b = int(np.searchsorted(cumulative, min(lower_index + 2, count))) + minimum
                return float(np.percentile(np.asarray([a, b], dtype=np.float64), fraction * 100))

            bounds[channel] = (quantile(settings.lower), quantile(settings.upper))
        elif settings.normalization == "simple_max":
            bounds[channel] = (0.0, high)
        elif settings.normalization == "fixed_range":
            bounds[channel] = (float(settings.lower), float(settings.upper))
        elif settings.normalization == "dtype_range":
            dtype = np.dtype(volume.source_dtype)
            if np.issubdtype(dtype, np.integer):
                limits = np.iinfo(dtype)
                bounds[channel] = (float(limits.min), float(limits.max))
            elif np.issubdtype(dtype, np.bool_):
                bounds[channel] = (0.0, 1.0)
            else:
                raise ImageConversionError("Data type range is only defined for integer and boolean images.")
        else:
            bounds[channel] = (low, high)

    statistics = []
    for channel in channels:
        if axes.channel is not None and channel is None:
            statistics.append({"method": "empty", "lower": None, "upper": None})
        else:
            low, high = bounds[channel]
            statistics.append({"method": settings.normalization, "lower": low, "upper": high,
                               "filter_method": settings.filter_method, "filter_radius": settings.filter_radius})
    for y, x in blocks():
        converted = {}
        for channel in distinct:
            values = read(channel, y, x).astype(np.float64, copy=False)
            done += 1
            if progress is not None:
                progress(done, total)
            low, high = bounds[channel]
            if low is None or high is None or high <= low or not all(map(math.isfinite, (low, high))):
                result = np.zeros(values.shape, dtype=np.uint8)
            else:
                scaled = (values - low) / (high - low)
                scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
                result = np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
            if volume.invert:
                result = np.asarray(255 - result, dtype=np.uint8)
            converted[channel] = result
        rgb = np.zeros((y.stop - y.start, x.stop - x.start, 3), dtype=np.uint8)
        for color, channel in enumerate(channels):
            if axes.channel is None or channel is not None:
                rgb[..., color] = converted[channel]
        output[y, x, :] = rgb
    return tuple(statistics)
