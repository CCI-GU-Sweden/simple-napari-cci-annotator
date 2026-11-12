"""Experimental connected-component extraction across disk-backed chunks."""

from dataclasses import replace

import numpy as np
from skimage.measure import label

from ._segmentation_io import SegmentationError
from ._volume_chunked_fusion import _update_geometry
from ._segmentation_tiling import _UnionFind
from ._tiled_inference import InferenceCancelled


def split_components_to_chunks(mask, scratch, output, records, *, chunk_xy=512, cancelled=None):
    """Split equal-valued 4-connected regions with deterministic slice IDs."""
    if (len(mask.shape) != 2 or tuple(scratch.shape) != tuple(mask.shape) or
            tuple(output.shape) != tuple(mask.shape) or np.dtype(mask.dtype) != np.dtype("uint32") or
            np.dtype(scratch.dtype) != np.dtype("uint64") or
            np.dtype(output.dtype) != np.dtype("uint32") or type(chunk_xy) is not int or chunk_xy < 1):
        raise SegmentationError("Invalid chunked component arrays.")
    height, width = mask.shape

    def check():
        if cancelled is not None and cancelled():
            raise InferenceCancelled("Chunked component extraction cancelled.")

    def blocks():
        for y in range(0, height, chunk_xy):
            for x in range(0, width, chunk_xy):
                yield slice(y, min(y + chunk_xy, height)), slice(x, min(x + chunk_xy, width))

    source_ids = {}
    first_pixels = {}
    next_id = 1
    for y, x in blocks():
        check()
        data = np.asarray(mask[y, x], dtype=np.uint32)
        components = label(data, background=0, connectivity=1)
        count = int(components.max(initial=0))
        if next_id + count > np.iinfo(np.uint64).max:
            raise SegmentationError("Chunk component IDs exceed uint64 capacity.")
        component_source = np.zeros(count + 1, dtype=np.uint32)
        component_source[components.ravel()] = data.ravel()
        positions = np.arange(data.size, dtype=np.int64).reshape(data.shape)
        positions = (positions // data.shape[1] + y.start) * width + positions % data.shape[1] + x.start
        first = np.full(count + 1, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(first, components.ravel(), positions.ravel())
        lookup = np.arange(next_id - 1, next_id + count, dtype=np.uint64)
        lookup[0] = 0
        scratch[y, x] = lookup[components]
        for local in range(1, count + 1):
            identifier = next_id + local - 1
            source = int(component_source[local])
            if source not in records:
                raise SegmentationError(f"Missing slice instance metadata: {source}.")
            source_ids[identifier] = source
            first_pixels[identifier] = int(first[local])
        next_id += count

    union = _UnionFind(source_ids)

    def connect(a, b, first, second):
        matched = (a != 0) & (a == b)
        if not np.any(matched):
            return
        pairs = np.unique(np.stack((first[matched], second[matched]), axis=1), axis=0)
        for left, right in pairs:
            union.union(int(left), int(right))

    for y in range(chunk_xy, height, chunk_xy):
        for x in range(0, width, chunk_xy):
            check()
            section = slice(x, min(x + chunk_xy, width))
            connect(mask[y - 1, section], mask[y, section], scratch[y - 1, section], scratch[y, section])
    for x in range(chunk_xy, width, chunk_xy):
        for y in range(0, height, chunk_xy):
            check()
            section = slice(y, min(y + chunk_xy, height))
            connect(mask[section, x - 1], mask[section, x], scratch[section, x - 1], scratch[section, x])
    groups = {}
    for identifier in source_ids:
        root = union.find(identifier)
        if root not in groups:
            groups[root] = (source_ids[identifier], first_pixels[identifier])
        else:
            source, first = groups[root]
            groups[root] = (source, min(first, first_pixels[identifier]))
    ordered = sorted(groups, key=lambda root: groups[root])
    if len(ordered) > np.iinfo(np.uint32).max:
        raise SegmentationError("Slice components exceed uint32 capacity.")
    root_ids = {root: index + 1 for index, root in enumerate(ordered)}
    keys = np.asarray(sorted(source_ids), dtype=np.uint64)
    values = np.asarray([root_ids[union.find(int(key))] for key in keys], dtype=np.uint32)
    geometry = {}
    for y, x in blocks():
        check()
        block = np.asarray(scratch[y, x], dtype=np.uint64)
        result = np.zeros(block.shape, dtype=np.uint32)
        active = block != 0
        if np.any(active):
            result[active] = values[np.searchsorted(keys, block[active])]
        output[y, x] = result
        _update_geometry(geometry, result, y.start, x.start)
    result_records, provenance, ordinals = {}, {}, {}
    for root in ordered:
        identifier = root_ids[root]
        source = groups[root][0]
        ordinals[source] = ordinals.get(source, 0) + 1
        area, bbox = geometry[identifier]
        result_records[identifier] = replace(records[source], instance_id=identifier, area=area, bbox=bbox)
        provenance[str(identifier)] = {"source_instance_id": source, "component_id": ordinals[source]}
    return result_records, provenance
