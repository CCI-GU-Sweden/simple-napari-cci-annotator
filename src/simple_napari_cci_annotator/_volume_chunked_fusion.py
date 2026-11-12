"""Experimental disk-backed seam fusion of temporary tile labels."""

from collections import defaultdict

import numpy as np
from scipy.ndimage import find_objects

from ._segmentation_io import InstanceRecord, SegmentationError
from ._segmentation_tiling import _UnionFind
from ._tiled_inference import InferenceCancelled


def fuse_tile_chunks(temporary, output, core_size, records, classes, *,
                     chunk_xy=512, cancelled=None):
    """Fuse matching seam IDs and write the cropped, compact mask in blocks.

    This preserves all branches, but the output still needs the separate
    connected-component splitting stage before it is a valid volume slice.
    """
    if (len(temporary.shape) != 2 or len(output.shape) != 2 or
            np.dtype(temporary.dtype) != np.dtype("uint32") or
            np.dtype(output.dtype) != np.dtype("uint32") or
            any(a > b for a, b in zip(output.shape, temporary.shape, strict=True)) or
            type(chunk_xy) is not int or chunk_xy < 1 or core_size < 1):
        raise SegmentationError("Invalid chunked fusion inputs.")

    def check():
        if cancelled is not None and cancelled():
            raise InferenceCancelled("Chunked mask fusion cancelled.")

    def blocks(shape):
        for y in range(0, shape[0], chunk_xy):
            for x in range(0, shape[1], chunk_xy):
                yield (slice(y, min(y + chunk_xy, shape[0])),
                       slice(x, min(x + chunk_xy, shape[1])))

    present = set()
    for y, x in blocks(temporary.shape):
        check()
        present.update(int(value) for value in np.unique(temporary[y, x]) if value)
    missing = present - set(records)
    if missing:
        raise SegmentationError(f"Tiled prediction produced IDs without metadata: {sorted(missing)}.")
    check()
    pairs, conflicts, ambiguous = _chunked_seams(temporary, core_size, records, chunk_xy, check)
    check()
    union = _UnionFind(present)
    for left, right in pairs:
        union.union(left, right)
    groups = defaultdict(list)
    for identifier in sorted(present):
        groups[union.find(identifier)].append(identifier)
    ordered = sorted(groups.values(), key=min)
    mapping = {}
    for final, members in enumerate(ordered, 1):
        if len({records[item].class_id for item in members}) != 1:
            raise SegmentationError("A fused instance crosses classes.")
        mapping.update({item: final for item in members})
    keys = np.asarray(sorted(mapping), dtype=np.uint32)
    values = np.asarray([mapping[int(key)] for key in keys], dtype=np.uint32)
    geometry = {}
    for y, x in blocks(output.shape):
        check()
        block = np.asarray(temporary[y, x], dtype=np.uint32)
        result = np.zeros(block.shape, dtype=np.uint32)
        active = block != 0
        if np.any(active):
            result[active] = values[np.searchsorted(keys, block[active])]
        output[y, x] = result
        _update_geometry(geometry, result, y.start, x.start)
    fused_records = {}
    for identifier, (area, bbox) in geometry.items():
        members = ordered[identifier - 1]
        class_id = records[members[0]].class_id
        confidences = [records[item].confidence for item in members if records[item].confidence is not None]
        fused_records[identifier] = InstanceRecord(
            instance_id=identifier, class_id=class_id, class_name=classes[class_id],
            confidence=max(confidences) if confidences else None, source="prediction",
            status="predicted", bbox=bbox, area=area, lineage=tuple(members),
        )
    return fused_records, mapping, pairs, conflicts, ambiguous


def _update_geometry(geometry, block, y_offset, x_offset):
    """Accumulate per-label areas and boxes in one pass over a chunk."""
    identifiers, inverse, counts = np.unique(block, return_inverse=True, return_counts=True)
    slices = find_objects((inverse.reshape(block.shape) + 1).astype(np.int32))
    for identifier, count, region in zip(identifiers, counts, slices, strict=True):
        if not identifier:
            continue
        y, x = region
        bounds = (y.start + y_offset, x.start + x_offset, y.stop + y_offset, x.stop + x_offset)
        key = int(identifier)
        if key in geometry:
            area, old = geometry[key]
            bounds = (min(old[0], bounds[0]), min(old[1], bounds[1]),
                      max(old[2], bounds[2]), max(old[3], bounds[3]))
            geometry[key] = (area + int(count), bounds)
        else:
            geometry[key] = (int(count), bounds)


def _chunked_seams(mask, core, records, chunk, check):
    pairs = set()
    conflicts = {}
    neighbours = defaultdict(set)

    def inspect(first, second, orientation, coordinate):
        active = (first != 0) & (second != 0) & (first != second)
        left = np.minimum(first[active], second[active]).astype(np.uint64)
        right = np.maximum(first[active], second[active]).astype(np.uint64)
        encoded = (left << np.uint64(32)) | right
        values, positions = np.unique(encoded, return_index=True)
        for encoded_id in values[np.argsort(positions)]:
            key = int(encoded_id)
            a, b = key >> 32, key & 0xffffffff
            first_record, second_record = records.get(a), records.get(b)
            if first_record is None or second_record is None:
                continue
            if first_record.class_id == second_record.class_id:
                pairs.add((a, b))
                neighbours[a].add(b)
                neighbours[b].add(a)
            else:
                classes = (first_record.class_id, second_record.class_id)
                conflict_key = ((a, b), classes, orientation, coordinate)
                conflicts.setdefault(conflict_key, {
                    "ids": [a, b], "classes": list(classes),
                    "orientation": orientation, "coordinate": coordinate,
                })

    for y in range(core, mask.shape[0], core):
        for x in range(0, mask.shape[1], chunk):
            check()
            region = slice(x, min(x + chunk, mask.shape[1]))
            inspect(mask[y - 1, region], mask[y, region], "horizontal", y)
    for x in range(core, mask.shape[1], core):
        for y in range(0, mask.shape[0], chunk):
            check()
            region = slice(y, min(y + chunk, mask.shape[0]))
            inspect(mask[region, x - 1], mask[region, x], "vertical", x)
    ambiguous = {identifier for identifier, matches in neighbours.items() if len(matches) > 1}
    return tuple(sorted(pairs)), list(conflicts.values()), ambiguous
