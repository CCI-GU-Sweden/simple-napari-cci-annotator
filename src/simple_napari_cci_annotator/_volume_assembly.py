"""Experimental adjacent-slice graph assembly for stored segmentation masks.

This first stage returns a graph in memory; it does not write or display a
3D label volume. Slice predictions are never modified.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from uuid import uuid4

import numpy as np

from ._tiled_inference import InferenceCancelled
from ._volume_store import VolumeMaskStore, VolumeStorageError, _atomic_json


@dataclass(frozen=True)
class AssemblySettings:
    minimum_pixels: int = 1
    minimum_fraction: float = 0.5
    minimum_iou: float = 0.0

    def validate(self):
        if type(self.minimum_pixels) is not int or self.minimum_pixels < 1:
            raise ValueError("Minimum overlap pixels must be a positive integer.")
        for value in (self.minimum_fraction, self.minimum_iou):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Overlap thresholds must be between 0 and 1.")


@dataclass(frozen=True)
class SliceNode:
    z: int
    local_id: int
    class_id: int
    area: int


@dataclass(frozen=True)
class SliceLink:
    source: tuple[int, int]
    target: tuple[int, int]
    intersection: int
    overlap_fraction: float
    iou: float
    coverage_source: float
    coverage_target: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class AssemblyResult:
    settings: AssemblySettings
    nodes: dict[tuple[int, int], SliceNode]
    links: tuple[SliceLink, ...]
    object_for_node: dict[tuple[int, int], int]


def assemble_volume(root, settings=None, *, cancelled=None, progress=None):
    """Build a deterministic graph from a complete, verified stored run.

    Overlaps are accumulated in XY blocks. At most two slice masks and block
    temporaries are resident, while graph memory grows with nodes and overlaps.
    """
    settings = settings or AssemblySettings()
    settings.validate()
    with VolumeMaskStore.open(root, cancelled=cancelled) as store:
        if not store.complete:
            raise VolumeStorageError("Assembly requires every slice to be complete.")
        entries = store.manifest["slices"]
        nodes = {}
        links = []
        parent = {}
        previous = None

        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for index, entry in enumerate(entries):
            _check_cancelled(cancelled)
            z = entry["z_index"]
            mask, metadata = store.read_slice(z)
            current = {}
            for local, record in metadata["instances"].items():
                key = (z, int(local))
                node = SliceNode(z, int(local), int(record["class_id"]), int(record["area"]))
                if node.area <= 0:
                    raise VolumeStorageError("Slice instance has an invalid area.")
                nodes[key] = current[node.local_id] = node
                parent[key] = key
            if previous is not None:
                previous_z, previous_mask, previous_nodes = previous
                for (left, right), count in sorted(_overlaps(previous_mask, mask, cancelled).items()):
                    a, b = previous_nodes[left], current[right]
                    fraction = count / min(a.area, b.area)
                    iou = count / (a.area + b.area - count)
                    reason = ("class mismatch" if a.class_id != b.class_id else
                              "below threshold" if count < settings.minimum_pixels or
                              fraction < settings.minimum_fraction or iou < settings.minimum_iou else "accepted")
                    source, target = (previous_z, left), (z, right)
                    accepted = reason == "accepted"
                    links.append(SliceLink(source, target, count, fraction, iou,
                                           count / a.area, count / b.area, accepted, reason))
                    if accepted:
                        root_a, root_b = find(source), find(target)
                        if root_a != root_b:
                            parent[max(root_a, root_b)] = min(root_a, root_b)
            previous = (z, mask, current)
            if progress is not None:
                progress(index + 1, len(entries))
        roots = sorted({find(key) for key in nodes})
        ids = {root: index + 1 for index, root in enumerate(roots)}
        return AssemblyResult(settings, nodes, tuple(links),
                              {key: ids[find(key)] for key in sorted(nodes)})


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise InferenceCancelled("Volume assembly cancelled.")


def _overlaps(first, second, cancelled, block_size=512):
    counts = {}
    for y in range(0, first.shape[0], block_size):
        for x in range(0, first.shape[1], block_size):
            _check_cancelled(cancelled)
            a = first[y:y + block_size, x:x + block_size]
            b = second[y:y + block_size, x:x + block_size]
            both = (a != 0) & (b != 0)
            pairs = (a[both].astype(np.uint64) << np.uint64(32)) | b[both].astype(np.uint64)
            values, frequencies = np.unique(pairs, return_counts=True)
            for pair, count in zip(values, frequencies, strict=True):
                value = int(pair)
                key = (value >> 32, value & 0xffffffff)
                counts[key] = counts.get(key, 0) + int(count)
    return counts


def save_assembly(root, settings=None, *, cancelled=None, progress=None):
    """Publish a new assembly only after its graph is fully written."""
    root = Path(root).expanduser().resolve()
    run_hash = hashlib.sha256((root / "run.json").read_bytes()).hexdigest()
    result = assemble_volume(root, settings, cancelled=cancelled, progress=progress)
    _check_cancelled(cancelled)
    if hashlib.sha256((root / "run.json").read_bytes()).hexdigest() != run_hash:
        raise VolumeStorageError("Source run changed during assembly.")
    folder = root / "assemblies" / f"assembly_{uuid4().hex}"
    folder.mkdir(parents=True, exist_ok=False)
    graph = {
        "settings": asdict(result.settings),
        "nodes": [{**asdict(node), "object_id": result.object_for_node[key]}
                  for key, node in sorted(result.nodes.items())],
        "links": [asdict(link) for link in result.links],
    }
    _atomic_json(folder / "graph.json", graph)
    _check_cancelled(cancelled)
    _atomic_json(folder / "assembly.json", {
        "format_version": 1, "run_sha256": run_hash,
        "graph_sha256": hashlib.sha256((folder / "graph.json").read_bytes()).hexdigest(),
    })
    return folder


def open_assembly(folder, *, cancelled=None, progress=None):
    """Verify the saved graph and source run, and return a lazy label view."""
    folder = Path(folder).expanduser().resolve()
    root = folder.parent.parent
    try:
        manifest = json.loads((folder / "assembly.json").read_text(encoding="utf-8"))
        graph_bytes = (folder / "graph.json").read_bytes()
        if (manifest["format_version"] != 1 or
                hashlib.sha256(graph_bytes).hexdigest() != manifest["graph_sha256"] or
                hashlib.sha256((root / "run.json").read_bytes()).hexdigest() != manifest["run_sha256"]):
            raise ValueError("Assembly checksum mismatch")
        graph = json.loads(graph_bytes)
        settings = AssemblySettings(**graph["settings"])
        settings.validate()
        nodes = graph["nodes"]
        mappings = {}
        for node in nodes:
            z, local, obj = node["z"], node["local_id"], node["object_id"]
            if (any(type(value) is not int or value < 1 or value > np.iinfo(np.uint32).max
                    for value in (local, obj)) or type(z) is not int):
                raise ValueError("Invalid assembly IDs")
            if local in mappings.setdefault(z, {}):
                raise ValueError("Duplicate assembly node")
            mappings[z][local] = obj
        with VolumeMaskStore.open(root, cancelled=cancelled, progress=progress) as store:
            if not store.complete:
                raise ValueError("Incomplete source run")
            run = store.manifest
            for entry in run["slices"]:
                if set(mappings.get(entry["z_index"], {})) != set(range(1, entry["instance_count"] + 1)):
                    raise ValueError("Assembly nodes do not match source run")
            if set(mappings) - {entry["z_index"] for entry in run["slices"]}:
                raise ValueError("Assembly has unexpected slices")
            source = store.mask_array()
        start = run["contract"]["volume"]["selection"]["z_range"][0]

        def relabel(block, block_info=None):
            z0 = block_info[None]["array-location"][0][0]
            output = np.zeros(block.shape, dtype=np.uint32)
            for i in range(block.shape[0]):
                mapping = mappings.get(start + z0 + i, {})
                if mapping:
                    lookup = np.zeros(max(mapping) + 1, dtype=np.uint32)
                    for local, obj in mapping.items():
                        lookup[local] = obj
                    output[i] = lookup[block[i]]
            return output

        array = source.map_blocks(relabel, dtype=np.uint32, meta=np.empty((0, 0, 0), dtype=np.uint32))
        return root, run, graph, array
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise VolumeStorageError(f"Cannot open assembly: {exc}") from exc
