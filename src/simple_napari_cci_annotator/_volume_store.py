"""Disk-backed slice masks with atomic completion records and lazy reads."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import dask.array as da
import numpy as np

from ._instance_mask import ComposedInstances
from ._tiled_inference import InferenceCancelled


class VolumeStorageError(RuntimeError):
    """A stored run is incomplete, invalid, or cannot be written safely."""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected an object")
        return value
    except (OSError, ValueError) as exc:
        raise VolumeStorageError(f"Cannot read stored metadata: {path.name}.") from exc


def _mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def _writer_lock(root: Path):
    """OS locks release on process exit, so crashes leave no stale lock."""
    stream = (root / ".writer.lock").open("a+b")
    try:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise VolumeStorageError("This volume run already has an active writer.") from exc
    return stream


class VolumeMaskStore:
    """One writer per run; manifest entries are the slice commit markers.

    Local IDs restart at 1 in each slice. A lazy display view can assign unique
    IDs across slices without modifying the saved masks. Close the writer
    before reopening a run. Concurrent writers are not supported.
    """

    def __init__(self, root, manifest, masks, *, writable=False, writer=None):
        self.root = Path(root)
        self._manifest = manifest
        self._masks = masks
        self._writable = writable
        self._closed = False
        self._writer = writer

    @classmethod
    def create(cls, root: Path, contract: dict[str, Any]) -> VolumeMaskStore:
        root = Path(root).expanduser().resolve()
        shape = tuple(contract["volume"]["geometry"]["shape"])
        start = contract["volume"]["selection"]["z_range"][0]
        if len(shape) != 3 or any(type(size) is not int or size < 1 for size in shape):
            raise VolumeStorageError("Stored mask shape must be positive ZYX dimensions.")
        # Never overwrite an existing run, including incomplete ones.
        root.mkdir(parents=True, exist_ok=False)
        writer = _writer_lock(root)
        try:
            (root / "slices").mkdir()
            masks = np.lib.format.open_memmap(
                root / "slice_masks.npy", mode="w+", dtype=np.uint32, shape=shape
            )
        except Exception:
            writer.close()
            raise
        manifest = {
            "format_version": 1,
            "contract": deepcopy(contract),
            "status": "pending",
            "error": None,
            "slices": [
                {"z_index": start + index, "complete": False}
                for index in range(shape[0])
            ],
        }
        store = cls(root, manifest, masks, writable=True, writer=writer)
        try:
            store._save()
        except Exception:
            store.close()
            raise
        return store

    @classmethod
    def open(
        cls, root: Path, *, writable: bool = False, recover: bool = False,
        cancelled=None, progress=None,
    ) -> VolumeMaskStore:
        """Verify committed masks and records one plane at a time.

        Recovery removes damaged slice commits from the in-memory manifest.
        Nothing on disk changes until the caller starts the validated run.
        """
        root = Path(root).expanduser().resolve()
        writer = _writer_lock(root) if writable else None
        masks = None
        try:
            manifest = _read_json(root / "run.json")
            if manifest["format_version"] != 1:
                raise ValueError("Unsupported format version")
            shape = tuple(manifest["contract"]["volume"]["geometry"]["shape"])
            start, stop = manifest["contract"]["volume"]["selection"]["z_range"]
            if (
                len(shape) != 3
                or any(type(size) is not int or size < 1 for size in shape)
                or stop - start != shape[0]
                or len(manifest["slices"]) != shape[0]
                or manifest["status"] not in {
                    "pending", "running", "completed", "cancelled", "failed"
                }
            ):
                raise ValueError("Invalid volume manifest")
            masks = np.load(
                root / "slice_masks.npy", mmap_mode="r+" if writable else "r",
                allow_pickle=False,
            )
            if masks.shape != shape or masks.dtype != np.dtype("uint32"):
                raise ValueError("Stored array does not match the manifest")
            store = cls(root, manifest, masks, writable=writable, writer=writer)
            for index, entry in enumerate(manifest["slices"]):
                if cancelled is not None and cancelled():
                    raise InferenceCancelled("Stored run verification cancelled.")
                if entry["z_index"] != start + index or type(entry["complete"]) is not bool:
                    raise ValueError("Invalid slice commit")
                if not entry["complete"]:
                    continue
                try:
                    store._verify_slice(index)
                except (VolumeStorageError, OSError, TypeError, ValueError):
                    if not recover:
                        raise
                    manifest["slices"][index] = {
                        "z_index": start + index, "complete": False
                    }
                if progress is not None:
                    progress(index + 1, shape[0])
            if manifest["status"] == "completed" and not store.complete:
                if not recover:
                    raise ValueError("Completed run has missing slices")
                manifest["status"] = "failed"
            return store
        except (KeyError, TypeError, ValueError, OSError) as exc:
            if masks is not None:
                masks._mmap.close()
            if writer is not None:
                writer.close()
            raise VolumeStorageError("Stored volume manifest or mask is invalid.") from exc
        except Exception:
            if masks is not None:
                masks._mmap.close()
            if writer is not None:
                writer.close()
            raise

    @property
    def manifest(self) -> dict[str, Any]:
        return deepcopy(self._manifest)

    @property
    def completed_count(self) -> int:
        return sum(entry["complete"] for entry in self._manifest["slices"])

    @property
    def complete(self) -> bool:
        return self.completed_count == len(self._manifest["slices"])

    def is_complete(self, z_index: int) -> bool:
        return self._manifest["slices"][self._index(z_index)]["complete"]

    def _index(self, z_index: int) -> int:
        start = self._manifest["contract"]["volume"]["selection"]["z_range"][0]
        index = z_index - start
        if (
            isinstance(z_index, bool) or not isinstance(z_index, (int, np.integer))
            or not 0 <= index < len(self._manifest["slices"])
        ):
            raise VolumeStorageError("Z index is outside the stored run.")
        return int(index)

    def _metadata_path(self, index: int) -> Path:
        z_index = self._manifest["slices"][index]["z_index"]
        return self.root / "slices" / f"z{z_index:06d}.json"

    def _verify_slice(self, index: int) -> None:
        entry = self._manifest["slices"][index]
        path = self._metadata_path(index)
        try:
            metadata_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise VolumeStorageError("Committed slice metadata is missing.") from exc
        if metadata_digest != entry.get("metadata_sha256"):
            raise VolumeStorageError("Slice metadata checksum does not match.")
        metadata = _read_json(path)
        mask = self._masks[index]
        if (
            metadata.get("z_index") != entry["z_index"]
            or metadata.get("shape") != list(mask.shape)
            or metadata.get("mask_sha256") != entry.get("mask_sha256")
            or _mask_digest(mask) != entry.get("mask_sha256")
        ):
            raise VolumeStorageError("Slice mask checksum or metadata does not match.")
        records = metadata.get("instances", {})
        present = {str(int(value)) for value in np.unique(mask) if value}
        if set(records) != present or len(records) != entry.get("instance_count"):
            raise VolumeStorageError("Slice instance records do not match the mask.")

    def read_slice(self, z_index: int) -> tuple[np.ndarray, dict[str, Any]]:
        self._ensure_open()
        index = self._index(z_index)
        if not self._manifest["slices"][index]["complete"]:
            raise VolumeStorageError("Slice has not been committed.")
        return self._masks[index].copy(), _read_json(self._metadata_path(index))

    def write_slice(
        self, z_index: int, result: ComposedInstances, conversion: dict[str, Any]
    ) -> None:
        self._ensure_writer()
        index = self._index(z_index)
        if self._manifest["slices"][index]["complete"]:
            raise VolumeStorageError("Cannot overwrite a committed slice.")
        mask = np.asarray(result.mask)
        if mask.shape != self._masks.shape[1:] or mask.dtype != np.dtype("uint32"):
            raise VolumeStorageError("Prediction mask must match the stored uint32 XY plane.")
        present = {int(value) for value in np.unique(mask) if value}
        if present != set(result.instances) or present != set(range(1, len(present) + 1)):
            raise VolumeStorageError("Slice IDs must be compact and match instance records.")
        mask_digest = _mask_digest(mask)
        metadata = {
            "z_index": int(z_index), "shape": list(mask.shape),
            "mask_sha256": mask_digest,
            "instances": {
                str(key): value.to_mapping()
                for key, value in sorted(result.instances.items())
            },
            "conversion": conversion,
            "provenance": result.provenance,
            "cleanup": [
                {
                    "component_count": item.component_count,
                    "removed_components": item.removed_components,
                    "removed_pixels": item.removed_pixels,
                }
                for item in result.cleanup
            ],
        }
        # Data and metadata must become durable before the commit marker.
        self._masks[index] = mask
        self._masks.flush()
        with (self.root / "slice_masks.npy").open("r+b") as stream:
            os.fsync(stream.fileno())
        path = self._metadata_path(index)
        _atomic_json(path, metadata)
        entry = {
            "z_index": int(z_index), "complete": True,
            "mask_sha256": mask_digest,
            "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "instance_count": len(result.instances),
        }
        previous = self._manifest["slices"][index]
        self._manifest["slices"][index] = entry
        try:
            self._save()
        except Exception:
            self._manifest["slices"][index] = previous
            raise

    def set_status(self, status: str, *, error: str | None = None) -> None:
        self._ensure_writer()
        if status not in {"running", "completed", "cancelled", "failed"}:
            raise VolumeStorageError("Invalid run status.")
        if status == "completed" and not self.complete:
            raise VolumeStorageError("Cannot complete a run with missing slices.")
        self._manifest["status"] = status
        self._manifest["error"] = error
        self._save()

    def mask_array(
        self, *, unique_ids: bool = False, allow_partial: bool = False,
        xy_chunk_size: int = 512,
    ) -> da.Array:
        """Return a lazy read-only snapshot. Missing slices are explicit zeros.

        Partial views require opt-in; their manifest must accompany the layer.
        Opening another view after more slices finish refreshes its snapshot.
        """
        self._ensure_open()
        if not self.complete and not allow_partial:
            raise VolumeStorageError("Run has missing slices; opt in to a partial view.")
        if type(xy_chunk_size) is not int or xy_chunk_size < 1:
            raise VolumeStorageError("XY chunk size must be a positive integer.")
        counts = [entry.get("instance_count", 0) for entry in self._manifest["slices"]]
        if unique_ids and sum(counts) > np.iinfo(np.uint32).max:
            raise VolumeStorageError("Display instance IDs exceed uint32 capacity.")
        source = _StoredMaskArray(
            self.root / "slice_masks.npy", self._masks.shape,
            tuple(entry["complete"] for entry in self._manifest["slices"]),
            tuple(np.cumsum([0] + counts[:-1])) if unique_ids else (0,) * len(counts),
        )
        return da.from_array(
            source, chunks=(1, xy_chunk_size, xy_chunk_size),
            asarray=False, name=False, fancy=False,
            meta=np.empty((0, 0, 0), dtype=np.uint32),
        )

    def _save(self) -> None:
        _atomic_json(self.root / "run.json", self._manifest)

    def _ensure_open(self) -> None:
        if self._closed:
            raise VolumeStorageError("Stored run is closed.")

    def _ensure_writer(self) -> None:
        self._ensure_open()
        if not self._writable:
            raise VolumeStorageError("Stored run is read-only.")

    def close(self) -> None:
        if not self._closed:
            self._masks._mmap.close()
            if self._writer is not None:
                self._writer.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _StoredMaskArray:
    """Array-like reader that prevents Dask copying/hashing the full memmap."""

    dtype = np.dtype("uint32")
    ndim = 3

    def __init__(self, path, shape, completed, offsets):
        self.path = path
        self.shape = shape
        self.completed = completed
        self.offsets = offsets

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 3:
            raise VolumeStorageError("Stored view expects ZYX chunk indexing.")
        z_index, y, x = index
        if not isinstance(z_index, slice):
            raise VolumeStorageError("Stored view expects Z slices.")
        indices = range(*z_index.indices(self.shape[0]))
        masks = np.load(self.path, mmap_mode="r", allow_pickle=False)
        try:
            result = masks[index].copy()
            for target, source in enumerate(indices):
                if not self.completed[source]:
                    result[target] = 0
                elif self.offsets[source]:
                    block = result[target]
                    block[block > 0] += np.uint32(self.offsets[source])
            return result
        finally:
            masks._mmap.close()
