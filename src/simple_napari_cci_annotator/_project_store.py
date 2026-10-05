from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


PROJECT_FILENAME = "project.yaml"
PROJECT_SCHEMA_VERSION = 1
PROJECT_TASKS = {"detect", "segment"}
STARTER_MODELS = {
    "detect": "yolo26n.pt",
    "segment": "yolo26n-seg.pt",
}


class ProjectError(RuntimeError):
    """Base exception for project storage errors."""


class ProjectConflictError(ProjectError):
    """Raised when a directory cannot safely be initialized as a project."""


class InvalidProjectError(ProjectError):
    """Raised when an existing project is incomplete or invalid."""


class ImageProcessingLockedError(ProjectError):
    """Raised when code attempts to change locked image-processing settings."""


class TrainingPatchLockedError(ProjectError):
    """Raised when code attempts to change a locked training patch contract."""


class ClassMapError(ProjectError):
    """Raised when a class-map edit would invalidate saved annotations."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    models: Path
    annotations: Path
    images: Path
    labels: Path
    masks: Path
    instances: Path
    audit: Path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        root = Path(root).expanduser().resolve()
        annotations = root / "annotations"
        return cls(
            root=root,
            config=root / PROJECT_FILENAME,
            models=root / "models",
            annotations=annotations,
            images=annotations / "images",
            labels=annotations / "labels",
            masks=annotations / "masks",
            instances=annotations / "instances",
            audit=annotations / "audit.jsonl",
        )


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: int
    name: str
    created_at: str
    task: str
    classes: dict[int, str]
    image_processing: dict[str, Any]
    training_patch: dict[str, Any]
    dataset_split: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> ProjectConfig:
        if not isinstance(value, dict):
            raise InvalidProjectError("project.yaml must contain a mapping.")

        schema_version = value.get("schema_version")
        if schema_version != PROJECT_SCHEMA_VERSION:
            raise InvalidProjectError(
                "Unsupported project schema_version "
                f"{schema_version!r}; expected {PROJECT_SCHEMA_VERSION}."
            )

        name = value.get("name")
        created_at = value.get("created_at")
        task = str(value.get("task", "detect"))
        raw_classes = value.get("classes")
        image_processing = value.get("image_processing")
        training_patch = value.get(
            "training_patch",
            {"size": 1024, "padding_value": 114, "locked": False},
        )
        dataset_split = value.get(
            "dataset_split",
            {
                "seed": 42,
                "validation_fraction": 0.2,
                "assignments": {},
            },
        )

        if not isinstance(name, str) or not name.strip():
            raise InvalidProjectError("Project name must be a non-empty string.")
        if not isinstance(created_at, str) or not created_at.strip():
            raise InvalidProjectError("Project created_at must be a string.")
        if task not in PROJECT_TASKS:
            raise InvalidProjectError(
                f"Project task must be one of {sorted(PROJECT_TASKS)}."
            )
        if not isinstance(raw_classes, dict) or not raw_classes:
            raise InvalidProjectError("Project classes must be a non-empty mapping.")

        classes: dict[int, str] = {}
        for raw_id, raw_name in raw_classes.items():
            try:
                class_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise InvalidProjectError(
                    f"Invalid class ID in project.yaml: {raw_id!r}."
                ) from exc
            if class_id < 0 or not isinstance(raw_name, str) or not raw_name.strip():
                raise InvalidProjectError(
                    f"Invalid class definition: {raw_id!r}: {raw_name!r}."
                )
            classes[class_id] = raw_name.strip()
        _validate_class_map(classes)

        if not isinstance(image_processing, dict):
            raise InvalidProjectError("image_processing must be a mapping.")
        if not isinstance(image_processing.get("locked"), bool):
            raise InvalidProjectError("image_processing.locked must be true or false.")
        training_patch = _validate_training_patch(training_patch)
        dataset_split = _validate_dataset_split(dataset_split)

        return cls(
            schema_version=schema_version,
            name=name.strip(),
            created_at=created_at,
            task=task,
            classes=classes,
            image_processing=dict(image_processing),
            training_patch=training_patch,
            dataset_split=dataset_split,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at": self.created_at,
            "task": self.task,
            "classes": dict(sorted(self.classes.items())),
            "image_processing": self.image_processing,
            "training_patch": self.training_patch,
            "dataset_split": self.dataset_split,
        }


class ProjectStore:
    """Owns an initialized annotation project's paths and configuration."""

    def __init__(self, paths: ProjectPaths, config: ProjectConfig):
        self.paths = paths
        self.config = config

    @classmethod
    def initialize(
        cls,
        root: Path,
        *,
        name: str | None = None,
        classes: dict[int, str] | None = None,
        task: str = "detect",
    ) -> ProjectStore:
        paths = ProjectPaths.from_root(root)
        paths.root.mkdir(parents=True, exist_ok=True)

        contents = list(paths.root.iterdir())
        if contents:
            if paths.config.exists():
                raise ProjectConflictError(
                    "This folder already contains a project. Use Open Project."
                )
            raise ProjectConflictError(
                "New projects require an empty folder. Choose an empty folder "
                "or open an existing project."
            )

        project_name = (name or paths.root.name).strip()
        if not project_name:
            raise ProjectConflictError("The project must have a name.")

        class_map = dict(classes) if classes is not None else {0: "LABEL"}
        _validate_class_map(class_map)
        if task not in PROJECT_TASKS:
            raise ProjectConflictError(
                f"Project task must be one of {sorted(PROJECT_TASKS)}."
            )
        config = ProjectConfig(
            schema_version=PROJECT_SCHEMA_VERSION,
            name=project_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            task=task,
            classes=dict(class_map),
            image_processing={
                "channels": "unset",
                "filter": "unset",
                "normalization": "unset",
                "locked": False,
            },
            training_patch={
                "size": 1024,
                "padding_value": 114,
                "locked": False,
            },
            dataset_split={
                "seed": 42,
                "validation_fraction": 0.2,
                "assignments": {},
            },
        )

        created_directories: list[Path] = []
        created_files: list[Path] = []
        try:
            for directory in (
                paths.models,
                paths.annotations,
                paths.images,
                paths.labels,
                paths.masks,
                paths.instances,
            ):
                directory.mkdir(parents=True, exist_ok=True)
                created_directories.append(directory)
            starter_model = paths.models / STARTER_MODELS[task]
            _copy_packaged_starter_model(task, starter_model)
            created_files.append(starter_model)
            paths.audit.touch(exist_ok=False)
            created_files.append(paths.audit)
            _atomic_write_yaml(paths.config, config.to_mapping())
            created_files.append(paths.config)
        except Exception:  # noqa: BLE001 - initialization cleanup must always run
            for path in reversed(created_files):
                path.unlink(missing_ok=True)
            for directory in reversed(created_directories):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            raise

        return cls.load(paths.root)

    @classmethod
    def load(cls, root: Path) -> ProjectStore:
        paths = ProjectPaths.from_root(root)
        if not paths.root.is_dir():
            raise InvalidProjectError(f"Project folder does not exist: {paths.root}")
        if not paths.config.is_file():
            raise InvalidProjectError(
                f"Not an initialized project: missing {PROJECT_FILENAME}."
            )

        try:
            raw_config = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise InvalidProjectError(f"Could not read project.yaml: {exc}") from exc

        config = ProjectConfig.from_mapping(raw_config)
        required_directories = [paths.models, paths.images]
        if config.task == "detect":
            required_directories.append(paths.labels)
        else:
            required_directories.extend((paths.masks, paths.instances))
        missing = [
            str(path.relative_to(paths.root))
            for path in required_directories
            if not path.is_dir()
        ]
        if missing:
            raise InvalidProjectError(
                "Project is incomplete; missing folder(s): " + ", ".join(missing)
            )
        if not paths.audit.is_file():
            raise InvalidProjectError(
                "Project is incomplete; missing annotations/audit.jsonl."
            )
        return cls(paths=paths, config=config)

    def refresh(self) -> None:
        refreshed = self.load(self.paths.root)
        self.config = refreshed.config

    def update_config(self, config: ProjectConfig) -> None:
        _atomic_write_yaml(self.paths.config, config.to_mapping())
        self.config = config

    def lock_image_processing(self, settings: dict[str, Any]) -> None:
        """Persist the immutable conversion settings used by project images."""
        requested = dict(settings)
        requested["locked"] = True
        if self.config.image_processing.get("locked"):
            if not _image_processing_equivalent(
                self.config.image_processing, requested
            ):
                raise ImageProcessingLockedError(
                    "Image-processing settings are locked for this project."
                )
            return

        updated = ProjectConfig(
            schema_version=self.config.schema_version,
            name=self.config.name,
            created_at=self.config.created_at,
            task=self.config.task,
            classes=dict(self.config.classes),
            image_processing=requested,
            training_patch=dict(self.config.training_patch),
            dataset_split=dict(self.config.dataset_split),
        )
        self.update_config(updated)

    def update_dataset_split(
        self,
        assignments: dict[str, str],
        *,
        seed: int,
        validation_fraction: float,
    ) -> None:
        dataset_split = _validate_dataset_split(
            {
                "seed": seed,
                "validation_fraction": validation_fraction,
                "assignments": assignments,
            }
        )
        updated = ProjectConfig(
            schema_version=self.config.schema_version,
            name=self.config.name,
            created_at=self.config.created_at,
            task=self.config.task,
            classes=dict(self.config.classes),
            image_processing=dict(self.config.image_processing),
            training_patch=dict(self.config.training_patch),
            dataset_split=dataset_split,
        )
        self.update_config(updated)

    def update_classes(self, classes: dict[int, str]) -> None:
        """Persist a class map without orphaning IDs used by saved labels."""
        _validate_class_map(classes)
        normalized = {
            class_id: name.strip() for class_id, name in classes.items()
        }
        used_ids = (
            _used_label_class_ids(self.paths.labels)
            if self.config.task == "detect"
            else _used_instance_class_ids(self.paths.instances)
        )
        removed_used_ids = sorted(used_ids - set(normalized))
        if removed_used_ids:
            raise ClassMapError(
                "Cannot remove class ID(s) used by saved annotations: "
                + ", ".join(str(class_id) for class_id in removed_used_ids)
            )
        updated = ProjectConfig(
            schema_version=self.config.schema_version,
            name=self.config.name,
            created_at=self.config.created_at,
            task=self.config.task,
            classes=normalized,
            image_processing=dict(self.config.image_processing),
            training_patch=dict(self.config.training_patch),
            dataset_split=dict(self.config.dataset_split),
        )
        self.update_config(updated)

    def lock_training_patch(self, size: int, *, padding_value: int = 114) -> None:
        """Persist the fixed-size contract used by canonical training images."""
        requested = _validate_training_patch(
            {"size": size, "padding_value": padding_value, "locked": True}
        )
        if self.config.training_patch.get("locked"):
            if self.config.training_patch != requested:
                raise TrainingPatchLockedError(
                    "Training patch size and padding are locked for this project."
                )
            return
        updated = ProjectConfig(
            schema_version=self.config.schema_version,
            name=self.config.name,
            created_at=self.config.created_at,
            task=self.config.task,
            classes=dict(self.config.classes),
            image_processing=dict(self.config.image_processing),
            training_patch=requested,
            dataset_split=dict(self.config.dataset_split),
        )
        self.update_config(updated)


def _validate_class_map(classes: dict[int, str]) -> None:
    if not classes:
        raise InvalidProjectError("Project classes must not be empty.")
    if any(
        isinstance(class_id, bool) or not isinstance(class_id, int)
        for class_id in classes
    ):
        raise InvalidProjectError("Class IDs must be integers.")
    expected = list(range(len(classes)))
    if sorted(classes) != expected:
        raise InvalidProjectError(
            "Class IDs must be contiguous and start at 0; expected "
            f"{expected}, found {sorted(classes)}."
        )
    if any(not isinstance(name, str) for name in classes.values()):
        raise InvalidProjectError("Class names must be strings.")
    names = [name.strip() for name in classes.values()]
    if any(not name for name in names):
        raise InvalidProjectError("Class names must not be empty.")
    folded = [name.casefold() for name in names]
    if len(folded) != len(set(folded)):
        raise InvalidProjectError("Class names must be unique.")


def _image_processing_equivalent(
    first: dict[str, Any], second: dict[str, Any]
) -> bool:
    """Treat pre-filter project settings as the explicit no-filter default."""
    left = dict(first)
    right = dict(second)
    left.setdefault("filter", {"method": "none", "radius": 1})
    right.setdefault("filter", {"method": "none", "radius": 1})
    return left == right


def _validate_training_patch(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidProjectError("training_patch must be a mapping.")
    size = value.get("size", 1024)
    padding_value = value.get("padding_value", 114)
    locked = value.get("locked", False)
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size not in {512, 1024}
    ):
        raise InvalidProjectError("training_patch.size must be 512 or 1024.")
    if (
        isinstance(padding_value, bool)
        or not isinstance(padding_value, int)
        or not 0 <= padding_value <= 255
    ):
        raise InvalidProjectError(
            "training_patch.padding_value must be an integer within 0..255."
        )
    if not isinstance(locked, bool):
        raise InvalidProjectError("training_patch.locked must be true or false.")
    return {
        "size": size,
        "padding_value": padding_value,
        "locked": locked,
    }


def _used_label_class_ids(labels_path: Path) -> set[int]:
    used: set[int] = set()
    for label_path in labels_path.glob("*.txt"):
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ClassMapError(f"Could not inspect {label_path.name}: {exc}") from exc
        for line_number, raw_line in enumerate(lines, start=1):
            fields = raw_line.split()
            if not fields:
                continue
            try:
                used.add(int(fields[0]))
            except ValueError as exc:
                raise ClassMapError(
                    f"Cannot edit classes while {label_path.name}, line "
                    f"{line_number} has an invalid class ID."
                ) from exc
    return used


def _used_instance_class_ids(instances_path: Path) -> set[int]:
    used: set[int] = set()
    for metadata_path in instances_path.glob("*.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            instances = payload.get("instances", {})
            if not isinstance(instances, dict):
                raise ValueError("instances must be a mapping")
            for value in instances.values():
                if not isinstance(value, dict):
                    raise ValueError("instance records must be mappings")
                used.add(int(value["class_id"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ClassMapError(
                f"Could not inspect segmentation metadata {metadata_path.name}: {exc}"
            ) from exc
    return used


def _copy_packaged_starter_model(task: str, destination: Path) -> None:
    """Atomically copy the task-compatible checkpoint from package resources."""
    try:
        filename = STARTER_MODELS[task]
    except KeyError as exc:
        raise ProjectConflictError(
            f"No starter model is defined for task {task!r}."
        ) from exc
    resource = resources.files("simple_napari_cci_annotator").joinpath(
        "models", filename
    )
    temporary: Path | None = None
    try:
        if not resource.is_file():
            raise ProjectConflictError(
                f"The installed package is missing starter model {filename!r}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as output, resource.open("rb") as source:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except ProjectConflictError:
        raise
    except OSError as exc:
        raise ProjectConflictError(
            f"Could not install starter model {filename!r}: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_dataset_split(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidProjectError("dataset_split must be a mapping.")
    seed = value.get("seed", 42)
    validation_fraction = value.get("validation_fraction", 0.2)
    assignments = value.get("assignments", {})
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise InvalidProjectError("dataset_split.seed must be a non-negative integer.")
    if not isinstance(validation_fraction, (int, float)) or not (
        0 <= float(validation_fraction) < 1
    ):
        raise InvalidProjectError(
            "dataset_split.validation_fraction must be in [0, 1)."
        )
    if not isinstance(assignments, dict):
        raise InvalidProjectError("dataset_split.assignments must be a mapping.")
    normalized: dict[str, str] = {}
    for sample_id, split in assignments.items():
        if not isinstance(sample_id, str) or not sample_id:
            raise InvalidProjectError("Split assignment sample IDs must be strings.")
        if split not in {"train", "val"}:
            raise InvalidProjectError(
                f"Invalid split {split!r} for sample {sample_id!r}."
            )
        normalized[sample_id] = split
    return {
        "seed": seed,
        "validation_fraction": float(validation_fraction),
        "assignments": dict(sorted(normalized.items())),
    }


def _atomic_write_yaml(path: Path, value: dict[str, Any]) -> None:
    text = yaml.safe_dump(
        value,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:  # noqa: BLE001 - remove temporary file on every write failure
        temporary_path.unlink(missing_ok=True)
        raise
