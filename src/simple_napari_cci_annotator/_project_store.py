from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_FILENAME = "project.yaml"
PROJECT_SCHEMA_VERSION = 1


class ProjectError(RuntimeError):
    """Base exception for project storage errors."""


class ProjectConflictError(ProjectError):
    """Raised when a directory cannot safely be initialized as a project."""


class InvalidProjectError(ProjectError):
    """Raised when an existing project is incomplete or invalid."""


class ImageProcessingLockedError(ProjectError):
    """Raised when code attempts to change locked image-processing settings."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    models: Path
    annotations: Path
    images: Path
    labels: Path
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
            audit=annotations / "audit.jsonl",
        )


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: int
    name: str
    created_at: str
    classes: dict[int, str]
    image_processing: dict[str, Any]

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
        raw_classes = value.get("classes")
        image_processing = value.get("image_processing")

        if not isinstance(name, str) or not name.strip():
            raise InvalidProjectError("Project name must be a non-empty string.")
        if not isinstance(created_at, str) or not created_at.strip():
            raise InvalidProjectError("Project created_at must be a string.")
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

        if not isinstance(image_processing, dict):
            raise InvalidProjectError("image_processing must be a mapping.")
        if not isinstance(image_processing.get("locked"), bool):
            raise InvalidProjectError("image_processing.locked must be true or false.")

        return cls(
            schema_version=schema_version,
            name=name.strip(),
            created_at=created_at,
            classes=classes,
            image_processing=dict(image_processing),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at": self.created_at,
            "classes": dict(sorted(self.classes.items())),
            "image_processing": self.image_processing,
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

        class_map = classes or {0: "LABEL"}
        config = ProjectConfig(
            schema_version=PROJECT_SCHEMA_VERSION,
            name=project_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            classes=dict(class_map),
            image_processing={
                "channels": "unset",
                "normalization": "unset",
                "locked": False,
            },
        )

        created_directories: list[Path] = []
        try:
            for directory in (
                paths.models,
                paths.annotations,
                paths.images,
                paths.labels,
            ):
                directory.mkdir(parents=True, exist_ok=True)
                created_directories.append(directory)
            paths.audit.touch(exist_ok=False)
            _atomic_write_yaml(paths.config, config.to_mapping())
        except Exception:  # noqa: BLE001 - initialization cleanup must always run
            if paths.audit.exists():
                paths.audit.unlink()
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
        required_directories = (paths.models, paths.images, paths.labels)
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
            if self.config.image_processing != requested:
                raise ImageProcessingLockedError(
                    "Image-processing settings are locked for this project."
                )
            return

        updated = ProjectConfig(
            schema_version=self.config.schema_version,
            name=self.config.name,
            created_at=self.config.created_at,
            classes=dict(self.config.classes),
            image_processing=requested,
        )
        self.update_config(updated)


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
