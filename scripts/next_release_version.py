"""Choose the next PyPI version from the project baseline and Git tags."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def next_release_version(
    baseline: str, tags: list[str]
) -> str:
    """Return the baseline, or one patch above the highest release tag."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", baseline):
        raise ValueError(f"Invalid baseline version: {baseline}")

    major, minor, patch = map(int, baseline.split("."))
    versions = {
        tuple(map(int, match.groups()))
        for tag in tags
        if (match := re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag))
    }
    latest = max(versions, default=None)
    if latest is not None and latest >= (major, minor, patch):
        major, minor, patch = latest
        patch += 1
    return f"{major}.{minor}.{patch}"


if __name__ == "__main__":
    import tomllib

    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as project_file:
        baseline = tomllib.load(project_file)["tool"]["setuptools_scm"][
            "fallback_version"
        ]
    tags = subprocess.check_output(
        ["git", "tag", "--list", "v*"], cwd=root, text=True
    ).splitlines()
    print(next_release_version(baseline, tags))
