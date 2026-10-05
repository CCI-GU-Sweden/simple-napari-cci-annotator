"""Release numbering must remain distinct from older plugin tags."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from next_release_version import next_release_version


def test_release_version_starts_at_project_baseline():
    assert next_release_version("0.10.0", ["v0.0.10", "v0.0.2"]) == "0.10.0"


def test_release_version_advances_after_existing_release():
    assert (
        next_release_version("0.10.0", ["v0.10.0", "v0.10.1", "v0.0.10"])
        == "0.10.2"
    )


def test_release_version_never_reuses_a_lower_gap():
    assert next_release_version("0.10.0", ["v0.10.2"]) == "0.10.3"


def test_release_version_follows_a_newer_minor_tag():
    assert next_release_version("0.10.0", ["v0.11.0"]) == "0.11.1"
