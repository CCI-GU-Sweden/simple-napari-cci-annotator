"""Expose the installed distribution version without importing the GUI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("simple-napari-cci-annotator")
except PackageNotFoundError:
    __version__ = "0.10.0"
