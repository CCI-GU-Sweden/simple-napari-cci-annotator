"""Package-wide diagnostic switches.

Set ``VERBOSE_PERFORMANCE_TIMING`` to ``True`` from a Python or napari
console to print timed inference, napari import, and output-saving phases.
"""

from time import perf_counter


VERBOSE_PERFORMANCE_TIMING = False


def print_performance_timing(scope: str, phase: str, started: float) -> None:
    """Print one elapsed phase when package-wide timing is enabled."""
    if VERBOSE_PERFORMANCE_TIMING:
        elapsed = perf_counter() - started
        print(f"[CCI timing] {scope} · {phase}: {elapsed:.3f} s", flush=True)

