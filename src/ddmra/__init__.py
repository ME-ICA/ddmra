"""DDMRA: A Python package for distance-dependent motion-related artifact analysis."""

from importlib.metadata import PackageNotFoundError, version

from .analysis import (
    highlow_analysis,
    qcrsfc_analysis,
    qcrsfc_summary,
    scrubbing_analysis,
)
from .plotting import plot_analysis, plot_results
from .workflows import run_analyses, run_pipeline_comparison

try:
    __version__ = version("ddmra")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "highlow_analysis",
    "qcrsfc_analysis",
    "qcrsfc_summary",
    "scrubbing_analysis",
    "run_analyses",
    "run_pipeline_comparison",
    "plot_analysis",
    "plot_results",
]
