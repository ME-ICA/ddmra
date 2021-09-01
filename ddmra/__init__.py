"""DDMRA: A Python package for distance-dependent motion-related artifact analysis."""

from ._version import get_versions
from .analysis import highlow_analysis, qcrsfc_analysis, scrubbing_analysis
from .plotting import plot_analysis, plot_results
from .workflows import run_analyses

__version__ = get_versions()["version"]

__all__ = [
    "__version__",
    "highlow_analysis",
    "qcrsfc_analysis",
    "scrubbing_analysis",
    "run_analyses",
    "plot_analysis",
    "plot_results",
]
