"""DDMRA: A Python package for distance-dependent motion-related artifact analysis."""

from .analysis import highlow_analysis, qcrsfc_analysis, scrubbing_analysis
from .plotting import plot_analysis, plot_results
from .workflows import run_analyses

__all__ = [
    "highlow_analysis",
    "qcrsfc_analysis",
    "scrubbing_analysis",
    "run_analyses",
    "plot_analysis",
    "plot_results",
]
