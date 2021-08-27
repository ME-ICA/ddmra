"""DDMRA: A Python package for distance-dependent motion-related artifact analysis."""

from .analysis import highlow_analysis, qcrsfc_analysis, scrubbing_analysis
from .workflows import run_analyses

__all__ = [
    "highlow_analysis",
    "qcrsfc_analysis",
    "scrubbing_analysis",
    "run_analyses",
]
