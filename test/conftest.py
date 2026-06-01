"""Shared pytest configuration and fixtures for the ddmra test suite."""

import matplotlib

# Force a non-interactive backend before anything imports matplotlib.pyplot
# (importing ``ddmra`` pulls in ``ddmra.plotting``, which imports pyplot).
matplotlib.use("Agg")
