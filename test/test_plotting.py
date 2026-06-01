"""Tests for the ddmra.plotting module.

These tests use the non-interactive "Agg" backend configured in conftest.py.
"""

import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ddmra import plotting


def _make_plot_inputs(n_edges=300, n_curve=100, n_iters=40, seed=0):
    rng = np.random.RandomState(seed)
    distances = np.sort(rng.uniform(0, 150, size=n_edges))
    data_points = rng.normal(size=n_edges)
    curve_distances = np.linspace(1, 149, n_curve)
    smoothing_curve = rng.normal(size=n_curve) * 0.1
    null_smoothing_curves = rng.normal(size=(n_iters, n_curve)) * 0.1
    return data_points, distances, smoothing_curve, curve_distances, null_smoothing_curves


def test_plot_analysis_returns_fig_and_ax():
    """plot_analysis returns a Figure/Axes pair and annotates the p-values."""
    data_points, distances, smoothing_curve, curve_distances, nulls = _make_plot_inputs()
    fig, ax = plotting.plot_analysis(
        data_points,
        distances,
        smoothing_curve,
        curve_distances,
        nulls,
        metric_name="test metric",
    )
    try:
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        # The significance annotation reports values at 35 mm and 35-100 mm.
        annotation_text = " ".join(a.get_text() for a in ax.texts)
        assert "mm" in annotation_text
    finally:
        plt.close(fig)


def test_plot_analysis_accepts_existing_axes():
    """Passing an existing fig/ax draws into them rather than creating new ones."""
    data_points, distances, smoothing_curve, curve_distances, nulls = _make_plot_inputs(seed=1)
    fig, ax = plt.subplots()
    try:
        out_fig, out_ax = plotting.plot_analysis(
            data_points,
            distances,
            smoothing_curve,
            curve_distances,
            nulls,
            fig=fig,
            ax=ax,
        )
        assert out_ax is ax
    finally:
        plt.close(fig)


def test_plot_analysis_shape_assertions():
    """Inconsistent input shapes raise a ValueError."""
    data_points, distances, smoothing_curve, curve_distances, nulls = _make_plot_inputs(seed=2)
    with pytest.raises(ValueError):
        # data_points and distances must have matching shapes.
        plotting.plot_analysis(
            data_points[:-1], distances, smoothing_curve, curve_distances, nulls
        )
    with pytest.raises(ValueError):
        # null_smoothing_curves must be 2D.
        plotting.plot_analysis(data_points, distances, smoothing_curve, curve_distances, nulls[0])


@pytest.mark.parametrize(
    "analyses",
    [
        ["qcrsfc"],  # single analysis -> plt.subplots returns a bare Axes
        ["qcrsfc", "highlow", "scrubbing"],
    ],
)
def test_plot_results_writes_png(tmp_path, analyses):
    """plot_results reads workflow outputs and writes analysis_results.png."""
    rng = np.random.RandomState(0)
    n_edges, n_curve, n_iters = 200, 100, 10

    distances = np.sort(rng.uniform(0, 150, size=n_edges))
    analysis_values = pd.DataFrame({"distance": distances})
    for name in analyses:
        analysis_values[name] = rng.normal(size=n_edges) * 0.1
    analysis_values.to_csv(op.join(tmp_path, "analysis_values.tsv.gz"), sep="\t", index=False)

    curve_distances = np.linspace(1, 149, n_curve)
    smoothing_curves = pd.DataFrame({"distance": curve_distances})
    for name in analyses:
        smoothing_curves[name] = rng.normal(size=n_curve) * 0.05
    smoothing_curves.to_csv(op.join(tmp_path, "smoothing_curves.tsv.gz"), sep="\t", index=False)

    null_curves = {name: rng.normal(size=(n_iters, n_curve)) * 0.05 for name in analyses}
    np.savez(op.join(tmp_path, "null_smoothing_curves.npz"), **null_curves)

    try:
        plotting.plot_results(str(tmp_path))
        assert op.isfile(op.join(tmp_path, "analysis_results.png"))
        if "qcrsfc" in analyses:
            ylabels = [ax.get_ylabel() for ax in plt.gcf().axes]
            assert any("mean QC" in label for label in ylabels)
            assert not any("mean FD" in label for label in ylabels)
    finally:
        plt.close("all")
