"""Generate distance-dependent motion-related artifact plots.

The rank for the intercept (smoothing curve at 35mm) indexes general dependence
on motion (i.e., a mix of global and focal effects), while the rank for the
slope (difference in smoothing curve at 100mm and 35mm) indexes distance
dependence (i.e., focal effects).
"""
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import assess_significance

sns.set_style("white")


def plot_analysis(
    data_points,
    distances,
    smoothing_curve,
    curve_distances,
    null_smoothing_curves,
    n_lines=50,
    metric_name=None,
    ylim=(-0.5, 0.5),
    fig=None,
    ax=None,
):
    """Generate plot for a DDMRA analysis.

    Parameters
    ----------
    data_points : numpy.ndarray of shape (n_edges,)
        DDMRA-metric values for all unique ROI-to-ROI edges, not including self-self edges.
    distances : numpy.ndarray of shape (n_edges,)
        All distances associated with data in data_points, in mm.
    smoothing_curve : numpy.ndarray of shape (n_unique_edge_distances,)
        Smoothing curve of data points, produced using a moving average.
    curve_distances : numpy.ndarray of shape (n_unique_edge_distances,)
        Edge distances, in mm, of data points in smoothing curve.
    null_smoothing_curves : numpy.ndarray of shape (n_iters, n_unique_edge_distances)
        Smoothing curves for all permutations, to be used as a null distribution.
    n_lines : int, optional
        Number of null smoothing curves to plot in the figure. Default is 50.
    metric_name : None or str, optional
        Label for the Y-axis of the figure, indicating the analysis' metric's name.
    ylim : tuple, optional
        Y-limits.
    fig : None or matplotlib.pyplot.Figure, optional
        Figure object.
    ax : None or matplotlib.pyplot.Axes, optional
        Axes object.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Updated Figure object.
    ax : matplotlib.pyplot.Axes
        Updated Axes object.
    """
    assert data_points.ndim == distances.ndim == smoothing_curve.ndim == curve_distances.ndim == 1
    assert null_smoothing_curves.ndim == 2
    assert data_points.shape == distances.shape
    assert smoothing_curve.shape[0] == curve_distances.shape[0] == null_smoothing_curves.shape[1]

    if not ax:
        fig, ax = plt.subplots(figsize=(16, 10))

    V1, V2 = 35, 100  # distances to evaluate
    n_lines = np.minimum(n_lines, null_smoothing_curves.shape[0])

    p_intercept, p_slope = assess_significance(
        smoothing_curve,
        null_smoothing_curves,
        curve_distances,
        V1,
        V2,
    )

    sns.regplot(
        x=distances,
        y=data_points,
        ax=ax,
        scatter=True,
        fit_reg=False,
        scatter_kws={"color": "red", "s": 5.0, "alpha": 1},
    )
    ax.axhline(0, xmin=0, xmax=np.max(distances) + 100, color="black", linewidth=3)

    for i_line in range(n_lines):
        ax.plot(curve_distances, null_smoothing_curves[i_line, :], color="black")

    ax.plot(curve_distances, smoothing_curve, color="white")

    ax.set_ylabel(metric_name, fontsize=32, labelpad=-30)
    ax.set_yticks(ylim)
    ax.set_yticklabels(ylim, fontsize=32)
    ax.set_ylim(ylim)

    ax.set_xlabel("Distance (mm)", fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, np.ceil(np.max(distances) / 10) * 10)

    ax.annotate(
        f"{V1} mm: {p_intercept:.04f}\n{V1}-{V2} mm: {p_slope:.04f}",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-20, 20),
        textcoords="offset pixels",
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=32,
    )
    fig.tight_layout()

    return fig, ax


def plot_results(in_dir):
    """Plot the results for all three analyses from a workflow run and save to a file.

    This function leverages the output file structure of :func:`workflows.run_analyses`.
    It writes out an image (analysis_results.png) to the output directory.

    Parameters
    ----------
    in_dir : str
        Path to the output directory of a ``run_analyses`` run.
    """
    METRIC_LABELS = {
        "qcrsfc": r"QC:RSFC $z_{r}$" + "\n(QC = mean FD)",
        "highlow": "High-low motion\n" + r"${\Delta}z_{r}$",
        "scrubbing": "Scrubbing\n" + r"${\Delta}z_{r}$",
    }
    YLIMS = {
        "qcrsfc": (-1.0, 1.0),
        "highlow": (-1.0, 1.0),
        "scrubbing": (-0.05, 0.05),
    }
    analysis_values = pd.read_table(op.join(in_dir, "analysis_values.tsv.gz"))
    smoothing_curves = pd.read_table(op.join(in_dir, "smoothing_curves.tsv.gz"))
    null_curves = np.load(op.join(in_dir, "null_smoothing_curves.npz"))

    fig, axes = plt.subplots(figsize=(8, 24), nrows=len(METRIC_LABELS))

    for i_analysis, (analysis_type, label) in enumerate(METRIC_LABELS.items()):
        values = analysis_values[analysis_type].values
        smoothing_curve = smoothing_curves[analysis_type].values

        fig, axes[i_analysis] = plot_analysis(
            values,
            analysis_values["distance"],
            smoothing_curve,
            smoothing_curves["distance"],
            null_curves[analysis_type],
            n_lines=50,
            ylim=YLIMS[analysis_type],
            metric_name=label,
            fig=fig,
            ax=axes[i_analysis],
        )

    fig.savefig(op.join(in_dir, "analysis_results.png"), dpi=100)
