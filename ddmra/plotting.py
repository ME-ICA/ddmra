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
    perm_smoothing_curves,
    n_lines=50,
    metric_name="QC:RSFC r\n(QC = mean FD)",
    ylim=(-0.5, 0.5),
    fig=None,
    ax=None,
):
    """Generate plot for a DDMRA analysis.

    Parameters
    ----------
    data_points : numpy.ndarray of shape (D,)
        DDMRA-metric values for all ROI-to-ROI edges.
    distances : numpy.ndarray of shape (D,)
        All distances associated with data in data_points, in mm.
    smoothing_curve : numpy.ndarray of shape (U,)
        Smoothing curve of data points, with U points.
        U being unique data points.
    curve_distances : numpy.ndarray of shape (U,)
    perm_smoothing_curves : numpy.ndarray of shape (P, U)
    n_lines : int, optional
    metric_name : str, optional
    ylim : tuple, optional
    fig : None or matplotlib.pyplot.Figure, optional
    ax : None or matplotlib.pyplot.Axes, optional

    Returns
    -------
    fig : matplotlib.pyplot.Figure
    ax : matplotlib.pyplot.Axes
    """
    assert data_points.ndim == distances.ndim == smoothing_curve.ndim == curve_distances.ndim == 1
    assert perm_smoothing_curves.ndim == 2
    assert data_points.shape == distances.shape
    assert smoothing_curve.shape[0] == curve_distances.shape[0] == perm_smoothing_curves.shape[1]

    if not ax:
        fig, ax = plt.subplots(figsize=(16, 10))

    V1, V2 = 35, 100  # distances to evaluate
    n_lines = np.minimum(n_lines, perm_smoothing_curves.shape[0])

    p_intercept, p_slope = assess_significance(
        smoothing_curve,
        perm_smoothing_curves,
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
        ax.plot(curve_distances, perm_smoothing_curves[i_line, :], color="black")

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

    fig, axes = plt.subplots(figsize=(8, 24), nrows=len(METRIC_LABELS))

    for i_analysis, (analysis_type, label) in enumerate(METRIC_LABELS.items()):
        values = analysis_values[analysis_type].values
        smoothing_curve = smoothing_curves[analysis_type].values
        perm_smoothing_curves = np.loadtxt(
            op.join(
                in_dir,
                f"{analysis_type}_analysis_null_smoothing_curves.txt",
            )
        )

        fig, axes[i_analysis] = plot_analysis(
            values,
            analysis_values["distance"],
            smoothing_curve,
            smoothing_curves["distance"],
            perm_smoothing_curves,
            n_lines=50,
            ylim=YLIMS[analysis_type],
            metric_name=label,
            fig=fig,
            ax=axes[i_analysis],
        )

    fig.savefig(op.join(in_dir, "analysis_results.png"), dpi=400)
