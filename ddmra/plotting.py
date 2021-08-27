"""Generate distance-dependent motion-related artifact plots.

The rank for the intercept (smoothing curve at 35mm) indexes general dependence
on motion (i.e., a mix of global and focal effects), while the rank for the
slope (difference in smoothing curve at 100mm and 35mm) indexes distance
dependence (i.e., focal effects).
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .ddmra import assess_significance

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
    assert smoothing_curve.shape == curve_distances.shape == perm_smoothing_curves.shape[1]

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
        distances,
        data_points,
        ax=ax,
        scatter=True,
        fit_reg=False,
        scatter_kws={"color": "red", "s": 2.0, "alpha": 1},
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
