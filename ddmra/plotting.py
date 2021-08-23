"""Generate distance-dependent motion-related artifact plots.

The rank for the intercept (smoothing curve at 35mm) indexes general dependence
on motion (i.e., a mix of global and focal effects), while the rank for the
slope (difference in smoothing curve at 100mm and 35mm) indexes distance
dependence (i.e., focal effects).
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import get_val, null_to_p

sns.set_style("white")


def plot_analysis(
    data_points,
    distances,
    smoothing_curve,
    curve_distances,
    perm_smoothing_curves,
    n_lines=50,
    metric_name="QC:RSFC r\n(QC = mean FD)",
    fig=None,
    ax=None,
):
    """Generate plot for a DDMRA analysis."""
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 14))

    v1, v2 = 35, 100  # distances to evaluate
    n_lines = np.minimum(n_lines, perm_smoothing_curves.shape[0])

    intercept = get_val(curve_distances, smoothing_curve, v1)
    slope = intercept - get_val(curve_distances, smoothing_curve, v2)
    perm_intercepts = get_val(curve_distances, perm_smoothing_curves, v1)
    perm_slopes = perm_intercepts - get_val(curve_distances, perm_smoothing_curves, v2)

    p_inter = null_to_p(intercept, perm_intercepts, tail="upper")
    p_slope = null_to_p(slope, perm_slopes, tail="upper")

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
    ax.set_yticks([-0.5, 0.5])
    ax.set_yticklabels([-0.5, 0.5], fontsize=32)
    ax.set_ylim(-0.5, 0.5)

    ax.set_xlabel("Distance (mm)", fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, np.ceil(np.max(distances) / 10) * 10)

    ax.annotate(
        f"{v1} mm: {p_inter:.04f}\n{v1}-{v2} mm: {p_slope:.04f}",
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
