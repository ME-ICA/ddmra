"""Perform distance-dependent motion-related artifact analyses."""
import logging
import os.path as op

import numpy as np
import pandas as pd
from nilearn import datasets, input_data
from scipy.spatial.distance import pdist, squareform

from .plotting import plot_analysis
from .utils import fast_pearson, moving_average

LGR = logging.getLogger("ddmra")


def scrubbing_analysis(qc_values, group_timeseries, sort_idx, qc_thresh=0.2, perm=True):
    """Perform Power scrubbing analysis.

    Note that correlations from scrubbed timeseries are subtracted from correlations from
    unscrubbed timeseries, which is the opposite to the original Power method.
    This reverses the signs of the results, but makes inference similar to that of the
    QCRSFC and high-low motion analyses.

    Parameters
    ----------
    qc_values : (N,) list
        List of (T,) arrays
    group_timeseries : (N,) list
        List of (R, T) arrays
    qc_thresh : float
        Threshold to apply to QC values for identifying bad volumes.
    perm : bool
        Whether the call is part of the null distribution permutations or for
        the real deal.

    Returns
    -------
    mean_delta_r : numpy.ndarray
        Average (across subjects) change in correlation coefficient from
        unscrubbed to scrubbed timeseries for each pair of ROIs. Length of
        array will be ((R * R) - R) / 2 (upper triangle of RxR correlation
        matrix).
    """
    n_rois = group_timeseries[0].shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    n_pairs = len(triu_idx[0])
    n_subjects = len(group_timeseries)
    delta_rs = np.zeros((n_subjects, n_pairs))
    c = 0  # included subject counter
    for subj in range(n_subjects):
        ts_arr = group_timeseries[subj]
        qc_arr = qc_values[subj]
        keep_idx = qc_arr <= qc_thresh
        # Subjects with no timepoints excluded or with more than 50% excluded
        # will be excluded from analysis.
        prop_incl = np.sum(keep_idx) / qc_arr.shape[0]
        if (prop_incl >= 0.5) and (prop_incl != 1.0):
            scrubbed_ts = ts_arr[:, keep_idx]
            raw_corrs = np.corrcoef(ts_arr)
            raw_corrs = raw_corrs[triu_idx]
            scrubbed_corrs = np.corrcoef(scrubbed_ts)
            scrubbed_corrs = scrubbed_corrs[triu_idx]
            delta_rs[c, :] = raw_corrs - scrubbed_corrs  # opposite of Power
            c += 1

    if not perm:
        LGR.info(f"{c} of {n_subjects} subjects retained in scrubbing analysis")

    delta_rs = delta_rs[:c, :]
    mean_delta_r = np.mean(delta_rs, axis=0)
    mean_delta_r = mean_delta_r[sort_idx]
    return mean_delta_r


def high_low_motion_analysis(mean_qcs, corr_mats, sort_idx):
    """Perform high-low motion analysis.

    Split the sample using a median split of the QC metric (generally mean FD).
    Then, for each pair of ROIs, calculate the difference between the
    mean across correlation coefficients for the high motion minus the low
    motion groups.

    Parameters
    ----------
    mean_qcs : numpy.ndarray of shape (n_subjects,)
    corr_mats : numpy.ndarray of shape (n_subjects, n_roi_pairs)
    sort_idx : numpy.ndarray of shape (n_roi_pairs,)
    """
    hm_idx = mean_qcs >= np.median(mean_qcs)
    lm_idx = mean_qcs < np.median(mean_qcs)
    hm_mean_corr = np.mean(corr_mats[hm_idx, :], axis=0)
    lm_mean_corr = np.mean(corr_mats[lm_idx, :], axis=0)
    hl_corr_diff = hm_mean_corr - lm_mean_corr
    hl_corr_diff = hl_corr_diff[sort_idx]
    return hl_corr_diff


def qcrsfc(mean_qcs, corr_mats, sort_idx):
    """Perform quality-control resting-state functional connectivity analysis."""
    # Correlate each ROI pair's z-value against QC measure (usually FD) across subjects.
    qcrsfc_rs = fast_pearson(corr_mats.T, mean_qcs)

    # Sort coefficients by distance
    qcrsfc_rs = qcrsfc_rs[sort_idx]

    return qcrsfc_rs


def scrubbing_null_distribution(
    qc_values,
    ts_all,
    smoothing_curve_distances,
    sort_idx,
    smoothing_curve_dist_idx,
    qc_thresh,
    window,
    n_iters=10000,
):
    """Generate null distribution smoothing curves for scrubbing analysis."""
    qc_values = [subj_qc_values.copy() for subj_qc_values in qc_values]
    perm_scrub_smoothing_curve = np.zeros((n_iters, len(smoothing_curve_distances)))
    for i_iter in range(n_iters):
        perm_qcs = [np.random.permutation(perm_qc) for perm_qc in qc_values]
        perm_mean_delta_r = scrubbing_analysis(
            perm_qcs,
            ts_all,
            sort_idx,
            qc_thresh,
            perm=True,
        )
        perm_scrub_smoothing_curve[i_iter, :] = moving_average(perm_mean_delta_r, window)[
            smoothing_curve_dist_idx
        ]

    return perm_scrub_smoothing_curve


def other_null_distributions(
    qc_values,
    corr_mats,
    smoothing_curve_distances,
    sort_idx,
    smoothing_curve_dist_idx,
    qc_thresh,
    window,
    n_iters=10000,
):
    """Generate null distribution smoothing curves for QC:RSFC and high-low motion analyses."""
    qc_values = [subj_qc_values.copy() for subj_qc_values in qc_values]
    mean_qcs = np.array([np.mean(subj_qc_values) for subj_qc_values in qc_values])

    perm_qcrsfc_smoothing_curve = np.zeros((n_iters, len(smoothing_curve_distances)))
    perm_hl_smoothing_curve = np.zeros((n_iters, len(smoothing_curve_distances)))
    for i_iter in range(n_iters):
        # Prep for QC:RSFC and high-low motion analyses
        perm_mean_qcs = np.random.permutation(mean_qcs)

        # QC:RSFC analysis
        perm_qcrsfc_rs = qcrsfc(perm_mean_qcs, corr_mats, sort_idx)
        perm_qcrsfc_smoothing_curve[i_iter, :] = moving_average(perm_qcrsfc_rs, window)[
            smoothing_curve_dist_idx
        ]

        # High-low analysis
        perm_hl_diff = high_low_motion_analysis(perm_mean_qcs, corr_mats, sort_idx)
        perm_hl_smoothing_curve[i_iter, :] = moving_average(perm_hl_diff, window)[
            smoothing_curve_dist_idx
        ]

    return perm_qcrsfc_smoothing_curve, perm_hl_smoothing_curve


def run(
    files,
    qc,
    out_dir=".",
    confounds=None,
    n_iters=10000,
    qc_thresh=0.2,
    window=1000,
):
    """Run scrubbing, high-low motion, and QCRSFC analyses.

    Parameters
    ----------
    files : (N,) list of nifti files
        List of 4D (X x Y x Z x T) images in MNI space.
    qc : (N,) list of array_like
        List of 1D (T) numpy arrays with QC metric values per img (e.g., FD or respiration).
    out_dir : str, optional
        Output directory. Default is current directory.
    confounds : None or (N,) list of array-like, optional
        List of 2D (T) numpy arrays with confounds per img.
        Default is None (no confounds are removed).
    n_iters : int, optional
        Number of iterations to run to generate null distributions. Default is 10000.
    qc_thresh : float, optional
        Threshold for QC metric used in scrubbing analysis. Default is 0.2 (for FD).
    window : int, optional
        Number of units (pairs of ROIs) to include when averaging to generate smoothing curve.
        Default is 1000.

    Notes
    -----
    This function writes out several files to out_dir:
    - ``analysis_values.tsv.gz``: Raw analysis values for analyses.
        Has four columns: distance, qcrsfc, scrubbing, and high_low.
    - ``smoothing_curves.tsv.gz``: Smoothing curve information for analyses.
        Has four columns: distance, qcrsfc, scrubbing, and high_low.
    - ``[analysis]_analysis_null_smoothing_curves.txt``:
        Null smoothing curves from each analysis.
        Contains 2D array, where number of columns is same
        size and order as distance column in ``smoothing_curves.tsv.gz``
        and number of rows is number of iterations for permutation analysis.
    - ``[analysis]_analysis.png``: Figure for each analysis.
    """
    # create LGR with 'spam_application'
    LGR.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(op.join(out_dir, "log.tsv"))
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s")
    fh.setFormatter(formatter)
    # add the handlers to the LGR
    LGR.addHandler(fh)

    LGR.info("Preallocating matrices")
    n_subjects = len(files)

    # Load atlas and associated masker
    atlas = datasets.fetch_coords_power_2011()
    coords = np.vstack((atlas.rois["x"], atlas.rois["y"], atlas.rois["z"])).T
    n_rois = coords.shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    distances = squareform(pdist(coords))
    distances = distances[triu_idx]

    # Sorting index for distances
    sort_idx = distances.argsort()
    distances = distances[sort_idx]
    unique_dists_idx = np.array([np.where(distances == i)[0][0] for i in np.unique(distances)])

    LGR.info("Creating masker")
    spheres_masker = input_data.NiftiSpheresMasker(
        seeds=coords,
        radius=5.0,
        t_r=None,
        smoothing_fwhm=4.0,
        detrend=False,
        standardize=False,
        low_pass=None,
        high_pass=None,
    )

    # prep for qcrsfc and high-low motion analyses
    mean_qc = np.array([np.mean(subj_qc) for subj_qc in qc])
    z_corr_mats = np.zeros((n_subjects, distances.size))

    # Get correlation matrices
    ts_all = []
    LGR.info("Building correlation matrices")
    if confounds:
        LGR.info("Regressing confounds out of data.")

    for i_sub in range(n_subjects):
        if confounds:
            raw_ts = spheres_masker.fit_transform(files[i_sub], confounds=confounds[i_sub]).T
        else:
            raw_ts = spheres_masker.fit_transform(files[i_sub]).T

        assert raw_ts.shape[0] == n_rois

        ts_all.append(raw_ts)
        raw_corrs = np.corrcoef(raw_ts)
        raw_corrs = raw_corrs[triu_idx]
        z_corr_mats[i_sub, :] = np.arctanh(raw_corrs)

    del (raw_corrs, raw_ts, spheres_masker, atlas, coords)

    smoothing_curves = pd.DataFrame(columns=["distance", "qcrsfc", "high_low", "scrubbing"])
    analysis_values = pd.DataFrame(columns=["distance", "qcrsfc", "high_low", "scrubbing"])
    analysis_values["distance"] = distances

    # QC:RSFC r analysis
    LGR.info("Performing QC:RSFC analysis")
    qcrsfc_values = qcrsfc(mean_qc, z_corr_mats, sort_idx)
    analysis_values["qcrsfc"] = qcrsfc_values
    qcrsfc_smoothing_curve = moving_average(qcrsfc_values, window)

    # Quick interlude to help reduce arrays
    # Identify unique distances that don't have NaNs in the smoothing curve
    smoothing_curve_dist_idx = np.intersect1d(
        np.where(~np.isnan(qcrsfc_smoothing_curve))[0], unique_dists_idx
    )
    smoothing_curve_distances = distances[smoothing_curve_dist_idx]
    smoothing_curves["distance"] = smoothing_curve_distances

    qcrsfc_smoothing_curve = qcrsfc_smoothing_curve[smoothing_curve_dist_idx]
    smoothing_curves["qcrsfc"] = qcrsfc_smoothing_curve
    del qcrsfc_values, qcrsfc_smoothing_curve

    # High-low motion analysis
    LGR.info("Performing high-low motion analysis")
    high_low_values = high_low_motion_analysis(mean_qc, z_corr_mats, sort_idx)
    analysis_values["high_low"] = high_low_values
    hl_smoothing_curve = moving_average(high_low_values, window)[smoothing_curve_dist_idx]
    smoothing_curves["high_low"] = hl_smoothing_curve
    del high_low_values, hl_smoothing_curve

    # Scrubbing analysis
    LGR.info("Performing scrubbing analysis")
    scrub_values = scrubbing_analysis(qc, ts_all, sort_idx, qc_thresh, perm=False)
    analysis_values["scrubbing"] = scrub_values
    scrub_smoothing_curve = moving_average(scrub_values, window)[smoothing_curve_dist_idx]
    smoothing_curves["scrubbing"] = scrub_smoothing_curve
    del scrub_values, scrub_smoothing_curve

    smoothing_curves.to_csv(
        op.join(out_dir, "smoothing_curves.tsv.gz"),
        sep="\t",
        line_terminator="\n",
        index=False,
    )
    analysis_values.to_csv(
        op.join(out_dir, "analysis_values.tsv.gz"),
        sep="\t",
        line_terminator="\n",
        index=False,
    )

    # Null distributions
    LGR.info("Building null distributions with permutations")
    perm_qcrsfc_smoothing_curves, perm_hl_smoothing_curves = other_null_distributions(
        qc,
        z_corr_mats,
        smoothing_curve_distances,
        sort_idx,
        smoothing_curve_dist_idx,
        qc_thresh=qc_thresh,
        window=window,
        n_iters=n_iters,
    )
    perm_scrub_smoothing_curves = scrubbing_null_distribution(
        qc,
        ts_all,
        smoothing_curve_distances,
        sort_idx,
        smoothing_curve_dist_idx,
        qc_thresh=qc_thresh,
        window=window,
        n_iters=n_iters,
    )

    np.savetxt(
        op.join(out_dir, "qcrsfc_analysis_null_smoothing_curves.txt"),
        perm_qcrsfc_smoothing_curves,
    )
    np.savetxt(
        op.join(out_dir, "highlow_analysis_null_smoothing_curves.txt"),
        perm_hl_smoothing_curves,
    )
    np.savetxt(
        op.join(out_dir, "scrubbing_analysis_null_smoothing_curves.txt"),
        perm_scrub_smoothing_curves,
    )
    del perm_qcrsfc_smoothing_curves, perm_hl_smoothing_curves, perm_scrub_smoothing_curves

    METRIC_LABELS = {
        "qcrsfc": "QC:RSFC r\n(QC = mean FD)",
        "high_low": r"High-low motion $\Delta$z",
        "scrubbing": r"Scrubbing $\Delta$z",
    }

    for analysis, label in METRIC_LABELS.items():
        values = analysis_values[analysis].values
        smoothing_curve = smoothing_curves[analysis].values
        perm_smoothing_curves = np.loadtxt(
            op.join(
                out_dir,
                f"{analysis}_analysis_null_smoothing_curves.txt",
            )
        )

        fig, ax = plot_analysis(
            values,
            distances,
            smoothing_curve,
            smoothing_curve_distances,
            perm_smoothing_curves,
            n_lines=50,
            metric_name=label,
            fig=None,
            ax=None,
        )
        fig.savefig(op.join(out_dir, f"{analysis}_analysis.png"), dpi=400)

    LGR.info("Workflow completed")
    LGR.shutdown()
