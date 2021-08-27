"""Perform distance-dependent motion-related artifact analyses."""
import logging

import numpy as np

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
        List of (T, R) arrays
    qc_thresh : float
        Threshold to apply to QC values for identifying bad volumes.
    perm : bool
        Whether the call is part of the null distribution permutations or for the real deal.

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


def highlow_analysis(mean_qcs, corr_mats, sort_idx):
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


def qcrsfc_analysis(mean_qcs, corr_mats, sort_idx):
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
        perm_scrub_smoothing_curve[i_iter, :] = moving_average(
            perm_mean_delta_r,
            window,
        )[smoothing_curve_dist_idx]

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
        perm_qcrsfc_rs = qcrsfc_analysis(perm_mean_qcs, corr_mats, sort_idx)
        perm_qcrsfc_smoothing_curve[i_iter, :] = moving_average(
            perm_qcrsfc_rs,
            window,
        )[smoothing_curve_dist_idx]

        # High-low analysis
        perm_hl_diff = highlow_analysis(perm_mean_qcs, corr_mats, sort_idx)
        perm_hl_smoothing_curve[i_iter, :] = moving_average(
            perm_hl_diff,
            window,
        )[smoothing_curve_dist_idx]

    return perm_qcrsfc_smoothing_curve, perm_hl_smoothing_curve
