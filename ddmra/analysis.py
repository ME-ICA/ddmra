"""Perform distance-dependent motion-related artifact analyses."""
import logging

import numpy as np
from joblib import Parallel, delayed

from .utils import average_across_distances, fast_pearson, moving_average

LGR = logging.getLogger("analysis")


def scrubbing_analysis(qc_values, group_timeseries, edge_sorting_idx, qc_thresh=0.2, perm=True):
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
    edge_sorting_idx : numpy.ndarray of shape (n_edges,)
        Sorting index for upper triangle (not including self-self edges) of correlation matrix.
        This will sort the 1D array by ascending physical distance of the ROI-ROI pairs.
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

    Notes
    -----
    The basic process for the scrubbing analysis is:
    1. Exclude any subjects with more than 50% excluded volumes or 0% excluded volumes.
    2. For each subject, correlate each ROI's time series with every other ROI's
       time series to produce standard correlation matrix.
    3. Apply QC metric threshold to "scrub" (i.e., remove bad volumes) time series,
       then compute scrubbing correlation matrix.
    4. Select the upper triangle (minus the diagonal) from the standard and scrubbing
       correlation matrices, flattening them both to 1D.
    5. Fisher's z-transform all correlation coefficients from both the standard and scrubbing
       vectors. **WARNING** The Power et al. version does not do this.
    6. Subtract the scrubbing z-values from the standard z-values.
       **WARNING** This is the opposite of how Power et al. did this!
    7. Average the difference values across participants.
    """
    n_rois = group_timeseries[0].shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    n_pairs = len(triu_idx[0])
    n_subjects = len(group_timeseries)
    delta_zs = np.zeros((n_subjects, n_pairs))
    c = 0  # included subject counter
    for i_subj in range(n_subjects):
        ts_arr = group_timeseries[i_subj]
        qc_arr = qc_values[i_subj]
        keep_idx = qc_arr <= qc_thresh

        # Subjects with no timepoints excluded or with more than 50% excluded
        # will be excluded from analysis.
        prop_incl = np.sum(keep_idx) / qc_arr.shape[0]
        if (prop_incl >= 0.5) and (prop_incl != 1.0):
            scrubbed_ts = ts_arr[:, keep_idx]
            raw_corrs = np.corrcoef(ts_arr)
            raw_corrs = raw_corrs[triu_idx]
            raw_zs = np.arctanh(raw_corrs)
            scrubbed_corrs = np.corrcoef(scrubbed_ts)
            scrubbed_corrs = scrubbed_corrs[triu_idx]
            scrubbed_zs = np.arctanh(scrubbed_corrs)
            delta_zs[c, :] = raw_zs - scrubbed_zs  # opposite of Power
            c += 1

    if not perm:
        LGR.info(f"{c} of {n_subjects} subjects retained in scrubbing analysis")

    # Remove extra rows corresponding to bad subjects
    delta_zs = delta_zs[:c, :]
    # Average over subjects
    mean_delta_z = np.mean(delta_zs, axis=0)
    # Sort by ascending distance
    mean_delta_z = mean_delta_z[edge_sorting_idx]
    return mean_delta_z


def highlow_analysis(mean_qcs, corr_mats):
    """Perform high-low QC analysis.

    Parameters
    ----------
    mean_qcs : numpy.ndarray of shape (n_subjects,)
        QC measure (typically mean framewise displacement) across participants.
    corr_mats : numpy.ndarray of shape (n_subjects, n_edges)
        Z-transformed correlation coefficients for ROI-ROI pairs.
        n_edges is the *unique* ROI-to-ROI edges, not including self-self edges.
        These coefficients must be sorted according to ascending distance along the second axis.

    Returns
    -------
    hl_corr_diff : numpy.ndarray of shape (n_edges,)
        ROI-ROI pair difference scores.

    Notes
    -----
    The basic process for the high-low analysis is:
    1. Average QC values within each participant.
    2. Split the participants into high-QC and low-QC groups using a median split.
    3. Calculate the average z-transformed correlation coefficient for each group.
    4. Subtract the low group's value from the high group's value, for each ROI-ROI pair.
    """
    highgroup_idx = mean_qcs >= np.median(mean_qcs)
    lowgroup_idx = mean_qcs < np.median(mean_qcs)
    highgroup_mean_z = np.mean(corr_mats[highgroup_idx, :], axis=0)
    lowgroup_mean_z = np.mean(corr_mats[lowgroup_idx, :], axis=0)
    hl_corr_diff = highgroup_mean_z - lowgroup_mean_z
    return hl_corr_diff


def qcrsfc_analysis(mean_qcs, corr_mats):
    """Perform quality-control resting-state functional connectivity analysis.

    Parameters
    ----------
    mean_qcs : numpy.ndarray of shape (n_subjects,)
        QC measure (typically mean framewise displacement) across participants.
    corr_mats : numpy.ndarray of shape (n_subjects, n_edges)
        Z-transformed correlation coefficients for ROI-ROI pairs.
        n_edges is the *unique* ROI-to-ROI edges, not including self-self edges.
        These coefficients must be sorted according to ascending distance along the second axis.

    Returns
    -------
    qcrsfc_zs : numpy.ndarray of shape (n_edges,)
        Z-transformed correlation coefficients for ROI-ROI pairs.

    Notes
    -----
    The basic process for the QC:RSFC analysis is:
    1. Average QC values within each participant.
    2. Correlate the mean QC values with z-transformed correlation coefficients
       across participants, for each ROI-ROI pair.
    3. Z-transform the edge-wise correlation coefficients.
    """
    # Correlate each ROI pair's z-value against QC measure (usually FD) across subjects.
    qcrsfc_rs = fast_pearson(corr_mats.T, mean_qcs)
    qcrsfc_zs = np.arctanh(qcrsfc_rs)
    return qcrsfc_zs


def _scrubbing_null_iter(
    qc_values,
    ts_all,
    distances,
    qc_thresh,
    edge_sorting_idx,
    window,
    seed=0,
):
    perm_qcs = [np.random.RandomState(seed=seed).permutation(perm_qc) for perm_qc in qc_values]
    perm_mean_delta_zs = scrubbing_analysis(
        perm_qcs,
        ts_all,
        edge_sorting_idx,
        qc_thresh,
        perm=True,
    )
    perm_scrub_smoothing_curve = moving_average(perm_mean_delta_zs, window)
    perm_scrub_smoothing_curve, _ = average_across_distances(
        perm_scrub_smoothing_curve,
        distances,
    )
    return perm_scrub_smoothing_curve


def scrubbing_null_distribution(
    qc_values,
    ts_all,
    distances,
    qc_thresh,
    edge_sorting_idx,
    window=1000,
    n_iters=10000,
    n_jobs=1,
):
    """Generate null distribution smoothing curves for scrubbing analysis.

    Parameters
    ----------
    qc_values : list of n_subjects length containing numpy.ndarray of shape (n_timepoints,)
        QC time series for each participant.
    ts_all : list of n_subjects length containing numpy.ndarray of shape (n_timepoints, n_rois)
        ROI time series for each participant.
    distances : numpy.ndarray of shape (n_edges,)
        Distances for edges, already sorted in ascending order.
    qc_thresh : float
        QC threshold used to identify bad volumes (i.e., scrub).
    edge_sorting_idx : numpy.ndarray of shape (n_edges,)
        Sorting index of the flattened upper triangle (minus the diagonal) of the correlation
        matrix, in order of ascending distance.
    window : int, optional
        Sliding window to use for smoothing curve. Default is 1000.
    n_iters : int, optional
        Number of iterations with which to build the null distributions. Default is 10000.
    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means 'all CPUs'. Default is 1.

    Returns
    -------
    perm_scrub_smoothing_curve : numpy.ndarray of shape (n_iters, n_unique_edge_distances)
        Smoothing curves for all permutations, to be used as a null distribution.
    """
    qc_values = [subj_qc_values.copy() for subj_qc_values in qc_values]

    scrub_null_smoothing_curves = Parallel(n_jobs=n_jobs)(
        delayed(_scrubbing_null_iter)(
            qc_values,
            ts_all,
            distances,
            qc_thresh,
            edge_sorting_idx,
            window,
            seed=seed,
        )
        for seed in range(n_iters)
    )

    scrub_null_smoothing_curves = np.vstack(scrub_null_smoothing_curves)

    return scrub_null_smoothing_curves


def _other_null_iter(mean_qcs, corr_mats, distances, window, seed=0):
    # Prep for QC:RSFC and high-low motion analyses
    perm_mean_qcs = np.random.RandomState(seed=seed).permutation(mean_qcs)

    # QC:RSFC analysis
    perm_qcrsfc_zs = qcrsfc_analysis(perm_mean_qcs, corr_mats)
    perm_qcrsfc_smoothing_curve = moving_average(perm_qcrsfc_zs, window)
    perm_qcrsfc_smoothing_curve, _ = average_across_distances(
        perm_qcrsfc_smoothing_curve,
        distances,
    )

    # High-low analysis
    perm_hl_diff = highlow_analysis(perm_mean_qcs, corr_mats)
    perm_hl_smoothing_curve = moving_average(perm_hl_diff, window)
    perm_hl_smoothing_curve, _ = average_across_distances(
        perm_hl_smoothing_curve,
        distances,
    )

    return perm_qcrsfc_smoothing_curve, perm_hl_smoothing_curve


def other_null_distributions(
    qc_values,
    corr_mats,
    distances,
    window=1000,
    n_iters=10000,
    n_jobs=1,
):
    """Generate null distribution smoothing curves for QC:RSFC and high-low analyses.

    Parameters
    ----------
    qc_values : list of n_subjects length containing numpy.ndarray of shape (n_timepoints,)
        QC time series for each participant.
    corr_mats : list of n_subjects length containing numpy.ndarray of shape (n_rois, n_rois)
        Z-transformed ROI-ROI correlation matrix for each participant.
    distances : numpy.ndarray of shape (n_edges,)
        Distances for edges, already sorted in ascending order.
    window : int, optional
        Sliding window to use for smoothing curve. Default is 1000.
    n_iters : int, optional
        Number of iterations with which to build the null distributions. Default is 10000.

    Returns
    -------
    perm_qcrsfc_smoothing_curve : numpy.ndarray of shape (n_iters, n_unique_edge_distances)
        Smoothing curves for all permutations, to be used as a null distribution.
    perm_hl_smoothing_curve : numpy.ndarray of shape (n_iters, n_unique_edge_distances)
        Smoothing curves for all permutations, to be used as a null distribution.
    """
    qc_values = [subj_qc_values.copy() for subj_qc_values in qc_values]
    mean_qcs = np.array([np.mean(subj_qc_values) for subj_qc_values in qc_values])

    results = Parallel(n_jobs=n_jobs)(
        delayed(_other_null_iter)(mean_qcs, corr_mats, distances, window, seed=seed)
        for seed in range(n_iters)
    )
    qcrsfc_null_smoothing_curves, hl_null_smoothing_curves = zip(*results)

    qcrsfc_null_smoothing_curves = np.vstack(qcrsfc_null_smoothing_curves)
    hl_null_smoothing_curves = np.vstack(hl_null_smoothing_curves)

    return qcrsfc_null_smoothing_curves, hl_null_smoothing_curves
