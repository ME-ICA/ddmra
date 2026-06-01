"""Perform distance-dependent motion-related artifact analyses."""

import logging

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import R2Z_CLIP, fast_pearson, r2z, tqdm_joblib

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
        List of (R, T) arrays, where rows are ROIs and columns are timepoints.
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
    if len(qc_values) != len(group_timeseries):
        raise ValueError(
            f"qc_values has {len(qc_values)} entries but group_timeseries has "
            f"{len(group_timeseries)}."
        )
    if not group_timeseries:
        raise ValueError("At least one subject is required for scrubbing analysis.")

    first_ts = np.asarray(group_timeseries[0])
    first_qc = np.asarray(qc_values[0])
    if first_ts.ndim != 2:
        raise ValueError("Subject 0 time series must be a 2D array.")
    if first_qc.ndim != 1:
        raise ValueError("Subject 0 QC values must be a 1D array.")
    if first_ts.shape[1] != first_qc.shape[0]:
        raise ValueError(
            "Scrubbing time series must be ROIs by timepoints, with one QC value per "
            f"timepoint. Subject 0 has time series shape {first_ts.shape} and "
            f"{first_qc.shape[0]} QC values."
        )

    n_rois = first_ts.shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    n_pairs = len(triu_idx[0])
    if edge_sorting_idx.size != n_pairs:
        raise ValueError(
            f"edge_sorting_idx has {edge_sorting_idx.size} entries, but {n_pairs} "
            f"ROI pairs are expected for {n_rois} ROIs."
        )
    n_subjects = len(group_timeseries)
    delta_zs = np.zeros((n_subjects, n_pairs))
    c = 0  # included subject counter
    n_clipped_raw = 0  # edge correlations clipped before Fisher z (full timeseries)
    n_clipped_scrubbed = 0  # edge correlations clipped before Fisher z (scrubbed timeseries)
    for i_subj in range(n_subjects):
        ts_arr = np.asarray(group_timeseries[i_subj])
        qc_arr = np.asarray(qc_values[i_subj])
        if ts_arr.ndim != 2:
            raise ValueError(f"Subject {i_subj} time series must be a 2D array.")
        if qc_arr.ndim != 1:
            raise ValueError(f"Subject {i_subj} QC values must be a 1D array.")
        if ts_arr.shape[0] != n_rois:
            raise ValueError("All subjects must have the same number of ROIs.")
        if ts_arr.shape[1] != qc_arr.shape[0]:
            raise ValueError(
                "Scrubbing time series must be ROIs by timepoints, with one QC value per "
                f"timepoint. Subject {i_subj} has time series shape {ts_arr.shape} and "
                f"{qc_arr.shape[0]} QC values."
            )
        keep_idx = qc_arr <= qc_thresh

        # Subjects with no timepoints excluded or with more than 50% excluded
        # will be excluded from analysis.
        prop_incl = np.sum(keep_idx) / qc_arr.shape[0]
        if (prop_incl >= 0.5) and (prop_incl != 1.0):
            scrubbed_ts = ts_arr[:, keep_idx]
            raw_corrs = np.corrcoef(ts_arr)
            raw_corrs = raw_corrs[triu_idx]
            raw_zs, n_clip_raw = r2z(raw_corrs, return_n_clipped=True)
            scrubbed_corrs = np.corrcoef(scrubbed_ts)
            scrubbed_corrs = scrubbed_corrs[triu_idx]
            scrubbed_zs, n_clip_scrubbed = r2z(scrubbed_corrs, return_n_clipped=True)
            n_clipped_raw += n_clip_raw
            n_clipped_scrubbed += n_clip_scrubbed
            delta_zs[c, :] = raw_zs - scrubbed_zs  # opposite of Power
            c += 1

    if not perm:
        LGR.info(f"{c} of {n_subjects} subjects retained in scrubbing analysis")
        # Short-distance edges can have near-perfect raw FC, so report how many edge
        # correlations were clipped before the Fisher z-transform (see utils.r2z).
        n_total = c * n_pairs
        LGR.info(
            f"Scrubbing analysis clipped {n_clipped_raw} full and {n_clipped_scrubbed} "
            f"scrubbed edge correlations to +/-{R2Z_CLIP} before Fisher z-transform "
            f"(out of {n_total} edge correlations each)."
        )

    # Remove extra rows corresponding to bad subjects
    delta_zs = delta_zs[:c, :]
    if c == 0:
        raise ValueError(
            "No subjects retained in scrubbing analysis. At least one subject must have "
            "some censored volumes and retain at least 50% of volumes."
        )

    # Average over subjects
    mean_delta_z = np.mean(delta_zs, axis=0)

    # Sort by ascending distance
    mean_delta_z = mean_delta_z[edge_sorting_idx]
    return mean_delta_z


def highlow_analysis(mean_qcs, z_corr_mats):
    """Perform high-low QC analysis.

    Parameters
    ----------
    mean_qcs : numpy.ndarray of shape (n_subjects,)
        QC measure (typically mean framewise displacement) across participants.
    z_corr_mats : numpy.ndarray of shape (n_subjects, n_edges)
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
    mean_qcs = np.asarray(mean_qcs, dtype=float)
    z_corr_mats = np.asarray(z_corr_mats, dtype=float)

    if mean_qcs.ndim != 1:
        raise ValueError(f"mean_qcs must be a 1D array, not {mean_qcs.ndim}D.")
    if z_corr_mats.ndim != 2:
        raise ValueError(f"z_corr_mats must be a 2D array, not {z_corr_mats.ndim}D.")
    if mean_qcs.shape[0] != z_corr_mats.shape[0]:
        raise ValueError(
            f"mean_qcs has {mean_qcs.shape[0]} runs but z_corr_mats has "
            f"{z_corr_mats.shape[0]}."
        )
    if not np.all(np.isfinite(mean_qcs)):
        raise ValueError("mean_qcs must contain only finite values.")
    if not np.all(np.isfinite(z_corr_mats)):
        raise ValueError("z_corr_mats must contain only finite values.")
    if np.isclose(np.var(mean_qcs), 0):
        raise ValueError("mean_qcs must have nonzero variance for high-low analysis.")

    highgroup_idx = mean_qcs >= np.median(mean_qcs)
    lowgroup_idx = mean_qcs < np.median(mean_qcs)
    if not np.any(highgroup_idx) or not np.any(lowgroup_idx):
        raise ValueError("High-low analysis requires at least one run in each QC group.")

    highgroup_mean_z = np.mean(z_corr_mats[highgroup_idx, :], axis=0)
    lowgroup_mean_z = np.mean(z_corr_mats[lowgroup_idx, :], axis=0)
    hl_corr_diff = highgroup_mean_z - lowgroup_mean_z
    return hl_corr_diff


def _residualize(values, covariates):
    """Remove covariate effects from one or more run-level variables."""
    values = np.asarray(values, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    if covariates.ndim != 2:
        raise ValueError("run_covariates must be a 2D array.")
    if covariates.shape[0] != values.shape[0]:
        raise ValueError(
            f"run_covariates has {covariates.shape[0]} rows, but {values.shape[0]} runs "
            "were provided."
        )
    if covariates.shape[1] == 0:
        return values.copy()
    if not np.all(np.isfinite(covariates)):
        raise ValueError("run_covariates must contain only finite values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Values to residualize must contain only finite values.")

    design = np.column_stack((np.ones(covariates.shape[0]), covariates))
    betas = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ betas


def _validate_qcrsfc_inputs(mean_qcs, z_corr_mats):
    """Validate that QC:RSFC inputs can produce defined correlations."""
    if not np.all(np.isfinite(mean_qcs)):
        raise ValueError("mean_qcs must contain only finite values.")
    if not np.all(np.isfinite(z_corr_mats)):
        raise ValueError("z_corr_mats must contain only finite values.")
    if np.isclose(np.var(mean_qcs), 0):
        raise ValueError("mean_qcs must have nonzero variance for QC:RSFC.")

    edge_variances = np.var(z_corr_mats, axis=0)
    if np.any(np.isclose(edge_variances, 0)):
        n_bad_edges = np.sum(np.isclose(edge_variances, 0))
        raise ValueError(f"{n_bad_edges} FC edge(s) have zero variance across runs.")


def qcrsfc_analysis(mean_qcs, z_corr_mats, run_covariates=None):
    """Perform quality-control resting-state functional connectivity analysis.

    Parameters
    ----------
    mean_qcs : numpy.ndarray of shape (n_subjects,)
        QC measure (typically mean framewise displacement) across participants.
    z_corr_mats : numpy.ndarray of shape (n_subjects, n_edges)
        Z-transformed correlation coefficients for ROI-ROI pairs.
        n_edges is the *unique* ROI-to-ROI edges, not including self-self edges.
        These coefficients must be sorted according to ascending distance along the second axis.
    run_covariates : None or numpy.ndarray of shape (n_subjects, n_covariates), optional
        Run-level covariates to adjust for before correlating QC and FC.

    Returns
    -------
    qcrsfc_zs : numpy.ndarray of shape (n_edges,)
        Z-transformed correlation coefficients for ROI-ROI pairs.

    Notes
    -----
    The basic process for the QC:RSFC analysis is:

    1. Average QC values within each participant.
    2. If run-level covariates are provided, regress them out of the mean QC values and
       z-transformed correlation coefficients.
    3. Correlate the mean QC values with z-transformed correlation coefficients
       across participants, for each ROI-ROI pair.
    4. Z-transform the edge-wise correlation coefficients.
    """
    mean_qcs = np.asarray(mean_qcs, dtype=float)
    z_corr_mats = np.asarray(z_corr_mats, dtype=float)

    if mean_qcs.ndim != 1:
        raise ValueError(f"mean_qcs must be a 1D array, not {mean_qcs.ndim}D.")
    if z_corr_mats.ndim != 2:
        raise ValueError(f"z_corr_mats must be a 2D array, not {z_corr_mats.ndim}D.")
    if mean_qcs.shape[0] != z_corr_mats.shape[0]:
        raise ValueError(
            f"mean_qcs has {mean_qcs.shape[0]} runs but z_corr_mats has "
            f"{z_corr_mats.shape[0]}."
        )
    _validate_qcrsfc_inputs(mean_qcs, z_corr_mats)

    if run_covariates is not None:
        mean_qcs = _residualize(mean_qcs, run_covariates)
        z_corr_mats = _residualize(z_corr_mats, run_covariates)
        _validate_qcrsfc_inputs(mean_qcs, z_corr_mats)

    # Correlate each ROI pair's z-value against QC measure (usually FD) across subjects.
    qcrsfc_rs = fast_pearson(z_corr_mats.T, mean_qcs)
    qcrsfc_zs = r2z(qcrsfc_rs)
    return qcrsfc_zs


def _scrubbing_null_iter(qc_values, ts_all, qc_thresh, edge_sorting_idx, seed=0):
    rng = np.random.RandomState(seed=seed)
    perm_qcs = [rng.permutation(perm_qc) for perm_qc in qc_values]
    ts_all = [ts.copy() for ts in ts_all]

    perm_mean_delta_zs = scrubbing_analysis(
        perm_qcs,
        ts_all,
        edge_sorting_idx,
        qc_thresh,
        perm=True,
    )

    return perm_mean_delta_zs


def scrubbing_null_distribution(
    qc_values,
    ts_all,
    qc_thresh,
    edge_sorting_idx,
    n_iters=10000,
    n_jobs=1,
):
    """Generate null distribution smoothing curves for scrubbing analysis.

    Parameters
    ----------
    qc_values : list of n_subjects length containing numpy.ndarray of shape (n_timepoints,)
        QC time series for each participant.
    ts_all : list of n_subjects length containing numpy.ndarray of shape (n_rois, n_timepoints)
        ROI time series for each participant.
    qc_thresh : float
        QC threshold used to identify bad volumes (i.e., scrub).
    edge_sorting_idx : numpy.ndarray of shape (n_edges,)
        Sorting index of the flattened upper triangle (minus the diagonal) of the correlation
        matrix, in order of ascending distance.
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
    ts_all = [subj_ts_values.copy() for subj_ts_values in ts_all]

    with tqdm_joblib(tqdm(desc="Scrubbing null distribution", total=n_iters)):
        scrub_null_values = Parallel(n_jobs=n_jobs)(
            delayed(_scrubbing_null_iter)(
                qc_values,
                ts_all,
                qc_thresh,
                edge_sorting_idx,
                seed=seed,
            )
            for seed in range(n_iters)
        )

    scrub_null_values = np.vstack(scrub_null_values)

    return scrub_null_values


def _other_null_iter(mean_qc, z_corr_mats, run_covariates=None, seed=0):
    # Prep for QC:RSFC and high-low motion analyses
    perm_mean_qc = np.random.RandomState(seed=seed).permutation(mean_qc)
    z_corr_mats = z_corr_mats.copy()

    # QC:RSFC analysis
    perm_qcrsfc_zs = qcrsfc_analysis(perm_mean_qc, z_corr_mats, run_covariates=run_covariates)

    # High-low analysis
    perm_hl_diff = highlow_analysis(perm_mean_qc, z_corr_mats)

    return perm_qcrsfc_zs, perm_hl_diff


def other_null_distributions(mean_qc, z_corr_mats, run_covariates=None, n_iters=10000, n_jobs=1):
    """Generate null distribution smoothing curves for QC:RSFC and high-low analyses.

    Parameters
    ----------
    mean_qc : numpy.ndarray of shape (n_subjects,)
        Mean QC value for each participant.
    z_corr_mats : numpy.ndarray of shape (n_subjects, n_roi_pairs)
        Z-transformed ROI-ROI correlation matrix for each participant.
    run_covariates : None or numpy.ndarray of shape (n_subjects, n_covariates), optional
        Run-level covariates to adjust for in QC:RSFC null distributions.
    n_iters : int, optional
        Number of iterations with which to build the null distributions. Default is 10000.

    Returns
    -------
    perm_qcrsfc_smoothing_curve : numpy.ndarray of shape (n_iters, n_unique_edge_distances)
        Smoothing curves for all permutations, to be used as a null distribution.
    perm_hl_smoothing_curve : numpy.ndarray of shape (n_iters, n_unique_edge_distances)
        Smoothing curves for all permutations, to be used as a null distribution.
    """
    with tqdm_joblib(tqdm(desc="QCRSFC/HL null distributions", total=n_iters)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_other_null_iter)(
                mean_qc, z_corr_mats, run_covariates=run_covariates, seed=seed
            )
            for seed in range(n_iters)
        )

    qcrsfc_null_values, hl_null_values = zip(*results)

    qcrsfc_null_values = np.vstack(qcrsfc_null_values)
    hl_null_values = np.vstack(hl_null_values)

    return qcrsfc_null_values, hl_null_values
