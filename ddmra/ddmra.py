"""
Perform distance-dependent motion-related artifact analyses.
"""
import logging
import os.path as op

import numpy as np
from nilearn import datasets, input_data
from scipy.spatial.distance import pdist, squareform

from .filter import filter_earl
from .utils import fast_pearson, get_fd_power, moving_average

LGR = logging.getLogger("ddmra")


def scrubbing_analysis(group_timeseries, fds_analysis, n_rois, qc_thresh=0.2, perm=True):
    """Perform Power scrubbing analysis.

    Note that correlations from scrubbed timeseries are subtracted from correlations from
    unscrubbed timeseries, which is the opposite to the original Power method.
    This reverses the signs of the results, but makes inference similar to that of the
    QCRSFC and high-low motion analyses.

    Parameters
    ----------
    group_timeseries : (N,) list
        List of (R, T) arrays
    fds_analysis : (N,) list
        List of (T,) arrays
    n_rois : int
        Equal to R.
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
    # double check this but should replace argument
    # n_rois = group_timeseries[0].shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    n_pairs = len(triu_idx[0])
    n_subjects = len(group_timeseries)
    delta_rs = np.zeros((n_subjects, n_pairs))
    c = 0  # included subject counter
    for subj in range(n_subjects):
        ts_arr = group_timeseries[subj]
        qc_arr = fds_analysis[subj]
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
        LGR.info("{0} of {1} subjects retained in scrubbing " "analysis".format(c, n_subjects))

    delta_rs = delta_rs[:c, :]
    mean_delta_r = np.mean(delta_rs, axis=0)
    return mean_delta_r


def high_low_motion_analysis(mean_qcs, raw_corr_mats, sort_idx):
    """Perform high-low motion analysis.

    Split the sample using a median split of the QC metric (generally mean FD).
    Then, for each pair of ROIs, calculate the difference between the
    mean across correlation coefficients for the high motion minus the low
    motion groups.

    Parameters
    ----------
    mean_qcs : numpy.ndarray of shape (n_subjects,)
    raw_corr_mats : numpy.ndarray of shape (n_subjects, n_roi_pairs)
    sort_idx : numpy.ndarray of shape (n_roi_pairs,)
    """
    hm_idx = mean_qcs >= np.median(mean_qcs)
    lm_idx = mean_qcs < np.median(mean_qcs)
    hm_mean_corr = np.mean(raw_corr_mats[hm_idx, :], axis=0)
    lm_mean_corr = np.mean(raw_corr_mats[lm_idx, :], axis=0)
    hl_corr_diff = hm_mean_corr - lm_mean_corr
    hl_corr_diff = hl_corr_diff[sort_idx]
    return hl_corr_diff


def qcrsfc(mean_qcs, z_corr_mats, sort_idx):
    """Perform quality-control resting-state functional connectivity analysis."""
    # Correlate each ROI pair's z-value against QC measure (usually FD) across subjects.
    qcrsfc_rs = fast_pearson(z_corr_mats.T, mean_qcs)

    # Sort coefficients by distance
    qcrsfc_rs = qcrsfc_rs[sort_idx]

    return qcrsfc_rs


def run(
    files,
    motpars,
    out_dir=".",
    n_iters=10000,
    qc_thresh=0.2,
    window=1000,
    earl=False,
    regress=False,
    t_r=None,
):
    """Run scrubbing, high-low motion, and QCRSFC analyses.

    Parameters
    ----------
    files : (N,) list of nifti files
        List of 4D (X x Y x Z x T) images in MNI space.
    fds_analysis : (N,) list of array-like
        List of 1D (T) numpy arrays with QC metric values per img (e.g., FD or respiration).
    out_dir : str, optional
        Output directory. Default is current directory.
    n_iters : int, optional
        Number of iterations to run to generate null distributions. Default is 10000.
    qc_thresh : float, optional
        Threshold for QC metric used in scrubbing analysis. Default is 0.2 (for FD).
    window : int, optional
        Number of units (pairs of ROIs) to include when averaging to generate smoothing curve.
        Default is 1000.
    earl : bool, optional
        Whether apply the Earl filter to the motion parameters or not. Default is False.
    regress : bool, optional
        Regress motion parameters from the time series or not. Default is False.
    t_r : float or None, optional
        Repetition time. Only necessary if `earl` is True.

    Notes
    -----
    This function writes out several files to out_dir:
    - all_sorted_distances.txt: Sorted distances between every pair of ROIs.
    - smc_sorted_distances.txt: Sorted distances for smoothing curves. Does not
        include duplicate distances or pairs of ROIs outside of smoothing curve
        (first and last window/2 pairs).
    - qcrsfc_analysis_values.txt: Results from QC:RSFC analysis. The QC:RSFC
        value for each pair of ROIs, of the same size and in the same order as
        all_sorted_distances.txt.
    - qcrsfc_analysis_smoothing_curve.txt: Smoothing curve for QC:RSFC
        analysis. Same size and order as smc_sorted_distances.txt.
    - qcrsfc_analysis_null_smoothing_curves.txt: Null smoothing curves from
        QC:RSFC analysis. Contains 2D array, where number of columns is same
        size and order as smc_sorted_distances.txt and number of rows is number
        of iterations for permutation analysis.
    - highlow_analysis_values.txt: Results from high-low analysis. The delta r
        value for each pair of ROIs, of the same size and in the same order as
        all_sorted_distances.txt.
    - highlow_analysis_smoothing_curve.txt: Smoothing curve for high-low
        analysis. Same size and order as smc_sorted_distances.txt.
    - highlow_analysis_null_smoothing_curves.txt: Null smoothing curves from
        high-low analysis. Contains 2D array, where number of columns is same
        size and order as smc_sorted_distances.txt and number of rows is number
        of iterations for permutation analysis.
    - scrubbing_analysis_values.txt: Results from scrubbing analysis. The
        mean delta r value for each pair of ROIs, of the same size and in the
        same order as all_sorted_distances.txt.
    - scrubbing_analysis_smoothing_curve.txt: Smoothing curve for scrubbing
        analysis. Same size and order as smc_sorted_distances.txt.
    - scrubbing_analysis_null_smoothing_curves.txt: Null smoothing curves from
        scrubbing analysis. Contains 2D array, where number of columns is same
        size and order as smc_sorted_distances.txt and number of rows is number
        of iterations for permutation analysis.
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
    assert len(files) == len(motpars)
    n_subjects = len(files)

    # Load atlas and associated masker
    atlas = datasets.fetch_coords_power_2011()
    coords = np.vstack((atlas.rois["x"], atlas.rois["y"], atlas.rois["z"])).T
    n_rois = coords.shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    dists = squareform(pdist(coords))
    dists = dists[triu_idx]

    # Sorting index for distances
    sort_idx = dists.argsort()
    all_sorted_dists = dists[sort_idx]
    np.savetxt(op.join(out_dir, "all_sorted_distances.txt"), all_sorted_dists)
    unique_dists_idx = np.array(
        [np.where(all_sorted_dists == i)[0][0] for i in np.unique(all_sorted_dists)]
    )

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

    # Apply filter is necessary
    motpars_analysis, fds_analysis = [], []
    for motpars_ in motpars:
        if earl:
            LGR.info("Filtering motion parameters")
            motpars_filt, fd_filt = filter_earl(motpars_, t_r)
            motpars_analysis.append(motpars_filt)
            fds_analysis.append(fd_filt)
        else:
            motpars_analysis.append(motpars_)
            fds_analysis.append(get_fd_power(motpars_, unit="rad"))

    # prep for qcrsfc and high-low motion analyses
    mean_qcs = np.array([np.mean(qc) for qc in fds_analysis])
    raw_corr_mats = np.zeros((n_subjects, dists.size))

    # Get correlation matrices
    ts_all = []
    LGR.info("Building correlation matrices")
    for i_sub in range(n_subjects):
        if regress:
            raw_ts = spheres_masker.fit_transform(
                files[i_sub],
                confounds=motpars_analysis[i_sub],
            ).T
        else:
            raw_ts = spheres_masker.fit_transform(files[i_sub]).T

        assert raw_ts.shape[0] == n_rois

        ts_all.append(raw_ts)
        raw_corrs = np.corrcoef(raw_ts)
        raw_corrs = raw_corrs[triu_idx]
        raw_corr_mats[i_sub, :] = raw_corrs

    z_corr_mats = np.arctanh(raw_corr_mats)
    del (raw_corrs, raw_ts, spheres_masker, atlas, coords)

    smoothing_curves = pd.DataFrame(
        columns=["distance", "qcrsfc", "high_low", "scrubbing"],
    )

    # QC:RSFC r analysis
    LGR.info("Performing QC:RSFC analysis")
    qcrsfc_rs = qcrsfc(
        mean_qcs,
        z_corr_mats,
        sort_idx
    )

    # Calculate smoothing curve over distance
    qcrsfc_smoothing_curve = moving_average(qcrsfc_rs, window)

    # Quick interlude to help reduce arrays
    keep_idx = np.intersect1d(np.where(~np.isnan(qcrsfc_smoothing_curve))[0], unique_dists_idx)
    keep_sorted_dists = all_sorted_dists[keep_idx]
    smoothing_curves["distance"] = keep_sorted_dists

    qcrsfc_smoothing_curve = qcrsfc_smoothing_curve[keep_idx]
    smoothing_curves["qcrsfc"] = qcrsfc_smoothing_curve
    np.savetxt(op.join(out_dir, "qcrsfc_analysis_values.txt"), qcrsfc_rs)

    # High-low motion analysis
    LGR.info("Performing high-low motion analysis")
    hl_corr_diff = high_low_motion_analysis(
        mean_qcs,
        raw_corr_mats,
        sort_idx
    )
    np.savetxt(op.join(out_dir, "highlow_analysis_values.txt"), hl_corr_diff)
    hl_smoothing_curve = moving_average(hl_corr_diff, window)
    hl_smoothing_curve = hl_smoothing_curve[keep_idx]
    smoothing_curves["high_low"] = hl_smoothing_curve

    # Scrubbing analysis
    LGR.info("Performing scrubbing analysis")
    mean_delta_r = scrubbing_analysis(ts_all, fds_analysis, n_rois, qc_thresh, perm=False)
    mean_delta_r = mean_delta_r[sort_idx]
    scrub_smoothing_curve = moving_average(mean_delta_r, window)
    np.savetxt(op.join(out_dir, "scrubbing_analysis_values.txt"), mean_delta_r)
    scrub_smoothing_curve = scrub_smoothing_curve[keep_idx]
    smoothing_curves["scrubbing"] = scrub_smoothing_curve
    del mean_delta_r, scrub_smoothing_curve

    smoothing_curves.to_csv(
        op.join(out_dir, "smoothing_curves.tsv"),
        sep="\t",
        line_terminator="\n",
    )

    # Null distributions
    LGR.info("Building null distributions with permutations")
    qcs_copy = [qc.copy() for qc in fds_analysis]
    perm_scrub_smoothing_curve = np.zeros((n_iters, len(keep_sorted_dists)))
    perm_qcrsfc_smoothing_curve = np.zeros((n_iters, len(keep_sorted_dists)))
    perm_hl_smoothing_curve = np.zeros((n_iters, len(keep_sorted_dists)))
    for i_iter in range(n_iters):
        # Prep for QC:RSFC and high-low motion analyses
        perm_mean_qcs = np.random.permutation(mean_qcs)

        # QC:RSFC analysis
        perm_qcrsfc_rs = qcrsfc(perm_mean_qcs, z_corr_mats, sort_idx)
        perm_qcrsfc_smoothing_curve[i_iter, :] = moving_average(perm_qcrsfc_rs, window)[keep_idx]

        # High-low analysis
        perm_hl_diff = high_low_motion_analysis(perm_mean_qcs, raw_corr_mats, sort_idx)
        perm_hl_smoothing_curve[i_iter, :] = moving_average(perm_hl_diff, window)[keep_idx]

        # Scrubbing analysis
        perm_qcs = [np.random.permutation(perm_qc) for perm_qc in qcs_copy]
        perm_mean_delta_r = scrubbing_analysis(ts_all, perm_qcs, n_rois, qc_thresh, perm=True)
        perm_mean_delta_r = perm_mean_delta_r[sort_idx]
        perm_scrub_smoothing_curve[i_iter, :] = moving_average(perm_mean_delta_r, window)[keep_idx]

    np.savetxt(
        op.join(out_dir, "qcrsfc_analysis_null_smoothing_curves.txt"),
        perm_qcrsfc_smoothing_curve,
    )
    np.savetxt(
        op.join(out_dir, "highlow_analysis_null_smoothing_curves.txt"),
        perm_hl_smoothing_curve,
    )
    np.savetxt(
        op.join(out_dir, "scrubbing_analysis_null_smoothing_curves.txt"),
        perm_scrub_smoothing_curve,
    )

    LGR.info("Workflow completed")
    LGR.shutdown()
