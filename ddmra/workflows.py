"""Perform distance-dependent motion-related artifact analyses."""
import logging
import os.path as op
from os import makedirs

import numpy as np
import pandas as pd
from nilearn import datasets, input_data
from scipy.spatial.distance import pdist, squareform

from . import analysis, plotting, utils

LGR = logging.getLogger("workflows")


def run_analyses(
    files,
    qc,
    out_dir=".",
    confounds=None,
    n_iters=10000,
    n_jobs=1,
    qc_thresh=0.2,
    window=1000,
    analyses=("qcrsfc", "highlow", "scrubbing"),
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
    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means 'all CPUs'. Default is 1.
    qc_thresh : float, optional
        Threshold for QC metric used in scrubbing analysis. Default is 0.2 (for FD).
    window : int, optional
        Number of units (pairs of ROIs) to include when averaging to generate smoothing curve.
        Default is 1000.

    Notes
    -----
    This function writes out several files to out_dir:
    - ``analysis_values.tsv.gz``: Raw analysis values for analyses.
        Has four columns: distance, qcrsfc, scrubbing, and highlow.
    - ``smoothing_curves.tsv.gz``: Smoothing curve information for analyses.
        Has four columns: distance, qcrsfc, scrubbing, and highlow.
    - ``null_smoothing_curves.npz``:
        Null smoothing curves from each analysis.
        Contains three 2D arrays, where number of columns is same
        size and order as distance column in ``smoothing_curves.tsv.gz``
        and number of rows is number of iterations for permutation analysis.
        The three arrays' keys are 'qcrsfc', 'highlow', and 'scrubbing'.
    - ``[analysis]_analysis.png``: Figure for each analysis.
    """
    ALLOWED_ANALYSES = ("qcrsfc", "highlow", "scrubbing")
    assert len(analyses) > 0, "At least one analysis must be selected."
    assert all([a in ALLOWED_ANALYSES for a in analyses]), (
        "Parameter 'analyses' must be a tuple of one or more of the following values: "
        f"{', '.join(ALLOWED_ANALYSES)}"
    )

    makedirs(out_dir, exist_ok=True)

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
    assert len(qc) == n_subjects, f"{len(qc)} != {n_subjects}"

    # Load atlas and associated masker
    atlas = datasets.fetch_coords_power_2011()
    coords = np.vstack((atlas.rois["x"], atlas.rois["y"], atlas.rois["z"])).T
    n_rois = coords.shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    distances = squareform(pdist(coords))
    distances = distances[triu_idx]

    # Round distances to the nearest thousandth to prevent float comparison issues later
    distances = np.round(distances, decimals=3)

    # Sorting index for distances
    edge_sorting_idx = distances.argsort()
    distances = distances[edge_sorting_idx]

    LGR.info("Creating masker")
    spheres_masker = input_data.NiftiSpheresMasker(
        seeds=coords,
        radius=5.0,
        t_r=None,
        smoothing_fwhm=None,
        detrend=False,
        standardize=False,
        low_pass=None,
        high_pass=None,
    )

    if ("qcrsfc" in analyses) or ("highlow" in analyses):
        # prep for qcrsfc and high-low motion analyses
        mean_qc = np.array([np.mean(subj_qc) for subj_qc in qc])
        z_corr_mats = np.zeros((n_subjects, distances.size))
        assert mean_qc.size == n_subjects

    # Get correlation matrices
    ts_all = []
    LGR.info("Building correlation matrices")
    if confounds:
        LGR.info("Regressing confounds out of data.")

    good_subjects = []
    for i_subj in range(n_subjects):
        skip_subject = False
        if confounds:
            raw_ts = spheres_masker.fit_transform(files[i_subj], confounds=confounds[i_subj]).T
        else:
            raw_ts = spheres_masker.fit_transform(files[i_subj]).T

        assert raw_ts.shape[0] == n_rois

        if np.any(np.isnan(raw_ts)):
            LGR.warning(f"Time series of {files[i_subj]} contains NaNs. Dropping from analysis.")
            skip_subject = True

        roi_variances = np.var(raw_ts, axis=1)
        if any(roi_variances == 0):
            bad_rois = np.where(roi_variances == 0)[0]
            LGR.warning(
                f"ROI(s) {bad_rois} for {files[i_subj]} have variance of 0. "
                "Dropping from analysis."
            )
            skip_subject = True

        if skip_subject:
            continue

        ts_all.append(raw_ts)
        raw_corrs = np.corrcoef(raw_ts)
        raw_corrs = raw_corrs[triu_idx]
        raw_corrs = raw_corrs[edge_sorting_idx]  # Sort from close to far ROI pairs
        z_corr_mats[i_subj, :] = utils.r2z(raw_corrs)
        good_subjects.append(i_subj)

    del (raw_corrs, raw_ts, spheres_masker, atlas, coords)

    good_subjects = np.array(good_subjects)
    qc = [qc[i] for i in good_subjects]

    if ("qcrsfc" in analyses) or ("highlow" in analyses):
        z_corr_mats = z_corr_mats[good_subjects, :]
        mean_qc = mean_qc[good_subjects]

    LGR.info(f"Retaining {len(good_subjects)}/{n_subjects} subjects for analysis.")
    if len(good_subjects) < 10:
        raise ValueError("Too few subjects remaining for analysis.")

    analysis_values = pd.DataFrame(columns=analyses, index=distances)
    analysis_values.index.name = "distance"

    # Create the smoothing_curves DataFrame
    ma_distances = utils.moving_average(distances, window)
    _, smoothing_curve_distances = utils.average_across_distances(ma_distances, distances)
    smoothing_curves = pd.DataFrame(columns=analyses, index=smoothing_curve_distances)
    smoothing_curves.index.name = "distance"

    if "qcrsfc" in analyses:
        # QC:RSFC r analysis
        LGR.info("Performing QC:RSFC analysis")
        analysis_values["qcrsfc"] = analysis.qcrsfc_analysis(mean_qc, z_corr_mats)
        qcrsfc_smoothing_curve = utils.moving_average(analysis_values["qcrsfc"], window)
        qcrsfc_smoothing_curve, qcrsfc_smoothing_curve_distances = utils.average_across_distances(
            qcrsfc_smoothing_curve,
            distances,
        )
        assert np.array_equal(smoothing_curve_distances, qcrsfc_smoothing_curve_distances)
        smoothing_curves.loc[smoothing_curve_distances, "qcrsfc"] = qcrsfc_smoothing_curve
        del qcrsfc_smoothing_curve

    if "highlow" in analyses:
        # High-low motion analysis
        LGR.info("Performing high-low motion analysis")
        analysis_values["highlow"] = analysis.highlow_analysis(mean_qc, z_corr_mats)
        hl_smoothing_curve = utils.moving_average(analysis_values["highlow"], window)
        hl_smoothing_curve, hl_smoothing_curve_distances = utils.average_across_distances(
            hl_smoothing_curve,
            distances,
        )
        assert np.array_equal(smoothing_curve_distances, hl_smoothing_curve_distances)
        smoothing_curves.loc[smoothing_curve_distances, "highlow"] = hl_smoothing_curve
        del hl_smoothing_curve

    if "scrubbing" in analyses:
        # Scrubbing analysis
        LGR.info("Performing scrubbing analysis")
        analysis_values["scrubbing"] = analysis.scrubbing_analysis(
            qc, ts_all, edge_sorting_idx, qc_thresh, perm=False
        )
        scrub_smoothing_curve = utils.moving_average(analysis_values["scrubbing"], window)
        scrub_smoothing_curve, scrub_smoothing_curve_distances = utils.average_across_distances(
            scrub_smoothing_curve,
            distances,
        )
        assert np.array_equal(smoothing_curve_distances, scrub_smoothing_curve_distances)
        smoothing_curves.loc[smoothing_curve_distances, "scrubbing"] = scrub_smoothing_curve
        del scrub_smoothing_curve

    analysis_values.reset_index(inplace=True)
    smoothing_curves.reset_index(inplace=True)

    analysis_values.to_csv(
        op.join(out_dir, "analysis_values.tsv.gz"),
        sep="\t",
        line_terminator="\n",
        index=False,
    )
    smoothing_curves.to_csv(
        op.join(out_dir, "smoothing_curves.tsv.gz"),
        sep="\t",
        line_terminator="\n",
        index=False,
    )

    # Null distributions
    LGR.info("Building null distributions with permutations")
    null_curves_dict = {}
    if ("qcrsfc" in analyses) or ("highlow" in analyses):
        qcrsfc_null_smoothing_curves, hl_null_smoothing_curves = analysis.other_null_distributions(
            qc,
            z_corr_mats,
            distances,
            window=window,
            n_iters=n_iters,
            n_jobs=n_jobs,
        )
        if "qcrsfc" in analyses:
            null_curves_dict["qcrsfc"] = qcrsfc_null_smoothing_curves.copy()
            del qcrsfc_null_smoothing_curves

        if "highlow" in analyses:
            null_curves_dict["highlow"] = hl_null_smoothing_curves.copy()
            del hl_null_smoothing_curves

    if "scrubbing" in analyses:
        null_curves_dict["scrubbing"] = analysis.scrubbing_null_distribution(
            qc,
            ts_all,
            distances,
            qc_thresh,
            edge_sorting_idx,
            window=window,
            n_iters=n_iters,
            n_jobs=n_jobs,
        )

    np.savez_compressed(op.join(out_dir, "null_smoothing_curves.npz"), **null_curves_dict)

    del null_curves_dict

    plotting.plot_results(out_dir)

    LGR.info("Workflow completed")
