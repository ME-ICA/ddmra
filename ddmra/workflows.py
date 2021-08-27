"""Perform distance-dependent motion-related artifact analyses."""
import logging
import os.path as op
from os import makedirs

import numpy as np
import pandas as pd
from nilearn import datasets, input_data
from scipy.spatial.distance import pdist, squareform

from . import ddmra, plotting, utils

LGR = logging.getLogger("workflows")


def run_analyses(
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
        Has four columns: distance, qcrsfc, scrubbing, and highlow.
    - ``smoothing_curves.tsv.gz``: Smoothing curve information for analyses.
        Has four columns: distance, qcrsfc, scrubbing, and highlow.
    - ``[analysis]_analysis_null_smoothing_curves.txt``:
        Null smoothing curves from each analysis.
        Contains 2D array, where number of columns is same
        size and order as distance column in ``smoothing_curves.tsv.gz``
        and number of rows is number of iterations for permutation analysis.
    - ``[analysis]_analysis.png``: Figure for each analysis.
    """
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

    for i_subj in range(n_subjects):
        if confounds:
            raw_ts = spheres_masker.fit_transform(files[i_subj], confounds=confounds[i_subj]).T
        else:
            raw_ts = spheres_masker.fit_transform(files[i_subj]).T

        assert raw_ts.shape[0] == n_rois

        ts_all.append(raw_ts)
        raw_corrs = np.corrcoef(raw_ts)
        raw_corrs = raw_corrs[triu_idx]
        raw_corrs = raw_corrs[sort_idx]  # Sort from close to far ROI pairs
        z_corr_mats[i_subj, :] = np.arctanh(raw_corrs)

    del (raw_corrs, raw_ts, spheres_masker, atlas, coords)

    smoothing_curves = pd.DataFrame(columns=["distance", "qcrsfc", "highlow", "scrubbing"])
    analysis_values = pd.DataFrame(columns=["distance", "qcrsfc", "highlow", "scrubbing"])
    analysis_values["distance"] = distances

    # QC:RSFC r analysis
    LGR.info("Performing QC:RSFC analysis")
    qcrsfc_values = ddmra.qcrsfc_analysis(mean_qc, z_corr_mats)
    analysis_values["qcrsfc"] = qcrsfc_values
    qcrsfc_smoothing_curve = utils.moving_average(qcrsfc_values, window)

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
    highlow_values = ddmra.highlow_analysis(mean_qc, z_corr_mats)
    analysis_values["highlow"] = highlow_values
    hl_smoothing_curve = utils.moving_average(highlow_values, window)[smoothing_curve_dist_idx]
    smoothing_curves["highlow"] = hl_smoothing_curve
    del highlow_values, hl_smoothing_curve

    # Scrubbing analysis
    LGR.info("Performing scrubbing analysis")
    scrub_values = ddmra.scrubbing_analysis(qc, ts_all, sort_idx, qc_thresh, perm=False)
    analysis_values["scrubbing"] = scrub_values
    scrub_smoothing_curve = utils.moving_average(scrub_values, window)[smoothing_curve_dist_idx]
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
    perm_qcrsfc_smoothing_curves, perm_hl_smoothing_curves = ddmra.other_null_distributions(
        qc,
        z_corr_mats,
        smoothing_curve_distances,
        smoothing_curve_dist_idx,
        qc_thresh=qc_thresh,
        window=window,
        n_iters=n_iters,
    )
    perm_scrub_smoothing_curves = ddmra.scrubbing_null_distribution(
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
        "qcrsfc": r"QC:RSFC $z_{r}$" + "\n(QC = mean FD)",
        "highlow": r"High-low motion ${\Delta}z_{r}$",
        "scrubbing": r"Scrubbing ${\Delta}z_{r}$",
    }
    YLIMS = {
        "qcrsfc": (-1.0, 1.0),
        "highlow": (-1.0, 1.0),
        "scrubbing": (-0.05, 0.05),
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

        fig, ax = plotting.plot_analysis(
            values,
            distances,
            smoothing_curve,
            smoothing_curve_distances,
            perm_smoothing_curves,
            n_lines=50,
            ylim=YLIMS[analysis],
            metric_name=label,
            fig=None,
            ax=None,
        )
        fig.savefig(op.join(out_dir, f"{analysis}_analysis.png"), dpi=400)

    LGR.info("Workflow completed")
