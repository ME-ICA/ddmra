"""Perform distance-dependent motion-related artifact analyses."""

import logging
import os.path as op
from os import PathLike, makedirs

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.plotting import find_parcellation_cut_coords
from scipy.spatial.distance import pdist, squareform

from . import analysis, plotting, utils

LGR = logging.getLogger("workflows")


def _masker_kwargs():
    """Return shared masker settings for atlas time series extraction."""
    return {
        "t_r": None,
        "smoothing_fwhm": None,
        "detrend": False,
        "standardize": False,
        "low_pass": None,
        "high_pass": None,
    }


def _sphere_fetcher_name(atlas):
    """Convert a user-facing sphere atlas name to a nilearn fetcher name."""
    atlas = atlas.lower().replace("-", "_")
    if atlas.startswith("fetch_coords_"):
        return atlas
    if atlas.startswith("coords_"):
        return f"fetch_{atlas}"
    return f"fetch_coords_{atlas}"


def _coords_from_sphere_atlas(atlas):
    """Fetch coordinate atlas centers from nilearn.datasets."""
    fetcher_name = _sphere_fetcher_name(atlas)
    fetcher = getattr(datasets, fetcher_name, None)
    if fetcher is None:
        fetchers = sorted(
            name.removeprefix("fetch_coords_")
            for name in dir(datasets)
            if name.startswith("fetch_coords_")
        )
        raise ValueError(
            f"Unknown sphere atlas '{atlas}'. Available nilearn coordinate atlases are: "
            f"{', '.join(fetchers)}."
        )

    fetched_atlas = fetcher()
    if hasattr(fetched_atlas, "rois"):
        rois = fetched_atlas.rois
        coords = np.vstack((rois["x"], rois["y"], rois["z"])).T
    elif all(hasattr(fetched_atlas, axis) for axis in ("x", "y", "z")):
        coords = np.vstack((fetched_atlas.x, fetched_atlas.y, fetched_atlas.z)).T
    else:
        raise ValueError(f"Coordinate atlas '{atlas}' does not expose x/y/z ROI coordinates.")

    return np.asarray(coords, dtype=float)


def _build_atlas_masker(atlas="power_2011", sphere_radius=5.0):
    """Create a Nilearn masker and ROI coordinates from an atlas specification."""
    kwargs = _masker_kwargs()

    if isinstance(atlas, PathLike) or (isinstance(atlas, str) and op.isfile(atlas)):
        coords = find_parcellation_cut_coords(atlas)
        masker = NiftiLabelsMasker(labels_img=atlas, **kwargs)
    elif isinstance(atlas, str):
        coords = _coords_from_sphere_atlas(atlas)
        masker = NiftiSpheresMasker(seeds=coords, radius=sphere_radius, **kwargs)
    else:
        raise TypeError("atlas must be a path to a labels image or a nilearn sphere atlas name.")

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Atlas coordinates must have shape (n_rois, 3), not {coords.shape}.")
    if coords.shape[0] < 2:
        raise ValueError("Atlas must define at least two ROIs.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("Atlas coordinates must all be finite.")

    return masker, coords


def _prepare_run_covariates(run_covariates, n_subjects):
    """Validate and encode run-level covariates for QC:RSFC adjustment."""
    if run_covariates is None:
        return None
    if not isinstance(run_covariates, pd.DataFrame):
        raise TypeError("run_covariates must be a pandas DataFrame.")
    if run_covariates.shape[0] != n_subjects:
        raise ValueError(
            f"run_covariates has {run_covariates.shape[0]} rows, but {n_subjects} runs "
            "were provided."
        )
    if run_covariates.shape[1] == 0:
        raise ValueError("run_covariates must include at least one column.")
    if run_covariates.isna().any().any():
        raise ValueError("run_covariates cannot contain missing values.")

    design = pd.get_dummies(run_covariates.reset_index(drop=True), drop_first=True, dtype=float)
    if design.shape[1] == 0:
        raise ValueError("run_covariates did not produce any usable covariate columns.")

    covariates = design.to_numpy(dtype=float)
    if not np.all(np.isfinite(covariates)):
        raise ValueError("run_covariates must encode to finite numeric values.")

    return covariates


def _select_n_pca_components(varex_cumsum, pca_threshold):
    """Select a one-based PCA component count from cumulative explained variance."""
    if isinstance(pca_threshold, float):
        n_components = np.where(varex_cumsum >= pca_threshold)[0][0] + 1
    else:
        n_components = pca_threshold

    if not 1 <= n_components <= varex_cumsum.size:
        raise ValueError(
            "pca_threshold must select between 1 and "
            f"{varex_cumsum.size} PCA components, not {n_components}."
        )

    perc_varex = varex_cumsum[n_components - 1] * 100
    return n_components, perc_varex


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
    verbose=False,
    pca_threshold=None,
    outlier_threshold=None,
    atlas="power_2011",
    sphere_radius=5.0,
    run_covariates=None,
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
    analyses : tuple, optional
        The analyses to run. Must be one or more of "qcrsfc", "highlow", "scrubbing".
    verbose : bool, optional
        If verbose, write out the correlation coefficients used by the QC:RSFC and high-low
        analyses. Default is False.
    pca_threshold : None or float or int, optional
        If None, do not perform outlier detection at all.
        If a float, perform PCA and retain components explain that proportion of the variance.
        If an int, perform PCA and retain that number of components.
    outlier_threshold : None or float, optional
        If None, do not perform outlier detection at all.
        If a float, flag any runs with Mahalanobis distance p-value < the float according to
        chi-squared distribution.
    atlas : str or path-like, optional
        Atlas to use for ROI time series extraction.
        If a path to an existing file, the file is treated as a labels image and loaded with
        :class:`nilearn.maskers.NiftiLabelsMasker`.
        Otherwise, the value is treated as the name of a coordinate atlas available through
        :mod:`nilearn.datasets`, such as ``"power_2011"``, ``"dosenbach_2010"``, or
        ``"seitzman_2018"``, and loaded with :class:`nilearn.maskers.NiftiSpheresMasker`.
        Default is ``"power_2011"``.
    sphere_radius : float, optional
        Radius in millimeters for sphere atlases. Ignored when ``atlas`` is a labels image.
        Default is 5.0.
    run_covariates : None or pandas.DataFrame, optional
        Run-level covariates to adjust for in the QC:RSFC analysis.
        Rows must correspond to ``files`` in order. Numeric columns are used directly, and
        categorical columns are dummy-coded with one reference level.
        Default is None.

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
    - ``ranks.tsv.gz``: Edgewise ranks of the observed analysis values against
        the edgewise null distributions.
    - ``[analysis]_analysis.png``: Figure for each analysis.

    If ``verbose`` is ``True``:
    - ``z_corrs.tsv.gz``: Z-transformed correlation coefficients for the good files,
        used by QC:RSFC and high-low analyses.
    - ``mean_qcs.tsv.gz``: Mean QC values for the good files, used by QC:RSFC and high-low
        analyses.
    """
    ALLOWED_ANALYSES = ("qcrsfc", "highlow", "scrubbing")
    assert len(analyses) > 0, "At least one analysis must be selected."
    assert all([a in ALLOWED_ANALYSES for a in analyses]), (
        "Parameter 'analyses' must be a tuple of one or more of the following values: "
        f"{', '.join(ALLOWED_ANALYSES)}"
    )

    if (pca_threshold is None) and (outlier_threshold is None):
        LGR.info("Not performing outlier detection.")
    elif (pca_threshold is None) or (outlier_threshold is None):
        raise ValueError("Both pca_threshold and outlier_threshold must be None or not None.")
    elif isinstance(pca_threshold, int) and isinstance(outlier_threshold, float):
        LGR.info(f"Performing outlier detection on first {pca_threshold} PCA components.")
    elif isinstance(pca_threshold, float) and isinstance(outlier_threshold, float):
        assert 0 < pca_threshold < 1, "Threshold must be between 0 and 1."
        LGR.info(
            "Performing outlier detection on PCA components explaining "
            f"{pca_threshold * 100}% of the variance."
        )
    else:
        raise ValueError("Bad inputs.")

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
    if run_covariates is not None and "qcrsfc" not in analyses:
        LGR.info("Ignoring run_covariates because QC:RSFC analysis was not requested.")
        run_covariates = None
    else:
        run_covariates = _prepare_run_covariates(run_covariates, n_subjects)

    # Load atlas and associated masker
    atlas_masker, coords = _build_atlas_masker(atlas=atlas, sphere_radius=sphere_radius)
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

    if ("qcrsfc" in analyses) or ("highlow" in analyses):
        # prep for qcrsfc and high-low motion analyses
        mean_qc = np.array([np.mean(subj_qc) for subj_qc in qc])
        z_corr_mats = np.zeros((n_subjects, distances.size))
        assert mean_qc.size == n_subjects, f"{mean_qc.size} != {n_subjects}"

    # Get correlation matrices
    ts_all = []
    LGR.info("Building correlation matrices")
    if confounds:
        LGR.info("Regressing confounds out of data.")

    good_subjects = []
    for i_subj in range(n_subjects):
        skip_subject = False
        if confounds:
            raw_ts = atlas_masker.fit_transform(files[i_subj], confounds=confounds[i_subj]).T
        else:
            raw_ts = atlas_masker.fit_transform(files[i_subj]).T

        assert raw_ts.shape[0] == n_rois, f"{raw_ts.shape[0]} != {n_rois}"

        if np.any(np.isnan(raw_ts)):
            LGR.warning(f"Time series of {files[i_subj]} contains NaNs. Dropping from analysis.")
            skip_subject = True

        roi_variances = np.var(raw_ts, axis=1)
        if any(np.isclose(roi_variances, 0)):
            bad_rois = np.where(np.isclose(roi_variances, 0))[0]
            LGR.warning(
                f"ROI(s) {bad_rois} for {files[i_subj]} have variance of 0. "
                "Dropping from analysis."
            )
            skip_subject = True

        if skip_subject:
            continue

        # This list will only include good subjects, so there's no need to reduce it later
        ts_all.append(raw_ts)

        if ("qcrsfc" in analyses) or ("highlow" in analyses):
            raw_corrs = np.corrcoef(raw_ts)
            raw_corrs = raw_corrs[triu_idx]
            raw_corrs = raw_corrs[edge_sorting_idx]  # Sort from close to far ROI pairs
            z_corr_mats[i_subj, :] = utils.r2z(raw_corrs)

            del raw_corrs

        good_subjects.append(i_subj)

    del (atlas_masker, coords)

    good_subjects = np.array(good_subjects)
    n_subjects_remaining = good_subjects.size
    LGR.info(f"Retaining {n_subjects_remaining}/{n_subjects} after loading data.")

    if "scrubbing" in analyses:
        qc = [qc[i] for i in good_subjects]

    if ("qcrsfc" in analyses) or ("highlow" in analyses):
        z_corr_mats = z_corr_mats[good_subjects, :]
        mean_qc = mean_qc[good_subjects]
        if run_covariates is not None:
            run_covariates = run_covariates[good_subjects, :]

        # Assumes no periods in the filename except for the extension
        file_names = [op.basename(files[i]).split(".")[0] for i in good_subjects]

    # Time to do some outlier detection
    if pca_threshold is not None:
        from scipy.stats import chi2
        from sklearn.covariance import MinCovDet
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Define the PCA object
        pca = PCA()

        # Run PCA on scaled data and obtain the scores array
        pca_components = pca.fit_transform(StandardScaler().fit_transform(z_corr_mats))

        varex_cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components, perc_varex = _select_n_pca_components(varex_cumsum, pca_threshold)
        LGR.info(f"{n_components} components selected ({perc_varex:.02f}% variance explained)")

        # Select the components
        pca_components = pca_components[:, :n_components]

        # Fit a Minimum Covariance Determinant (MCD) robust estimator to data
        robust_cov = MinCovDet().fit(pca_components)

        # Get the Mahalanobis distance
        mahalanobis_distances = robust_cov.mahalanobis(pca_components)

        # Use chi2 threshold, based on
        # https://towardsdatascience.com/multivariate-outlier-detection-in-python-e946cfc843b3
        # Cutoff (threshold) value from chi-square distribution for detecting outliers
        cutoff = chi2.ppf(1 - outlier_threshold, pca_components.shape[1])
        outlier_idx = np.where(mahalanobis_distances > cutoff)[0]
        keep_idx = np.where(mahalanobis_distances <= cutoff)[0]
        LGR.info(f"Removing {outlier_idx.size} outliers of {mahalanobis_distances.size} runs.")
        n_subjects_remaining = keep_idx.size

        if ("qcrsfc" in analyses) or ("highlow" in analyses):
            z_corr_mats = z_corr_mats[keep_idx, :]
            mean_qc = mean_qc[keep_idx]
            if run_covariates is not None:
                run_covariates = run_covariates[keep_idx, :]
            file_names = [file_names[i] for i in keep_idx]

        if "scrubbing" in analyses:
            ts_all = [ts_all[i] for i in keep_idx]
            qc = [qc[i] for i in keep_idx]

    if verbose and (("qcrsfc" in analyses) or ("highlow" in analyses)):
        LGR.info("Saving z-transformed file-wise correlation coefficients")
        corrs_df = pd.DataFrame(index=distances, columns=file_names, data=z_corr_mats.T)
        corrs_df.to_csv(
            op.join(out_dir, "z_corrs.tsv.gz"),
            sep="\t",
            lineterminator="\n",
            index=True,
            index_label="distance",
        )
        mean_qc_df = pd.DataFrame(index=file_names, columns=["mean_qc"], data=mean_qc[:, None])
        mean_qc_df.to_csv(
            op.join(out_dir, "mean_qcs.tsv.gz"),
            sep="\t",
            lineterminator="\n",
            index=True,
            index_label="filename",
        )
        del corrs_df, mean_qc_df, file_names

    LGR.info(f"Retaining {n_subjects_remaining}/{n_subjects} subjects for analysis.")
    if n_subjects_remaining < 10:
        raise ValueError("Too few subjects remaining for analysis.")

    analysis_values = pd.DataFrame(columns=analyses, index=distances, dtype=float)
    analysis_values.index.name = "distance"

    ranks_df = pd.DataFrame(columns=analyses, index=distances)
    ranks_df.index.name = "distance"

    # Create the smoothing_curves DataFrame
    ma_distances = utils.moving_average(distances, window)
    _, smoothing_curve_distances = utils.average_across_distances(ma_distances, distances)
    smoothing_curves = pd.DataFrame(columns=analyses, index=smoothing_curve_distances, dtype=float)
    smoothing_curves.index.name = "distance"

    if "qcrsfc" in analyses:
        # QC:RSFC r analysis
        LGR.info("Performing QC:RSFC analysis")
        analysis_values["qcrsfc"] = analysis.qcrsfc_analysis(
            mean_qc, z_corr_mats, run_covariates=run_covariates
        )
        qcrsfc_smoothing_curve = utils.moving_average(analysis_values["qcrsfc"], window)
        qcrsfc_smoothing_curve, qcrsfc_smoothing_curve_distances = utils.average_across_distances(
            qcrsfc_smoothing_curve,
            distances,
        )
        assert np.array_equal(smoothing_curve_distances, qcrsfc_smoothing_curve_distances), (
            f"{smoothing_curve_distances} != {qcrsfc_smoothing_curve_distances}"
        )
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
        assert np.array_equal(smoothing_curve_distances, hl_smoothing_curve_distances), (
            f"{smoothing_curve_distances} != {hl_smoothing_curve_distances}"
        )
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
        assert np.array_equal(smoothing_curve_distances, scrub_smoothing_curve_distances), (
            f"{smoothing_curve_distances} != {scrub_smoothing_curve_distances}"
        )
        smoothing_curves.loc[smoothing_curve_distances, "scrubbing"] = scrub_smoothing_curve
        del scrub_smoothing_curve

    analysis_values.reset_index(inplace=True)
    smoothing_curves.reset_index(inplace=True)

    analysis_values.to_csv(
        op.join(out_dir, "analysis_values.tsv.gz"),
        sep="\t",
        lineterminator="\n",
        index=False,
    )
    smoothing_curves.to_csv(
        op.join(out_dir, "smoothing_curves.tsv.gz"),
        sep="\t",
        lineterminator="\n",
        index=False,
    )

    # Null distributions
    LGR.info("Building null distributions with permutations")
    null_curves_dict = {}

    if ("qcrsfc" in analyses) or ("highlow" in analyses):
        qcrsfc_null_values, hl_null_values = analysis.other_null_distributions(
            mean_qc,
            z_corr_mats,
            run_covariates=run_covariates,
            n_iters=n_iters,
            n_jobs=n_jobs,
        )
        if "qcrsfc" in analyses:
            np.savez_compressed(op.join(out_dir, "qcrsfc_null.npz"), qcrsfc_null_values)
            ranks_df["qcrsfc"] = utils.get_rank(
                analysis_values["qcrsfc"].values, qcrsfc_null_values
            )

            qcrsfc_null_smoothing_curves = utils.calculate_smoothing_curve(
                qcrsfc_null_values,
                window,
                distances,
                smoothing_curve_distances,
            )
            null_curves_dict["qcrsfc"] = qcrsfc_null_smoothing_curves.copy()
            del qcrsfc_null_smoothing_curves

        if "highlow" in analyses:
            np.savez_compressed(op.join(out_dir, "hl_null.npz"), hl_null_values)
            ranks_df["highlow"] = utils.get_rank(analysis_values["highlow"].values, hl_null_values)

            hl_null_smoothing_curves = utils.calculate_smoothing_curve(
                hl_null_values,
                window,
                distances,
                smoothing_curve_distances,
            )
            null_curves_dict["highlow"] = hl_null_smoothing_curves.copy()
            del hl_null_smoothing_curves

    if "scrubbing" in analyses:
        scrubbing_null_values = analysis.scrubbing_null_distribution(
            qc,
            ts_all,
            qc_thresh,
            edge_sorting_idx,
            n_iters=n_iters,
            n_jobs=n_jobs,
        )
        np.savez_compressed(op.join(out_dir, "scrubbing_null.npz"), scrubbing_null_values)
        ranks_df["scrubbing"] = utils.get_rank(
            analysis_values["scrubbing"].values, scrubbing_null_values
        )

        scrubbing_null_smoothing_curves = utils.calculate_smoothing_curve(
            scrubbing_null_values,
            window,
            distances,
            smoothing_curve_distances,
        )
        null_curves_dict["scrubbing"] = scrubbing_null_smoothing_curves.copy()
        del scrubbing_null_smoothing_curves

    ranks_df.reset_index(inplace=True)
    ranks_df.to_csv(
        op.join(out_dir, "ranks.tsv.gz"),
        sep="\t",
        lineterminator="\n",
        index=False,
    )

    for ana, null_curve in null_curves_dict.items():
        assert null_curve.shape == (
            n_iters,
            smoothing_curve_distances.size,
        ), f"{ana}: {null_curve.shape} != ({n_iters}, {smoothing_curve_distances.size})"

    for analysis_name in analyses:
        p_inter, p_slope = utils.assess_significance(
            smoothing_curves[analysis_name].values,
            null_curves_dict[analysis_name],
            smoothing_curves["distance"].values,
            35,
            100,
        )
        LGR.info(
            f"For {analysis_name} analysis, intercept (p = {p_inter}); "
            f"slope (p = {p_slope})."
        )

    np.savez_compressed(op.join(out_dir, "null_smoothing_curves.npz"), **null_curves_dict)

    del null_curves_dict

    plotting.plot_results(out_dir)

    LGR.info("Workflow completed")
