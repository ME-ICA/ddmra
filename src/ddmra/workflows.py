"""Perform distance-dependent motion-related artifact analyses."""

import logging
import os.path as op
import re
import warnings
from itertools import combinations
from os import PathLike, makedirs

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.plotting import find_parcellation_cut_coords
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from . import analysis, plotting, utils

LGR = logging.getLogger("workflows")
ALLOWED_ANALYSES = ("qcrsfc", "highlow", "scrubbing")
COMPARISON_INTERCEPT_DISTANCE = 35
COMPARISON_SLOPE_DISTANCE = 100
# Minimum number of runs required to attempt analysis.
MIN_SUBJECTS = 10
# QC-FC estimates are unstable in small samples. Parkes et al. (2018) and Ciric
# et al. (2017) show sampling variability dominates below roughly 30 runs, so warn
# (but do not error) when the retained sample is in the MIN_SUBJECTS-to-this range.
QCFC_STABILITY_N = 30


def _reset_workflow_log_handler():
    """Remove the previous workflow file handler, if one exists."""
    for handler in list(LGR.handlers):
        if getattr(handler, "_ddmra_workflow_handler", False):
            LGR.removeHandler(handler)
            handler.close()


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


def _validate_analyses(analyses):
    """Validate and return requested DDMRA analyses."""
    if len(analyses) == 0:
        raise ValueError("At least one analysis must be selected.")
    if not all(a in ALLOWED_ANALYSES for a in analyses):
        raise ValueError(
            "Parameter 'analyses' must be a tuple of one or more of the following values: "
            f"{', '.join(ALLOWED_ANALYSES)}"
        )
    return tuple(analyses)


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


def _validate_qc_inputs(qc):
    """Validate run-level QC time series."""
    qc_arrays = []
    for i_subj, subj_qc in enumerate(qc):
        qc_arr = np.asarray(subj_qc, dtype=float)
        if qc_arr.ndim != 1:
            raise ValueError(f"QC values for run {i_subj} must be a 1D array.")
        if qc_arr.size == 0:
            raise ValueError(f"QC values for run {i_subj} cannot be empty.")
        if not np.all(np.isfinite(qc_arr)):
            raise ValueError(f"QC values for run {i_subj} must contain only finite values.")
        qc_arrays.append(qc_arr)

    return qc_arrays


def _prepare_run_denoising_metrics(run_denoising_metrics, n_subjects):
    """Validate optional run-level denoising and data-loss metrics."""
    if run_denoising_metrics is None:
        return None
    if not isinstance(run_denoising_metrics, pd.DataFrame):
        raise TypeError("run_denoising_metrics must be a pandas DataFrame.")
    if run_denoising_metrics.shape[0] != n_subjects:
        raise ValueError(
            "run_denoising_metrics has "
            f"{run_denoising_metrics.shape[0]} rows, but {n_subjects} runs were provided."
        )
    if run_denoising_metrics.isna().any().any():
        raise ValueError("run_denoising_metrics cannot contain missing values.")
    non_numeric = [
        column
        for column in run_denoising_metrics.columns
        if not pd.api.types.is_numeric_dtype(run_denoising_metrics[column])
    ]
    if non_numeric:
        raise TypeError(
            "run_denoising_metrics columns must be numeric. "
            f"Non-numeric columns: {', '.join(non_numeric)}."
        )

    return run_denoising_metrics.reset_index(drop=True)


def _count_confounds(confounds, n_subjects):
    """Count run-level confound regressors."""
    if confounds is None:
        return np.zeros(n_subjects, dtype=int)
    if len(confounds) != n_subjects:
        raise ValueError(
            f"confounds has {len(confounds)} runs, but {n_subjects} files were provided."
        )

    counts = np.zeros(n_subjects, dtype=int)
    for i_subj, subj_confounds in enumerate(confounds):
        confounds_arr = np.asarray(subj_confounds)
        if confounds_arr.ndim == 1:
            counts[i_subj] = 1
        elif confounds_arr.ndim == 2:
            counts[i_subj] = confounds_arr.shape[1]
        else:
            raise ValueError(f"Confounds for run {i_subj} must be a 1D or 2D array.")

    return counts


def _build_run_denoising_summary(files, qc, confounds, qc_thresh, run_denoising_metrics):
    """Build run-level tDOF and data-loss accounting table."""
    n_subjects = len(files)
    qc_arrays = [np.asarray(subj_qc) for subj_qc in qc]
    n_volumes = np.array([subj_qc.size for subj_qc in qc_arrays], dtype=int)
    n_qc_missing = np.array([np.sum(~np.isfinite(subj_qc)) for subj_qc in qc_arrays], dtype=int)
    n_volumes_at_or_below_thresh = np.array(
        [np.sum(subj_qc <= qc_thresh) for subj_qc in qc_arrays], dtype=int
    )
    n_volumes_above_thresh = np.array(
        [np.sum(subj_qc > qc_thresh) for subj_qc in qc_arrays], dtype=int
    )
    n_confounds = _count_confounds(confounds, n_subjects)

    summary = pd.DataFrame(
        {
            "input_index": np.arange(n_subjects),
            "filename": [op.basename(file_) for file_ in files],
            "n_volumes": n_volumes,
            "mean_qc": [np.mean(subj_qc) for subj_qc in qc_arrays],
            "qc_thresh": qc_thresh,
            "n_qc_missing": n_qc_missing,
            "n_volumes_at_or_below_qc_thresh": n_volumes_at_or_below_thresh,
            "n_volumes_above_qc_thresh": n_volumes_above_thresh,
            "proportion_volumes_at_or_below_qc_thresh": n_volumes_at_or_below_thresh / n_volumes,
            "proportion_volumes_above_qc_thresh": n_volumes_above_thresh / n_volumes,
            "n_confounds": n_confounds,
            "nominal_t_dof_after_confounds": n_volumes - n_confounds,
            "retained_after_loading": False,
            "retained_for_analysis": False,
            "drop_reason": "",
        }
    )

    if run_denoising_metrics is not None:
        overlapping_columns = set(summary.columns).intersection(run_denoising_metrics.columns)
        if overlapping_columns:
            raise ValueError(
                "run_denoising_metrics columns overlap with built-in summary columns: "
                f"{', '.join(sorted(overlapping_columns))}."
            )
        summary = pd.concat([summary, run_denoising_metrics], axis=1)

    return summary


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


def _safe_pipeline_name(pipeline):
    """Convert a pipeline label to a safe directory name."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(pipeline).strip())
    safe_name = safe_name.strip("._-")
    if not safe_name:
        raise ValueError(f"Pipeline name {pipeline!r} does not produce a valid directory name.")
    return safe_name


def _is_nifti_path(path):
    """Return True when a path names a NIfTI image."""
    path = str(path).lower()
    cifti_suffixes = (
        ".dconn.nii",
        ".dlabel.nii",
        ".dscalar.nii",
        ".dtseries.nii",
        ".pconn.nii",
        ".pdconn.nii",
        ".plabel.nii",
        ".pscalar.nii",
        ".ptseries.nii",
        ".sdseries.nii",
    )
    return not path.endswith(cifti_suffixes) and path.endswith((".nii", ".nii.gz"))


def _load_pipeline_file_table(pipeline_file_table):
    """Load a run-by-pipeline file table from a DataFrame or TSV path."""
    if isinstance(pipeline_file_table, pd.DataFrame):
        table = pipeline_file_table.copy()
        base_dir = op.abspath(".")
    else:
        table_path = op.abspath(pipeline_file_table)
        table = pd.read_table(table_path)
        base_dir = op.dirname(table_path)

    if table.empty:
        raise ValueError("pipeline_file_table must contain at least one run and one pipeline.")

    return table.reset_index(drop=True), base_dir


def _prepare_pipeline_file_table(pipeline_file_table, pipeline_columns=None):
    """Validate and resolve paths in a run-by-pipeline file table."""
    table, base_dir = _load_pipeline_file_table(pipeline_file_table)

    if pipeline_columns is None:
        pipeline_columns = list(table.columns)
    else:
        pipeline_columns = list(pipeline_columns)

    if not pipeline_columns:
        raise ValueError("At least one pipeline column must be selected.")

    missing_columns = [column for column in pipeline_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Pipeline columns not found: {', '.join(missing_columns)}.")

    safe_names = [_safe_pipeline_name(column) for column in pipeline_columns]
    if len(set(safe_names)) != len(safe_names):
        raise ValueError("Pipeline column names must produce unique output directory names.")

    file_table = pd.DataFrame(index=table.index)
    for column in pipeline_columns:
        if table[column].isna().any():
            raise ValueError(f"Pipeline column '{column}' contains missing file paths.")

        paths = []
        for value in table[column].astype(str):
            path = value if op.isabs(value) else op.join(base_dir, value)
            path = op.abspath(path)
            if not op.isfile(path):
                raise FileNotFoundError(f"File not found for pipeline '{column}': {path}")
            if not _is_nifti_path(path):
                raise ValueError(
                    f"Pipeline column '{column}' contains a non-NIfTI file path: {path}"
                )
            paths.append(path)
        file_table[column] = paths

    return file_table, dict(zip(pipeline_columns, safe_names))


def _extract_pipeline_data(files, qc, analyses, atlas, sphere_radius, window, confounds=None):
    """Extract analysis-ready run-level data for direct pipeline comparisons."""
    n_subjects = len(files)
    if confounds is not None and len(confounds) != n_subjects:
        raise ValueError(
            f"confounds has {len(confounds)} rows, but {n_subjects} runs were provided."
        )

    atlas_masker, coords = _build_atlas_masker(atlas=atlas, sphere_radius=sphere_radius)
    n_rois = coords.shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    distances = squareform(pdist(coords))[triu_idx]
    distances = np.round(distances, decimals=3)
    edge_sorting_idx = distances.argsort()
    distances = distances[edge_sorting_idx]
    n_edges = distances.size

    needs_z_corrs = ("qcrsfc" in analyses) or ("highlow" in analyses)
    z_corr_mats = np.full((n_subjects, n_edges), np.nan) if needs_z_corrs else None
    ts_all = [None] * n_subjects if "scrubbing" in analyses else None
    valid = np.zeros(n_subjects, dtype=bool)

    for i_subj, file_ in enumerate(files):
        if confounds is None:
            raw_ts = atlas_masker.fit_transform(file_).T
        else:
            raw_ts = atlas_masker.fit_transform(file_, confounds=confounds[i_subj]).T

        if raw_ts.shape[0] != n_rois:
            raise ValueError(
                f"{file_} produced {raw_ts.shape[0]} ROIs, but {n_rois} are expected."
            )
        if np.any(np.isnan(raw_ts)):
            continue
        if np.any(np.isclose(np.var(raw_ts, axis=1), 0)):
            continue

        if needs_z_corrs:
            raw_corrs = np.corrcoef(raw_ts)[triu_idx]
            raw_corrs = raw_corrs[edge_sorting_idx]
            z_corr_mats[i_subj, :] = utils.r2z(raw_corrs)
        if ts_all is not None:
            ts_all[i_subj] = raw_ts
        valid[i_subj] = True

    ma_distances = utils.moving_average(distances, window)
    _, smoothing_curve_distances = utils.average_across_distances(ma_distances, distances)

    return {
        "valid": valid,
        "z_corr_mats": z_corr_mats,
        "ts_all": ts_all,
        "distances": distances,
        "edge_sorting_idx": edge_sorting_idx,
        "smoothing_curve_distances": smoothing_curve_distances,
    }


def _subset_pipeline_data(pipeline_data, idx):
    """Subset extracted pipeline data to paired runs."""
    return {
        "z_corr_mats": (
            None if pipeline_data["z_corr_mats"] is None else pipeline_data["z_corr_mats"][idx, :]
        ),
        "ts_all": (
            None
            if pipeline_data["ts_all"] is None
            else [pipeline_data["ts_all"][i] for i in np.where(idx)[0]]
        ),
    }


def _compute_pipeline_analysis_values(
    analysis_name,
    mean_qc,
    qc,
    pipeline_data,
    edge_sorting_idx,
    qc_thresh,
    run_covariates=None,
):
    """Compute one DDMRA analysis for one pipeline on an already-paired run set."""
    if analysis_name == "qcrsfc":
        return analysis.qcrsfc_analysis(
            mean_qc,
            pipeline_data["z_corr_mats"],
            run_covariates=run_covariates,
        )
    if analysis_name == "highlow":
        return analysis.highlow_analysis(mean_qc, pipeline_data["z_corr_mats"])
    if analysis_name == "scrubbing":
        return analysis.scrubbing_analysis(
            qc,
            pipeline_data["ts_all"],
            edge_sorting_idx,
            qc_thresh,
            perm=False,
        )
    raise ValueError(f"Unknown analysis '{analysis_name}'.")


def _compute_pipeline_analysis_curves(
    analyses,
    mean_qc,
    qc,
    pipeline_data,
    edge_sorting_idx,
    qc_thresh,
    window,
    distances,
    smoothing_curve_distances,
    run_covariates=None,
):
    """Compute smoothing curves for all requested analyses for one pipeline."""
    curves = {}
    for analysis_name in analyses:
        values = _compute_pipeline_analysis_values(
            analysis_name,
            mean_qc,
            qc,
            pipeline_data,
            edge_sorting_idx,
            qc_thresh,
            run_covariates=run_covariates,
        )
        curves[analysis_name] = utils.calculate_smoothing_curve(
            values,
            window,
            distances,
            smoothing_curve_distances,
        )
    return curves


def _curve_intercept_and_slope(curve, distances):
    """Return DDMRA intercept and local-to-long distance slope for a smoothing curve."""
    intercept = utils.get_val(distances, curve, COMPARISON_INTERCEPT_DISTANCE)
    slope = intercept - utils.get_val(distances, curve, COMPARISON_SLOPE_DISTANCE)
    return intercept, slope


def _swap_paired_pipeline_data(first_data, second_data, swap_idx):
    """Swap paired run-level pipeline labels according to a boolean index."""
    swapped_first = {}
    swapped_second = {}

    if first_data["z_corr_mats"] is None:
        swapped_first["z_corr_mats"] = None
        swapped_second["z_corr_mats"] = None
    else:
        first_z = first_data["z_corr_mats"].copy()
        second_z = second_data["z_corr_mats"].copy()
        first_z[swap_idx, :], second_z[swap_idx, :] = (
            second_z[swap_idx, :].copy(),
            first_z[swap_idx, :].copy(),
        )
        swapped_first["z_corr_mats"] = first_z
        swapped_second["z_corr_mats"] = second_z

    if first_data["ts_all"] is None:
        swapped_first["ts_all"] = None
        swapped_second["ts_all"] = None
    else:
        first_ts = list(first_data["ts_all"])
        second_ts = list(second_data["ts_all"])
        for i_run, do_swap in enumerate(swap_idx):
            if do_swap:
                first_ts[i_run], second_ts[i_run] = second_ts[i_run], first_ts[i_run]
        swapped_first["ts_all"] = first_ts
        swapped_second["ts_all"] = second_ts

    return swapped_first, swapped_second


def _pipeline_pairwise_null_iter(
    seed,
    analyses,
    mean_qc,
    qc,
    first_data,
    second_data,
    edge_sorting_idx,
    qc_thresh,
    window,
    distances,
    smoothing_curve_distances,
    run_covariates=None,
):
    """One paired within-run pipeline-label swap permutation."""
    rng = np.random.RandomState(seed=seed)
    swap_idx = rng.random(mean_qc.shape[0]) < 0.5
    swapped_first, swapped_second = _swap_paired_pipeline_data(first_data, second_data, swap_idx)

    first_curves = _compute_pipeline_analysis_curves(
        analyses,
        mean_qc,
        qc,
        swapped_first,
        edge_sorting_idx,
        qc_thresh,
        window,
        distances,
        smoothing_curve_distances,
        run_covariates=run_covariates,
    )
    second_curves = _compute_pipeline_analysis_curves(
        analyses,
        mean_qc,
        qc,
        swapped_second,
        edge_sorting_idx,
        qc_thresh,
        window,
        distances,
        smoothing_curve_distances,
        run_covariates=run_covariates,
    )

    null_values = np.zeros((len(analyses), 2))
    for i_analysis, analysis_name in enumerate(analyses):
        first_intercept, first_slope = _curve_intercept_and_slope(
            first_curves[analysis_name],
            smoothing_curve_distances,
        )
        second_intercept, second_slope = _curve_intercept_and_slope(
            second_curves[analysis_name],
            smoothing_curve_distances,
        )
        null_values[i_analysis, 0] = first_intercept - second_intercept
        null_values[i_analysis, 1] = first_slope - second_slope

    return null_values


def _read_pipeline_retention(pipeline_outputs, pipeline, n_runs):
    """Read the retained run mask from a pipeline workflow output directory."""
    summary_path = op.join(pipeline_outputs[pipeline], "run_denoising_summary.tsv")
    summary = pd.read_table(summary_path)
    if summary.shape[0] != n_runs:
        raise ValueError(
            f"{summary_path} has {summary.shape[0]} rows, but {n_runs} runs were provided."
        )
    return summary["retained_for_analysis"].astype(bool).to_numpy()


def _run_pipeline_pairwise_comparisons(
    file_table,
    qc,
    out_dir,
    pipeline_outputs,
    safe_names,
    analyses,
    confounds,
    n_iters,
    n_jobs,
    qc_thresh,
    window,
    atlas,
    sphere_radius,
    run_covariates,
):
    """Perform direct paired statistical comparisons between processing pipelines."""
    n_runs = file_table.shape[0]
    if n_iters < 1:
        raise ValueError("comparison_n_iters must be at least 1.")

    qc = _validate_qc_inputs(qc)
    mean_qc = np.array([np.mean(subj_qc) for subj_qc in qc])
    if "qcrsfc" in analyses:
        run_covariates = _prepare_run_covariates(run_covariates, n_runs)
    else:
        run_covariates = None

    pipeline_data = {}
    retained = {}
    for pipeline in file_table.columns:
        pipeline_data[pipeline] = _extract_pipeline_data(
            file_table[pipeline].tolist(),
            qc,
            analyses,
            atlas,
            sphere_radius,
            window,
            confounds=confounds,
        )
        retained[pipeline] = _read_pipeline_retention(pipeline_outputs, pipeline, n_runs)

    distances = next(iter(pipeline_data.values()))["distances"]
    smoothing_curve_distances = next(iter(pipeline_data.values()))["smoothing_curve_distances"]
    edge_sorting_idx = next(iter(pipeline_data.values()))["edge_sorting_idx"]
    curve_rows = []
    comparison_rows = []
    null_arrays = {}

    for first_pipeline, second_pipeline in combinations(file_table.columns, 2):
        common_idx = (
            retained[first_pipeline]
            & retained[second_pipeline]
            & pipeline_data[first_pipeline]["valid"]
            & pipeline_data[second_pipeline]["valid"]
        )
        n_common = int(np.sum(common_idx))
        if n_common < 10:
            raise ValueError(
                f"Too few paired runs remaining for {first_pipeline} vs {second_pipeline}: "
                f"{n_common}."
            )

        pair_mean_qc = mean_qc[common_idx]
        pair_qc = [qc[i] for i in np.where(common_idx)[0]]
        pair_run_covariates = None if run_covariates is None else run_covariates[common_idx, :]
        first_data = _subset_pipeline_data(pipeline_data[first_pipeline], common_idx)
        second_data = _subset_pipeline_data(pipeline_data[second_pipeline], common_idx)

        first_curves = _compute_pipeline_analysis_curves(
            analyses,
            pair_mean_qc,
            pair_qc,
            first_data,
            edge_sorting_idx,
            qc_thresh,
            window,
            distances,
            smoothing_curve_distances,
            run_covariates=pair_run_covariates,
        )
        second_curves = _compute_pipeline_analysis_curves(
            analyses,
            pair_mean_qc,
            pair_qc,
            second_data,
            edge_sorting_idx,
            qc_thresh,
            window,
            distances,
            smoothing_curve_distances,
            run_covariates=pair_run_covariates,
        )

        with utils.tqdm_joblib(
            tqdm(
                desc=f"{first_pipeline} vs {second_pipeline} paired swaps",
                total=n_iters,
            )
        ):
            null_values = Parallel(n_jobs=n_jobs)(
                delayed(_pipeline_pairwise_null_iter)(
                    seed,
                    analyses,
                    pair_mean_qc,
                    pair_qc,
                    first_data,
                    second_data,
                    edge_sorting_idx,
                    qc_thresh,
                    window,
                    distances,
                    smoothing_curve_distances,
                    run_covariates=pair_run_covariates,
                )
                for seed in range(n_iters)
            )
        null_values = np.stack(null_values, axis=0)

        pair_key = f"{safe_names[first_pipeline]}__vs__{safe_names[second_pipeline]}"
        for i_analysis, analysis_name in enumerate(analyses):
            first_curve = first_curves[analysis_name]
            second_curve = second_curves[analysis_name]
            for distance, first_value, second_value in zip(
                smoothing_curve_distances,
                first_curve,
                second_curve,
            ):
                curve_rows.append(
                    {
                        "pipeline_1": first_pipeline,
                        "pipeline_2": second_pipeline,
                        "analysis": analysis_name,
                        "distance": distance,
                        "pipeline_1_value": first_value,
                        "pipeline_2_value": second_value,
                        "difference": first_value - second_value,
                    }
                )

            first_intercept, first_slope = _curve_intercept_and_slope(
                first_curve,
                smoothing_curve_distances,
            )
            second_intercept, second_slope = _curve_intercept_and_slope(
                second_curve,
                smoothing_curve_distances,
            )
            observed = {
                "intercept_35mm": (
                    first_intercept,
                    second_intercept,
                    first_intercept - second_intercept,
                    null_values[:, i_analysis, 0],
                ),
                "slope_35_to_100mm": (
                    first_slope,
                    second_slope,
                    first_slope - second_slope,
                    null_values[:, i_analysis, 1],
                ),
            }

            for contrast, (first_value, second_value, diff_value, null_array) in observed.items():
                p_value = utils.null_to_p(
                    diff_value,
                    null_array,
                    tail="two",
                    symmetric=True,
                )
                comparison_rows.append(
                    {
                        "pipeline_1": first_pipeline,
                        "pipeline_2": second_pipeline,
                        "analysis": analysis_name,
                        "contrast": contrast,
                        "pipeline_1_value": first_value,
                        "pipeline_2_value": second_value,
                        "difference": diff_value,
                        "p_value": p_value,
                        "tail": "two-sided",
                        "null_method": "paired run-wise pipeline-label swaps",
                        "n_permutations": n_iters,
                        "n_paired_runs": n_common,
                    }
                )
                null_arrays[f"{pair_key}__{analysis_name}__{contrast}"] = null_array

    pd.DataFrame(comparison_rows).to_csv(
        op.join(out_dir, "pipeline_pairwise_comparisons.tsv"),
        sep="\t",
        lineterminator="\n",
        index=False,
    )
    pd.DataFrame(curve_rows).to_csv(
        op.join(out_dir, "pipeline_pairwise_smoothing_curves.tsv.gz"),
        sep="\t",
        lineterminator="\n",
        index=False,
    )
    np.savez_compressed(op.join(out_dir, "pipeline_pairwise_nulls.npz"), **null_arrays)


def run_pipeline_comparison(
    pipeline_file_table,
    qc,
    out_dir=".",
    pipeline_columns=None,
    compare_pipelines=True,
    comparison_n_iters=None,
    comparison_n_jobs=None,
    **run_analyses_kwargs,
):
    """Run DDMRA analyses for multiple processing pipelines.

    Parameters
    ----------
    pipeline_file_table : path-like or pandas.DataFrame
        TSV file or DataFrame with one row per run and one column per processing pipeline.
        Each selected cell must contain a path to a NIfTI image file for that run and pipeline.
        Relative paths in TSV files are resolved relative to the TSV's directory.
    qc : (N,) list of array_like
        List of 1D QC metric arrays, one per run. The same QC values are used for every
        pipeline so the processing pipelines are compared over the same motion/quality axis.
    out_dir : str, optional
        Output directory. Each pipeline is written to a subdirectory of this directory.
        Default is current directory.
    pipeline_columns : None or list of str, optional
        Pipeline columns to run. If None, all columns in ``pipeline_file_table`` are used.
    compare_pipelines : bool, optional
        If True and at least two pipelines are selected, perform direct paired
        between-pipeline comparisons using run-wise pipeline-label swaps.
        Default is True.
    comparison_n_iters : None or int, optional
        Number of paired label-swap permutations for direct pipeline comparisons. If None,
        use the same ``n_iters`` value passed to :func:`run_analyses`.
    comparison_n_jobs : None or int, optional
        Number of jobs for direct pipeline comparisons. If None, use the same ``n_jobs`` value
        passed to :func:`run_analyses`.
    **run_analyses_kwargs
        Additional keyword arguments passed through to :func:`run_analyses`.

    Returns
    -------
    pipeline_outputs : dict
        Mapping from input pipeline column names to output directories.

    Notes
    -----
    This workflow writes each pipeline's DDMRA outputs to a pipeline-specific subdirectory.
    When ``compare_pipelines`` is True, it also writes direct pairwise comparisons of each
    requested analysis' smoothing-curve intercept and slope. These comparisons use paired
    run-wise pipeline-label swaps, which preserve the run-level QC values and matched
    processing-pipeline structure. Pairwise comparisons are conditional on the intersection
    of runs retained for both pipelines being compared.

    Direct comparison outputs are written to:
    - ``pipeline_pairwise_comparisons.tsv``: intercept/slope differences and p-values.
    - ``pipeline_pairwise_smoothing_curves.tsv.gz``: observed paired difference curves.
    - ``pipeline_pairwise_nulls.npz``: paired label-swap null distributions.
    """
    file_table, safe_names = _prepare_pipeline_file_table(pipeline_file_table, pipeline_columns)
    if len(qc) != file_table.shape[0]:
        raise ValueError(
            f"qc has {len(qc)} runs, but the pipeline file table has {file_table.shape[0]}."
        )
    analyses = _validate_analyses(
        run_analyses_kwargs.get("analyses", ("qcrsfc", "highlow", "scrubbing"))
    )

    makedirs(out_dir, exist_ok=True)
    pipeline_outputs = {}
    summary_rows = []

    for pipeline, pipeline_safe_name in safe_names.items():
        pipeline_out_dir = op.join(out_dir, pipeline_safe_name)
        run_analyses(
            file_table[pipeline].tolist(),
            qc,
            out_dir=pipeline_out_dir,
            **run_analyses_kwargs,
        )
        pipeline_outputs[pipeline] = pipeline_out_dir
        summary_rows.append(
            {
                "pipeline": pipeline,
                "output_dir": pipeline_out_dir,
                "n_runs": file_table.shape[0],
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        op.join(out_dir, "pipeline_comparison_summary.tsv"),
        sep="\t",
        lineterminator="\n",
        index=False,
    )

    if compare_pipelines and len(file_table.columns) > 1:
        if comparison_n_iters is None:
            comparison_n_iters = run_analyses_kwargs.get("n_iters", 10000)
        if comparison_n_jobs is None:
            comparison_n_jobs = run_analyses_kwargs.get("n_jobs", 1)
        _run_pipeline_pairwise_comparisons(
            file_table,
            qc,
            out_dir,
            pipeline_outputs,
            safe_names,
            analyses,
            run_analyses_kwargs.get("confounds", None),
            comparison_n_iters,
            comparison_n_jobs,
            run_analyses_kwargs.get("qc_thresh", 0.2),
            run_analyses_kwargs.get("window", 1000),
            run_analyses_kwargs.get("atlas", "power_2011"),
            run_analyses_kwargs.get("sphere_radius", 5.0),
            run_analyses_kwargs.get("run_covariates", None),
        )

    return pipeline_outputs


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
    run_denoising_metrics=None,
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
    run_denoising_metrics : None or pandas.DataFrame, optional
        Run-level denoising and data-loss metrics to include in
        ``run_denoising_summary.tsv``. Rows must correspond to ``files`` in order, and
        columns must be numeric. Useful columns include upstream retained-volume counts,
        censored-volume counts, or temporal degrees of freedom after denoising.
        Default is None.

    Notes
    -----
    At least ``MIN_SUBJECTS`` (10) runs must be retained for analysis, or a
    :class:`ValueError` is raised. When the QC:RSFC or high-low analyses are requested and
    fewer than ``QCFC_STABILITY_N`` (30) runs are retained, a :class:`UserWarning` is issued
    because QC-FC estimates are unstable in small samples (Parkes et al., 2018; Ciric et al.,
    2017); the analyses still run, but their intercept and slope summaries should be
    interpreted with caution.

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
    - ``ranks.tsv.gz``: Diagnostic edgewise ranks of the observed analysis values against
        the edgewise null distributions. These ranks are not inferential p-values.
    - ``qcrsfc_summary.tsv`` (only when the ``qcrsfc`` analysis is requested):
        Descriptive QC-FC benchmark summaries, including the median absolute QC-FC
        correlation and the percentage of edges with a significant QC-FC correlation
        (Ciric et al., 2017; Parkes et al., 2018). These are diagnostics, not the
        package's inferential result.
    - ``run_denoising_summary.tsv``: Run-level volume, confound-regressor, retention,
        and optional user-provided tDOF/data-loss accounting.
    - ``[analysis]_analysis.png``: Figure for each analysis.

    If ``verbose`` is ``True``:
    - ``z_corrs.tsv.gz``: Z-transformed correlation coefficients for the good files,
        used by QC:RSFC and high-low analyses.
    - ``mean_qcs.tsv.gz``: Mean QC values for the good files, used by QC:RSFC and high-low
        analyses.
    """
    analyses = _validate_analyses(analyses)

    if (pca_threshold is None) and (outlier_threshold is None):
        LGR.info("Not performing outlier detection.")
    elif (pca_threshold is None) or (outlier_threshold is None):
        raise ValueError("Both pca_threshold and outlier_threshold must be None or not None.")
    elif isinstance(pca_threshold, int) and isinstance(outlier_threshold, float):
        LGR.info(f"Performing outlier detection on first {pca_threshold} PCA components.")
    elif isinstance(pca_threshold, float) and isinstance(outlier_threshold, float):
        if not 0 < pca_threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")
        LGR.info(
            "Performing outlier detection on PCA components explaining "
            f"{pca_threshold * 100}% of the variance."
        )
    else:
        raise ValueError("Bad inputs.")

    makedirs(out_dir, exist_ok=True)

    # create LGR with 'spam_application'
    LGR.setLevel(logging.DEBUG)
    _reset_workflow_log_handler()
    # create file handler which logs even debug messages
    fh = logging.FileHandler(op.join(out_dir, "log.tsv"))
    fh.setLevel(logging.DEBUG)
    fh._ddmra_workflow_handler = True
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s")
    fh.setFormatter(formatter)
    # add the handlers to the LGR
    LGR.addHandler(fh)

    LGR.info("Preallocating matrices")
    n_subjects = len(files)
    if len(qc) != n_subjects:
        raise ValueError(f"qc has {len(qc)} runs, but {n_subjects} files were provided.")
    qc = _validate_qc_inputs(qc)
    if run_covariates is not None and "qcrsfc" not in analyses:
        LGR.info("Ignoring run_covariates because QC:RSFC analysis was not requested.")
        run_covariates = None
    else:
        run_covariates = _prepare_run_covariates(run_covariates, n_subjects)
    run_denoising_metrics = _prepare_run_denoising_metrics(run_denoising_metrics, n_subjects)
    run_denoising_summary = _build_run_denoising_summary(
        files,
        qc,
        confounds,
        qc_thresh,
        run_denoising_metrics,
    )
    drop_reasons = [[] for _ in range(n_subjects)]

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

        if raw_ts.shape[0] != n_rois:
            raise ValueError(
                f"{files[i_subj]} produced {raw_ts.shape[0]} ROIs, but {n_rois} are expected."
            )

        if np.any(np.isnan(raw_ts)):
            LGR.warning(f"Time series of {files[i_subj]} contains NaNs. Dropping from analysis.")
            drop_reasons[i_subj].append("timeseries_nan")
            skip_subject = True

        roi_variances = np.var(raw_ts, axis=1)
        if any(np.isclose(roi_variances, 0)):
            bad_rois = np.where(np.isclose(roi_variances, 0))[0]
            LGR.warning(
                f"ROI(s) {bad_rois} for {files[i_subj]} have variance of 0. "
                "Dropping from analysis."
            )
            drop_reasons[i_subj].append("zero_variance_roi")
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
    run_denoising_summary.loc[good_subjects, "retained_after_loading"] = True
    analysis_subjects = good_subjects.copy()

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
        outlier_subjects = analysis_subjects[outlier_idx]
        LGR.info(f"Removing {outlier_idx.size} outliers of {mahalanobis_distances.size} runs.")
        n_subjects_remaining = keep_idx.size
        for i_subj in outlier_subjects:
            drop_reasons[i_subj].append("pca_mcd_outlier")
        analysis_subjects = analysis_subjects[keep_idx]

        if ("qcrsfc" in analyses) or ("highlow" in analyses):
            z_corr_mats = z_corr_mats[keep_idx, :]
            mean_qc = mean_qc[keep_idx]
            if run_covariates is not None:
                run_covariates = run_covariates[keep_idx, :]
            file_names = [file_names[i] for i in keep_idx]

        if "scrubbing" in analyses:
            ts_all = [ts_all[i] for i in keep_idx]
            qc = [qc[i] for i in keep_idx]

    run_denoising_summary.loc[analysis_subjects, "retained_for_analysis"] = True
    run_denoising_summary["drop_reason"] = [";".join(reasons) for reasons in drop_reasons]
    run_denoising_summary.to_csv(
        op.join(out_dir, "run_denoising_summary.tsv"),
        sep="\t",
        lineterminator="\n",
        index=False,
    )

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
    if n_subjects_remaining < MIN_SUBJECTS:
        raise ValueError("Too few subjects remaining for analysis.")

    # QC-FC and high-low are cross-run associations whose estimates are unstable in
    # small samples. Warn (but proceed) so users can interpret marginal-N results with care.
    cross_run_analyses = ("qcrsfc" in analyses) or ("highlow" in analyses)
    if cross_run_analyses and n_subjects_remaining < QCFC_STABILITY_N:
        stability_msg = (
            f"Only {n_subjects_remaining} runs were retained for analysis. QC-FC and "
            f"high-low estimates are unstable below ~{QCFC_STABILITY_N} runs (Parkes et al., "
            "2018; Ciric et al., 2017); interpret intercept and slope summaries with caution "
            "and prefer larger samples."
        )
        LGR.warning(stability_msg)
        warnings.warn(stability_msg, UserWarning, stacklevel=2)

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

        # Descriptive QC-FC benchmark summaries (Ciric et al., 2017; Parkes et al., 2018).
        qcrsfc_summary_metrics = analysis.qcrsfc_summary(
            mean_qc, z_corr_mats, run_covariates=run_covariates
        )
        pd.DataFrame([qcrsfc_summary_metrics]).to_csv(
            op.join(out_dir, "qcrsfc_summary.tsv"),
            sep="\t",
            lineterminator="\n",
            index=False,
        )
        LGR.info(
            f"QC:RSFC summary: median |QC-FC| r = "
            f"{qcrsfc_summary_metrics['median_abs_qcfc']:.04f}; "
            f"{qcrsfc_summary_metrics['percent_significant_edges']:.02f}% of edges "
            f"significant at p < {qcrsfc_summary_metrics['alpha']} (uncorrected)."
        )

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
            f"For {analysis_name} analysis, intercept (p = {p_inter}); slope (p = {p_slope})."
        )

    np.savez_compressed(op.join(out_dir, "null_smoothing_curves.npz"), **null_curves_dict)

    del null_curves_dict

    plotting.plot_results(out_dir)

    LGR.info("Workflow completed")
