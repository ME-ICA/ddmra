"""Tests for the ddmra.workflows module.

The input-validation tests are fast and have no external dependencies. The
end-to-end ``test_run_analyses_*`` test exercises the full pipeline against a
small synthetic atlas (patched in to avoid any network access) and synthetic
4D images, and is marked as an integration test.
"""

import os.path as op
import re
import types

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.maskers import NiftiLabelsMasker, NiftiSpheresMasker

from ddmra import workflows

# ---------------------------------------------------------------------------- #
#                            Input validation                                  #
# ---------------------------------------------------------------------------- #


def test_run_analyses_requires_at_least_one_analysis(tmp_path):
    """An empty ``analyses`` tuple is rejected before any heavy work."""
    with pytest.raises(AssertionError, match="At least one analysis"):
        workflows.run_analyses(["a"], [np.zeros(5)], out_dir=str(tmp_path), analyses=())


def test_run_analyses_rejects_unknown_analysis(tmp_path):
    """Unknown analysis names are rejected."""
    with pytest.raises(AssertionError, match="must be a tuple"):
        workflows.run_analyses(["a"], [np.zeros(5)], out_dir=str(tmp_path), analyses=("bogus",))


def test_run_analyses_requires_both_outlier_thresholds(tmp_path):
    """Specifying only one of pca/outlier thresholds is an error."""
    with pytest.raises(ValueError, match="Both pca_threshold and outlier_threshold"):
        workflows.run_analyses(
            ["a"], [np.zeros(5)], out_dir=str(tmp_path), pca_threshold=5, outlier_threshold=None
        )


def test_run_analyses_pca_threshold_out_of_range(tmp_path):
    """A float pca_threshold must be between 0 and 1."""
    with pytest.raises(AssertionError, match="between 0 and 1"):
        workflows.run_analyses(
            ["a"],
            [np.zeros(5)],
            out_dir=str(tmp_path),
            pca_threshold=1.5,
            outlier_threshold=0.05,
        )


def test_run_analyses_bad_threshold_types(tmp_path):
    """Unsupported threshold type combinations raise a 'Bad inputs' error."""
    with pytest.raises(ValueError, match="Bad inputs"):
        workflows.run_analyses(
            ["a"],
            [np.zeros(5)],
            out_dir=str(tmp_path),
            pca_threshold="not-a-number",
            outlier_threshold=0.05,
        )


def test_run_analyses_qc_length_mismatch(tmp_path):
    """The number of QC arrays must match the number of files."""
    with pytest.raises(AssertionError):
        workflows.run_analyses(["a", "b"], [np.zeros(5)], out_dir=str(tmp_path))


def test_run_analyses_run_covariates_length_mismatch(tmp_path):
    """Run-level covariates must have one row per input run."""
    run_covariates = pd.DataFrame({"age": [20, 30]})
    with pytest.raises(ValueError, match="2 rows"):
        workflows.run_analyses(
            ["a"],
            [np.zeros(5)],
            out_dir=str(tmp_path),
            analyses=("qcrsfc",),
            run_covariates=run_covariates,
        )


def test_run_analyses_run_denoising_metrics_length_mismatch(tmp_path):
    """Run-level denoising metrics must have one row per input run."""
    run_denoising_metrics = pd.DataFrame({"temporal_degrees_of_freedom": [30, 31]})
    with pytest.raises(ValueError, match="2 rows"):
        workflows.run_analyses(
            ["a"],
            [np.zeros(5)],
            out_dir=str(tmp_path),
            run_denoising_metrics=run_denoising_metrics,
        )


def test_select_n_pca_components_float_threshold():
    """Float thresholds retain the first component count that reaches the target."""
    n_components, perc_varex = workflows._select_n_pca_components(
        np.array([0.75, 0.95, 1.0]), 0.75
    )
    assert n_components == 1
    assert perc_varex == 75.0

    n_components, perc_varex = workflows._select_n_pca_components(
        np.array([0.75, 0.95, 1.0]), 0.90
    )
    assert n_components == 2
    assert perc_varex == 95.0


def test_select_n_pca_components_int_threshold():
    """Integer thresholds are already component counts, not zero-based indices."""
    n_components, perc_varex = workflows._select_n_pca_components(
        np.array([0.25, 0.60, 0.85, 1.0]), 3
    )
    assert n_components == 3
    assert perc_varex == 85.0


def test_select_n_pca_components_int_threshold_out_of_range():
    """Integer component counts must select at least one available component."""
    with pytest.raises(ValueError, match="between 1 and 3"):
        workflows._select_n_pca_components(np.array([0.25, 0.60, 1.0]), 0)
    with pytest.raises(ValueError, match="between 1 and 3"):
        workflows._select_n_pca_components(np.array([0.25, 0.60, 1.0]), 4)


# ---------------------------------------------------------------------------- #
#                            End-to-end pipeline                               #
# ---------------------------------------------------------------------------- #


# 12 ROI coordinates (in mm) laid out along a line with varied gaps. This gives
# ROI-pair distances spanning ~15 mm up to ~240 mm, so the smoothing curve
# straddles the 35 mm and 100 mm points that run_analyses evaluates. The minimum
# gap (15 mm) exceeds twice the 5 mm sphere radius, so no spheres overlap.
_X_POSITIONS = [0, 15, 30, 50, 75, 100, 120, 140, 160, 180, 210, 240]
ATLAS_COORDS = np.array([[x, 90.0, 90.0] for x in _X_POSITIONS], dtype=float)
IMG_SHAPE = (26, 12, 12, 40)
IMG_AFFINE = np.diag([10.0, 10.0, 10.0, 1.0])  # 10 mm voxels -> FOV spans the coords above
_WINDOW = 4  # small smoothing window so the curve retains its extreme distances


def _fake_power_atlas():
    """Build a tiny stand-in for nilearn's Power 2011 coordinate atlas."""
    rois = {"x": ATLAS_COORDS[:, 0], "y": ATLAS_COORDS[:, 1], "z": ATLAS_COORDS[:, 2]}
    return types.SimpleNamespace(rois=rois)


def _write_synthetic_label_atlas(tmp_path):
    """Write a tiny labels atlas aligned to the synthetic test images."""
    labels = np.zeros(IMG_SHAPE[:3], dtype=np.int16)
    inv_affine = np.linalg.inv(IMG_AFFINE)
    for label, coord in enumerate(ATLAS_COORDS, start=1):
        ijk = np.rint(nib.affines.apply_affine(inv_affine, coord)).astype(int)
        labels[tuple(ijk)] = label

    atlas_img = nib.Nifti1Image(labels, IMG_AFFINE)
    atlas_path = tmp_path / "synthetic_labels_atlas.nii.gz"
    atlas_img.to_filename(atlas_path)
    return atlas_path


def _write_synthetic_images(tmp_path, n_subjects=12, seed=1):
    """Write ``n_subjects`` random 4D NIfTI files and return their paths + QC arrays."""
    rng = np.random.RandomState(seed)
    files, qc = [], []
    for i in range(n_subjects):
        data = rng.normal(size=IMG_SHAPE).astype(np.float32)
        img = nib.Nifti1Image(data, IMG_AFFINE)
        path = op.join(tmp_path, f"sub-{i:02d}_bold.nii.gz")
        img.to_filename(path)
        files.append(path)
        qc.append(rng.uniform(0, 0.4, size=IMG_SHAPE[-1]))
    return files, qc


def test_build_atlas_masker_uses_nilearn_sphere_fetcher(monkeypatch):
    """A string atlas name is fetched as coordinates and wrapped in a spheres masker."""
    monkeypatch.setattr(
        workflows.datasets, "fetch_coords_power_2011", lambda *a, **k: _fake_power_atlas()
    )
    masker, coords = workflows._build_atlas_masker("power_2011", sphere_radius=4.0)

    assert isinstance(masker, NiftiSpheresMasker)
    assert masker.radius == 4.0
    assert np.array_equal(coords, ATLAS_COORDS)


def test_coords_from_sphere_atlas_accepts_xyz_attributes(monkeypatch):
    """Coordinate atlas fetchers may expose x/y/z directly instead of rois."""
    fetched = types.SimpleNamespace(x=[1, 2], y=[3, 4], z=[5, 6])
    monkeypatch.setattr(
        workflows.datasets,
        "fetch_coords_custom",
        lambda *a, **k: fetched,
        raising=False,
    )

    coords = workflows._coords_from_sphere_atlas("custom")

    assert np.array_equal(coords, np.array([[1, 3, 5], [2, 4, 6]], dtype=float))


def test_coords_from_sphere_atlas_rejects_unknown_or_malformed(monkeypatch):
    """Coordinate atlas lookup failures raise clear errors."""
    with pytest.raises(ValueError, match="Unknown sphere atlas"):
        workflows._coords_from_sphere_atlas("definitely_not_an_atlas")

    monkeypatch.setattr(
        workflows.datasets,
        "fetch_coords_bad",
        lambda *a, **k: types.SimpleNamespace(labels=["a", "b"]),
        raising=False,
    )
    with pytest.raises(ValueError, match="does not expose"):
        workflows._coords_from_sphere_atlas("bad")


def test_build_atlas_masker_uses_labels_file(tmp_path):
    """An existing atlas file is loaded as a labels masker."""
    atlas_path = _write_synthetic_label_atlas(tmp_path)
    masker, coords = workflows._build_atlas_masker(atlas_path)

    assert isinstance(masker, NiftiLabelsMasker)
    assert coords.shape == ATLAS_COORDS.shape


def test_build_atlas_masker_validates_atlas_specification(monkeypatch):
    """Atlas specifications and coordinates are validated before analysis."""
    with pytest.raises(TypeError, match="atlas must"):
        workflows._build_atlas_masker(atlas=object())

    monkeypatch.setattr(workflows, "_coords_from_sphere_atlas", lambda atlas: np.ones((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        workflows._build_atlas_masker("bad_shape")

    monkeypatch.setattr(workflows, "_coords_from_sphere_atlas", lambda atlas: np.ones((1, 3)))
    with pytest.raises(ValueError, match="at least two"):
        workflows._build_atlas_masker("too_small")

    monkeypatch.setattr(
        workflows, "_coords_from_sphere_atlas", lambda atlas: np.array([[0, 0, 0], [np.nan, 1, 1]])
    )
    with pytest.raises(ValueError, match="finite"):
        workflows._build_atlas_masker("nonfinite")


def test_prepare_run_covariates_encodes_numeric_and_categorical_columns():
    """Run covariates are validated and categorical columns are dummy-coded."""
    run_covariates = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "site": ["a", "a", "b", "c"],
        }
    )

    result = workflows._prepare_run_covariates(run_covariates, n_subjects=4)

    assert result.shape == (4, 3)
    assert result.dtype == float


def test_prepare_run_covariates_rejects_invalid_tables():
    """Run covariate validation catches unsupported or unusable inputs."""
    with pytest.raises(TypeError, match="DataFrame"):
        workflows._prepare_run_covariates({"age": [20]}, n_subjects=1)
    with pytest.raises(ValueError, match="at least one column"):
        workflows._prepare_run_covariates(pd.DataFrame(index=[0, 1]), n_subjects=2)
    with pytest.raises(ValueError, match="missing"):
        workflows._prepare_run_covariates(pd.DataFrame({"age": [20, np.nan]}), n_subjects=2)
    with pytest.raises(ValueError, match="usable"):
        workflows._prepare_run_covariates(pd.DataFrame({"site": ["a", "a"]}), n_subjects=2)
    with pytest.raises(ValueError, match="finite"):
        workflows._prepare_run_covariates(pd.DataFrame({"age": [20, np.inf]}), n_subjects=2)


def test_validate_qc_inputs_rejects_nonfinite_values():
    """Workflow QC validation raises before NaNs can propagate into analyses."""
    with pytest.raises(ValueError, match="finite"):
        workflows._validate_qc_inputs([np.array([0.1, np.nan, 0.2])])


def test_validate_qc_inputs_rejects_bad_shapes_and_empty_arrays():
    """Workflow QC validation requires non-empty 1D arrays."""
    with pytest.raises(ValueError, match="1D"):
        workflows._validate_qc_inputs([np.ones((2, 2))])
    with pytest.raises(ValueError, match="cannot be empty"):
        workflows._validate_qc_inputs([np.array([])])


def test_build_run_denoising_summary_includes_inferred_and_user_metrics():
    """Run denoising summaries include volume, confound, and user-provided metrics."""
    qc = [np.array([0.1, 0.3, 0.4]), np.array([0.0, 0.2, 0.5])]
    confounds = [np.zeros((3, 2)), np.zeros((3, 1))]
    run_denoising_metrics = pd.DataFrame(
        {
            "n_volumes_retained_after_denoising": [2, 3],
            "temporal_degrees_of_freedom": [1, 2],
        }
    )

    summary = workflows._build_run_denoising_summary(
        ["sub-01_bold.nii.gz", "sub-02_bold.nii.gz"],
        qc,
        confounds,
        qc_thresh=0.2,
        run_denoising_metrics=run_denoising_metrics,
    )

    assert summary["n_volumes"].tolist() == [3, 3]
    assert summary["n_volumes_at_or_below_qc_thresh"].tolist() == [1, 2]
    assert summary["n_confounds"].tolist() == [2, 1]
    assert summary["nominal_t_dof_after_confounds"].tolist() == [1, 2]
    assert summary["temporal_degrees_of_freedom"].tolist() == [1, 2]


def test_count_confounds_supports_1d_and_rejects_bad_inputs():
    """Confound counting validates shape and run count."""
    counts = workflows._count_confounds([np.zeros(3), np.zeros((3, 2))], n_subjects=2)

    assert counts.tolist() == [1, 2]
    with pytest.raises(ValueError, match="2 files"):
        workflows._count_confounds([np.zeros(3)], n_subjects=2)
    with pytest.raises(ValueError, match="1D or 2D"):
        workflows._count_confounds([np.zeros((2, 2, 2))], n_subjects=1)


def test_prepare_run_denoising_metrics_rejects_invalid_tables():
    """Optional denoising metrics must be complete numeric run-level tables."""
    with pytest.raises(TypeError, match="DataFrame"):
        workflows._prepare_run_denoising_metrics({"tdof": [1]}, n_subjects=1)
    with pytest.raises(ValueError, match="missing"):
        workflows._prepare_run_denoising_metrics(pd.DataFrame({"tdof": [1, np.nan]}), 2)
    with pytest.raises(TypeError, match="Non-numeric"):
        workflows._prepare_run_denoising_metrics(pd.DataFrame({"strategy": ["a"]}), 1)


def test_build_run_denoising_summary_rejects_overlapping_metric_columns():
    """User-supplied denoising metrics cannot replace built-in summary columns."""
    metrics = pd.DataFrame({"n_volumes": [10]})
    with pytest.raises(ValueError, match="overlap"):
        workflows._build_run_denoising_summary(
            ["sub-01.nii.gz"],
            [np.ones(10)],
            confounds=None,
            qc_thresh=0.2,
            run_denoising_metrics=metrics,
        )


def test_prepare_pipeline_file_table_resolves_relative_paths(tmp_path):
    """Pipeline TSV paths are resolved relative to the TSV file."""
    image_path = tmp_path / "sub-01_bold.nii.gz"
    image_path.write_text("placeholder")
    table_path = tmp_path / "pipelines.tsv"
    pd.DataFrame({"preprocessed": ["sub-01_bold.nii.gz"]}).to_csv(
        table_path, sep="\t", index=False
    )

    table, safe_names = workflows._prepare_pipeline_file_table(table_path)

    assert table["preprocessed"].tolist() == [str(image_path)]
    assert safe_names == {"preprocessed": "preprocessed"}


def test_prepare_pipeline_file_table_rejects_bad_column_selections(tmp_path):
    """Pipeline table validation catches empty, missing, duplicate, and null selections."""
    image_path = tmp_path / "sub-01_bold.nii.gz"
    image_path.write_text("placeholder")
    table = pd.DataFrame({"a b": [str(image_path)], "a_b": [str(image_path)]})

    with pytest.raises(ValueError, match="At least one"):
        workflows._prepare_pipeline_file_table(table, pipeline_columns=[])
    with pytest.raises(ValueError, match="not found"):
        workflows._prepare_pipeline_file_table(table, pipeline_columns=["missing"])
    with pytest.raises(ValueError, match="unique"):
        workflows._prepare_pipeline_file_table(table)
    with pytest.raises(ValueError, match="does not produce"):
        workflows._prepare_pipeline_file_table(
            pd.DataFrame({"   ": [str(image_path)]}),
        )
    with pytest.raises(ValueError, match="missing file paths"):
        workflows._prepare_pipeline_file_table(pd.DataFrame({"preprocessed": [np.nan]}))


def test_load_pipeline_file_table_rejects_empty_tables():
    """Pipeline table inputs must include at least one run and pipeline."""
    with pytest.raises(ValueError, match="at least one"):
        workflows._load_pipeline_file_table(pd.DataFrame())


def test_prepare_pipeline_file_table_rejects_missing_file(tmp_path):
    """Missing image paths in a pipeline table raise a clear error."""
    table_path = tmp_path / "pipelines.tsv"
    pd.DataFrame({"preprocessed": ["missing.nii.gz"]}).to_csv(table_path, sep="\t", index=False)

    with pytest.raises(FileNotFoundError, match="missing.nii.gz"):
        workflows._prepare_pipeline_file_table(table_path)


def test_prepare_pipeline_file_table_rejects_non_nifti_file(tmp_path):
    """Pipeline comparison currently supports only NIfTI files."""
    cifti_path = tmp_path / "sub-01_bold.dtseries.nii"
    cifti_path.write_text("placeholder")
    table_path = tmp_path / "pipelines.tsv"
    pd.DataFrame({"preprocessed": ["sub-01_bold.dtseries.nii"]}).to_csv(
        table_path, sep="\t", index=False
    )

    with pytest.raises(ValueError, match="non-NIfTI"):
        workflows._prepare_pipeline_file_table(table_path)


def test_prepare_pipeline_file_table_selects_columns(tmp_path):
    """Users can select a subset of pipeline columns for comparison."""
    first_path = tmp_path / "first.nii.gz"
    second_path = tmp_path / "second.nii.gz"
    first_path.write_text("placeholder")
    second_path.write_text("placeholder")
    table = pd.DataFrame(
        {
            "preprocessed": [str(first_path)],
            "tedana": [str(second_path)],
        }
    )

    selected, safe_names = workflows._prepare_pipeline_file_table(
        table, pipeline_columns=["tedana"]
    )

    assert list(selected.columns) == ["tedana"]
    assert selected["tedana"].tolist() == [str(second_path)]
    assert safe_names == {"tedana": "tedana"}


def test_swap_paired_pipeline_data_swaps_matrices_and_timeseries():
    """Paired label swaps exchange the selected run-level pipeline data."""
    first = {
        "z_corr_mats": np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        "ts_all": ["a", "b", "c"],
    }
    second = {
        "z_corr_mats": np.array([[10, 20], [30, 40], [50, 60]], dtype=float),
        "ts_all": ["x", "y", "z"],
    }

    swapped_first, swapped_second = workflows._swap_paired_pipeline_data(
        first, second, np.array([True, False, True])
    )

    assert np.array_equal(swapped_first["z_corr_mats"], [[10, 20], [3, 4], [50, 60]])
    assert np.array_equal(swapped_second["z_corr_mats"], [[1, 2], [30, 40], [5, 6]])
    assert swapped_first["ts_all"] == ["x", "b", "z"]
    assert swapped_second["ts_all"] == ["a", "y", "c"]


def test_swap_paired_pipeline_data_handles_missing_modalities():
    """Paired label swaps can operate when an analysis family is absent."""
    first = {"z_corr_mats": None, "ts_all": None}
    second = {"z_corr_mats": None, "ts_all": None}

    swapped_first, swapped_second = workflows._swap_paired_pipeline_data(
        first, second, np.array([True])
    )

    assert swapped_first == first
    assert swapped_second == second


def test_compute_pipeline_analysis_values_rejects_unknown_analysis():
    """Comparison helpers reject unknown analysis names."""
    with pytest.raises(ValueError, match="Unknown analysis"):
        workflows._compute_pipeline_analysis_values(
            "bogus",
            mean_qc=np.array([0, 1]),
            qc=[np.zeros(2), np.ones(2)],
            pipeline_data={"z_corr_mats": np.ones((2, 1)), "ts_all": []},
            edge_sorting_idx=np.array([0]),
            qc_thresh=0.2,
        )


def test_pipeline_pairwise_comparisons_rejects_invalid_permutation_count(tmp_path):
    """Pairwise comparison permutation count must be positive."""
    with pytest.raises(ValueError, match="at least 1"):
        workflows._run_pipeline_pairwise_comparisons(
            file_table=pd.DataFrame({"a": ["a.nii.gz"], "b": ["b.nii.gz"]}),
            qc=[np.ones(3)],
            out_dir=str(tmp_path),
            pipeline_outputs={"a": str(tmp_path), "b": str(tmp_path)},
            safe_names={"a": "a", "b": "b"},
            analyses=("qcrsfc",),
            confounds=None,
            n_iters=0,
            n_jobs=1,
            qc_thresh=0.2,
            window=_WINDOW,
            atlas="power_2011",
            sphere_radius=5.0,
            run_covariates=None,
        )


def test_run_pipeline_comparison_validates_qc_length(tmp_path):
    """Pipeline comparison requires one QC array per table row."""
    image_path = tmp_path / "sub-01_bold.nii.gz"
    image_path.write_text("placeholder")

    with pytest.raises(ValueError, match="pipeline file table"):
        workflows.run_pipeline_comparison(
            pd.DataFrame({"preprocessed": [str(image_path)]}),
            qc=[],
            out_dir=str(tmp_path / "out"),
        )


@pytest.mark.integration
def test_run_analyses_end_to_end(tmp_path, monkeypatch):
    """Run the full workflow and confirm all expected output files are written."""
    # Patch the atlas fetch so the pipeline uses a tiny, network-free atlas.
    monkeypatch.setattr(
        workflows.datasets, "fetch_coords_power_2011", lambda *a, **k: _fake_power_atlas()
    )

    out_dir = tmp_path / "out"
    files, qc = _write_synthetic_images(tmp_path)

    workflows.run_analyses(
        files,
        qc,
        out_dir=str(out_dir),
        n_iters=2,
        n_jobs=1,
        window=_WINDOW,
        analyses=("qcrsfc", "highlow", "scrubbing"),
    )

    expected = [
        "analysis_values.tsv.gz",
        "smoothing_curves.tsv.gz",
        "null_smoothing_curves.npz",
        "ranks.tsv.gz",
        "run_denoising_summary.tsv",
        "analysis_results.png",
        "log.tsv",
    ]
    for name in expected:
        assert (out_dir / name).is_file(), f"Missing expected output: {name}"

    assert not (out_dir / "rank_smoothing_curves.tsv.gz").exists()

    log_text = (out_dir / "log.tsv").read_text()
    assert "rank =" not in log_text
    logged_ps = [float(p) for p in re.findall(r"p = ([0-9.]+)", log_text)]
    assert logged_ps
    assert all(0 <= p <= 1 for p in logged_ps)


@pytest.mark.integration
def test_run_analyses_with_labels_atlas_file(tmp_path):
    """The full workflow supports a local labels image atlas."""
    out_dir = tmp_path / "out"
    files, qc = _write_synthetic_images(tmp_path)
    atlas_path = _write_synthetic_label_atlas(tmp_path)

    workflows.run_analyses(
        files,
        qc,
        out_dir=str(out_dir),
        n_iters=2,
        n_jobs=1,
        window=_WINDOW,
        analyses=("qcrsfc", "highlow"),
        atlas=atlas_path,
    )

    assert (out_dir / "analysis_values.tsv.gz").is_file()
    assert (out_dir / "analysis_results.png").is_file()


@pytest.mark.integration
def test_run_pipeline_comparison_end_to_end(tmp_path, monkeypatch):
    """A run-by-pipeline TSV launches one workflow per processing pipeline."""
    monkeypatch.setattr(
        workflows.datasets, "fetch_coords_power_2011", lambda *a, **k: _fake_power_atlas()
    )
    files, qc = _write_synthetic_images(tmp_path)
    table_path = tmp_path / "pipelines.tsv"
    pd.DataFrame(
        {
            "preprocessed": [op.relpath(file_, tmp_path) for file_ in files],
            "XCP-D": [op.relpath(file_, tmp_path) for file_ in files],
        }
    ).to_csv(table_path, sep="\t", index=False)
    out_dir = tmp_path / "comparison"

    outputs = workflows.run_pipeline_comparison(
        table_path,
        qc,
        out_dir=str(out_dir),
        n_iters=2,
        n_jobs=1,
        window=_WINDOW,
        analyses=("qcrsfc", "highlow", "scrubbing"),
    )

    assert set(outputs) == {"preprocessed", "XCP-D"}
    assert (out_dir / "preprocessed" / "analysis_values.tsv.gz").is_file()
    assert (out_dir / "XCP-D" / "analysis_values.tsv.gz").is_file()
    summary = pd.read_table(out_dir / "pipeline_comparison_summary.tsv")
    assert summary["pipeline"].tolist() == ["preprocessed", "XCP-D"]
    pairwise = pd.read_table(out_dir / "pipeline_pairwise_comparisons.tsv")
    assert pairwise["pipeline_1"].unique().tolist() == ["preprocessed"]
    assert pairwise["pipeline_2"].unique().tolist() == ["XCP-D"]
    assert set(pairwise["analysis"]) == {"qcrsfc", "highlow", "scrubbing"}
    assert set(pairwise["contrast"]) == {"intercept_35mm", "slope_35_to_100mm"}
    assert np.allclose(pairwise["difference"], 0)
    assert np.allclose(pairwise["p_value"], 1)
    assert (out_dir / "pipeline_pairwise_smoothing_curves.tsv.gz").is_file()
    nulls = np.load(out_dir / "pipeline_pairwise_nulls.npz")
    assert set(nulls.files) == {
        "preprocessed__vs__XCP-D__qcrsfc__intercept_35mm",
        "preprocessed__vs__XCP-D__qcrsfc__slope_35_to_100mm",
        "preprocessed__vs__XCP-D__highlow__intercept_35mm",
        "preprocessed__vs__XCP-D__highlow__slope_35_to_100mm",
        "preprocessed__vs__XCP-D__scrubbing__intercept_35mm",
        "preprocessed__vs__XCP-D__scrubbing__slope_35_to_100mm",
    }


@pytest.mark.integration
def test_run_analyses_verbose_writes_extra_files(tmp_path, monkeypatch):
    """verbose=True additionally writes the z-correlation and mean-QC tables."""
    monkeypatch.setattr(
        workflows.datasets, "fetch_coords_power_2011", lambda *a, **k: _fake_power_atlas()
    )
    out_dir = tmp_path / "out"
    files, qc = _write_synthetic_images(tmp_path)
    workflows.run_analyses(
        files,
        qc,
        out_dir=str(out_dir),
        n_iters=2,
        n_jobs=1,
        window=_WINDOW,
        analyses=("qcrsfc",),
        verbose=True,
    )
    assert (out_dir / "z_corrs.tsv.gz").is_file()
    assert (out_dir / "mean_qcs.tsv.gz").is_file()


@pytest.mark.integration
def test_run_analyses_with_confounds(tmp_path, monkeypatch):
    """Confounds are regressed out and the pipeline still completes."""
    monkeypatch.setattr(
        workflows.datasets, "fetch_coords_power_2011", lambda *a, **k: _fake_power_atlas()
    )
    out_dir = tmp_path / "out"
    files, qc = _write_synthetic_images(tmp_path)
    rng = np.random.RandomState(3)
    confounds = [rng.normal(size=(IMG_SHAPE[-1], 2)) for _ in files]
    workflows.run_analyses(
        files,
        qc,
        out_dir=str(out_dir),
        confounds=confounds,
        n_iters=2,
        n_jobs=1,
        window=_WINDOW,
        analyses=("qcrsfc", "highlow"),
    )
    assert (out_dir / "analysis_values.tsv.gz").is_file()


@pytest.mark.integration
def test_run_analyses_with_run_covariates(tmp_path, monkeypatch):
    """Run-level covariates are accepted by the QC:RSFC workflow."""
    monkeypatch.setattr(
        workflows.datasets, "fetch_coords_power_2011", lambda *a, **k: _fake_power_atlas()
    )
    out_dir = tmp_path / "out"
    files, qc = _write_synthetic_images(tmp_path)
    run_covariates = pd.DataFrame(
        {
            "age": np.linspace(20, 40, len(files)),
            "site": np.repeat(["a", "b", "c"], 4),
        }
    )
    workflows.run_analyses(
        files,
        qc,
        out_dir=str(out_dir),
        n_iters=2,
        n_jobs=1,
        window=_WINDOW,
        analyses=("qcrsfc",),
        run_covariates=run_covariates,
    )
    assert (out_dir / "analysis_values.tsv.gz").is_file()


@pytest.mark.integration
def test_run_analyses_writes_run_denoising_summary(tmp_path, monkeypatch):
    """The workflow writes inferred and user-supplied denoising accounting."""
    monkeypatch.setattr(
        workflows.datasets, "fetch_coords_power_2011", lambda *a, **k: _fake_power_atlas()
    )
    out_dir = tmp_path / "out"
    files, qc = _write_synthetic_images(tmp_path)
    run_denoising_metrics = pd.DataFrame(
        {
            "n_volumes_retained_after_denoising": [IMG_SHAPE[-1] - 1] * len(files),
            "temporal_degrees_of_freedom": [IMG_SHAPE[-1] - 3] * len(files),
        }
    )

    workflows.run_analyses(
        files,
        qc,
        out_dir=str(out_dir),
        n_iters=2,
        n_jobs=1,
        window=_WINDOW,
        analyses=("qcrsfc",),
        run_denoising_metrics=run_denoising_metrics,
    )

    summary = pd.read_table(out_dir / "run_denoising_summary.tsv")
    assert summary.shape[0] == len(files)
    assert summary["n_volumes"].eq(IMG_SHAPE[-1]).all()
    assert summary["n_volumes_retained_after_denoising"].eq(IMG_SHAPE[-1] - 1).all()
    assert summary["temporal_degrees_of_freedom"].eq(IMG_SHAPE[-1] - 3).all()
    assert summary["retained_for_analysis"].all()


@pytest.mark.integration
def test_run_analyses_skips_bad_subjects(tmp_path, monkeypatch):
    """A run with a zero-variance ROI is dropped, and the log records it."""
    monkeypatch.setattr(
        workflows.datasets, "fetch_coords_power_2011", lambda *a, **k: _fake_power_atlas()
    )
    out_dir = tmp_path / "out"
    files, qc = _write_synthetic_images(tmp_path, n_subjects=12)

    # Append a 13th subject whose image is constant -> every ROI has 0 variance.
    bad_path = op.join(tmp_path, "sub-bad_bold.nii.gz")
    nib.Nifti1Image(np.ones(IMG_SHAPE, dtype=np.float32), IMG_AFFINE).to_filename(bad_path)
    files.append(bad_path)
    qc.append(np.random.RandomState(9).uniform(0, 0.4, size=IMG_SHAPE[-1]))

    workflows.run_analyses(
        files, qc, out_dir=str(out_dir), n_iters=2, n_jobs=1, window=_WINDOW, analyses=("qcrsfc",)
    )
    assert (out_dir / "analysis_values.tsv.gz").is_file()
    log_text = (out_dir / "log.tsv").read_text()
    assert "variance of 0" in log_text
    summary = pd.read_table(out_dir / "run_denoising_summary.tsv")
    bad_run = summary.loc[summary["filename"] == "sub-bad_bold.nii.gz"].iloc[0]
    assert not bad_run["retained_after_loading"]
    assert "zero_variance_roi" in bad_run["drop_reason"]


@pytest.mark.integration
def test_run_analyses_pca_outlier_detection(tmp_path, monkeypatch):
    """The PCA/Mahalanobis outlier-detection branch runs and completes."""
    monkeypatch.setattr(
        workflows.datasets, "fetch_coords_power_2011", lambda *a, **k: _fake_power_atlas()
    )
    out_dir = tmp_path / "out"
    files, qc = _write_synthetic_images(tmp_path, n_subjects=15)
    workflows.run_analyses(
        files,
        qc,
        out_dir=str(out_dir),
        n_iters=2,
        n_jobs=1,
        window=_WINDOW,
        analyses=("qcrsfc", "highlow", "scrubbing"),
        pca_threshold=3,
        outlier_threshold=0.05,
    )
    assert (out_dir / "analysis_values.tsv.gz").is_file()


@pytest.mark.integration
def test_run_analyses_too_few_subjects(tmp_path, monkeypatch):
    """Fewer than 10 retained subjects raises a clear error."""
    monkeypatch.setattr(
        workflows.datasets, "fetch_coords_power_2011", lambda *a, **k: _fake_power_atlas()
    )
    out_dir = tmp_path / "out"
    files, qc = _write_synthetic_images(tmp_path, n_subjects=5)
    with pytest.raises(ValueError, match="Too few subjects"):
        workflows.run_analyses(
            files, qc, out_dir=str(out_dir), n_iters=2, n_jobs=1, window=_WINDOW
        )
