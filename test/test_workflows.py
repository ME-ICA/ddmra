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


def test_build_atlas_masker_uses_labels_file(tmp_path):
    """An existing atlas file is loaded as a labels masker."""
    atlas_path = _write_synthetic_label_atlas(tmp_path)
    masker, coords = workflows._build_atlas_masker(atlas_path)

    assert isinstance(masker, NiftiLabelsMasker)
    assert coords.shape == ATLAS_COORDS.shape


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
