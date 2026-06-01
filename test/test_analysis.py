"""Tests for the ddmra.analysis module."""

import numpy as np
import pytest

from ddmra import analysis, utils


def test_highlow_analysis():
    """High-low analysis returns (high-group mean) - (low-group mean) per edge."""
    mean_qcs = np.array([1.0, 2.0, 3.0, 4.0])  # median 2.5
    z_corr_mats = np.array(
        [
            [1.0, 1.0],  # low
            [3.0, 3.0],  # low
            [10.0, 10.0],  # high
            [20.0, 20.0],  # high
        ]
    )
    result = analysis.highlow_analysis(mean_qcs, z_corr_mats)
    # high mean = [15, 15]; low mean = [2, 2]; diff = [13, 13]
    assert result.shape == (2,)
    assert np.allclose(result, [13.0, 13.0])


def test_highlow_analysis_shape_assertions():
    """highlow_analysis enforces 1D QC, 2D corr matrices, and matching subjects."""
    with pytest.raises(AssertionError):
        analysis.highlow_analysis(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(AssertionError):
        analysis.highlow_analysis(np.ones(2), np.ones(3))
    with pytest.raises(AssertionError):
        analysis.highlow_analysis(np.ones(2), np.ones((3, 4)))


def test_qcrsfc_analysis_signs_and_values():
    """QC:RSFC correlates each edge with QC across subjects, then z-transforms."""
    mean_qcs = np.array([1.0, 2.0, 3.0, 4.0])
    z_corr_mats = np.array(
        [
            [1.0, 4.0, 1.0],
            [2.0, 3.0, 2.0],
            [3.0, 2.0, 3.0],
            [4.0, 1.0, 5.0],
        ]
    )
    result = analysis.qcrsfc_analysis(mean_qcs, z_corr_mats)
    assert result.shape == (3,)
    # Edge 0 is perfectly correlated with QC -> clipped to arctanh(0.999).
    assert np.isclose(result[0], np.arctanh(0.999))
    # Edge 1 is perfectly anti-correlated -> -arctanh(0.999).
    assert np.isclose(result[1], np.arctanh(-0.999))
    # Edge 2 is positively (but not perfectly) correlated.
    assert 0 < result[2] < np.arctanh(0.999)


def test_qcrsfc_analysis_shape_assertions():
    """qcrsfc_analysis enforces 1D QC, 2D corr matrices, and matching subjects."""
    with pytest.raises(AssertionError):
        analysis.qcrsfc_analysis(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(AssertionError):
        analysis.qcrsfc_analysis(np.ones(2), np.ones(3))


def test_scrubbing_analysis_inclusion_and_value():
    """Only subjects with 0.5 <= proportion-kept < 1.0 contribute to the mean."""
    rng = np.random.RandomState(7)
    n_rois, n_tps = 3, 20
    qc_thresh = 0.2
    edge_sorting_idx = np.arange(3)

    ts_included = rng.normal(size=(n_rois, n_tps))
    # 10 of 20 volumes kept -> proportion 0.5 -> included.
    qc_included = np.concatenate([np.full(10, 0.1), np.full(10, 0.9)])
    keep_idx = qc_included <= qc_thresh

    ts_all_kept = rng.normal(size=(n_rois, n_tps))
    qc_all_kept = np.full(n_tps, 0.1)  # proportion 1.0 -> excluded

    ts_mostly_scrubbed = rng.normal(size=(n_rois, n_tps))
    qc_mostly_scrubbed = np.concatenate([np.full(2, 0.1), np.full(18, 0.9)])  # 0.1 -> excluded

    result = analysis.scrubbing_analysis(
        [qc_included, qc_all_kept, qc_mostly_scrubbed],
        [ts_included, ts_all_kept, ts_mostly_scrubbed],
        edge_sorting_idx,
        qc_thresh=qc_thresh,
    )

    triu_idx = np.triu_indices(n_rois, k=1)
    raw = utils.r2z(np.corrcoef(ts_included)[triu_idx])
    scrubbed = utils.r2z(np.corrcoef(ts_included[:, keep_idx])[triu_idx])
    expected = (raw - scrubbed)[edge_sorting_idx]

    assert result.shape == (3,)
    # Only the single included subject contributes, so the mean equals its delta.
    assert np.allclose(result, expected)


def test_scrubbing_analysis_edge_sorting():
    """The output is reordered according to edge_sorting_idx."""
    rng = np.random.RandomState(11)
    n_rois, n_tps = 3, 20
    qc = np.concatenate([np.full(10, 0.1), np.full(10, 0.9)])
    ts = rng.normal(size=(n_rois, n_tps))

    identity = analysis.scrubbing_analysis([qc], [ts], np.arange(3), qc_thresh=0.2)
    reversed_idx = np.array([2, 1, 0])
    reversed_result = analysis.scrubbing_analysis([qc], [ts], reversed_idx, qc_thresh=0.2)
    assert np.allclose(reversed_result, identity[::-1])


def test_scrubbing_analysis_length_assertion():
    """Mismatched QC and timeseries list lengths raise."""
    ts = np.random.RandomState(0).normal(size=(3, 20))
    qc = np.concatenate([np.full(10, 0.1), np.full(10, 0.9)])
    with pytest.raises(AssertionError):
        analysis.scrubbing_analysis([qc, qc], [ts], np.arange(3))


def _make_scrubbing_inputs(n_subjects=4, n_rois=3, n_tps=20, seed=0):
    rng = np.random.RandomState(seed)
    qc_values = [np.concatenate([np.full(10, 0.1), np.full(10, 0.9)]) for _ in range(n_subjects)]
    ts_all = [rng.normal(size=(n_rois, n_tps)) for _ in range(n_subjects)]
    edge_sorting_idx = np.arange((n_rois * (n_rois - 1)) // 2)
    return qc_values, ts_all, edge_sorting_idx


def test_scrubbing_null_distribution_shape_and_determinism():
    """The scrubbing null distribution has shape (n_iters, n_pairs) and is reproducible."""
    qc_values, ts_all, edge_sorting_idx = _make_scrubbing_inputs()
    n_iters = 3
    out1 = analysis.scrubbing_null_distribution(
        qc_values, ts_all, qc_thresh=0.2, edge_sorting_idx=edge_sorting_idx, n_iters=n_iters
    )
    out2 = analysis.scrubbing_null_distribution(
        qc_values, ts_all, qc_thresh=0.2, edge_sorting_idx=edge_sorting_idx, n_iters=n_iters
    )
    assert out1.shape == (n_iters, 3)
    # Seeds are fixed per-iteration, so repeated calls are identical.
    assert np.array_equal(out1, out2)


def test_other_null_distributions_shape_and_determinism():
    """QC:RSFC and high-low null distributions have the expected shapes and reproduce."""
    rng = np.random.RandomState(2)
    n_subjects, n_edges = 12, 5
    mean_qc = rng.normal(size=n_subjects)
    z_corr_mats = rng.normal(size=(n_subjects, n_edges))
    n_iters = 3

    qcrsfc1, hl1 = analysis.other_null_distributions(mean_qc, z_corr_mats, n_iters=n_iters)
    qcrsfc2, hl2 = analysis.other_null_distributions(mean_qc, z_corr_mats, n_iters=n_iters)

    assert qcrsfc1.shape == (n_iters, n_edges)
    assert hl1.shape == (n_iters, n_edges)
    assert np.array_equal(qcrsfc1, qcrsfc2)
    assert np.array_equal(hl1, hl2)
