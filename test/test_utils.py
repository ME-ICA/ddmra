"""Tests for the ddmra.utils module."""

import math

import joblib
import numpy as np
import pytest
from tqdm import tqdm

from ddmra import utils


def test_null_to_p_float():
    """Test utils.null_to_p with single float input, assuming asymmetric null dist."""
    null = [-10, -9, -9, -3, -2, -1, -1, 0, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9]
    denom = len(null) + 1

    # Two-tailed
    assert math.isclose(utils.null_to_p(0, null, "two"), 18 / denom)
    assert math.isclose(utils.null_to_p(9, null, "two"), 4 / denom)
    assert math.isclose(utils.null_to_p(10, null, "two"), 2 / denom)
    assert math.isclose(utils.null_to_p(-9, null, "two"), 8 / denom)
    assert math.isclose(utils.null_to_p(-10, null, "two"), 4 / denom)
    # The two-tailed minimum is doubled because it is based on two one-tailed p-values.
    result = utils.null_to_p(20, null, "two")
    assert result == utils.null_to_p(-20, null, "two")
    assert math.isclose(result, 2 / denom)

    # Left/lower-tailed
    assert math.isclose(utils.null_to_p(9, null, "lower"), 1)
    assert math.isclose(utils.null_to_p(-9, null, "lower"), 4 / denom)
    assert math.isclose(utils.null_to_p(0, null, "lower"), 9 / denom)

    # Right/upper-tailed
    assert math.isclose(utils.null_to_p(9, null, "upper"), 2 / denom)
    assert math.isclose(utils.null_to_p(-9, null, "upper"), 20 / denom)
    assert math.isclose(utils.null_to_p(0, null, "upper"), 14 / denom)

    # Test the plus-one correction with extreme values.
    nulldist = np.random.normal(size=10000)
    assert math.isclose(utils.null_to_p(20, nulldist, "two"), 2 / 10001)
    assert math.isclose(utils.null_to_p(20, nulldist, "upper"), 1 / 10001)
    assert math.isclose(utils.null_to_p(20, nulldist, "lower"), 1)


def test_null_to_p_float_symmetric():
    """Test utils.null_to_p with single float input, assuming symmetric null dist."""
    null = [-10, -9, -9, -3, -2, -1, -1, 0, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9]
    denom = len(null) + 1

    # Only need to test two-tailed; symmetry is irrelevant for one-tailed
    assert math.isclose(utils.null_to_p(0, null, "two", symmetric=True), 1)
    result = utils.null_to_p(9, null, "two", symmetric=True)
    assert result == utils.null_to_p(-9, null, "two", symmetric=True)
    assert math.isclose(result, 5 / denom)
    result = utils.null_to_p(10, null, "two", symmetric=True)
    assert result == utils.null_to_p(-10, null, "two", symmetric=True)
    assert math.isclose(result, 2 / denom)
    result = utils.null_to_p(20, null, "two", symmetric=True)
    assert result == utils.null_to_p(-20, null, "two", symmetric=True)
    assert math.isclose(result, 1 / denom)


def test_null_to_p_array():
    """Test utils.null_to_p with 1d array input."""
    N = 10000
    nulldist = np.random.normal(size=N)
    t = np.sort(np.random.normal(size=N))
    p = np.sort(utils.null_to_p(t, nulldist))
    assert p.shape == (N,)
    assert (p <= 1).all()
    assert (p > 0).all()
    # Resulting distribution should be roughly uniform
    assert np.abs(p.mean() - 0.5) < 0.02
    assert np.abs(p.var() - 1 / 12) < 0.02


def test_null_to_p_invalid_tail():
    """An unrecognized ``tail`` argument should raise a ValueError."""
    with pytest.raises(ValueError, match="tail"):
        utils.null_to_p(0, [0, 1, 2, 3], tail="middle")


def test_null_to_p_empty_null_raises():
    """An empty null distribution cannot produce an empirical p-value."""
    with pytest.raises(ValueError, match="at least one"):
        utils.null_to_p(0, [])


def test_null_to_p_large_array_reconstruction():
    """The >1000-value unique/reconstruct path should match the naive path."""
    rng = np.random.RandomState(42)
    null = rng.normal(size=500)
    # Build a >1000 length test array with duplicates to exercise np.unique path.
    test_value = np.repeat(rng.normal(size=600), 2)
    assert test_value.size > 1000
    result = utils.null_to_p(test_value, null)
    assert result.shape == test_value.shape
    # Duplicated entries must map to identical p-values after reconstruction.
    assert np.array_equal(result[0::2], result[1::2])


def test_moving_average():
    """Test utils.moving_average against hand-computed windowed means."""
    arr = np.random.random(1000) * 100
    sma_10 = utils.moving_average(arr, window=10)
    assert all(np.isnan(sma_10[:5]))
    assert math.isclose(sma_10[50], np.mean(arr[45:55]))
    sma_50 = utils.moving_average(arr, window=50)
    assert all(np.isnan(sma_50[:25]))
    assert math.isclose(sma_50[50], np.mean(arr[25:75]))


def test_moving_average_odd_window_raises():
    """An odd window size is not supported and should raise."""
    with pytest.raises(ValueError, match="divisible by 2"):
        utils.moving_average(np.arange(10.0), window=3)


def test_moving_average_preserves_length():
    """Output length matches the input length."""
    arr = np.arange(100.0)
    out = utils.moving_average(arr, window=10)
    assert out.shape == arr.shape


def test_get_val_1d_exact_match():
    """Exact x_val matches return the corresponding y value(s)."""
    x_arr = np.array([0.0, 1.0, 2.0, 3.0])
    y_arr = np.array([10.0, 20.0, 30.0, 40.0])
    assert math.isclose(utils.get_val(x_arr, y_arr, 2.0), 30.0)


def test_get_val_1d_interpolation():
    """Values between grid points are linearly interpolated."""
    x_arr = np.array([0.0, 1.0, 2.0, 3.0])
    y_arr = np.array([10.0, 20.0, 30.0, 40.0])
    assert math.isclose(utils.get_val(x_arr, y_arr, 1.5), 25.0)


def test_get_val_1d_duplicate_x_averaged():
    """Duplicate x values are averaged on an exact match."""
    x_arr = np.array([0.0, 1.0, 1.0, 2.0])
    y_arr = np.array([10.0, 20.0, 40.0, 50.0])
    assert math.isclose(utils.get_val(x_arr, y_arr, 1.0), 30.0)


def test_get_val_2d():
    """A 2D y_arr returns a per-row value array."""
    x_arr = np.array([0.0, 1.0, 2.0, 3.0])
    # shape (X, N); will be transposed internally to (N, X)
    y_arr = np.array([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])
    # Exact match
    exact = utils.get_val(x_arr, y_arr, 2.0)
    assert np.allclose(exact, [30.0, 3.0])
    # Interpolated
    interp = utils.get_val(x_arr, y_arr, 1.5)
    assert np.allclose(interp, [25.0, 2.5])


def test_fast_pearson_matches_numpy():
    """fast_pearson should equal row-wise numpy correlations."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(5, 50))
    y = rng.normal(size=50)
    fast = utils.fast_pearson(X, y)
    expected = np.array([np.corrcoef(row, y)[0, 1] for row in X])
    assert fast.shape == (5,)
    assert np.allclose(fast, expected)


def test_fast_pearson_shape_assertions():
    """fast_pearson enforces 2D X, 1D y, and matching lengths."""
    with pytest.raises(AssertionError):
        utils.fast_pearson(np.ones(10), np.ones(10))  # X not 2D
    with pytest.raises(AssertionError):
        utils.fast_pearson(np.ones((2, 10)), np.ones((2, 10)))  # y not 1D
    with pytest.raises(AssertionError):
        utils.fast_pearson(np.ones((2, 10)), np.ones(9))  # length mismatch


def test_get_fd_power_translation():
    """A unit translation step produces a framewise displacement of 1."""
    motion = np.zeros((3, 6))
    motion[1:, 0] = 1.0  # x shifts by 1 at second timepoint and stays
    fd = utils.get_fd_power(motion, unit="deg")
    assert np.allclose(fd, [0.0, 1.0, 0.0])


def test_get_fd_power_rotation_units():
    """Rotation scaling differs between degrees and radians."""
    motion_deg = np.zeros((2, 6))
    motion_deg[1, 3] = 1.0  # 1 unit of rotation
    fd_deg = utils.get_fd_power(motion_deg.copy(), unit="deg", radius=50)
    assert math.isclose(fd_deg[1], 1.0 * 50 * (np.pi / 180.0))

    motion_rad = np.zeros((2, 6))
    motion_rad[1, 3] = 1.0
    fd_rad = utils.get_fd_power(motion_rad.copy(), unit="rad", radius=50)
    assert math.isclose(fd_rad[1], 1.0 * 50)


def test_get_fd_power_invalid_unit():
    """An unrecognized rotation unit raises."""
    with pytest.raises(Exception, match="degrees or radians"):
        utils.get_fd_power(np.zeros((3, 6)), unit="furlongs")


def test_get_fd_power_order_independent():
    """Reordering motion columns + ``order`` yields identical FD."""
    rng = np.random.RandomState(1)
    motion = rng.normal(size=(20, 6))
    fd_default = utils.get_fd_power(motion.copy(), unit="rad")

    custom_order = ["r", "p", "ya", "x", "y", "z"]
    # Rearrange the columns to match the custom order, then declare that order.
    des_order = ["x", "y", "z", "r", "p", "ya"]
    col_for_custom = [des_order.index(name) for name in custom_order]
    motion_reordered = motion[:, col_for_custom]
    fd_custom = utils.get_fd_power(motion_reordered.copy(), order=custom_order, unit="rad")
    assert np.allclose(fd_default, fd_custom)


def test_average_across_distances():
    """Values are averaged within unique distance bins, ignoring NaNs."""
    distances = np.array([1.0, 1.0, 2.0, 3.0])
    values = np.array([10.0, 20.0, 30.0, np.nan])
    means, unique_distances = utils.average_across_distances(values, distances)
    assert np.allclose(unique_distances, [1.0, 2.0])  # 3.0 dropped (NaN value)
    assert np.allclose(means, [15.0, 30.0])


def test_r2z():
    """Fisher's r-to-z transform with clipping of perfect correlations."""
    assert math.isclose(utils.r2z(np.array([0.0]))[0], 0.0)
    assert math.isclose(utils.r2z(np.array([0.5]))[0], np.arctanh(0.5))
    # Perfect correlations are clipped to +/-0.999 before the transform.
    assert math.isclose(utils.r2z(np.array([1.0]))[0], np.arctanh(0.999))
    assert math.isclose(utils.r2z(np.array([-1.0]))[0], np.arctanh(-0.999))
    # r=2 (out of range) is clipped to the same value as r=1.
    assert utils.r2z(np.array([2.0]))[0] == utils.r2z(np.array([1.0]))[0]


def test_get_rank():
    """get_rank returns searchsorted ranks of values within null columns."""
    values = np.array([2.5, 0.0, 5.0])
    null_values = np.array(
        [
            [4.0, 4.0, 4.0],
            [3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
        ]
    )
    ranks = utils.get_rank(values, null_values)
    assert np.array_equal(ranks, [2, 0, 4])


def test_get_rank_shape_assertions():
    """get_rank enforces 1D values, 2D null, and matching widths."""
    with pytest.raises(AssertionError):
        utils.get_rank(np.ones((2, 2)), np.ones((4, 2)))  # values not 1D
    with pytest.raises(AssertionError):
        utils.get_rank(np.ones(2), np.ones(4))  # null not 2D
    with pytest.raises(AssertionError):
        utils.get_rank(np.ones(3), np.ones((4, 2)))  # width mismatch


def test_assess_significance():
    """A curve that exceeds every null curve gets the smallest possible p-values."""
    distances = np.array([0.0, 35.0, 50.0, 100.0, 150.0])
    curve = np.array([5.0, 10.0, 7.0, 0.0, -2.0])
    n_perms = 20
    null_curves = np.zeros((n_perms, distances.size))  # all-zero nulls
    p_inter, p_slope = utils.assess_significance(
        curve, null_curves, distances, intercept_distance=35, v2=100
    )
    # intercept (10) and slope (10 - 0 = 10) both exceed all nulls -> minimum p.
    assert math.isclose(p_inter, 1 / (n_perms + 1))
    assert math.isclose(p_slope, 1 / (n_perms + 1))


def test_assess_significance_shape_assertions():
    """assess_significance enforces consistent dimensions."""
    distances = np.array([0.0, 35.0, 100.0])
    curve = np.array([1.0, 2.0, 3.0])
    null_curves = np.zeros((5, 3))
    # Mismatched curve/distance length
    with pytest.raises(AssertionError):
        utils.assess_significance(curve[:2], null_curves, distances, 35, 100)
    # 1D null_curves
    with pytest.raises(AssertionError):
        utils.assess_significance(curve, np.zeros(3), distances, 35, 100)


def test_calculate_smoothing_curve_1d_matches_helpers():
    """The 1D smoothing curve equals moving_average + average_across_distances."""
    rng = np.random.RandomState(3)
    distances = np.sort(rng.randint(0, 6, size=40).astype(float))
    values = rng.normal(size=40)
    window = 4

    ma_distances = utils.moving_average(distances, window)
    expected, true_distances = utils.average_across_distances(
        utils.moving_average(values, window), distances
    )
    _, expected_distances = utils.average_across_distances(ma_distances, distances)
    assert np.allclose(true_distances, expected_distances)

    result = utils.calculate_smoothing_curve(values, window, distances, true_distances)
    assert result.shape == expected.shape
    assert np.allclose(result, expected)


def test_calculate_smoothing_curve_2d():
    """A 2D input with identical rows yields identical per-row smoothing curves."""
    rng = np.random.RandomState(4)
    distances = np.sort(rng.randint(0, 6, size=40).astype(float))
    values_1d = rng.normal(size=40)
    window = 4
    _, true_distances = utils.average_across_distances(
        utils.moving_average(distances, window), distances
    )

    curve_1d = utils.calculate_smoothing_curve(values_1d, window, distances, true_distances)
    values_2d = np.vstack([values_1d, values_1d])
    curve_2d = utils.calculate_smoothing_curve(values_2d, window, distances, true_distances)

    assert curve_2d.shape == (2, true_distances.size)
    assert np.allclose(curve_2d[0], curve_1d)
    assert np.allclose(curve_2d[1], curve_1d)


def _double(x):
    """Module-level helper so it is picklable by joblib."""
    return x * 2


def test_tqdm_joblib_restores_callback_and_runs():
    """tqdm_joblib runs jobs and restores joblib's batch-completion callback."""
    original_callback = joblib.parallel.BatchCompletionCallBack
    with utils.tqdm_joblib(tqdm(total=3, disable=True)):
        results = joblib.Parallel(n_jobs=1)(joblib.delayed(_double)(i) for i in range(3))
    assert results == [0, 2, 4]
    # The patched callback must be restored after the context exits.
    assert joblib.parallel.BatchCompletionCallBack is original_callback
