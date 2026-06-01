"""Tests for the ddmra.utils module."""
import math

import numpy as np

from ddmra import utils


def test_null_to_p_float():
    """Test utils.null_to_p with single float input, assuming asymmetric null dist."""
    null = [-10, -9, -9, -3, -2, -1, -1, 0, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9]

    # Two-tailed
    assert math.isclose(utils.null_to_p(0, null, "two"), 0.8)
    assert math.isclose(utils.null_to_p(9, null, "two"), 0.1)
    assert math.isclose(utils.null_to_p(10, null, "two"), 0.05)
    assert math.isclose(utils.null_to_p(-9, null, "two"), 0.3)
    assert math.isclose(utils.null_to_p(-10, null, "two"), 0.1)
    # Still 0.05 because minimum valid p-value is 1 / len(null)
    result = utils.null_to_p(20, null, "two")
    assert result == utils.null_to_p(-20, null, "two")
    assert math.isclose(result, 0.05)

    # Left/lower-tailed
    assert math.isclose(utils.null_to_p(9, null, "lower"), 0.95)
    assert math.isclose(utils.null_to_p(-9, null, "lower"), 0.15)
    assert math.isclose(utils.null_to_p(0, null, "lower"), 0.4)

    # Right/upper-tailed
    assert math.isclose(utils.null_to_p(9, null, "upper"), 0.05)
    assert math.isclose(utils.null_to_p(-9, null, "upper"), 0.95)
    assert math.isclose(utils.null_to_p(0, null, "upper"), 0.65)

    # Test that 1/n(null) is preserved with extreme values
    nulldist = np.random.normal(size=10000)
    assert math.isclose(utils.null_to_p(20, nulldist, "two"), 1 / 10000)
    assert math.isclose(utils.null_to_p(20, nulldist, "lower"), 1 - 1 / 10000)


def test_null_to_p_float_symmetric():
    """Test utils.null_to_p with single float input, assuming symmetric null dist."""
    null = [-10, -9, -9, -3, -2, -1, -1, 0, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9]

    # Only need to test two-tailed; symmetry is irrelevant for one-tailed
    assert math.isclose(utils.null_to_p(0, null, "two", symmetric=True), 0.95)
    result = utils.null_to_p(9, null, "two", symmetric=True)
    assert result == utils.null_to_p(-9, null, "two", symmetric=True)
    assert math.isclose(result, 0.2)
    result = utils.null_to_p(10, null, "two", symmetric=True)
    assert result == utils.null_to_p(-10, null, "two", symmetric=True)
    assert math.isclose(result, 0.05)
    # Still 0.05 because minimum valid p-value is 1 / len(null)
    result = utils.null_to_p(20, null, "two", symmetric=True)
    assert result == utils.null_to_p(-20, null, "two", symmetric=True)
    assert math.isclose(result, 0.05)


def test_null_to_p_array():
    """Test nimare.stats.utils.null_to_p with 1d array input."""
    N = 10000
    nulldist = np.random.normal(size=N)
    t = np.sort(np.random.normal(size=N))
    p = np.sort(utils.null_to_p(t, nulldist))
    assert p.shape == (N,)
    assert (p < 1).all()
    assert (p > 0).all()
    # Resulting distribution should be roughly uniform
    assert np.abs(p.mean() - 0.5) < 0.02
    assert np.abs(p.var() - 1 / 12) < 0.02


def test_moving_average():
    arr = np.random.random(1000) * 100
    sma_10 = utils.moving_average(arr, window=10)
    assert all(np.isnan(sma_10[:5]))
    assert math.isclose(sma_10[50], np.mean(arr[45:55]))
    sma_50 = utils.moving_average(arr, window=50)
    assert all(np.isnan(sma_50[:25]))
    assert math.isclose(sma_50[50], np.mean(arr[25:75]))
