"""Miscellaneous utility functions for the DDMRA package."""

import numpy as np
from scipy import stats


def get_val(x_arr, y_arr, x_val):
    """Perform interpolation to get value from y_arr at index of x_val based on x_arr.

    TODO: Better documentation

    Parameters
    ----------
    x_arr : (N,) numpy.ndarray

    y_arr : ([X], N) numpy.ndarray

    x_val : float
        Position in x_arr for which to estimate value in y_arr.

    Returns
    -------
    y_val : float or (X,) numpy.ndarray
        Value in y_arr corresponding to position x_val in x_arr.
    """
    if y_arr.ndim == 2:
        y_arr = y_arr.T
    assert x_arr.shape[0] == y_arr.shape[0]

    temp_idx = np.where(x_arr == x_val)[0]
    if len(temp_idx) > 0:
        if y_arr.ndim == 2:
            y_val = np.mean(y_arr[temp_idx, :], axis=0)
        else:
            y_val = np.mean(y_arr[temp_idx])
    else:
        val1 = x_arr[np.where(x_arr <= x_val)[0][-1]]
        val2 = x_arr[np.where(x_arr >= x_val)[0][0]]
        temp_idx1 = np.where(x_arr == val1)[0]
        temp_idx2 = np.where(x_arr == val2)[0]
        temp_idx = np.unique(np.hstack((temp_idx1, temp_idx2)))
        if y_arr.ndim == 2:
            y_val = np.zeros(y_arr.shape[1])
            for i in range(y_val.shape[0]):
                temp_y_arr = y_arr[:, i]
                y_val[i] = np.interp(xp=x_arr[temp_idx], fp=temp_y_arr[temp_idx], x=x_val)
        else:
            y_val = np.interp(xp=x_arr[temp_idx], fp=y_arr[temp_idx], x=x_val)
    return y_val


def rank_p(test_value, null_array, tail="two"):
    """Return rank-based p-value for test value against null array.

    Parameters
    ----------
    test_value : float
        Value for which to determine rank-based p-value.
    null_array : 1D numpy.ndarray
        Values to use as null distribution.
    tail : {'two', 'upper', 'lower'}
        Tail for computing p-value.
    """
    if tail == "two":
        p_value = (
            (50 - np.abs(stats.percentileofscore(null_array, test_value) - 50.0)) * 2.0 / 100.0
        )
    elif tail == "upper":
        p_value = 1 - (stats.percentileofscore(null_array, test_value) / 100.0)
    elif tail == "lower":
        p_value = stats.percentileofscore(null_array, test_value) / 100.0
    else:
        raise ValueError('Argument "tail" must be one of ["two", "upper", ' '"lower"]')
    return p_value


def fast_pearson(X, y):
    """Fast correlations between y and each row of X, for QC-RSFC.

    Checked for accuracy. From http://qr.ae/TU1B9C

    Parameters
    ----------
    X : (i, j) numpy.ndarray
        Matrix for which each row is correlated with y.
    y : (j,) numpy.ndarray
        Array with which to correlate each row of X.

    Returns
    -------
    pearsons : (i,) numpy.ndarray
        Row-wise correlations between X and y.
    """
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert y.shape[0] == X.shape[1]
    y_bar = np.mean(y)
    y_intermediate = y - y_bar
    X_bar = np.mean(X, axis=1)[:, np.newaxis]
    X_intermediate = X - X_bar
    nums = X_intermediate.dot(y_intermediate)
    y_sq = np.sum(np.square(y_intermediate))
    X_sqs = np.sum(np.square(X_intermediate), axis=1)
    denoms = np.sqrt(y_sq * X_sqs)
    pearsons = nums / denoms
    return pearsons


def get_fd_power(motion, order=["x", "y", "z", "r", "p", "ya"], unit="deg", radius=50):
    """Calculate Framewise Displacement (Power version).

    This is meant to be very flexible, but is really only here for the test ADHD dataset.

    Parameters
    ----------
    motion : (T, 6) numpy.ndarray
        Translation and rotation motion parameters.
    order : (6,) list
        Order of the motion parameters in motion array. Some combination of
        'x', 'y', 'z', 'r', 'p', and 'ya'. Default has translations then
        rotations.
    unit : {'deg', 'rad'}
        Which unit the rotation parameters are in. 'deg' for degree (default in
        SPM and AfNI), 'rad' for radian (default in FSL). Default is 'deg'.
    radius : float or int
        Radius of brain in millimeters. Default is 50, as used by Power.

    Returns
    -------
    fd : (T,) numpy.ndarray
        Framewise displacement values.
    """
    des_order = ["x", "y", "z", "r", "p", "ya"]
    reorder = [order.index(i) for i in des_order]
    motion_reordered = motion[:, reorder]
    if unit == "deg":
        motion_reordered[:, 3:] = motion_reordered[:, 3:] * radius * (np.pi / 180.0)
    elif unit == "rad":
        motion_reordered[:, 3:] = motion_reordered[:, 3:] * radius
    else:
        raise Exception("Rotation units must be degrees or radians.")

    deriv = np.vstack((np.array([[0, 0, 0, 0, 0, 0]]), np.diff(motion_reordered, axis=0)))
    fd = np.sum(np.abs(deriv), axis=1)
    return fd


def moving_average(values, window):
    """Calculate running average along values array.

    Parameters
    ----------
    values : (N,) numpy.ndarray
        Values over which to compute average.
    window : int
        Sliding window size over which to average values.

    Returns
    -------
    sma : (N,) numpy.ndarray
        Each value in sma is the average of the `window` values in values
        surrounding that position. So sma[1000] (with a window of 500) will be
        the mean of values[750:1250]. Positions at the beginning and end will
        be NaNs.
    """
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, "valid")
    buff = np.zeros(int(window / 2)) * np.nan
    sma = np.hstack((buff, sma, buff))[:-1]
    return sma
