"""Miscellaneous utility functions for the DDMRA package."""
import numpy as np


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


def null_to_p(test_value, null_array, tail="two", symmetric=False):
    """Return p-value for test value(s) against null array.

    Parameters
    ----------
    test_value : 1D array_like
        Values for which to determine p-value.
    null_array : 1D array_like
        Null distribution against which test_value is compared.
    tail : {'two', 'upper', 'lower'}, optional
        Whether to compare value against null distribution in a two-sided
        ('two') or one-sided ('upper' or 'lower') manner.
        If 'upper', then higher values for the test_value are more significant.
        If 'lower', then lower values for the test_value are more significant.
        Default is 'two'.
    symmetric : bool
        When tail="two", indicates how to compute p-values. When False (default),
        both one-tailed p-values are computed, and the two-tailed p is double
        the minimum one-tailed p. When True, it is assumed that the null
        distribution is zero-centered and symmetric, and the two-tailed p-value
        is computed as P(abs(test_value) >= abs(null_array)).

    Returns
    -------
    p_value : :obj:`float`
        P-value(s) associated with the test value when compared against the null
        distribution. Return type matches input type (i.e., a float if
        test_value is a single float, and an array if test_value is an array).

    Notes
    -----
    P-values are clipped based on the number of elements in the null array.
    Therefore no p-values of 0 or 1 should be produced.
    When the null distribution is known to be symmetric and centered on zero,
    and two-tailed p-values are desired, use symmetric=True, as it is
    approximately twice as efficient computationally, and has lower variance.
    """
    if tail not in {"two", "upper", "lower"}:
        raise ValueError('Argument "tail" must be one of ["two", "upper", "lower"]')

    return_first = isinstance(test_value, (float, int))
    test_value = np.atleast_1d(test_value)
    null_array = np.array(null_array)

    # For efficiency's sake, if there are more than 1000 values, pass only the unique
    # values through percentileofscore(), and then reconstruct.
    if len(test_value) > 1000:
        reconstruct = True
        test_value, uniq_idx = np.unique(test_value, return_inverse=True)
    else:
        reconstruct = False

    def compute_p(t, null):
        null = np.sort(null)
        idx = np.searchsorted(null, t, side="left").astype(float)
        return 1 - idx / len(null)

    if tail == "two":
        if symmetric:
            p = compute_p(np.abs(test_value), np.abs(null_array))
        else:
            p_l = compute_p(test_value, null_array)
            p_r = compute_p(test_value * -1, null_array * -1)
            p = 2 * np.minimum(p_l, p_r)
    elif tail == "lower":
        p = compute_p(test_value * -1, null_array * -1)
    else:
        p = compute_p(test_value, null_array)

    # ensure p_value in the following range:
    # smallest_value <= p_value <= (1.0 - smallest_value)
    smallest_value = np.maximum(np.finfo(float).eps, 1.0 / len(null_array))
    result = np.maximum(smallest_value, np.minimum(p, 1.0 - smallest_value))

    if reconstruct:
        result = result[uniq_idx]

    return result[0] if return_first else result


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
    assert window % 2 == 0, "Window must be divisible by 2."

    weights = np.ones(window) / window
    sma = np.convolve(values, weights, "valid")
    buffer = np.full(shape=int(window / 2), fill_value=np.nan)
    sma = np.hstack((buffer, sma, buffer))[:-1]

    return sma


def average_across_distances(values, distances):
    nonnan_idx = np.where(~np.isnan(values))[0]
    unique_retained_distances, retained_distance_idxs = np.unique(
        distances[nonnan_idx],
        return_inverse=True,
    )
    distance_val_sums = np.bincount(retained_distance_idxs, values[nonnan_idx])
    distance_val_means = distance_val_sums / np.bincount(retained_distance_idxs)
    return distance_val_means, unique_retained_distances


def assess_significance(curve, null_curves, distances, intercept_distance, v2):
    """Measure significance of smoothing curve against null curves.

    Parameters
    ----------
    curve : numpy.ndarray of shape (U,)
        U is distance value.
    null_curves : numpy.ndarray of shape (P, U)
        U is distance value, P is permutation.
    distances : numpy.ndarray of shape (U,)
        Distances in mm.
    intercept_distance : float
    v2 : float

    Returns
    -------
    p_inter : float
        P-value for intercept. Higher intercepts with lower p-values
        indicate both focal and global effects of the QC metric on functional connectivity.
    p_slope : float
        P-value for slope.
        Slope, in this case, is calculated as the intercept's value minus the second location's
        value, so that negative slopes go *up* and positive slopes go *down*.
        Higher slopes (i.e., those going down) with lower p-values
        indicate focal effects of the QC metric on functional connectivity
        (i.e., enhanced local connectivity and decrease long-distance connectivity due to the
        QC measure).
    """
    assert curve.ndim == distances.ndim == 1
    assert null_curves.ndim == 2
    assert curve.shape[0] == null_curves.shape[1] == distances.shape[0]

    intercept = get_val(distances, curve, intercept_distance)
    slope = intercept - get_val(distances, curve, v2)
    perm_intercepts = get_val(distances, null_curves, intercept_distance)
    perm_slopes = perm_intercepts - get_val(distances, null_curves, v2)

    p_inter = null_to_p(intercept, perm_intercepts, tail="upper")
    p_slope = null_to_p(slope, perm_slopes, tail="upper")
    return p_inter, p_slope
