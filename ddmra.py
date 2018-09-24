"""
Perform distance-dependent motion-related artifact analyses.
"""
import numpy as np
from nilearn import input_data, datasets
from scipy import stats
from scipy.spatial.distance import pdist, squareform


def get_val(x_arr, y_arr, x_val):
    """
    Perform interpolation to get value from y_arr at index of x_val based on
    x_arr.
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
                y_val[i] = np.interp(xp=x_arr[temp_idx],
                                     fp=temp_y_arr[temp_idx], x=x_val)
        else:
            y_val = np.interp(xp=x_arr[temp_idx], fp=y_arr[temp_idx], x=x_val)
    return y_val


def rank_p(test_value, null_array, tail='two'):
    """
    Return rank-based p-value for test value against null array.
    """
    if tail == 'two':
        p_value = (50 - np.abs(stats.percentileofscore(null_array, test_value) - 50.)) * 2. / 100.
    elif tail == 'upper':
        p_value = 1 - (stats.percentileofscore(null_array, test_value) / 100.)
    elif tail == 'lower':
        p_value = stats.percentileofscore(null_array, test_value) / 100.
    else:
        raise ValueError('Argument "tail" must be one of ["two", "upper", "lower"]')
    return p_value


def fast_pearson(X, y):
    """
    Fast correlations between y and each row of X. For QC-RSFC. Checked for
    accuracy. From http://qr.ae/TU1B9C
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


def get_fd(motion):
    # assuming rotations in degrees
    motion[:, :3] = motion[:, :3] * (np.pi/180.) * 50
    motion = np.vstack((np.array([[0, 0, 0, 0, 0, 0]]),
                        np.diff(motion, axis=0)))
    fd = np.sum(np.abs(motion), axis=1)
    return fd


def moving_average(values, window):
    """Calculate running average along values array
    """
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    buff = np.zeros(int(window / 2)) * np.nan
    sma = np.hstack((buff, sma, buff))[:-1]
    return sma


def scrubbing_analysis(group_timeseries, qcs, n_rois, qc_thresh=0.2, perm=True):
    """
    Perform Power scrubbing analysis. Note that correlations from scrubbed
    timeseries are subtracted from correlations from unscrubbed timeseries,
    which is the opposite to the original Power method. This reverses the
    signs of the results, but makes inference similar to that of the
    QCRSFC and high-low motion analyses.

    Parameters
    ----------
    group_timeseries : list
        List of (R, T) arrays
    qcs : list
        List of (T,) arrays
    """
    mat_idx = np.triu_indices(n_rois, k=1)
    n_pairs = len(mat_idx[0])
    n_subjects = len(group_timeseries)
    delta_rs = np.zeros((n_subjects, n_pairs))
    c = 0  # included subject counter
    for subj in range(n_subjects):
        ts_arr = group_timeseries[subj]
        qc_arr = qcs[subj]
        keep_idx = qc_arr <= qc_thresh
        # Subjects with no timepoints excluded or with more than 50% excluded
        # will be excluded from analysis.
        prop_incl = np.sum(keep_idx) / qc_arr.shape[0]
        if (prop_incl >= 0.5) and (prop_incl != 1.):
            scrubbed_ts = ts_arr[:, keep_idx]
            raw_corrs = np.corrcoef(ts_arr)
            raw_corrs = raw_corrs[mat_idx]
            scrubbed_corrs = np.corrcoef(scrubbed_ts)
            scrubbed_corrs = scrubbed_corrs[mat_idx]
            delta_rs[c, :] = raw_corrs - scrubbed_corrs  # opposite of Power
            c += 1

    if perm is False:
        print('{0} of {1} subjects retained in scrubbing '
              'analysis'.format(c, n_subjects))
    delta_rs = delta_rs[:c, :]
    mean_delta_r = np.mean(delta_rs, axis=0)
    return mean_delta_r


def run(imgs, qcs, n_iters=10000, qc_thresh=0.2, window=1000):
    """
    Run scrubbing, high-low motion, and QCRSFC analyses.

    Parameters
    ----------
    imgs : list of img-like
        List of 4D (X x Y x Z x T) images in MNI space.
    qcs : list of array-like
        List of 1D (T) numpy arrays with QC metric values per img (e.g., FD or
        respiration).
    n_iters : int
        Number of iterations to run to generate null distributions.
    qc_thresh : float
        Threshold for QC metric used in scrubbing analysis. Default is 0.2
        (for FD).
    window : int
        Number of units (pairs of ROIs) to include when averaging to generate
        smoothing curve.

    Returns
    -------
    results : dict
        Dictionary of results arrays:
        - sorted_dists: Sorted distances between ROIs
        - [name]_y: Measure value for each pair of ROIs.
            Same order as sorted_dists.
        - [name]_smc: Smoothing curve for analysis. Same size as other
            arrays, but contains NaNs for ROI pairs outside of window.
            Same order as sorted_dists.
        - [name]_null: Null distribution smoothing curves. Contains NaNs for
            ROI pairs outside of window. Same order as sorted_dists.
    """
    assert len(imgs) == len(qcs)
    n_subjects = len(imgs)
    atlas = datasets.fetch_coords_power_2011()
    coords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T
    n_rois = coords.shape[0]
    mat_idx = np.triu_indices(n_rois, k=1)
    dists = squareform(pdist(coords))
    dists = dists[mat_idx]
    sort_idx = dists.argsort()
    sorted_dists = dists[sort_idx]

    t_r = imgs[0].header.get_zooms()[-1]
    spheres_masker = input_data.NiftiSpheresMasker(
        seeds=coords, radius=5., t_r=t_r, smoothing_fwhm=4., detrend=False,
        standardize=False, low_pass=None, high_pass=None)

    # prep for qcrsfc and high-low motion analyses
    mean_qcs = np.array([np.mean(qc) for qc in qcs])
    raw_corr_mats = np.zeros((n_subjects, len(mat_idx[0])))

    # Get correlation matrices
    ts_all = []
    for i_sub in range(n_subjects):
        raw_ts = spheres_masker.fit_transform(imgs[i_sub]).T
        ts_all.append(raw_ts)
        raw_corrs = np.corrcoef(raw_ts)
        raw_corrs = raw_corrs[mat_idx]
        raw_corr_mats[i_sub, :] = raw_corrs
    z_corr_mats = np.arctanh(raw_corr_mats)

    # QC:RSFC r analysis
    # For each pair of ROIs, correlate the z-transformed correlation
    # coefficients across subjects with the subjects' mean QC (generally FD)
    # values.
    qcrsfc_rs = fast_pearson(z_corr_mats.T, mean_qcs)
    qcrsfc_rs = qcrsfc_rs[sort_idx]
    qcrsfc_smc = moving_average(qcrsfc_rs, window)

    # High-low motion analysis
    # Split the sample using a median split of the QC metric (generally mean
    # FD). Then, for each pair of ROIs, calculate the difference between the
    # mean across correlation coefficients for the high motion minus the low
    # motion groups.
    hm_idx = mean_qcs >= np.median(mean_qcs)
    lm_idx = mean_qcs < np.median(mean_qcs)
    hm_mean_corr = np.mean(raw_corr_mats[hm_idx, :], axis=0)
    lm_mean_corr = np.mean(raw_corr_mats[lm_idx, :], axis=0)
    hl_corr_diff = hm_mean_corr - lm_mean_corr
    hl_corr_diff = hl_corr_diff[sort_idx]
    hl_smc = moving_average(hl_corr_diff, window)

    # Scrubbing analysis
    mean_delta_r = scrubbing_analysis(ts_all, qcs, n_rois, qc_thresh, perm=False)
    mean_delta_r = mean_delta_r[sort_idx]
    scrub_smc = moving_average(mean_delta_r, window)

    # Null distributions
    qcs_copy = [qc.copy() for qc in qcs]
    perm_scrub_smc = np.zeros((n_iters, len(dists)))
    perm_qcrsfc_smc = np.zeros((n_iters, len(dists)))
    perm_hl_smc = np.zeros((n_iters, len(dists)))
    for i in range(n_iters):
        # Prep for QC:RSFC and high-low motion analyses
        perm_mean_qcs = np.random.permutation(mean_qcs)

        # QC:RSFC analysis
        perm_qcrsfc_rs = fast_pearson(z_corr_mats.T, perm_mean_qcs)
        perm_qcrsfc_rs = perm_qcrsfc_rs[sort_idx]
        perm_qcrsfc_smc[i, :] = moving_average(perm_qcrsfc_rs, window)

        # High-low analysis
        perm_hm_idx = perm_mean_qcs >= np.median(perm_mean_qcs)
        perm_lm_idx = perm_mean_qcs < np.median(perm_mean_qcs)
        perm_hm_corr = np.mean(raw_corr_mats[perm_hm_idx, :], axis=0)
        perm_lm_corr = np.mean(raw_corr_mats[perm_lm_idx, :], axis=0)
        perm_hl_diff = perm_hm_corr - perm_lm_corr
        perm_hl_diff = perm_hl_diff[sort_idx]
        perm_hl_smc[i, :] = moving_average(perm_hl_diff, window)

        # Scrubbing analysis
        perm_qcs = [np.random.permutation(perm_qc) for perm_qc in qcs_copy]
        perm_mean_delta_r = scrubbing_analysis(ts_all, perm_qcs, n_rois,
                                               qc_thresh, perm=True)
        perm_mean_delta_r = perm_mean_delta_r[sort_idx]
        perm_scrub_smc[i, :] = moving_average(perm_mean_delta_r, window)

    results = {'sorted_dists': sorted_dists,
               'qcrsfc_y': qcrsfc_rs,
               'qcrsfc_smc': qcrsfc_smc,
               'qcrsfc_null': perm_qcrsfc_smc,
               'hl_y': hl_corr_diff,
               'hl_smc': hl_smc,
               'hl_null': perm_hl_smc,
               'scrub_y': mean_delta_r,
               'scrub_smc': scrub_smc,
               'scrub_null': perm_scrub_smc}
    return results
