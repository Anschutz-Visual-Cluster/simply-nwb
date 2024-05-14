import numpy as np


def interpolate_flat_arr(arr):
    # Given an arr of shape (N,) find nan values and interpolate them
    nan_mask = np.isnan(arr)
    nan_idxs = np.where(nan_mask)[0]
    xvals = np.where(np.invert(nan_mask))[0]  # Where non-nan values are
    yvals = arr[np.invert(nan_mask)]  # Actual values of the array that are not nan

    interp = np.interp(nan_idxs, xvals, yvals)  # get arr of interpolated values
    arr[nan_idxs] = interp  # Fill in original arr with interp vals

    return arr


def smooth_flat_arr(arr, window_size=5, window_type='hanning'):
    """
    Smooth array by convolving with a sliding window

    :param arr: Input array to smooth
    :param window_size: Size of the smoothing window (in samples)
    :param window_type: Type of window to use for smoothing, one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    :returns: Smoothed array
    """

    if window_size < 3:
        raise ValueError('Window size must be odd and greater than or equal to 3')

    window_types = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    if window_type not in window_types:
        raise ValueError(f"Invalid window type: '{window_type}' Must be one of '{window_types}'")

    if window_type == 'flat':
        w = np.ones(window_size, 'd')
    else:
        w = getattr(np, window_type)(window_size)

    if arr.size < window_size:
        raise ValueError('Input array is smaller than the smoothing window')

    s = np.r_[arr[window_size - 1: 0: -1], arr, arr[-2: -window_size - 1: -1]]
    smoothed = np.convolve(w / w.sum(), s, mode='valid')
    centered = smoothed[int((window_size - 1) / 2): int(-1 * (window_size - 1) / 2)]
    a2 = centered
    return a2

