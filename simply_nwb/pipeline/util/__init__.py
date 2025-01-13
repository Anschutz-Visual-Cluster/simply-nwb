import numpy as np


def resample_interp(myarr, num_feat):
    """
    Create a linear space of equally spaced parts total num_feat
    interpolate these values, essentially resampling the given array
    if num_feat > myarr.size, this is an 'up sampling'
    elif num_feat <= myarr.size, this is a 'down sampling'

    :param myarr:
    :param num_feat:
    :return: interpolated x values, matching len interpolated y values
    """
    interpd_xs = np.linspace(0, myarr.size - 1, num_feat)
    actual_xs = np.linspace(0, myarr.size - 1, myarr.size)
    actual_ys = myarr

    interpd_ys = np.interp(interpd_xs, actual_xs, actual_ys)
    return interpd_xs, interpd_ys


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


class SkippedListDict(object):
    def __init__(self, data: dict[str, list], skip_idx: int):
        self.data = data
        self.skip_idx = skip_idx

    def __getitem__(self, item):
        # return self.data[item][self.skip_idx:]
        return self.data[item]


# class LazyLoadObj(object):
#     def __init__(self, cls, *args, **kwargs):
#         self._asdf_asdf_args = args
#         self._asdf_asdf_kwargs = kwargs
#         self._asdf_asdf_cls = cls
#         self._asdf_asdf_instance = None
#
#     def __getattribute__(self, name):
#         if name.startswith("_asdf_asdf_"):
#             return object.__getattribute__(self, name)
#         object.__getattribute__(self, "_asdf_asdf_load")()
#         return object.__getattribute__(object.__getattribute__(self, "_asdf_asdf_instance"), name)
#
#     def _asdf_asdf_load(self):
#         inst = object.__getattribute__(self, "_asdf_asdf_instance")
#         if inst is None:
#             cls = object.__getattribute__(self, "_asdf_asdf_cls")
#             args = object.__getattribute__(self, "_asdf_asdf_args")
#             kwargs = object.__getattribute__(self, "_asdf_asdf_kwargs")
#             object.__setattr__(self, "_asdf_asdf_instance", cls(*args, **kwargs))
#
#
# def load_lazy_obj(obj: LazyLoadObj):
#     obj._asdf_asdf_load()
#     return obj._asdf_asdf_instance
