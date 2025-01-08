import warnings
import numpy as np
warnings.simplefilter("always", UserWarning)  # Python being fucking stupid and not showing multiple warnings unless explicitly being told to


def startstop_of_squarewave(arr: np.ndarray, minval=None, maxval=None, epsilon=.000003, handle_dropped_signal=True, dropped_width=3, warn_on_large_gaps=1.3) -> np.ndarray:
    """
    Take a square wave and find the start and stop times of each pulse, along with the state
    So a wave lke ⨅__⨅__⨅ would give [(0,1,1),(1,3,0),(3,4,1),(4,6,0),(6,7,1)]


    :param arr: Array of the squarewave
    :param minval: minimum value for the state change threshold, if None will infer
    :param maxval: maximum value for the state change threshold, if None will infer
    :param epsilon: minimum difference between minval/maxval and current index of the wave
    :param handle_dropped_signal: Sometimes during a pulse a signal can be dropped. e.g. a signal that is supposed to look like 000111000 can end up as 000101000 and will need to be corrected
    :param dropped_width: Any change in signal less than or equal to 3 indexes long will be smoothed out
    :param warn_on_large_gaps: If set, if there is any gap larger than 'warn_on_large_gaps' times larger than the median gap size, it will throw a warning message, set to None to silence

    :returns: np.ndarray of shape (num states, 2, 1)
    """
    assert len(arr.shape) == 1, "array must be one dimensional"

    if minval is None:
        minval = np.min(arr)
    if maxval is None:
        maxval = np.max(arr)

    startstop = []
    last_found_value = 0  # 0 is no edge, -1 is min, 1 is max
    current_window = [0]  # [start, stop]
    close_to = lambda val1, val2: np.abs(val1 - val2) < epsilon

    def check_flipped(chk_arr, chk_idx):
        if chk_idx >= len(chk_arr):  # If we try to check ahead of our arr len, return no change (essentially padding end)
            return 0
        chk_val = chk_arr[chk_idx]

        if close_to(chk_val, minval):
            cur_edge = -1
        elif close_to(chk_val, maxval):
            cur_edge = 1
        else:
            cur_edge = 0
        return cur_edge

    for idx, value in enumerate(arr):
        current_edge = check_flipped(arr, idx)

        if idx == 0:
            last_found_value = current_edge

        if current_edge * -1 == last_found_value and current_edge != 0 or idx == len(arr) - 1:  # We have flipped (or ended)
            if handle_dropped_signal:
                skip = False
                for look_ahead_idx in range(dropped_width):  # Look ahead spaces to see if the signal recovers or is truly changing
                    newflip = check_flipped(arr, idx + look_ahead_idx)
                    if newflip == last_found_value:  # Within our window the signal has returned to the previous value, need to smooth out by skipping this edge
                        skip = True
                        break
                if skip:
                    continue

            current_window.append(idx)
            startstop.append([*current_window, last_found_value])  # Append [start, stop, state]
            last_found_value = current_edge
            current_window = [idx]

        tw = 2
    result = np.array(startstop)

    # Sanity checks
    base_states = result[np.where(result[:,2] == -1)]

    if warn_on_large_gaps is not None:
        gaps = base_states[:, 1] - base_states[:, 0]
        median = np.median(gaps)
        threshold = median * warn_on_large_gaps
        large_gap_idxs = np.where(gaps > threshold)[0]
        if len(large_gap_idxs) > 0:
            warnings.warn(f"WARNING! DETECTED '{len(large_gap_idxs)}' LARGE GAPS (gaps larger than {warn_on_large_gaps}x the median gap size) BETWEEN PULSES, DATA MAY BE CORRUPTED!")


    # tmp = np.zeros(len(arr))
    # for ss in startstop:
    #     tmp[ss[0]:ss[1] + 1] = ss[2]
    #
    # import plotly.express as px
    # px.line(tmp).show()

    return result
