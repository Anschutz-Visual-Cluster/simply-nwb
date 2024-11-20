import numpy as np


def startstop_of_squarewave(arr: np.ndarray, minval=None, maxval=None, epsilon=.000003) -> np.ndarray:
    """
    Take a square wave and find the start and stop times of each pulse, along with the state
    So a wave lke ⨅__⨅__⨅ would give [(0,1,1),(1,3,0),(3,4,1),(4,5,0)]


    :param arr: Array of the squarewave
    :param minval: minimum value for the state change threshold, if None will infer
    :param maxval: maximum value for the state change threshold, if None will infer
    :param epsilon: minimum difference between minval/maxval and current index of the wave

    :returns: np.ndarray of shape (num states, 2, 1)
    """
    assert len(arr.shape) == 1, "array must be one dimensional"
    if minval is None:
        minval = np.min(arr)
    if maxval is None:
        maxval = np.max(arr)

    startstop = []
    foundedge = 0  # 0 is no edge, -1 is min, 1 is max
    current_window = [0]  # [start, stop]

    for idx, value in enumerate(arr):
        if np.abs(value - minval) < epsilon:
            current_edge = -1
        elif np.abs(value - maxval) < epsilon:
            current_edge = 1
        else:
            current_edge = 0
        if idx == 0:
            foundedge = current_edge

        if current_edge * -1 == foundedge and current_edge != 0 or idx == len(arr) - 1:  # We have flipped (or ended)
            current_window.append(idx)
            startstop.append([*current_window, foundedge])  # Append [start, stop, state]
            foundedge = current_edge
            current_window = [idx]

        tw = 2
    result = np.array(startstop)

    # tmp = np.zeros(len(arr))
    # for ss in startstop:
    #     tmp[ss[0]:ss[1] + 1] = ss[2]
    #
    # import plotly.express as px
    # px.line(tmp).show()

    return result
