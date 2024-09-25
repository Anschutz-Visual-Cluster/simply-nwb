import numpy as np

from simply_nwb.pipeline.util.saccade_gui.consts import PERISACCADIC_WINDOW_IN_SECONDS

BASELINE_IDX_LEN = 30


def flip_waveforms(orig_wvs):
    return np.copy(orig_wvs)[:, ::-1]


def mirror_waveforms(orig_wvs):
    baselines = np.mean(orig_wvs[:, :BASELINE_IDX_LEN], axis=1)
    mirrored = ((np.copy(orig_wvs) - baselines[:, None]) * -1) + baselines[:, None]
    return mirrored


def noise_waveforms(orig_wvs):
    stds = np.std(orig_wvs[:, :BASELINE_IDX_LEN], axis=1)
    noise_wvs = np.copy(orig_wvs)
    for idx in range(noise_wvs.shape[0]):
        noise_wvs[idx] += np.random.normal(0, stds[idx] / 2, size=noise_wvs.shape[1])
    return noise_wvs


def scale_waveforms(orig_wvs, scale_factor):
    return np.copy(orig_wvs) * scale_factor


class DirectionDataGenerator(object):

    def __init__(self, waveforms, predictions):
        self.wv = waveforms  # (numsaccades, time)
        self.pred = predictions  # [[0],[1],[-1], ..etc]

    def _to_nearest_one(self, val):
        if val < 0:
            return -1
        elif val > 0:
            return 1
        else:
            return 0

    def _flip(self, orig_wvs, orig_preds) -> tuple[np.ndarray, np.ndarray]:
        flipped = flip_waveforms(orig_wvs)
        flipped_pred = np.copy(orig_preds) * -1
        # import matplotlib.pyplot as plt
        # [plt.plot(f, color="orange") for f in flipped]
        # [plt.plot(f, color="blue") for f in orig_wvs]
        # plt.show()
        return flipped, flipped_pred

    def _mirror(self, orig_wvs, orig_preds) -> tuple[np.ndarray, np.ndarray]:
        mirrored = mirror_waveforms(orig_wvs)
        mirror_pred = np.copy(orig_preds) * -1
        # import matplotlib.pyplot as plt
        # [plt.plot(f, color="orange") for f in mirrored]
        # [plt.plot(f, color="blue") for f in orig_wvs]
        # plt.show()
        return mirrored, mirror_pred

    def _noise(self, orig_wvs, orig_preds):
        noise_wvs = noise_waveforms(orig_wvs)
        # import matplotlib.pyplot as plt
        # [plt.plot(f, color="orange") for f in noise_wvs]
        # [plt.plot(f, color="blue") for f in orig_wvs]
        # plt.show()
        return noise_wvs, orig_preds

    def _scaling_funcs(self, scale_factor_list):
        funcs = []
        for scale_factor in scale_factor_list:
            def func(orig_wvs, orig_preds):
                wvs = scale_waveforms(orig_wvs, scale_factor)
                preds = np.copy(orig_preds) * self._to_nearest_one(scale_factor)
                return wvs, preds
            funcs.append(func)

        return funcs

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        funcs = [
            self._flip,
            self._noise,
            self._mirror,
            self._noise,
            *self._scaling_funcs([.1, 1.8, .5]),
            self._noise
        ]

        wvs = self.wv
        preds = self.pred
        for func in funcs:
            w, p = func(wvs, preds)
            wvs = np.vstack([wvs, w])
            preds = np.vstack([preds, p])

        # import matplotlib.pyplot as plt
        # [plt.plot(f) for f in wvs[preds[:, 0] == 1]]
        # plt.title("preds == 1")
        # plt.show()
        # [plt.plot(f) for f in wvs[preds[:, 0] == -1]]
        # plt.title("preds == -1")
        # plt.show()
        # [plt.plot(f) for f in wvs[preds[:, 0] == 0]]
        # plt.title("preds == 0")
        # plt.show()
        #
        # distrib = np.mean(wvs[preds[:, 0] == 1][:, :30], axis=1)
        # plt.hist(distrib, bins=100)
        # plt.show()
        # amounts = np.unique(preds, return_counts=True)
        return wvs, preds


class EpochDataGenerator(object):
    def __init__(self, waveforms, epochs):
        self.wv = waveforms
        self.ep = epochs

    def _flip(self, orig_wvs, orig_ep):
        flip = flip_waveforms(orig_wvs)
        flip_ep = []

        for idx in range(orig_ep.shape[0]):
            start = orig_ep[idx][0]
            end = orig_ep[idx][1]
            assert start < 0, "Start epoch must be negative to flip around center of saccadic window"
            assert end > 0, "End epoch must be positive to flip around center of saccadic window"
            flip_ep.append([end * -1, start * -1])
        flip_ep = np.array(flip_ep)

        # import matplotlib.pyplot as plt
        # [plt.plot(f) for f in orig_wvs]
        # [(plt.vlines(f[0] * 200 + 40, -5, 5, color="green"), plt.vlines(f[1] * 200 + 40, -5, 5, color="red")) for f in orig_ep]
        # plt.show()
        # # Flipped
        # [plt.plot(f) for f in flip]
        # [(plt.vlines(f[0] * 200 + 40, -5, 5, color="green"), plt.vlines(f[1] * 200 + 40, -5, 5, color="red")) for f in flip_ep]
        # plt.show()
        return flip, flip_ep  # flipping is over yaxis so is reversed epochs

    def _mirror(self, orig_wvs, orig_ep):
        mirror = mirror_waveforms(orig_wvs)
        # import matplotlib.pyplot as plt
        # [plt.plot(f) for f in mirror]
        # [(plt.vlines(f[0] * 200 + 40, -5, 5, color="green"), plt.vlines(f[1] * 200 + 40, -5, 5, color="red")) for f in orig_ep]
        # plt.show()
        return mirror, orig_ep  # Mirroring doesn't change epoch ordering, mirroring over xaxis

    def _noise(self, orig_wvs, orig_ep):
        return noise_waveforms(orig_wvs), orig_ep

    def _scaling_funcs(self, scale_factor_list):
        funcs = []
        for scale_factor in scale_factor_list:
            def func(orig_wvs, orig_eps):
                wvs = scale_waveforms(orig_wvs, scale_factor)
                return wvs, orig_eps
            funcs.append(func)

        return funcs

    def generate(self):
        funcs = [
            self._flip,
            self._noise,
            self._mirror,
            self._noise,
            *self._scaling_funcs([.1, 1.8, .5]),
            self._noise
        ]

        wvs = self.wv
        eps = self.ep
        for func in funcs:
            w, ep = func(wvs, eps)
            wvs = np.vstack([wvs, w])
            eps = np.vstack([eps, ep])

        return wvs, eps
