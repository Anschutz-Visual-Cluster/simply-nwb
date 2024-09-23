import numpy as np


class DirectionalClassifier(object):
    def is_too_noisy(self, full_wv):
        if np.min(full_wv) < 0:
            full_wv = full_wv + np.min(full_wv)*-1
        velocity = np.diff(full_wv)

        pos_rnge = np.abs(np.max(full_wv) - np.min(full_wv))
        vel_rnge = np.abs(np.max(velocity) - np.min(velocity))
        full_wv = full_wv / pos_rnge

        def checksubwv(waveform, rnge):
            std = np.std(waveform)
            ratio = np.abs(std/rnge)

            if ratio > .15:
                return True

            return False

        def checkbounds(wv):
            def checkside(waveform):
                mean = np.mean(waveform)
                std = np.mean(waveform)
                mx = np.max(full_wv)
                mn = np.min(full_wv)
                if np.abs(mx - mean) < np.abs(mn - mean):
                    is_max = True
                else:
                    is_max = False

                if is_max:
                    val = np.max(full_wv)
                else:
                    val = np.min(full_wv)
                diff = np.abs(val - mean)/std
                if diff > .15:
                    return True, is_max
                return False, is_max
            start, smax = checkside(wv[:30])
            end, emax = checkside(wv[-30:])
            if (emax and smax) or (not smax and not emax):
                return True  # Both are close to the same value
            return start and end

        cases = [
            checksubwv(full_wv[:30], pos_rnge),
            checksubwv(full_wv[-30:], pos_rnge),
            checksubwv(velocity[:30], vel_rnge),
            checksubwv(velocity[-30:], vel_rnge),
            checkbounds(full_wv)
        ]
        result = any(cases)
        return result

    def predict(self, data):
        # data is (saccadenum, t)
        predicts = []
        for idx in range(data.shape[0]):
            sacc = data[idx, :]
            # import matplotlib.pyplot as plt
            # plt.plot(sacc)
            # plt.show()
            if self.is_too_noisy(sacc):
                predicts.append(0)
                continue

            start_mean = np.mean(sacc[:35])
            end_mean = np.mean(sacc[-35:])
            # pos is temporal=-1 (end - start), neg is nasal=1, other is noise
            diff = end_mean - start_mean
            if diff >= 0:
                predicts.append(-1)
            else:
                predicts.append(1)

        return np.array(predicts)[:, None]
