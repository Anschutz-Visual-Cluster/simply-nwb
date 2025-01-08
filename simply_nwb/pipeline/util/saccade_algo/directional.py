import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots
import scipy

# NOT CURRENTLY USED OR FULLY IMPLEMENTED!!
class DirectionalClassifier(object):
    def is_too_noisy(self, full_wv, print_debug=False, plot_debug=False):
        def doprint(text_to_print):
            if print_debug:
                print(text_to_print)

        def to_plotdict(dataval):
            return {"x": np.arange(len(dataval)), "y": dataval}

        if np.min(full_wv) < 0:  # If any part of our waveform is negative, make it positive before standardizing
            full_wv = full_wv + np.min(full_wv)*-1

        pos_rnge = np.abs(np.max(full_wv) - np.min(full_wv))

        full_wv = full_wv / pos_rnge
        velocity = np.diff(full_wv)
        vel_rnge = np.abs(np.max(velocity) - np.min(velocity))
        # px.line(full_wv).show()

        def check_flatness(waveform, full_range):
            # Check that the subwave standard deviation ratio is within 15%
            if plot_debug:
                tw = 2

            std = np.std(waveform)
            ratio = np.abs(std/full_range)  # range is 1 since it is expecting a normalized waveform
            if ratio > .15:
                return True

            subwave_range = np.max(waveform) - np.min(waveform)
            subwave_range_ratio = subwave_range / full_range
            if subwave_range_ratio > .45:  # If the range of our wave is outside the total range, its too noisy
                return True
            return False
        #
        # def checkbounds(wv):
        #     def checkside(waveform):
        #         mean = np.mean(waveform)
        #         std = np.mean(waveform)
        #         mx = np.max(full_wv)
        #         mn = np.min(full_wv)
        #         if np.abs(mx - mean) < np.abs(mn - mean):
        #             is_max = True
        #         else:
        #             is_max = False
        #
        #         if is_max:
        #             val = np.max(full_wv)
        #         else:
        #             val = np.min(full_wv)
        #         diff = np.abs(val - mean)/std
        #         if diff > .15:
        #             return True, is_max
        #         return False, is_max
        #     start, smax = checkside(wv[:30])
        #     end, emax = checkside(wv[-30:])
        #     if (emax and smax) or (not smax and not emax):
        #         return True  # Both are close to the same value
        #     return start and end
        #
        # cases = [
        #     [checksubwv, [full_wv[:30], pos_rnge]],
        #     [checksubwv, [full_wv[-30:], pos_rnge]],
        #     [checksubwv, [velocity[:30], vel_rnge]],
        #     [checksubwv, [velocity[-30:], vel_rnge]],
        #     [checkbounds, [full_wv]]
        # ]

        def check_velocity_peaks(waveform):
            # Ensure that there is only one peak, and that it is large enough comparatively
            waveform_range = np.max(waveform) - np.min(waveform)
            normalized_wave = waveform / waveform_range  # Normalize size to 1
            normalized_waveform_velocity = np.diff(normalized_wave)
            if np.min(normalized_waveform_velocity) < 0:
                normalized_waveform_velocity = normalized_waveform_velocity + np.abs(np.min(normalized_waveform_velocity))

            # Any peaks within 15% of difference between eachother
            smoothed_velocity = scipy.ndimage.gaussian_filter(normalized_waveform_velocity, 4)
            smoothed_velocity = smoothed_velocity / (np.max(smoothed_velocity) - np.min(smoothed_velocity))
            smoothed_velocity = smoothed_velocity + np.abs(np.min(smoothed_velocity))
            smoothed_velocity = smoothed_velocity - np.mean(smoothed_velocity)
            if np.abs(np.min(smoothed_velocity)) > np.abs(np.max(smoothed_velocity)):
                smoothed_velocity = smoothed_velocity * -1  # Flip to detect the trough as a peak instead
            peaks, properties = scipy.signal.find_peaks(smoothed_velocity, prominence=(None, 1))
            if plot_debug:
                fig = plotly.subplots.make_subplots(rows=2, cols=1)
                fig.add_trace(go.Scatter(**to_plotdict(waveform), fillcolor="blue", name="raw waveform"), row=1, col=1)
                fig.add_trace(go.Scatter(**to_plotdict(smoothed_velocity), fillcolor="red", name="smoothed normalized velocity (flipped)"), row=2, col=1)
                fig.add_trace(go.Scatter(**to_plotdict(normalized_waveform_velocity), fillcolor="green", name="normalized raw velocity"), row=2, col=1)
                fig.show()
                tw = 2

            # Normalize the prominence values so that the highest is 1, to ensure there is only one peak we don't want any peaks with prominence ratio > .25
            max_prominence = np.max(properties["prominences"])
            prom_ratio = properties["prominences"] / max_prominence
            doprint(f"Peaks found: {peaks}")
            doprint(f"Prominences: {properties['prominences']}")
            doprint(f"Prominence Ratio: {prom_ratio}")
            proms = np.where(prom_ratio > .25)[0]
            if len(proms) > 1:
                return True
            else:
                # Need to check flipped, in case it is a multi peaked/trough'd

                if plot_debug:
                    tw = 2

                flipped_peaks, flipped_properties = scipy.signal.find_peaks(-1*smoothed_velocity, prominence=(None, 1))
                flipped_ratio = flipped_properties["prominences"] / max_prominence
                flipped = np.where(flipped_ratio > .25)[0]
                if len(flipped) > 0:  # We have a waveform that has a significant trough and peak
                    return True

                return False

        full_rnge = np.max(full_wv) - np.min(full_wv)
        cases = [
            ["check_velocity_names", check_velocity_peaks, [full_wv]],
            ["beginning_flatness", check_flatness, [full_wv[:30], full_rnge]],
            ["ending_flatness", check_flatness, [full_wv[-30:], full_rnge]],
        ]
        results = []
        for case in cases:
            name, func, args = case
            doprint("-"*20)
            doprint(name)
            result = func(*args)
            doprint(f"Noisy: {result}")
            if result:
                tw = 2
                # func(*args)
                doprint("-" * 20)
                return True

        tw = 2
        for case in cases:
            name, func, args = case
            # result = func(*args)

        doprint("-" * 20)
        return False

    def predict(self, data):
        # data is (saccadenum, t)
        predicts = []
        for idx in range(data.shape[0]):
            sacc = data[idx, :]
            # import matplotlib.pyplot as plt
            # plt.plot(sacc)
            # plt.show()
            if self.is_too_noisy(sacc, True, True):
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

        noise = data[np.where(np.array(predicts) == 0)[0]]
        pos = data[np.where(np.array(predicts) == 1)[0]]
        neg = data[np.where(np.array(predicts) == -1)[0]]

        def show(arr):
            px.line(arr.T).show()

        def debug_run(arr, num):
            self.is_too_noisy(arr[num, :], True, True)

        return np.array(predicts)[:, None]
