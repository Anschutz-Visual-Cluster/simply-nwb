import base64
import math
import os
import random
from io import BytesIO


import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from spencer_funcs.lazy import LazyObject, lazy_init

from simply_nwb import SimpleNWB
from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.enrichments.saccades import PredictSaccadesEnrichment, PutativeSaccadesEnrichment
import pickle
import warnings

from simply_nwb.pipeline.util.models import ModelReader
from simply_nwb.pipeline.util.saccade_gui.data_generator import DirectionDataGenerator
from simply_nwb.transforms import eyetracking_load_dlc, csv_load_dataframe


class WrappedMLModel(object):
    def __init__(self, model):
        self.model = model
    def predict(self, xvals):
        # Func expects an inputs of (80,) eyepositions
        # Turns those into a (79,) arr of velocities, feeds into model
        return self.raw_predict(np.diff(xvals))

    def raw_predict(self, xvals):
        return self.model.predict(xvals)


class PredictSaccadeMLEnrichment(PredictSaccadesEnrichment):
    def __init__(self, direction_model=None, epoch_nasal_regressor=None, epoch_temporal_regressor=None, epoch_nasal_classifier=None, epoch_temporal_classifier=None):
        data = [  # Order matches superclass init arg order
            [direction_model, "direction_model"],
            [epoch_temporal_regressor, "temporal_epoch_regressor"],
            [epoch_temporal_classifier, "temporal_epoch_transformer"],
            [epoch_nasal_regressor, "nasal_epoch_regressor"],
            [epoch_nasal_classifier, "nasal_epoch_transformer"]
        ]

        args = []
        for value, name in data:
            if value is None:
                value = lazy_init(ModelReader.get_model, name)
            else:
                if not isinstance(value, str):  # Assume value passed in is a filename
                    raise ValueError(f"Invalid arg '{name}' Expected a filepath!")
                with open(value, "rb") as f:
                    value = LazyObject(lambda: pickle.load(f))

            args.append(value)
        super().__init__(*args, use_mlp_input=True)

    @staticmethod
    def _load_model(filepath):
        if not os.path.exists(filepath):
            # TODO auto-create model here
            raise FileNotFoundError(f"Cannot find model file '{filepath}'!")

        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def center_saccade(waveform, start, end, pad_len=40):
        assert start-pad_len > 0, f"Must have padding '{pad_len}' around the window to center! LeftPad Error"
        assert end+pad_len < len(waveform), f"Must have padding '{pad_len}' around the window to center! RightPad Error"

        vel = np.diff(waveform[start:end])  # Bound of 20 error around user-labeled data
        mx = max(vel)
        mn = min(vel)
        val = mn if np.abs(mx) < np.abs(mn) else mx
        center_idx = np.where(vel == val)[0][0] + start

        pstart = center_idx-pad_len
        pend = center_idx+pad_len

        wv = waveform[pstart:pend]
        wvv = np.diff(wv)
        return wvv, pstart, pend

    @staticmethod
    def process_trainingdata(training_datas: list[tuple[str, str, str]], xkey="center_x", ykey="center_y", likeli="center_likelihood", generate_data=True):
        training_x = []  # will be a numpy array of size (N, 80) N = training samples given
        training_y = []  # numpy array like (N,) with
        print("Loading training data..")
        for labeled_csv, timestamps_txt, dlc_csv in training_datas:
            print(f"Loading '{labeled_csv}'..")
            # raw_eyepos = eyetracking_load_dlc(dlc_csv)[xkey].to_numpy()
            labeled = csv_load_dataframe(
                labeled_csv)  # time,eyepos,nasaltemporal,orig_time (note columns may not be named the same, just assume order is correct
            labeled_cols = list(labeled.columns)
            timecol = labeled_cols[0]  # Time column name
            direction_col = labeled_cols[2]  # name of column that decides nasal v temporal, 0 = nasal, 1 = temporal
            directions = labeled[direction_col].to_numpy()
            directions = ((directions * 2) - 1) * -1  # Swap 0 -> 1 and 1 -> -1
            waveform_windows = []  # [[start, end], ..] of each saccade, used to find noise from nonlabeled
            # current standard temporal is -1, nasal is 1 need to convert the 0,1

            # with open(timestamps_txt, "r") as f:
            #     timestamps = {int(t.strip()): idx for idx, t in enumerate(f.readlines())}
            sess = NWBSession(SimpleNWB.test_nwb())
            sess.enrich(PutativeSaccadesEnrichment.from_raw(
                sess.nwb, dlc_csv, timestamps_txt,
                units=["idx", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px",
                       "likelihood", "px", "px", "likelihood"],
                x_center=xkey,
                y_center=ykey,
                likelihood=likeli
            ))
            # Process all labeled saccades
            eyepos = sess.pull("PutativeSaccades.processed_eyepos")[:, 0]  # Grab first dim since its x/y
            likelihoods = sess.pull("PutativeSaccades.raw_likelihoods")
            for idx, labelval in enumerate(labeled[timecol].to_numpy()):
                if labelval + 80 > len(eyepos):
                    warnings.warn(
                        f"Found a labeled saccade outside of the eyeposition index range, not using for training! Index: '{labelval}'")
                    continue
                # If we wanted to make a model including the likelyhoods, this could be useful
                # waveform = [*eyepos[startidx:endidx], *likelihoods[startidx:endidx]]

                # Give a buffer of size 70+120 = 190 -> |--start--<80>--end--| we need to find start/end to center the saccade
                # 80 is the len of the waveform. We start at 60 -> 60 + -70 = -10 and 80 -> 80 - 70 = 10, so we have a
                # window around the labeled spot, -10 before and 10 after, where we check for a peak, then center on it
                #|(-70)----(-10)-----(10)----(120)| space between 10s is the peak finding window, -70 to 120 is buffer

                vel_waveform, rel_start, rel_end = PredictSaccadeMLEnrichment.center_saccade(eyepos[labelval-70:labelval+120], 60, 60+20)  # Center
                start_eyeidx = labelval - 70 + rel_start
                end_eyeidx = labelval - 70 + rel_end  # Subtract 1 since the indicies are meant for velocity calc

                if len(vel_waveform) != 79:
                    tw = 2
                    raise ValueError(
                        f"Invalid vel waveform extracted from training data! Expected waveform to be len 79, actual '{len(vel_waveform)}' In file '{labeled_csv}' Row '{idx}'")

                direction = directions[idx]

                training_x.append(vel_waveform)
                training_y.append(direction)
                waveform_windows.append([start_eyeidx, end_eyeidx])

            # Sample noise for training
            noise = PredictSaccadeMLEnrichment.select_noise_vel_waveforms(waveform_windows, eyepos, likelihoods,
                                                                      len(waveform_windows))
            training_x.extend(noise)
            training_y.extend([0] * len(noise))  # 0 for noise

        training_x = np.array(training_x)
        training_y = np.array(training_y)

        if generate_data:
            print("Generating extra data")
            training_x, training_y = DirectionDataGenerator(training_x, training_y[:, None]).generate()

        return training_x, training_y

    @staticmethod
    def retrain(training_datas: list[tuple[str, str, str]], save_filename: str, xkey="center_x", ykey="center_y", likeli="center_likelihood", save_to_default_model=False, generate_data=True):
        """
        Re-train a model using given a list of training data files like
        [('saccade_times.csv', 'timestamps.txt', dlc.csv'), ...] where each (..) is it's own training dataset

        'saccade_times.csv' is formatted in csv like the following, this is a list of filenames that are manually
        labeled csvs used to train a model

        time,eye_pos,nasal0_temporal1,original_time
        14148,-1.850164897,1,0
        ...

        'timestamps.txt' and 'dlc.csv' are outputs from DLC, 'dlc.csv' is the eye positions
        save_to_default_model is used to include the model in the package by default, writing to direction_model.py file
        which can be copied into simply_nwb/pipeline/util/models to replace the model that comes installed with the
        package. For the epoch models, you can use test/save_models_to_py.py and move those into that folder as well
        """

        # training_x, training_y = PredictSaccadeMLEnrichment.process_trainingdata(training_datas, xkey, ykey, likeli, generate_data=False)
        # training_x = training_x[:20]
        # training_y = training_y[:20]
        training_x, training_y = PredictSaccadeMLEnrichment.process_trainingdata(training_datas, xkey, ykey, likeli, generate_data=generate_data)

        mlp_clf = MLPClassifier(
            activation='tanh',
            solver='adam',
            hidden_layer_sizes=[32, 16, 8, 4],
            learning_rate="adaptive",
            # 80 xpoints
            max_iter=10000,
            verbose=True,
            shuffle=True,
            n_iter_no_change=10000
        )
        mlp_clf.out_activation_ = "softmax"

        # pipe = Pipeline(steps=[
        #     ("scale", StandardScaler()),
        #   # ("mlpc", MultiOutputClassifier(mlp_clf))
        #     ("mlpc", mlp_clf)
        # ])

        # hidden_layer_sizes = [[x, 3] for x in [int(math.pow(2, v)) for v in range(2, 6)]]  # Try layer sizes 4,8,16,32
        # search = GridSearchCV(pipe, {
        #     # 'mlpc__estimator__hidden_layer_sizes': hidden_layer_sizes,
        #     # 'mlpc__estimator__max_iter': [
        #     #     1000,
        #     # ],
        #     # 'mlpc__estimator__activation': ['tanh', 'relu'],
        #     # 'mlpc__estimator__solver': ['adam'],
        #     # 'mlpc__estimator__alpha': [0.0001, 0.05],
        #     # 'mlpc__estimator__learning_rate': ['constant', 'adaptive'],
        #     'mlpc__hidden_layer_sizes': hidden_layer_sizes,
        #     'mlpc__max_iter': [
        #         10000,
        #     ],
        #     'mlpc__activation': ['tanh', 'relu'],
        #     'mlpc__solver': ['adam'],
        #     'mlpc__alpha': [0.0001, 0.05],
        #     'mlpc__learning_rate': ['constant', 'adaptive'],
        # })
        # search = GridSearchCV(pipe, {
        #     'mlpc__max_iter': [
        #         100000,
        #     ],
        #     'mlpc__activation': ['tanh'],
        #     'mlpc__solver': ['adam'],
        #     'mlpc__alpha': [0.0001],
        #     'mlpc__learning_rate': ['adaptive'],
        # })
        #
        # search.fit(training_x, training_y)
        # trained_model = WrappedMLModel(search)

        print("Starting Neural Network training, might take a bit..")
        mlp_clf.fit(training_x, training_y)
        trained_model = WrappedMLModel(mlp_clf)

        if save_to_default_model:
            byts = pickle.dumps(trained_model)
            b64 = base64.b64encode(byts)
            with open("direction_model.py", "w") as f:  # TODO write this directly into the package location? or leave as manually move..
                f.write(f"MODEL_DATA = {str(b64)}")
        else:
            with open(save_filename, "wb") as f:
                pickle.dump(trained_model, f)

        return trained_model, training_x, training_y

    # @staticmethod
    # def select_noise_vel_waveforms(waveform_windows, eyepos, likelihoods, num_samples):
    #     count = 0
    #     loops = 0
    #
    #     def within(x, w):
    #         return (x <= w[1]) and (x >= w[0])
    #
    #     samples = []
    #
    #     while True:  # Could make this better but its not worth the time lol
    #         loops = loops + 1
    #         if loops >= 10000000:
    #             raise ValueError("Unable to sample noise waveforms!")  # TODO if this happens fix the sampling lol
    #
    #         if count >= num_samples:
    #             return samples
    #
    #         found = False
    #         idx = random.randint(0, len(eyepos))
    #         for wind in waveform_windows:
    #             if within(idx, wind) or within(idx + 80, wind):
    #                 found = True
    #                 break
    #         if not found:
    #             # samples.append([*eyepos[idx:idx+80], *likelihoods[idx:idx+80]])  # TODO include likelihoods?
    #             samples.append(np.diff(eyepos[idx:idx + 80]))
    #             count = count + 1

    @staticmethod
    def select_noise_vel_waveforms(waveform_windows, eyepos, likelihoods, num_samples):
        """
        Select some waveforms from the full eyeposition to label as noise.
        Takes the peaks and troughs of the eyeposition, then randomly selects one, if that value intersects with
        a labeled saccade, re-pick. Do this until all requested samples are pulled. (could be optimized)
        """

        count = 0
        loops = 0
        eyevel = np.diff(eyepos)  # Eye velocity, forward difference
        up_peaks = find_peaks(eyevel)
        down_peaks = find_peaks(eyevel*-1)

        # Combine and make unique
        peaks = [*list(up_peaks[0]), *list(down_peaks[0])]
        peaks = np.array(list(set(peaks)))

        if len(peaks) < num_samples*3/4:
            raise ValueError("Cannot select noise waveforms, num samples must be at most 3/4 of the peaks found!")

        def within(x, w):
            return (x <= w[1]) and (x >= w[0])

        samples = []

        while True:  # Could make this better but its not worth the time lol
            loops = loops + 1
            if loops >= 10000000:
                raise ValueError("Unable to sample noise waveforms!")  # TODO if this happens fix the sampling lol

            if count >= num_samples:
                return samples

            found = False
            p_idx = random.randint(0, len(peaks))
            idx = peaks[p_idx]

            for wind in waveform_windows:
                if within(idx - 40, wind) or within(idx + 40, wind):
                    found = True
                    break
            if not found:
                # samples.append([*eyepos[idx:idx+80], *likelihoods[idx:idx+80]])  # TODO include likelihoods?
                wv = eyepos[idx - 40:idx + 40]
                if len(wv) == 80:  # .diff will turn 80 -> 79
                    samples.append(np.diff(wv))
                    count = count + 1
