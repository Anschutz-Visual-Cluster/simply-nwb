import base64
import os
import random
from io import BytesIO

import numpy as np
from sklearn.neural_network import MLPClassifier

from simply_nwb import SimpleNWB
from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.enrichments.saccades import PredictSaccadesEnrichment, PutativeSaccadesEnrichment
import pickle
import warnings

from simply_nwb.pipeline.util.models import ModelReader
from simply_nwb.transforms import eyetracking_load_dlc, csv_load_dataframe


class PredictSaccadeMLEnrichment(PredictSaccadesEnrichment):
    def __init__(self, direction_model=None, epoch_nasal_regressor=None, epoch_temporal_regressor=None, epoch_nasal_classifier=None, epoch_temporal_classifier=None):
        data = [  # Order matches superclass init arg order
            [direction_model, "direction_model"],
            [epoch_temporal_regressor, "epoch_temporal_regressor"],
            [epoch_temporal_classifier, "epoch_temporal_classifier"],
            [epoch_nasal_regressor, "epoch_nasal_regressor"],
            [epoch_nasal_classifier, "epoch_nasal_classifier"]
        ]

        args = []
        for value, name in data:
            if value is None:
                value = ModelReader.get_model(name)  # TODO Lazy load these
            else:
                if not isinstance(value, str):  # Assume value passed in is a filename
                    raise ValueError(f"Invalid arg '{name}' Expected a filepath!")
                with open(value, "rb") as f:
                    value = pickle.load(f)

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
    def retrain(training_datas: list[tuple[str, str, str]], save_filename: str, xkey="center_x", ykey="center_y", likeli="center_likelihood", save_to_default_model=False):
        """
        Re-train a model using given a list of training data files like
        [('saccade_times.csv', 'timestamps.csv', dlc.csv'), ...] where each (..) is it's own training dataset

        'saccade_times.csv' is formatted in csv like the following, this is a list of filenames that are manually
        labeled csvs used to train a model

        time,eye_pos,nasal0_temporal1,original_time
        14148,-1.850164897,1,0
        ...

        'timestamps.txt' and 'dlc.csv' are outputs from DLC, 'dlc.csv' is the eye positionss
        save_to_default_model is used to include the model in the package by default, writing to default_model.py file
        """

        training_x = []  # will be a numpy array of size (240, N) N = training samples given
        training_y = []  # numpy array like (N,) with
        print("Loading training data..")
        for labeled_csv, timestamps_txt, dlc_csv in training_datas:
            print(f"Loading '{labeled_csv}'..")
            # raw_eyepos = eyetracking_load_dlc(dlc_csv)[xkey].to_numpy()
            labeled = csv_load_dataframe(labeled_csv)  # time,eyepos,nasaltemporal,orig_time (note columns may not be named the same, just assume order is correct
            labeled_cols = list(labeled.columns)
            timecol = labeled_cols[0]  # Time column name
            direction_col = labeled_cols[2]  # name of column that decides nasal v temporal, 0 = nasal, 1 = temporal
            directions = labeled[direction_col].to_numpy()
            directions = ((directions*2)-1)*-1  # Swap 0 -> 1 and 1 -> -1
            waveform_windows = []  # [[start, end], ..] of each saccade, used to find noise from nonlabeled
            # current standard temporal is -1, nasal is 1 need to convert the 0,1

            # with open(timestamps_txt, "r") as f:
            #     timestamps = {int(t.strip()): idx for idx, t in enumerate(f.readlines())}
            sess = NWBSession(SimpleNWB.test_nwb())
            sess.enrich(PutativeSaccadesEnrichment.from_raw(
                sess.nwb, dlc_csv, timestamps_txt,
                units=["idx", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood"],
                x_center=xkey,
                y_center=ykey,
                likelihood=likeli
            ))
            # Process all labeled saccades
            eyepos = sess.pull("PutativeSaccades.processed_eyepos")[:, 0]  # Grab first dim since its x/y
            likelihoods = sess.pull("PutativeSaccades.raw_likelihoods")
            for idx, labelval in enumerate(labeled[timecol].to_numpy()):
                if labelval + 80 > len(eyepos):
                    warnings.warn(f"Found a labeled saccade outside of the eyeposition index range, not using for training! Index: '{labelval}'")
                    continue
                startidx = labelval
                endidx = startidx + 80

                # If we wanted to make a model including the likelyhoods, this could be useful
                # waveform = [*eyepos[startidx:endidx], *likelihoods[startidx:endidx]]
                waveform = eyepos[startidx:endidx]
                direction = directions[idx]

                training_x.append(waveform)
                training_y.append(direction)
                waveform_windows.append([startidx, endidx])

            # Sample noise for training
            noise = PredictSaccadeMLEnrichment.select_noise_waveforms(waveform_windows, eyepos, likelihoods, len(waveform_windows))
            training_x.extend(noise)
            training_y.extend([0]*len(noise))  # 0 for noise

            tw = 2
        print("Starting Neural Network training, might take a bit..")
        training_x = np.array(training_x)
        training_y = np.array(training_y)

        clf = MLPClassifier(
            activation='tanh',
            solver='adam',
            learning_rate="adaptive",
            # 80 xpoints
            hidden_layer_sizes=(80, 12, 2),
            max_iter=10000,
            verbose=True,
            shuffle=True,
            n_iter_no_change=1000
        )

        clf.out_activation_ = "softmax"
        clf.fit(training_x, training_y)

        if save_to_default_model:
            byts = pickle.dumps(clf)
            b64 = base64.b64encode(byts)
            with open("default_model.py", "w") as f:
                f.write(f"MODEL_DATA = {str(b64)}")
        else:
            with open(save_filename, "wb") as f:
                pickle.dump(clf, f)

        return clf

    @staticmethod
    def select_noise_waveforms(waveform_windows, eyepos, likelihoods, num_samples):
        count = 0
        loops = 0

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
            idx = random.randint(0, len(eyepos))
            for wind in waveform_windows:
                if within(idx, wind) or within(idx + 80, wind):
                    found = True
                    break
            if not found:
                # samples.append([*eyepos[idx:idx+80], *likelihoods[idx:idx+80]])  # TODO include likelihoods?
                samples.append(eyepos[idx:idx + 80])
                count = count + 1

