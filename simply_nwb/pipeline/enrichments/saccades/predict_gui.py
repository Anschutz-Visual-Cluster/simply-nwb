import os
import pickle

import numpy as np
from pynwb import NWBHDF5IO
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from simply_nwb.pipeline import Enrichment
from simply_nwb.pipeline.enrichments.saccades import PredictSaccadesEnrichment, PutativeSaccadesEnrichment
from simply_nwb.pipeline.util.saccade_gui.direction import SaccadeDirectionLabelingGUI
from simply_nwb.pipeline.util.saccade_gui.epochs import SaccadeEpochLabelingGUI
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

"""
Use me as a starter point to make your own enrichment
"""


class PredictedSaccadeGUIEnrichment(PredictSaccadesEnrichment):
    NUM_TRAINING_SAMPLES = 5  # Needs to be at least 5

    def __init__(self, recording_fps, list_of_putative_nwbs_filenames):
        super().__init__(None, None, None, None, None)
        self.putat_nwbs = []
        self.putat_nwb_fps = []  # TODO atexit close?

        putat = PutativeSaccadesEnrichment()
        for nwbfn in list_of_putative_nwbs_filenames:
            if not os.path.exists(nwbfn):
                raise ValueError(f"Cannot find file '{nwbfn}'!")
            nwb_fp = NWBHDF5IO(nwbfn, mode="r")
            self.putat_nwb_fps.append(nwb_fp)

            nwb = nwb_fp.read()
            try:
                putat.validate(nwb)
            except Exception as e:
                print(f"ERROR VALIDATING FILE '{nwbfn}'!")
                raise e
            self.putat_nwbs.append(nwb)

        self.recording_fps = recording_fps

    def _get_direction_gui_traindata(self):
        if self._check_for_pretained_direction_model():
            print("Model found, skipping training phase")
            return [0, 0]

        putative_waveforms_list = []

        for nwb in self.putat_nwbs:
            putative_waveforms_list.append(Enrichment.get_val("PutativeSaccades", "saccades_putative_waveforms", nwb))

        putative_waveforms = np.vstack(putative_waveforms_list)
        putative_idxs = np.random.choice(
            np.arange(putative_waveforms.shape[0]),
            size=PredictedSaccadeGUIEnrichment.NUM_TRAINING_SAMPLES,
        )

        gui = SaccadeDirectionLabelingGUI()
        gui.inputSamples(putative_waveforms[putative_idxs, :, 0])  # 0 is x
        while gui.isRunning():
            continue

        return gui.trainingData

    def _get_epoch_gui_traindata(self, pred_waveforms, pred_labels):
        models = self._check_for_epoch_models()
        if models:
            print("Loading pretrained models for epochs..")
            return [[0], [0], [0]]

        pred_idxs = np.random.choice(
            np.arange(pred_waveforms.shape[0]),
            size=PredictedSaccadeGUIEnrichment.NUM_TRAINING_SAMPLES,
        )
        gui = SaccadeEpochLabelingGUI()
        gui.inputSamples(pred_waveforms[pred_idxs, :, 0], pred_labels[pred_idxs])  # 0 is x
        while gui.isRunning():
            continue

        train_x, train_y, train_z = gui.trainingData
        train_y = train_y / self.recording_fps  # Divide by recording fps to get epochs in units of frames
        return train_x, train_y, train_z

    def _format_epoch_trainingdata(self, training_data, direction):
        train_x, train_y, train_z = training_data
        # direction_idxs = np.where(train_z == direction)[0]
        # tx = train_x[direction_idxs]
        # ty = train_y[direction_idxs]
        tx = train_x
        ty = train_y
        return tx, ty

    def _check_for_epoch_models(self):
        prefix = "predict_gui_"
        filenames = [
            "temporal_epoch_regressor",  # temporal is -1
            "temporal_epoch_transformer",
            "nasal_epoch_regressor",  # nasal is 1
            "nasal_epoch_transformer"
        ]
        found_models = []

        for fn in filenames:
            filename = f"{prefix}{fn}.pickle"
            if os.path.exists(filename):
                with open(filename, "rb") as fp:
                    found_models.append(pickle.load(fp))
            else:
                found_models = False
                break
        return found_models

    def _get_epoch_models(self, training_data):
        prefix = "predict_gui_"
        filenames = [
            "temporal_epoch_regressor",  # temporal is -1
            "temporal_epoch_transformer",
            "nasal_epoch_regressor",  # nasal is 1
            "nasal_epoch_transformer"
        ]

        models = self._check_for_epoch_models()
        if models:
            print("Loading pretrained models for epochs..")
            return models

        # Didn't find models, need to train ourselves
        models = []
        for direc in [-1, 1]:
            tx, ty = self._format_epoch_trainingdata(training_data, direc)
            if len(ty) == 0:
                raise ValueError(f"Can't train model with direction = {direc} no epoch training data!")

            reg, tra = self.get_pretrained_epoch_models(tx, ty)
            models.append(reg)
            models.append(tra)

        for idx, fn in enumerate(filenames):
            filename = f"{prefix}{fn}.pickle"
            with open(filename, "wb") as fp:
                pickle.dump(models[idx], fp)
        return models

    def _run(self, pynwb_obj):
        # Directional model training
        print("Collecting directional training data..")
        direction_training_data = self._get_direction_gui_traindata()
        print("Training model..")
        direction_model = PredictedSaccadeGUIEnrichment.get_pretrained_direction_model(*direction_training_data)
        self._direction_cls = direction_model
        print("Predicting directions..")
        pred_labels, pred_waveforms = self._predict_saccade_direction(pynwb_obj)

        # Epoch model training
        print("Collecting directional training data..")
        epoch_training_data = self._get_epoch_gui_traindata(pred_waveforms, pred_labels)
        print("Training models..")
        epoch_models = self._get_epoch_models(epoch_training_data)

        self._temporal_epoch_regressor = epoch_models[0]
        self._temporal_epoch_transformer = epoch_models[1]
        self._nasal_epoch_regressor = epoch_models[2]
        self._nasal_epoch_transformer = epoch_models[3]

        print("Predicting epochs..")
        self._predict_saccade_epochs(pynwb_obj, pred_labels, pred_waveforms)

    @staticmethod
    def _check_for_pretained_direction_model():
        fn = "predict_gui_directional_model.pickle"
        return os.path.exists(fn)

    @staticmethod
    def get_pretrained_direction_model(wv, y_vals):
        fn = "predict_gui_directional_model.pickle"
        if PredictedSaccadeGUIEnrichment._check_for_pretained_direction_model():
            print(f"Pretrained data found at '{fn}' using that..")
            with open(fn, "rb") as fp:
                return pickle.load(fp)

        tmp_wv = np.broadcast_to(wv[:, :, None], shape=(*wv.shape, 2))
        x_velocities, idxs = PredictSaccadesEnrichment.preformat_waveforms(tmp_wv)
        x_velocities = np.array(x_velocities)
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_velocities, y_vals)

        with open(fn, "wb") as fp:
            pickle.dump(lda, fp)

        return lda

    @staticmethod
    def get_pretrained_epoch_models(wv, epoch_labels):
        num_features = 30
        # x vals of the waveforms for training on, handpicked and corresponding to the epoch_labels below
        tmp_wv = np.broadcast_to(wv[:, :, None], shape=(*wv.shape, 2))  # pretend this epoch waveform is a direction to use the same preprocessing func
        training_x_waveforms, idxs = PredictSaccadesEnrichment.preformat_waveforms(tmp_wv, num_features=num_features)

        # Transformer
        transformer = StandardScaler().fit(epoch_labels)
        standardized_epoch_labels = transformer.transform(epoch_labels)

        # Smaller grid for faster (but worse) training, used for testing
        hidden_layer_sizes = [(4,)]
        grid = {
            'estimator__hidden_layer_sizes': hidden_layer_sizes,
            'estimator__max_iter': [
                1000,
            ],
            'estimator__activation': ['tanh'],  # , 'relu'],
            'estimator__solver': ['sgd'],  # , 'adam'],
            'estimator__alpha': [0.0001],  # , 0.05],
            'estimator__learning_rate': ['constant']  # , 'adaptive'],
        }

        # Regressor
        # hidden_layer_sizes = [(int(n),) for n in np.arange(2, num_features, 1)]
        # grid = {
        #     'estimator__hidden_layer_sizes': hidden_layer_sizes,
        #     'estimator__max_iter': [
        #         1000000,
        #     ],
        #     'estimator__activation': ['tanh', 'relu'],
        #     'estimator__solver': ['sgd', 'adam'],
        #     'estimator__alpha': [0.0001, 0.05],
        #     'estimator__learning_rate': ['constant', 'adaptive'],
        # }

        reg = MultiOutputRegressor(MLPRegressor(verbose=True))
        search = GridSearchCV(reg, grid)

        search.fit(training_x_waveforms, standardized_epoch_labels)
        regressor = search.best_estimator_

        return regressor, transformer
