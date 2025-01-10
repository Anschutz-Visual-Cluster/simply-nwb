import math
import os
import pickle

import numpy as np
from pynwb import NWBHDF5IO
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from simply_nwb.pipeline import Enrichment
from simply_nwb.pipeline.enrichments.saccades import PredictSaccadesEnrichment, PutativeSaccadesEnrichment
from simply_nwb.pipeline.util.saccade_algo.directional import DirectionalClassifier
from simply_nwb.pipeline.util.saccade_gui.data_generator import DirectionDataGenerator, EpochDataGenerator
from simply_nwb.pipeline.util.saccade_gui.direction import SaccadeDirectionLabelingGUI
from simply_nwb.pipeline.util.saccade_gui.epochs import SaccadeEpochLabelingGUI
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class PredictedSaccadeGUIEnrichment(PredictSaccadesEnrichment):
    def __init__(self, recording_fps, list_of_putative_nwbs_filenames, num_training_samples, putative_kwargs):
        super().__init__(None, None, None, None, None)
        self.putat_nwbs = []
        self.putat_nwb_fps = []  # TODO atexit close?
        self.num_training_samples = num_training_samples
        if self.num_training_samples < 20:
            raise ValueError("Must have at least 20 training samples!")

        putat = PutativeSaccadesEnrichment(**putative_kwargs)
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

    def _get_direction_prelabeled_data_name(self):
        return "predict_gui_directional_trainingdata.pickle"

    def _save_direction_prelabeled_data(self, x, y):
        fn = self._get_direction_prelabeled_data_name()
        with open(fn, "wb") as f:
            print("Saving pre-labeled direction data..")
            pickle.dump((x, y), f)

    def _get_direction_prelabeled_data(self):
        fn = self._get_direction_prelabeled_data_name()
        if os.path.exists(fn):
            print("Loading direction labeled data..")
            with open(fn, "rb") as f:
                return pickle.load(f)
        else:
            print("No labeled data found for direction model found..")
            return False

    def _get_direction_gui_traindata(self):
        result = self._get_direction_prelabeled_data()
        if result:
            return result

        putative_waveforms_list = []

        for nwb in self.putat_nwbs:
            putative_waveforms_list.append(Enrichment.get_val("PutativeSaccades", "saccades_putative_waveforms", nwb))
        if len(self.putat_nwbs) == 0:
            raise ValueError("Cannot train model without any putative NWB training sets!")

        putative_waveforms = np.vstack(putative_waveforms_list)
        putative_idxs = np.random.choice(
            np.arange(putative_waveforms.shape[0]),
            size=self.num_training_samples,
        )

        gui = SaccadeDirectionLabelingGUI()
        gui.inputSamples(putative_waveforms[putative_idxs, :, 0])  # 0 is x
        while gui.isRunning():
            continue
        x, y = gui.trainingData

        # debug testing
        # x = np.array([[20.0,40.0]]*len(putative_idxs))
        # y = np.array([1.0]*len(putative_idxs))

        self._save_direction_prelabeled_data(x, y)
        return x, y

    def _get_epoch_prelabeled_data_name(self):
        return "predict_gui_epoch_trainingdata.pickle"

    def _save_epoch_prelabeled_data(self, x, y, z):
        fn = self._get_epoch_prelabeled_data_name()
        with open(fn, "wb") as f:
            print("Saving pre-labeled epoch data..")
            pickle.dump((x, y, z), f)

    def _get_epoch_prelabeled_data(self):
        fn = self._get_epoch_prelabeled_data_name()
        if os.path.exists(fn):
            print("Loading epoch labeled data..")
            with open(fn, "rb") as f:
                return pickle.load(f)
        else:
            print("No labeled data found for epoch model found..")
            return False

    def _get_epoch_gui_traindata(self, pred_waveforms, pred_labels):
        result = self._get_epoch_prelabeled_data()
        if result:
            return result

        gui = SaccadeEpochLabelingGUI()
        nonzero_idxs = np.logical_not(pred_labels == 0)[:, 0]  # Dont label epochs of noise
        gui.inputSamples(pred_waveforms[nonzero_idxs], pred_labels[nonzero_idxs])
        while gui.isRunning():
            continue

        train_x, train_y, train_z = gui.trainingData
        train_y = train_y / self.recording_fps  # Divide by recording fps to get epochs in units of frames

        self._save_epoch_prelabeled_data(train_x, train_y, train_z)
        return train_x, train_y, train_z

    def _format_epoch_trainingdata(self, epoch_training_waveforms, epoch_training_epochs, direction):
        # train_x, train_y, train_z = training_data
        # direction_idxs = np.where(train_z == direction)[0]
        # tx = train_x[direction_idxs]
        # ty = train_y[direction_idxs]
        # tx = train_x
        # ty = train_y
        # return tx, ty
        return epoch_training_waveforms, epoch_training_epochs

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

    def _get_epoch_models(self, epoch_training_waveforms, epoch_training_epochs):
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
            tx, ty = self._format_epoch_trainingdata(epoch_training_waveforms, epoch_training_epochs, direc)
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
        print("="*50)
        print("MAKE SURE YOU HAVE AT LEAST 10 EXAMPLES OF LEFT AND RIGHT SACCADES!!")
        print("MAKE SURE YOU LABEL AT LEAST 5 EPOCHS!!")
        print("="*50)

        found_pretrained = PredictedSaccadeGUIEnrichment._check_for_pretained_direction_model()
        epoch_models = self._check_for_epoch_models()
        if not found_pretrained:
            print("Collecting directional training data..")
            direction_userlabeled_waveforms, direction_userlabeled_labels = self._get_direction_gui_traindata()
            print("Generating more training data..")
            direction_training_waveforms, direction_training_labels = DirectionDataGenerator(direction_userlabeled_waveforms, direction_userlabeled_labels).generate()

            print("Training model..")
            self._direction_cls = self.get_pretrained_direction_model(direction_training_waveforms, direction_training_labels)
        else:
            self._direction_cls = self.load_pretrained_direction_model()

        if not epoch_models:
            # Epoch model training
            print("Collecting directional training data..")
            direction_userlabeled_waveforms, direction_userlabeled_labels = self._get_direction_gui_traindata()
            epoch_userlabeled_waveforms, epoch_userlabeled_epochs, ignorezvalfornow = self._get_epoch_gui_traindata(
                direction_userlabeled_waveforms, direction_userlabeled_labels)
            epoch_training_waveforms, epoch_training_epochs = EpochDataGenerator(epoch_userlabeled_waveforms,
                                                                                 epoch_userlabeled_epochs).generate()
            print("Training models..")
            epoch_models = self._get_epoch_models(epoch_training_waveforms, epoch_training_epochs)

        print("Predicting directions..")
        pred_labels, pred_waveforms, pred_sacc_indices = self._predict_saccade_direction(pynwb_obj)

        self._temporal_epoch_regressor = epoch_models[0]
        self._temporal_epoch_transformer = epoch_models[1]
        self._nasal_epoch_regressor = epoch_models[2]
        self._nasal_epoch_transformer = epoch_models[3]

        print("Predicting epochs..")
        self._predict_saccade_epochs(pynwb_obj, pred_labels, pred_waveforms, pred_sacc_indices)

        # vals = {}
        # for ky in self.saved_keys():
        #     vals[ky] = Enrichment.get_val(self.get_name(), ky, pynwb_obj)
        #
        # import matplotlib.pyplot as plt
        # [plt.plot(f) for f in vals["saccades_predicted_nasal_waveforms"][:, :, 0]]
        # plt.title("Nasal")
        # plt.show()
        #
        # [plt.plot(f) for f in vals["saccades_predicted_temporal_waveforms"][:, :, 0]]
        # plt.title("Temporal")
        # plt.show()
        #
        # [plt.plot(f) for f in vals["saccades_predicted_noise_waveforms"][:, :, 0]]
        # plt.title("Noise")
        # plt.show()
        tw = 2

    @staticmethod
    def _check_for_pretained_direction_model():
        fn = "predict_gui_directional_model.pickle"
        return os.path.exists(fn)

    @staticmethod
    def load_pretrained_direction_model():
        fn = "predict_gui_directional_model.pickle"
        if PredictedSaccadeGUIEnrichment._check_for_pretained_direction_model():
            print(f"Pretrained data found at '{fn}' using that..")
            with open(fn, "rb") as fp:
                return pickle.load(fp)
        return False

    @staticmethod
    def get_pretrained_direction_model(wv, y_vals):
        load_model = PredictedSaccadeGUIEnrichment.load_pretrained_direction_model()
        if load_model:
            return load_model
        print("No pretrained model found for direction, training..")
        # TODO do we want to use LDA?
        tmp_wv = np.broadcast_to(wv[:, :, None], shape=(*wv.shape, 2))
        x_velocities, idxs = PredictSaccadesEnrichment.preformat_waveforms(tmp_wv)
        x_velocities = np.array(x_velocities)
        non_nan_yvals = np.array(y_vals)[idxs]

        lda = LinearDiscriminantAnalysis()
        lda.fit(x_velocities, non_nan_yvals)

        fn = "predict_gui_directional_model.pickle"
        with open(fn, "wb") as fp:
            pickle.dump(lda, fp)
        return lda

        # return DirectionalClassifier()

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
        if "NWB_DEBUG" in os.environ and os.environ["NWB_DEBUG"] == "True":
            hidden_layer_sizes = [(5,)]
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
        else:
            # Regressor
            hidden_layer_sizes = [(int(math.pow(2, v)),) for v in range(3, 6)]  # Try layer sizes 8,16,32
            # 1 2 4 8 16 32 64 128
            # 0 1 2 3  4  5  6  7
            grid = {
                'estimator__hidden_layer_sizes': hidden_layer_sizes,
                'estimator__max_iter': [
                    1000000,
                ],
                'estimator__activation': ['tanh', 'relu'],
                'estimator__solver': ['adam'],
                'estimator__alpha': [0.0001, 0.05],
                'estimator__learning_rate': ['constant', 'adaptive'],
            }

        reg = MultiOutputRegressor(MLPRegressor(verbose=True))
        search = GridSearchCV(reg, grid)

        search.fit(training_x_waveforms, standardized_epoch_labels)
        regressor = search.best_estimator_

        return regressor, transformer
