import os

from sklearn.neural_network import MLPClassifier

from simply_nwb.pipeline.enrichments.saccades import PredictSaccadesEnrichment
from simply_nwb.pipeline.util.saccade_algo.directional import DirectionalClassifier
from simply_nwb.pipeline.util.saccade_algo.epochs import EpochRegressor, EpochTransformer
import pickle

from simply_nwb.transforms import eyetracking_load_dlc, csv_load_dataframe


class PredictSaccadeMLEnrichment(PredictSaccadesEnrichment):
    DEFAULT_MODEL_FILENAME = "saccade_extraction_model.pickle"

    def __init__(self):
        super().__init__(
            PredictSaccadeMLEnrichment._load_model(PredictSaccadeMLEnrichment.DEFAULT_MODEL_FILENAME),
            EpochRegressor(),
            EpochTransformer(),
            EpochRegressor(),
            EpochTransformer()
        )

    @staticmethod
    def _load_model(filepath):
        if not os.path.exists(filepath):
            # TODO auto-create model here
            raise FileNotFoundError(f"Cannot find model file '{filepath}'!")

        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def retrain(training_datas: list[tuple[str, str, str]], xkey="center_x", ykey="center_y", likeli="center_likelihood"):
        """
        Re-train a model using given a list of training data files like
        [('saccade_times.csv', 'timestamps.csv', dlc.csv'), ...] where each (..) is it's own training dataset

        'saccade_times.csv' is formatted in csv like the following, this is a list of filenames that are manually
        labeled csvs used to train a model

        time,eye_pos,nasal0_temporal1,original_time
        14148,-1.850164897,1,0
        ...

        'timestamps.txt' and 'dlc.csv' are outputs from DLC, 'dlc.csv' is the eye positions
        """
        clf = MLPClassifier(
            activation='tanh',
            solver='adam',
            learning_rate="adaptive",
            # 80 xpoints + 80y + 80likelihoods = 240 inputs, other layer sizes are arbitrary
            hidden_layer_sizes=(240, 80, 4, 2),
            random_state=1
        )
        clf.out_activation_ = "softmax"

        training_x = []  # will be a numpy array of size (240, N) N = training samples given
        training_y = []  # numpy array like (N,) with
        for labeled_csv, timestamps_txt, dlc_csv in training_datas:
            eyepos = eyetracking_load_dlc(dlc_csv)
            labeled = csv_load_dataframe(labeled_csv)  # time,eyepos,nasaltemporal,orig_time (note columns may not be named the same, just assume order is correct
            with open(timestamps_txt, "r") as f:
                timestamps = [int(t.strip()) for t in f.readlines()]
            # TODO extract each training example's x/y positions and likelihoods for each timestamp value in the 'saccade_times.csv'
            # format them all into a matrix
            # generate noisedata/randomly sample eyepositions for noise examples
            # train and then test
            tw = 2




        raise NotImplementedError
