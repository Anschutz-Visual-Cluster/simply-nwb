from simply_nwb.pipeline.enrichments.saccades import PredictSaccadesEnrichment, PutativeSaccadesEnrichment
from simply_nwb.pipeline.util.saccade_algo.directional import DirectionalClassifier
from simply_nwb.pipeline.util.saccade_algo.epochs import EpochRegressor, EpochTransformer


class PredictedSaccadeAlgoEnrichment(PredictSaccadesEnrichment):
    def __init__(self):
        # TODO load ML model from somewhere (memory? package include?) prepackaged but allow override
        super().__init__(
            DirectionalClassifier(),
            EpochRegressor(),
            EpochTransformer(),
            EpochRegressor(),
            EpochTransformer()
        )

    def retrain(self, training_data_paths: list[str]):
        """
        Re-train a model using given a list of training data files formatted in csv like the following

        time,eye_pos,nasal0_temporal1,original_time
        14148,-1.850164897,1,0
        ...

        This is a list of filenames that are manually labeled csvs used to train a model
        """

        raise NotImplementedError
