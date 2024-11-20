from simply_nwb.pipeline.enrichments.saccades import PredictSaccadesEnrichment, PutativeSaccadesEnrichment
from simply_nwb.pipeline.util.saccade_algo.directional import DirectionalClassifier
from simply_nwb.pipeline.util.saccade_algo.epochs import EpochRegressor, EpochTransformer


class PredictedSaccadeAlgoEnrichment(PredictSaccadesEnrichment):
    def __init__(self):
        super().__init__(
            DirectionalClassifier(),
            EpochRegressor(),
            EpochTransformer(),
            EpochRegressor(),
            EpochTransformer()
        )
