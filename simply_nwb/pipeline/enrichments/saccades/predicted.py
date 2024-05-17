from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.value_mapping import EnrichmentReference


class PredictSaccadesEnrichment(Enrichment):
    def __init__(self, classifier):
        super().__init__(NWBValueMapping({
            "PutativeSaccades": EnrichmentReference("PutativeSaccades")
        }))
        self._classifier = classifier

    @staticmethod
    def saved_keys() -> list[str]:
        return []

    def _run(self, pynwb_obj):
        self._get_req_val("PutativeSaccades.pose_corrected", pynwb_obj)
        # TODO here
        tw = 2
        pass

    @staticmethod
    def get_name() -> str:
        return "PredictSaccades"

    def _predict_saccade_direction(self):

        pass

