from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.transforms import drifting_grating_metadata_read

"""
Use me as a starter point to make your own enrichment
"""


class DriftingGratingEnrichment(Enrichment):
    def __init__(self,  drifting_grating_metadata_filename):
        super().__init__(NWBValueMapping({}))
        self.data = drifting_grating_metadata_read(drifting_grating_metadata_filename)
        tw = 2

    def _run(self, pynwb_obj):
        # get var from req
        # self._get_req_val("PutativeSaccades.saccades_putative_waveforms", pynwb_obj)
        # self._get_req_val("myvariable_i_need", pynwb_obj)
        for k, v in self.data.items():
            if isinstance(v, str):
                v = [v]
            self._save_val(k, v, pynwb_obj)

    @staticmethod
    def get_name() -> str:
        return "DriftingGrating"

    @staticmethod
    def saved_keys() -> list[str]:
        # Keys may or may not be dynamically named, assuming they aren't
        return [
            "Spatial frequency",
            "Velocity",
            "Orientation",
            "Baseline contrast",
            "Columns",
            "Event (1=Grating, 2=Motion, 3=Probe, 4=ITI)",
            "Motion direction",
            "Probe contrast",
            "Probe phase",
            "Timestamp"
        ]

    @staticmethod
    def descriptions() -> dict[str, str]:
        return {
            "Spatial frequency": "How many cycles/degree",
            "Velocity": "How many degrees/second",
            "Orientation": "Orientation degrees",
            "Baseline contrast": "0 or 1 for contrast value",
            "Columns": "Column names for the data",
            "Event (1=Grating, 2=Motion, 3=Probe, 4=ITI)": "Event data number",
            "Motion direction": "Motion direction",
            "Probe contrast": "Contrast number",
            "Probe phase": "Phase number",
            "Timestamp": "Timestamp value"
        }
