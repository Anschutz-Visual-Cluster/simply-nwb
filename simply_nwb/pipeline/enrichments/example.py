from simply_nwb.pipeline import Enrichment, NWBValueMapping

"""
Use me as a starter point to make your own enrichment
"""


class ExampleEnrichment(Enrichment):
    def __init__(self):
        super().__init__(NWBValueMapping({}))

    def _run(self, pynwb_obj):
        pass

    @staticmethod
    def get_name() -> str:
        return "Example"

    @staticmethod
    def saved_keys() -> list[str]:
        return []

