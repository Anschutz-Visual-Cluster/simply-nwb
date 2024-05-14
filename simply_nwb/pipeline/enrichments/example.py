from simply_nwb.pipeline import Enrichment, NWBValueMapping


class ExampleEnrichment(Enrichment):
    def __init__(self):
        super().__init__(NWBValueMapping({}))

    def _run(self, pynwb_obj):
        pass

    @staticmethod
    def get_name() -> str:
        return "Example"
