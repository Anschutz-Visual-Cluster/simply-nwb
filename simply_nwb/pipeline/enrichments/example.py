from simply_nwb.pipeline import Enrichment


class ExampleEnrichment(Enrichment):
    def __init__(self):
        super().__init__()

        pass

    def run(self, pynwb_obj):
        pass

    @staticmethod
    def get_name() -> str:
        return "Example"
