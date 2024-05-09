from simply_nwb.pipeline import Enrichment


class ExampleEnrichment(Enrichment):
    @staticmethod
    def get_name() -> str:
        return "example"

    pass
