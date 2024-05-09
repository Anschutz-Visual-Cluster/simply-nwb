
# This file is used for testing things during development

from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.enrichments.example import ExampleEnrichment


class T(object):
    def a(self):
        print("hi")

    def g(self):
        return self.a


def main():
    # first = T().g()
    # second = T.a
    from simply_nwb.pipeline.enrichments import Enrichment

    class CustEnrich(Enrichment):
        @staticmethod
        def get_name():
            return "custom"

    v = NWBSession("a", custom_enrichments=[CustEnrich])

    v.enrich(ExampleEnrichment())

    tw = 2


if __name__ == "__main__":
    main()

