class Enrichment(object):
    def __init__(self):
        pass

    def run(self, pynwb_obj):
        raise NotImplemented("Cannot run baseclass! Override in a subclass")

    @staticmethod
    def get_name() -> str:
        raise NotImplemented

