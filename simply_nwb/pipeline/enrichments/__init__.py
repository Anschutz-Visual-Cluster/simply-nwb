from typing import Any

from pynwb import NWBFile

from pipeline.value_mapping import NWBValueMapping


class Enrichment(object):
    def __init__(self):
        pass

    def run(self, pynwb_obj):
        raise NotImplemented("Cannot run baseclass! Override in a subclass")

    @staticmethod
    def get_name() -> str:
        raise NotImplemented

    @staticmethod
    def default_mapping() -> NWBValueMapping:
        raise NotImplemented

    def get_val(self, val_key: str, nwb: NWBFile) -> Any:
        """
        Get a value from this enrichment in a given NWB

        :param val_key: key for the value in this enrichment's namespace
        :param nwb: nwbfile to pull from
        :return: value or error if it doesn't exist
        """
        raise NotImplemented
