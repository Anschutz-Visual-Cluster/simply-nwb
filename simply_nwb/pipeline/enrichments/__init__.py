from typing import Any
from pynwb import NWBFile
from simply_nwb.pipeline.value_mapping import NWBValueMapping


class Enrichment(object):
    def __init__(self):
        pass

    def run(self, pynwb_obj):
        """
        Run the enrichment, adding it to the nwb object
        :param pynwb_obj: nwb object to modify
        """
        raise NotImplemented("Cannot run baseclass! Override in a subclass")

    @staticmethod
    def get_name() -> str:
        """
        Unique CamelCase name for the enrichment
        """

        raise NotImplemented

    @staticmethod
    def default_mapping() -> NWBValueMapping:
        """
        Return a NWBValueMapping for keys under this Enrichment's namespace to a path within the NWB

        """
        raise NotImplemented

    def get_val(self, val_key: str, nwb: NWBFile) -> Any:
        """
        Get a value from this enrichment in a given NWB

        :param val_key: key for the value in this enrichment's namespace
        :param nwb: nwbfile to pull from
        :return: value or error if it doesn't exist
        """
        raise NotImplemented
