from typing import Any
from pynwb import NWBFile, TimeSeries

from simply_nwb import SimpleNWB
from simply_nwb.pipeline.value_mapping import NWBValueMapping


class Enrichment(object):
    def __init__(self, required_vals_map: NWBValueMapping):
        self._required_vals_map = required_vals_map

    def run(self, pynwb_obj):
        # TODO Check required vals here
        enrichment_name = self.get_name()
        for k in self._required_vals_map.keys():
            try:
                self._get_req_val(k, pynwb_obj)
            except Exception as e:
                raise ValueError(f"Unable to find required key '{k}' for enrichment '{enrichment_name}' Error: '{str(e)}'")

        return self._run(pynwb_obj)

    def _save_val(self, key: str, value: Any, nwb: NWBFile):
        """
        Internal func to save a value from an enrichment to the NWB
        """
        ts = TimeSeries(name=key, data=value, unit="val", rate=1.0)
        SimpleNWB.add_to_processing_module(nwb, ts, f"Enrichment.{self.get_name()}")
        tw = 2

    @staticmethod
    def keys(enrichment_name: str, nwb: NWBFile) -> list[str]:
        module = nwb.processing[f"Enrichment.{enrichment_name}"]
        return list(module.containers.keys())

    @staticmethod
    def get_val(enrichment_name: str, key: str, nwb: NWBFile):
        module = nwb.processing[f"Enrichment.{enrichment_name}"]
        available_keys = Enrichment.keys(enrichment_name, nwb)

        if key not in module.containers.keys():
            raise ValueError(f"Unable to find key '{key}' in Enrichment '{enrichment_name}' Available keys '{available_keys}'")
        val = module[key].data[:]
        return val

    def _get_req_val(self, val_key: str, nwb: NWBFile) -> Any:
        """
        Get a value from this enrichment in a given NWB that is required for the enrichment

        :param val_key: key for the value in this enrichment's namespace
        :param nwb: nwbfile to pull from
        :return: value or error if it doesn't exist
        """
        return self._required_vals_map.get(val_key, nwb)

    @staticmethod
    def saved_keys() -> list[str]:
        """
        Return a list of the names of keys that this enrichment will add, used for determining requirements

        :returns: list of str keynames
        """
        raise NotImplemented

    @staticmethod
    def get_name() -> str:
        """
        Unique CamelCase name for the enrichment
        """
        raise NotImplemented

    def _run(self, pynwb_obj):
        """
        Run the enrichment, adding it to the nwb object
        :param pynwb_obj: nwb object to modify
        """
        raise NotImplemented("Cannot run baseclass! Override in a subclass")

