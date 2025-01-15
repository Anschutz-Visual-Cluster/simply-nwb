import logging
import warnings
from typing import Any, Callable

import pynwb.base
from av.video.format import names
from pynwb import NWBFile, TimeSeries

from simply_nwb import SimpleNWB
from simply_nwb.pipeline.funcinfo import FuncInfo
from simply_nwb.pipeline.value_mapping import NWBValueMapping


class _PrintLogger(object):
    def __init__(self, name):
        self.name = name

    def info(self, msg):
        print(f"[{self.name}] {msg}")


class Enrichment(object):
    def __init__(self, required_vals_map: NWBValueMapping):
        self._required_vals_map = required_vals_map
        self.logger = _PrintLogger(self.get_name())

    def validate(self, pynwb_obj):
        enrichment_name = self.get_name()
        for k in self._required_vals_map.keys():
            try:
                self._get_req_val(k, pynwb_obj)
            except Exception as e:
                self.logger.info(f"Unable to find required key '{k}' for enrichment '{enrichment_name}' Error: '{str(e)}'")
                raise e

    def post_validate(self, pynwb_obj):
        # validate that all the keys we said were saved, are saved, and no others
        enrichment_name = self.get_name()
        module = pynwb_obj.processing[f"Enrichment.{enrichment_name}"]
        available_keys = self.saved_keys()
        desc_keys = self.descriptions()

        keys = list(module.containers.keys())
        for key in keys:
            if key not in available_keys:
                warnings.warn(f"Key '{key}' found, but not defined in enrichment! Defined '{list(available_keys)}' In NWB '{list(keys)}'")
            if key not in desc_keys:
                txt = f"Key '{key}' not found in descriptions! Defined '{desc_keys}'"
                if key in available_keys:
                    raise ValueError(txt)
                else:
                    warnings.warn(txt)

    def run(self, pynwb_obj):
        self.validate(pynwb_obj)
        val = self._run(pynwb_obj)
        self.post_validate(pynwb_obj)
        return val

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
        try:
            val = self._required_vals_map.get(val_key, nwb)
            return val
        except KeyError as e:
            # Couldn't find the key, could be extra key not defined in config
            try:
                val = nwb.processing[f"Enrichment.{val_key.split(".")[0]}"][f"{"".join(val_key.split(".")[1:])}"]
                if isinstance(val, pynwb.base.TimeSeries):
                    val = val.data[:]
                return val
            except Exception as e2:
                print(f"Attempting to find missing key '{val_key}' failed!")
                raise e2

    @staticmethod
    def saved_keys() -> list[str]:
        """
        Return a list of the names of keys that this enrichment will add, used for determining requirements

        :returns: list of str keynames
        """
        raise NotImplementedError

    @staticmethod
    def descriptions() -> dict[str, str]:
        """
        Get the descriptions for all keys

        :returns: Return a dict of the names and descriptions for each saved key
        """
        raise NotImplementedError

    @staticmethod
    def get_name() -> str:
        """
        Unique CamelCase name for the enrichment
        """
        raise NotImplementedError

    @staticmethod
    def func_list() -> list[FuncInfo]:
        """
        Form of functions should be f(nwbobj, args, kwargs) -> Any
        This is a list of simply_nwb.pipeline.funcinfo.FuncInfo objects
        """
        return []

    def _run(self, pynwb_obj):
        """
        Run the enrichment, adding it to the nwb object
        :param pynwb_obj: nwb object to modify
        """
        raise NotImplementedError("Cannot run baseclass! Override in a subclass")

