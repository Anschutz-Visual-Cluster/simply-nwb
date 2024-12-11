import functools
import logging
from typing import Optional, Any, Callable

import pynwb
from pynwb import NWBHDF5IO
from simply_nwb import SimpleNWB
from simply_nwb.pipeline.enrichments import Enrichment
from spencer_funcs.autodiscovery import discover_wrapper
from simply_nwb.pipeline.value_mapping import NWBValueMapping


class NWBSession(object):
    def __init__(self, filename_or_nwbobj, custom_enrichments: Optional[list[type]] = None):
        """
        Create a new NWB Session object from a given nwb filename. Will automatically detect enrichments in the NWB
        and compare to available. Can pass a list of custom enrichments to load in if they're not in this library

        :param filename_or_nwbobj: filepath to the nwb file or pynwb.NWBFile object
        :param custom_enrichments: list of class types for classes inheriting the Enrichment class
        """

        if isinstance(filename_or_nwbobj, pynwb.NWBFile):
            self.nwb = filename_or_nwbobj
            self._nwb_fp = None
        elif isinstance(filename_or_nwbobj, str):
            print(f"Reading NWB file '{filename_or_nwbobj}'..")
            self._nwb_fp = NWBHDF5IO(filename_or_nwbobj)
            self.nwb = self._nwb_fp.read()

        self.__builtin_enrichments: dict[str, Enrichment.__class__] = discover_enrichments()  # str: EnrichmentClass
        if custom_enrichments is not None:
            for cust in custom_enrichments:
                if not isinstance(cust, type):
                    raise ValueError(f"Custom Enrichment {cust} passed in must be a classtype and inherit from Enrichment! Use [MyEnrichment, ..] NOT [MyEnrichment(), ..]")
                # Not going to bother to check if the object type passed is actually a subclass TODO?
                self.__builtin_enrichments[cust.get_name()] = cust

        self.__enrichments = set()  # list of str names of current enrichments in the nwb file
        for k in list(self.nwb.processing.keys()):
            if k.startswith("Enrichment."):
                self.__enrichments.add(k[len("Enrichment."):])

    def __del__(self):
        if self._nwb_fp is not None:
            self._nwb_fp.close()

    def available_enrichments(self):
        return list(self.__enrichments)

    def to_dict(self):
        d = {}
        for enrich in self.available_enrichments():
            d[enrich] = {}
            for ky in self.available_keys(enrich):
                d[enrich][ky] = self.pull(f"{enrich}.{ky}")
        return d

    def _check_enrichment_name(self, enrichment_name):
        if enrichment_name not in self.__enrichments:
            raise ValueError(f"Enrichment '{enrichment_name}' not found in NWB, found '{self.available_enrichments()}'")

    def available_keys(self, enrichment_name):
        self._check_enrichment_name(enrichment_name)
        return Enrichment.keys(enrichment_name, self.nwb)

    def enrich(self, enrichment: Enrichment):
        if not isinstance(enrichment, Enrichment):
            raise ValueError(f"Invalid enrichment type received! Got {type(enrichment)}")

        # TODO requirement checking, for fields that are needed for adding specific enrichments
        enrichment.run(self.nwb)
        self.__enrichments.add(enrichment.get_name())

    def description(self, namespace: str) -> dict[str, str]:
        self._check_enrichment_name(namespace)
        return self.__builtin_enrichments[namespace].descriptions()

    def _parse_namespaced_key(self,  namespaced_key: str) -> (str, str):
        namespace = namespaced_key.split(".")[0]  # namespace, eg 'ExampleEnrichment' from 'ExampleEnrichment.myvar'
        key = ".".join(namespaced_key.split(".")[1:])  # The rest, 'myvar'
        self._check_enrichment_name(namespace)
        return namespace, key

    def pull(self, namespaced_key: str) -> Any:
        """
        Pull data from the NWB using namespaced valued from the enrichments

        :param namespaced_key: Key for the value to retrieve, namespaced. ie ExampleEnrichment.myvar
        """
        namespace, key = self._parse_namespaced_key(namespaced_key)
        val = self.__builtin_enrichments[namespace].get_val(namespace, key, self.nwb)
        return val

    def get_funclist(self, namespace: str) -> list[str]:
        self._check_enrichment_name(namespace)
        funcs = self.__builtin_enrichments[namespace].func_list()
        return [str(f) for f in funcs]

    def print_funclist(self, namespace: str):
        funcs = self.get_funclist(namespace)
        for f in funcs:
            print(f)

    def func(self, namespaced_key: str) -> Callable:
        namespace, key = self._parse_namespaced_key(namespaced_key)
        funcs = self.__builtin_enrichments[namespace].func_list()

        found = False
        for f in funcs:
            if key == f.name:
                found = True
                break

        if not found:
            raise ValueError(f"Function '{key}' not found in Enrichment '{namespace}' Available functions '{[str(f) for f in funcs]}'")

        myfunc = getattr(self.__builtin_enrichments[namespace], key)
        newfunc = functools.partial(myfunc, self.nwb)
        return newfunc

    def save(self, filename):
        v = self.nwb
        SimpleNWB.write(v, filename)


@discover_wrapper
def discover_enrichments():
    def get_enrichment_name(cls) -> dict[str, type]:
        if hasattr(cls, "get_name"):
            return cls.get_name()
        else:
            return None

    return [
        "simply_nwb.pipeline.enrichments",
        Enrichment,
        get_enrichment_name
    ]
