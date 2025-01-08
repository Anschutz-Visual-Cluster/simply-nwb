import types
from typing import Any, Callable, Union


class NWBValueMapping(object):
    def __init__(self, mapping: dict[str, Union['EnrichmentReference', list[Union[str, Callable[[Any], Any]]]]]):
        """
        Mapping is a dict of the form
        {"value_key": [ <mapping path> ] }
        Where mapping path is a list that can contain functions or strings

        The following mapping path will give nwb.processing["RightCamStim"]["pupilCenter_x"]
        [lambda x: x.processing, "RightCamStim", "pupilCenter_x",]

        :param mapping: mapping dict
        """

        def getkey(ky):
            def kfunc(obj):
                try:
                    result = obj[ky]
                    return result
                except KeyError as e:
                    raise KeyError(f"Unable to access required subkey '{ky}' in object '{str(obj)}' Error '{str(e)}'")
            return kfunc

        def wrap_nested(func1, func2):
            # Wrap the output of func 1 into func 2 into a single function
            def func3(*args, **kwargs):
                vfunc1 = func1(*args, **kwargs)
                vfunc2 = func2(vfunc1)
                return vfunc2
            return func3

        self._mapping = {}

        def _get_val(enrich_name, enrich_ky):
            def func(mynwb):
                if f"Enrichment.{enrich_name}" not in mynwb.processing:
                    raise ValueError(f"Required Enrichment '{enrich_name}' not found in NWB! Are you using the right version of the NWB?")
                if enrich_ky not in mynwb.processing[f"Enrichment.{enrich_name}"].containers:
                    raise KeyError(f"Cannot find key '{enrich_ky}' in the NWB!")

                myvall = mynwb.processing[f"Enrichment.{enrich_name}"][enrich_ky].data[:]
                return myvall
            return func

        for k, v in mapping.items():
            if isinstance(v, EnrichmentReference):
                cls = v.get_classtype()
                name = cls.get_name()
                for ks in cls.saved_keys():  # Check that the keys required actually exist
                    self._mapping[f"{name}.{ks}"] = _get_val(name, ks)
            else:
                if not isinstance(v, list):
                    raise ValueError(f"Invalid NWBValueMapping, key '{k}' has a non-list type value '{v}'")

                func_mapping_path = lambda x: x

                for vv in v:
                    if isinstance(vv, types.FunctionType):
                        func_mapping_path = wrap_nested(func_mapping_path, vv)
                    elif isinstance(vv, str):
                        func_mapping_path = wrap_nested(func_mapping_path, getkey(vv))
                    else:
                        raise ValueError(f"Invalid mapping path '{v}' entry '{vv}' must be a function or string!")

                self._mapping[k] = func_mapping_path

    def get(self, key, nwb) -> Any:
        if key not in self._mapping:
            raise KeyError(f"Key '{key}' not found in mapping! Keys available '{self.keys()}'")
        return self._mapping[key](nwb)

    def keys(self) -> list[str]:
        return list(self._mapping.keys())


class EnrichmentReference(object):
    ENRICHS = None

    def __init__(self, enrichment_name):
        if EnrichmentReference.ENRICHS is None:
            from simply_nwb.pipeline import discover_enrichments
            EnrichmentReference.ENRICHS = discover_enrichments()

        if enrichment_name not in EnrichmentReference.ENRICHS:
            raise ValueError(f"Unknown Enrichment referenced '{enrichment_name}' Available: '{list(EnrichmentReference.ENRICHS.keys())}'")

        self._cls = EnrichmentReference.ENRICHS[enrichment_name]

    def get_classtype(self):
        return self._cls

