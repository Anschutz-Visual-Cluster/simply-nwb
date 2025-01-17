from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.funcinfo import FuncInfo

"""
Use me as a starter point to make your own enrichment
If you have a folder of enrichments, ie 'saccades/' make sure you import each one in the __init__.py
see simply_nwb.pipeline.enrichments.saccades.__init__.py
"""


class ExampleEnrichment(Enrichment):
    def __init__(self):
        super().__init__(NWBValueMapping({
            # "PutativeSaccades": EnrichmentReference("PutativeSaccades"),  # Reference another enrichment as required
            # "myvariable_i_need": [list of keys/functions to find it]
            # So [lambda x: x.processing, "MyContainer"] would result in nwb.processing["MyContainer"]
        }))

    def _run(self, pynwb_obj):
        # get var from req
        # self._get_req_val("PutativeSaccades.saccades_putative_waveforms", pynwb_obj)
        # self._get_req_val("myvariable_i_need", pynwb_obj)
        self._save_val("mykeyname", [1, 2, 3], pynwb_obj)

    @staticmethod
    def get_name() -> str:
        return "Example"

    @staticmethod
    def saved_keys() -> list[str]:
        return ["mykeyname"]

    @staticmethod
    def descriptions() -> dict[str, str]:
        return {
            "mykeyname": "test example key desc"
        }

    @staticmethod
    def func_list() -> list[FuncInfo]:
        return [
            #(self, funcname: str, funcdescription: str, arg_and_description_list: dict[str, str], example_str: str):
            # Form of functions should be f(nwbobj, args, kwargs) -> Any
            FuncInfo("test", "test function", {
                "args": "list of args to pass to the function",
                "kwargs": "dict of keyword arguments to pass to the function"
            }, "test(['myarg'], {'mykwarg': 8})")
        ]

    @staticmethod
    def test(pynwb_obj, args, kwargs):
        # Called by NWBSession(..).func("ExampleEnrichment.test", args, kwargs)
        print(f"test func being called with obj {pynwb_obj} args {args} kwargs {kwargs}")
