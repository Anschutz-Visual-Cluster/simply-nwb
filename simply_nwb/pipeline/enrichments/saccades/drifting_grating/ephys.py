import os
import types
import warnings

import numpy as np
from population_analysis.processors.kilosort import KilosortProcessor

from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.funcinfo import FuncInfo
from simply_nwb.pipeline.util.waves import startstop_of_squarewave
from simply_nwb.pipeline.value_mapping import EnrichmentReference
from simply_nwb.transforms import drifting_grating_metadata_read_from_filelist, labjack_concat_files


# TODO Create graph code for analyzing the labjack data?


class DriftingGratingEPhysEnrichment(Enrichment):
    """
    Enrich the saccade data with metadata about the drifting grating using labjack as the global clock, including ephys data

    Default labjack signal mapping is as follows
    y0 'barcode' of a count
    y1 default drifting grating event happened, align driftingGrating-0.txt, starts at zero, goes to 1
    Assumes that the first 'pulse' in the labjack data corresponds to the first event in the driftingGrating-0.txt

    y2 video camera timing acquisition frame, timestamps for video frames
    a frame is 0 or 1, each time it flips is a new frame, 0 to 1, 1 to 0 etc..
    y3 misc analogue signal, per usecase

    Requires the Neuropixels timestamps for labjack signals aka 'barcode' to sync to labjack time

    """
    def __init__(self, np_barcode_fn, np_spike_clusts_fn, np_spike_times_fn, labjack_barcode_channel="", squarewave_args={}):
        super().__init__(NWBValueMapping({
            "DriftingGratingLabjack": EnrichmentReference("DriftingGratingLabjack")  # Required that the saccades, labjack and driftingGrating are already in file
        }))

        self.np_barcode_fn = np_barcode_fn
        self.np_spike_clusts_fn = np_spike_clusts_fn
        self.np_spike_times_fn = np_spike_times_fn
        self.squarewave_args = squarewave_args
        for fn in [np_barcode_fn, np_spike_clusts_fn, np_spike_times_fn]:
            assert os.path.exists(fn)

        self._spike_clusts = None
        self._spike_times = None
        self._np_barcode = None

    @property
    def spike_clusts(self):
        if self._spike_clusts is None:
            self._spike_clusts = np.load(self.np_spike_clusts_fn)
        return self._spike_clusts

    @property
    def spike_times(self):
        if self._spike_times is None:
            self._spike_times = np.load(self.np_spike_times_fn)
        return self._spike_times

    @property
    def np_barcode(self):
        if self._np_barcode is None:
            self._np_barcode = np.load(self.np_barcode_fn)
        return self._np_barcode

    def _run(self, pynwb_obj):
        # kp = KilosortProcessor(self.spike_clusters, self.spike_timings)
        # raw_spike_times = kp.calculate_spikes(load_precalculated)
        # raw_firing_rates, fr_bins = kp.calculate_firingrates(SPIKE_BIN_MS, load_precalculated)
        self._save_val("asdf", [8], pynwb_obj)
        tw = 2
        pass

    @staticmethod
    def get_name() -> str:
        return "DriftingGratingEPhys"

    @staticmethod
    def saved_keys() -> list[str]:
        return ["asdf"]

    @staticmethod
    def descriptions() -> dict[str, str]:
        return {"asdf": "asdf"}

    @staticmethod
    def func_list() -> list[FuncInfo]:
        return [
            #(self, funcname: str, funcdescription: str, arg_and_description_list: dict[str, str], example_str: str):
            # Form of functions should be f(nwbobj, args, kwargs) -> Any
            FuncInfo(
                "test",
                "test function",
                {
                    "args": "list of args to pass to the function",
                    "kwargs": "dict of keyword arguments to pass to the function"
                },
                "test(['myarg'], {'mykwarg': 8})"
            )
        ]

    @staticmethod
    def test(pynwb_obj, args, kwargs):
        # Called by NWBSession(..).func("ExampleEnrichment.test", args, kwargs)
        print(f"test func being called with obj {pynwb_obj} args {args} kwargs {kwargs}")

