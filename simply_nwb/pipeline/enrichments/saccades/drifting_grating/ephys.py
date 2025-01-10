import types
import warnings

import numpy as np

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
    def __init__(self, np_timestamps, drifting_grating_channel="y1", video_frame_channel="y2", squarewave_args={}):
        super().__init__(NWBValueMapping({
            "DriftingGratingLabjack": EnrichmentReference("DriftingGratingLabjack")  # Required that the saccades, labjack and driftingGrating are already in file
        }))


        if isinstance(dat_filenames, types.GeneratorType):
            dat_filenames = list(dat_filenames)
        assert len(dat_filenames) > 0, "List of given labjack filenames is empty!"

        self._dat_filenames = dat_filenames
        self._labjack_kwargs = labjack_kwargs
        self._dats = None
        self.grating_channel = drifting_grating_channel
        self.frames_channel = video_frame_channel
        self.squarewave_args = squarewave_args

    @property
    def dats(self):
        if self._dats is None:
            self._dats = labjack_concat_files(self._dat_filenames, **self._labjack_kwargs)
        return self._dats

    def get_video_startstop(self):
        # Get an array of (time, 2) for the start/stop of the frames
        print("Processing video frame labjack data wave pulses..")
        return startstop_of_squarewave(self.dats[self.frames_channel], **self.squarewave_args)[:, :2]  # Chop off the state value, only want start/stop

    def get_gratings_startstop(self):
        # Filter by rising state
        print("Processing grating pulses from labjack data wave pulses..")
        grating_data = self.dats[self.grating_channel]
        grating_wave = startstop_of_squarewave(grating_data, **self.squarewave_args) # come back as [[start, stop, state], ...]
        inbetween_idxs = np.where(grating_wave[:, 2] == 1)  # 1 is where wave is went up, duration of pulse
        inbetweens = grating_wave[inbetween_idxs]

        startstop = inbetweens[:, :2]  # Chop off state with :2

        return startstop

    def _run(self, pynwb_obj):
        super()._run(pynwb_obj)
        for k, v in self.dats.items():
            self._save_val(k, v, pynwb_obj)

    @staticmethod
    def get_name() -> str:
        return "DriftingGratingLabjack"

    @staticmethod
    def saved_keys() -> list[str]:
        saved = DriftingGratingEnrichment.saved_keys()
        labjack_keys = ["Time", "v0", "v1", "v2", "v3", "y0", "y1", "y2", "y3"]
        saved.extend(labjack_keys)
        return saved

    @staticmethod
    def descriptions() -> dict[str, str]:
        descs = DriftingGratingEnrichment.descriptions()
        descs.update({
            "Time": "Labjack times array",
            "v0": "Labjack channel v0 (currently not used)",
            "v1": "Labjack channel v1 (currently not used)",
            "v2": "Labjack channel v2 (currently not used)",
            "v3": "Labjack channel v3 (currently not used)",
            "y0": "Labjack channel y0 (currently not used)",
            "y1": "Labjack channel y1, this is the default channel for the drifting grating signal pulses for block alignment",
            "y2": "Labjack channel y2, this is the default channel for the video recording signal pulse for determining when a frame in the video has been recorded",
            "y3": "Labjack channel y3 (currently not used)"
        })
        return descs

    @staticmethod
    def nasal_saccade_info(pynwb_obj, saccade_index=None):
        return DriftingGratingEnrichment._saccade_info(pynwb_obj, "nasal", saccade_index, DriftingGratingLabjackEnrichment.get_name())

    @staticmethod
    def temporal_saccade_info(pynwb_obj, saccade_index=None):
        return DriftingGratingEnrichment._saccade_info(pynwb_obj, "temporal", saccade_index, DriftingGratingLabjackEnrichment.get_name())
