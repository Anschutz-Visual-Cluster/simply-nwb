import types
import warnings

import numpy as np

from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.enrichments.saccades.drifting_grating.base import DriftingGratingEnrichment
from simply_nwb.pipeline.funcinfo import FuncInfo
from simply_nwb.pipeline.util.waves import startstop_of_squarewave
from simply_nwb.pipeline.value_mapping import EnrichmentReference
from simply_nwb.transforms import drifting_grating_metadata_read_from_filelist, labjack_concat_files


# TODO Create graph code for analyzing the labjack data?


class DriftingGratingLabjackEnrichment(DriftingGratingEnrichment):
    """
    Enrich the saccade data with metadata about the drifting grating using labjack as the global clock

    Default labjack signal mapping is as follows
    y0 'barcode' of a count
    y1 default drifting grating event happened, align driftingGrating-0.txt, starts at zero, goes to 1
    Assumes that the first 'pulse' in the labjack data corresponds to the first event in the driftingGrating-0.txt

    y2 video camera timing acquisition frame, timestamps for video frames
    a frame is 0 or 1, each time it flips is a new frame, 0 to 1, 1 to 0 etc..
    y3 misc analogue signal, per usecase

    """
    def __init__(self,  drifting_grating_metadata_filenames, dat_filenames, drifting_grating_channel="y1", video_frame_channel="y2", drifting_kwargs={}, labjack_kwargs={}, squarewave_args={}, skip_sparse_noise=False):
        # If skip_sparse_noise is True, will find a gap in the grating signal and truncate up to it, to account for the
        # sparse noise in the first part of the recording, TODO integrate and parse sparse noise

        super().__init__(drifting_grating_metadata_filenames, drifting_kwargs=drifting_kwargs)

        if isinstance(dat_filenames, types.GeneratorType):
            dat_filenames = list(dat_filenames)
        assert len(dat_filenames) > 0, "List of given labjack filenames is empty!"

        self.skip_sparse_noise = skip_sparse_noise
        self._sparse_skip = 0  # Calculated upon labjack load
        self._sparse_skip_calcd = False  # Sparse value has not been calculated yet (only applicable when skip_sparse_noise=True)
        self._gratings_startstop = None  # gratings waveform starts and stops, cached
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
            # If we have a sparse noise stimulus in the beginning of the labjack signal, we will want to skip over it
        if self.skip_sparse_noise:
            self._sparse_skip = self.find_sparse_noise_offset_value()

        return self._dats[self._sparse_skip:]

    def find_sparse_noise_offset_value(self):
        if self._sparse_skip_calcd:
            return self._sparse_skip
        else:
            # Calc here
            tw = 2
            gratings = self.get_gratings_startstop()
            self._sparse_skip_calcd = True

        return self._sparse_skip

    def get_video_startstop(self):
        # Get an array of (time, 2) for the start/stop of the frames
        print("Processing video frame labjack data wave pulses..")
        return startstop_of_squarewave(self.dats[self.frames_channel], **self.squarewave_args)[:, :2]  # Chop off the state value, only want start/stop

    def get_gratings_startstop(self):
        # Filter by rising state
        print("Processing grating pulses from labjack data wave pulses..")
        if self._gratings_startstop is None:
            grating_data = self.dats[self.grating_channel]
            grating_wave = startstop_of_squarewave(grating_data, **self.squarewave_args) # come back as [[start, stop, state], ...]
            inbetween_idxs = np.where(grating_wave[:, 2] == 1)  # 1 is where wave is went up, duration of pulse
            inbetweens = grating_wave[inbetween_idxs]

            startstop = inbetweens[:, :2]  # Chop off state with :2
            self._gratings_startstop = startstop

        return self._gratings_startstop

    def _run(self, pynwb_obj):
        super()._run(pynwb_obj)
        self._save_val("sparse_skip_count", self._sparse_skip, pynwb_obj)
        sparse = self._sparse_skip
        self._sparse_skip = 0  # We want to put all the data in the NWB, including the spare noise stim data we skipped
        for k, v in self.dats.items():
            self._save_val(k, v, pynwb_obj)

        self._sparse_skip = sparse

    @staticmethod
    def get_name() -> str:
        return "DriftingGratingLabjack"

    @staticmethod
    def saved_keys() -> list[str]:
        saved = DriftingGratingEnrichment.saved_keys()
        labjack_keys = ["Time", "v0", "v1", "v2", "v3", "y0", "y1", "y2", "y3"]
        saved.extend(labjack_keys)
        saved.append("sparse_skip_count")
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
            "y3": "Labjack channel y3, if present, keeps track of neuropixels' counting barcode signal)",
            "sparse_skip_count": "How far into the labjack array (in indexes) does the driftingGrating stimulus starts, used to skip past initial sparse noise data in the labjack"
        })
        return descs

    @staticmethod
    def nasal_saccade_info(pynwb_obj, saccade_index=None):
        return DriftingGratingEnrichment._saccade_info(pynwb_obj, "nasal", saccade_index, DriftingGratingLabjackEnrichment.get_name())

    @staticmethod
    def temporal_saccade_info(pynwb_obj, saccade_index=None):
        return DriftingGratingEnrichment._saccade_info(pynwb_obj, "temporal", saccade_index, DriftingGratingLabjackEnrichment.get_name())
