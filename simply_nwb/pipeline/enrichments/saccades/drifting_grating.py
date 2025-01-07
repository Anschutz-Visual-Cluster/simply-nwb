import types

import numpy as np

from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.funcinfo import FuncInfo
from simply_nwb.pipeline.util.waves import startstop_of_squarewave
from simply_nwb.pipeline.value_mapping import EnrichmentReference
from simply_nwb.transforms import drifting_grating_metadata_read_from_filelist, labjack_concat_files

"""
Use me as a starter point to make your own enrichment
"""

# TODO Create graph code for analyzing the labjack data


class DriftingGratingEnrichment(Enrichment):
    def __init__(self, drifting_grating_metadata_filenames, drifting_kwargs={}, drifting_grating_filename_str: str = "filename", drifting_timestamp_key: str = "Timestamp"):
        super().__init__(NWBValueMapping({
            "PredictSaccades": EnrichmentReference("PredictSaccades")  # Required that the saccades are already in file
        }))

        if isinstance(drifting_grating_metadata_filenames, types.GeneratorType):
            drifting_grating_metadata_filenames = list(drifting_grating_metadata_filenames)
        drifting_kwargs["expand_file_keys"] = True
        assert len(drifting_grating_metadata_filenames) > 0, "Must provide at least one driftingGratingMetadata.txt file!"
        self._drifting_grating_metadata_filenames = drifting_grating_metadata_filenames
        self._drifting_kwargs = drifting_kwargs

        self.drifting_grating_filename_str = drifting_grating_filename_str
        self._meta = None
        self.drifting_timestamp_key = drifting_timestamp_key
        tw = 2

    @property
    def meta(self):
        if self._meta is None:
            self._meta = drifting_grating_metadata_read_from_filelist(self._drifting_grating_metadata_filenames, **self._drifting_kwargs)
        return self._meta

    def get_video_startstop(self):
        # Get an array of (time, 2) for the start/stop of the frames
        # TODO, without labjack we can guess the video frames, will be less accurate though
        raise NotImplemented

    def get_gratings_startstop(self):
        # TODO get the start/stop times of each grating instance, on the same timescale as the video frames
        raise NotImplemented

    def _run(self, pynwb_obj):
        video_windows = self.get_video_startstop()
        grating_windows = self.get_gratings_startstop()
        grating_timestamps = self.meta[self.drifting_timestamp_key]
        if len(grating_timestamps) != len(grating_windows):
            # print("=" * 50) # TODO Disable this
            # print("DISABLE THIS TESTING CODE!!!!!!!!!!!!!!!!!!")
            # print("=" * 50)
            # if len(grating_timestamps) > len(grating_windows):
            #     grating_timestamps = grating_timestamps[:len(grating_windows)]
            # else:
            #     grating_windows = grating_windows[:len(grating_timestamps)]
            raise ValueError(f"Number of blocks in the driftingGrating.txt ({len(grating_timestamps)}) files do not match the number of grating windows ({len(grating_windows)})! Could this be malformatted labjack data?")
        # TODO handle a small number of mismatches between labjack and driftingGrating blocks?
        
        def process_saccade_epochs(saccade_epoch: np.ndarray):
            # saccade_epoch is a (numsaccade, 2) array for the start/stop of each saccade of a particular type

            # Bin each saccade epoch time into indexes into the video windows for which frame section it is within
            # interpolate the fractional frame within the signal's frame
            bins_idxs = np.digitize(saccade_epoch[:, 0], range(video_windows.shape[0]))
            # Convert the frames into absolute timestamps
            try:
                epochstart_framewindows = video_windows[bins_idxs]
            except IndexError as e:
                print("Error binning saccades into the video frame windows! Are labjack files missing? This could happen due to a saccade epoch being outside the recorded labjack time!")
                raise e

            # Use the start of the saccade [:, 0] to determine which grating bin it falls within
            # grating_windows[:, 1] is the end (right edge) of the grating bin
            epoch_grating_idxs = np.digitize(epochstart_framewindows[:, 0], grating_windows[:, 1], right=True)
            return epoch_grating_idxs

        nasal = self._get_req_val("PredictSaccades.saccades_predicted_nasal_epochs", pynwb_obj)
        temporal = self._get_req_val("PredictSaccades.saccades_predicted_temporal_epochs", pynwb_obj)

        print("Processing nasal saccades..")
        nasal_grating_idxs = process_saccade_epochs(nasal)
        print("Processing temporal saccades..")
        temporal_grating_idxs = process_saccade_epochs(temporal)
        # Which grating index does each saccade fall within, used with the grating metadata, we can determine info about each saccade

        self._save_val("nasal_grating_idxs", nasal_grating_idxs, pynwb_obj)
        self._save_val("temporal_grating_idxs", temporal_grating_idxs, pynwb_obj)

        # Drifting grating is in terms of pulses, different states of pulses
        # any pulse is a change in 'state'
        # 1 is grating (start), 2 is motion, 3 is probe, 4 is ITI (end)
        #Block(0, {event1,event2,event3,event3,event3, .., event4 }, "file1"), Block(1, {..}, "file1"), Block(2, {..}, "file2"), Block(3, {..}, "file2")

        # Want to check if the difference between blocks in the grating is
        # similar to the size of the length of the pulses
        # Save the drifting grating metadata
        for k, v in self.meta.items():
            if isinstance(v, str):
                v = [v]
            self._save_val(k, v, pynwb_obj)

    @staticmethod
    def get_name() -> str:
        return "DriftingGrating"

    @staticmethod
    def saved_keys() -> list[str]:
        # Keys may or may not be dynamically named, assuming they aren't
        # TODO key metadata?
        return [
            "file_len",
            "filename",
            "nasal_grating_idxs",
            "temporal_grating_idxs",
            "Spatial frequency",
            "Velocity",
            "Orientation",
            "Baseline contrast",
            "Columns",
            "Event (1=Grating, 2=Motion, 3=Probe, 4=ITI)",
            "Motion direction",
            "Probe contrast",
            "Probe phase",
            "Timestamp"
        ]

    @staticmethod
    def descriptions() -> dict[str, str]:
        return {
            "file_len": "Length of each drifting grating file entries, used to index into the other keys to determine which file contains which range of indexes",
            "filename": "filenames of each drifingGrating.txt",
            "nasal_grating_idxs": "Index into the grating windows for each nasal saccade",
            "temporal_grating_idxs": "Index into the grating windows for each temporal saccade",
            "Spatial frequency": "How many cycles/degree",
            "Velocity": "How many degrees/second",
            "Orientation": "Orientation degrees",
            "Baseline contrast": "0 or 1 for contrast value",
            "Columns": "Column names for the data",
            "Event (1=Grating, 2=Motion, 3=Probe, 4=ITI)": "Event data number",
            "Motion direction": "Motion direction",
            "Probe contrast": "Contrast number",
            "Probe phase": "Phase number",
            "Timestamp": "Timestamp value"
        }

    @staticmethod
    def func_list() -> list[FuncInfo]:
        return [
            # FuncInfo(self, funcname: str, funcdescription: str, arg_and_description_list: dict[str, str], example_str: str):
            FuncInfo(
                "nasal_saccade_info",
                "Get information about a given saccade by index",
                {"saccade_index": "Index of the saccade, if a list indexes is passed will return a list of info for each element"},
                "nasal_saccade_info([45, 89]) #Gets info about nasal saccade 45 and 89"
            ),
            FuncInfo(
                "temporal_saccade_info",
                "Get information about a given saccade by index",
                {
                    "saccade_index": "Index of the saccade, if a list indexes is passed will return a list of info for each element"},
                "temporal_saccade_info([45, 89]) #Gets info about temporal saccade 45 and 89"
            ),
        ]

    @staticmethod
    def grating_metadata_keys():
        return ["Baseline contrast", "Motion direction", "Orientation", "Probe contrast", "Probe phase", "Spatial frequency", "Timestamp", "Velocity", "filename"]

    @staticmethod
    def nasal_saccade_info(pynwb_obj, saccade_index=None):
        return DriftingGratingEnrichment._saccade_info(pynwb_obj, "nasal", saccade_index, DriftingGratingEnrichment.get_name())

    @staticmethod
    def temporal_saccade_info(pynwb_obj, saccade_index=None):
        return DriftingGratingEnrichment._saccade_info(pynwb_obj, "temporal", saccade_index, DriftingGratingEnrichment.get_name())

    @staticmethod
    def _saccade_info(pynwb_obj, saccade_name, indexdata, subname):
        # Get the drifting grating info for a given saccade or list of saccades
        if saccade_name not in ["nasal", "temporal"]:
            raise ValueError(f"Saccadename must be nasal or temporal, got '{saccade_name}'!")

        saccidxs = Enrichment.get_val(subname, f"{saccade_name}_grating_idxs", pynwb_obj)
        if indexdata is not None:
            saccidxs = saccidxs[indexdata]

        keys = DriftingGratingEnrichment.grating_metadata_keys()
        data = {}
        for k in keys:
            val = np.array(Enrichment.get_val(subname, k, pynwb_obj).data[:])[saccidxs]
            data[k] = val
        return data


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
    def __init__(self,  drifting_grating_metadata_filenames, dat_filenames, drifting_grating_channel="y1", video_frame_channel="y2", drifting_kwargs={}, labjack_kwargs={}, squarewave_args={}):
        super().__init__(drifting_grating_metadata_filenames, drifting_kwargs=drifting_kwargs)

        if isinstance(dat_filenames, types.GeneratorType):
            dat_filenames = list(dat_filenames)
        assert len(dat_filenames) > 0, "List of given labjack filenames is empty!"

        self.dats = labjack_concat_files(dat_filenames, **labjack_kwargs)
        self.grating_channel = drifting_grating_channel
        self.frames_channel = video_frame_channel
        self.squarewave_args = squarewave_args

    def get_video_startstop(self):
        # Get an array of (time, 2) for the start/stop of the frames
        print("Processing video frame labjack data wave pulses..")
        return startstop_of_squarewave(self.dats[self.frames_channel], **self.squarewave_args)[:, :2]  # Chop off the state value, only want start/stop

    def get_gratings_startstop(self):
        # Filter by rising state
        print("Processing grating pulses from labjack data wave pulses..")
        grating_wave = startstop_of_squarewave(self.dats[self.grating_channel], **self.squarewave_args) # come back as [[start, stop, state], ...]
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
