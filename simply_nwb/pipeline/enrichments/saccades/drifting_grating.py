import numpy as np

from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.funcinfo import FuncInfo
from simply_nwb.pipeline.util.waves import startstop_of_squarewave
from simply_nwb.pipeline.value_mapping import EnrichmentReference
from simply_nwb.transforms import drifting_grating_metadata_read_from_filelist, labjack_concat_files

"""
Use me as a starter point to make your own enrichment
"""

# TODO find an ideal session to test on
# TODO Create graph code for analyzing the labjack data
class DriftingGratingEnrichment(Enrichment):
    def __init__(self, drifting_grating_metadata_filenames, drifting_kwargs={}, drifting_grating_filename_str: str = "filename"):
        super().__init__(NWBValueMapping({
            "PredictSaccades": EnrichmentReference("PredictSaccades")  # Required that the saccades are already in file
        }))
        self.drifting_grating_filename_str = drifting_grating_filename_str
        self.meta = drifting_grating_metadata_read_from_filelist(drifting_grating_metadata_filenames, **drifting_kwargs)
        tw = 2

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

        def process_saccade_epochs(saccade_epoch: np.ndarray):
            # saccade_epoch is a (numsaccade, 2) array for the start/stop of each saccade of a particular type

            # Bin each saccade epoch time into indexes into the video windows for which frame section it is within
            # interpolate the fractional frame within the signal's frame
            bins_idxs = np.digitize(saccade_epoch[:, 0], range(video_windows.shape[0]))
            # Convert the frames into absolute timestamps
            epochstart_framewindows = video_windows[bins_idxs]

            # Use the start of the saccade [:, 0] to determine which grating bin it falls within
            # grating_windows[:, 1] is the end (right edge) of the grating bin
            epoch_grating_idxs = np.digitize(epochstart_framewindows[:, 0], grating_windows[:, 1], right=True)
            return epoch_grating_idxs

        nasal = self._get_req_val("PredictSaccades.saccades_predicted_nasal_epochs", pynwb_obj)
        temporal = self._get_req_val("PredictSaccades.saccades_predicted_temporal_epochs", pynwb_obj)

        nasal_grating_idxs = process_saccade_epochs(nasal)
        temporal_grating_idxs = process_saccade_epochs(temporal)

        self._save_val("nasal_grating_idxs", nasal_grating_idxs, pynwb_obj)
        self._save_val("temporal_grating_idxs", temporal_grating_idxs, pynwb_obj)

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
    def nasal_saccade_info(pynwb_obj, saccade_index):
        return DriftingGratingEnrichment._saccade_info(pynwb_obj, "nasal", saccade_index)

    @staticmethod
    def temporal_saccade_info(pynwb_obj, saccade_index):
        return DriftingGratingEnrichment._saccade_info(pynwb_obj, "temporal", saccade_index)

    @staticmethod
    def _saccade_info(pynwb_obj, saccade_name, indexdata):
        # Get the drifting grating info for a given saccade or list of saccades
        saccidxs = Enrichment.get_val(DriftingGratingEnrichment.get_name(), f"{saccade_name}_grating_idxs", pynwb_obj)

        tw = 2
        pass


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
    def __init__(self,  drifting_grating_metadata_filenames, dat_filenames, drifting_grating_channel="y1", video_frame_channel="y2", drifting_kwargs={}, labjack_kwargs={}):
        super().__init__(drifting_grating_metadata_filenames, drifting_kwargs=drifting_kwargs)

        self.dats = labjack_concat_files(dat_filenames, **labjack_kwargs)
        self.grating_channel = drifting_grating_channel
        self.frames_channel = video_frame_channel

    def get_video_startstop(self):
        # Get an array of (time, 2) for the start/stop of the frames
        # TODO Check for large gaps in frames, raise error
        return startstop_of_squarewave(self.dats[self.frames_channel])[:, :2]  # Chop off the state value, only want start/stop

    def get_gratings_startstop(self):
        # TODO threshold for artifact with multi pulses
        # Filter by rising state
        # Drifting grating is in terms of pulses, different states of pulses
        # 1 is start 4 is end, any pulse is a change in 'state'
        #Block(0, {}, "file1"), Block(1, {}, "file1"), Block(2, {}, "file2",0), Block(3, {}, "file2", 1)

        return startstop_of_squarewave(self.dats[self.grating_channel])[:, :2]  # Chop off state value

    def _run(self, pynwb_obj):
        super()._run(pynwb_obj)
        # TODO add labjack stuff here

    @staticmethod
    def get_name() -> str:
        return "DriftingGratingLabjack"

    @staticmethod
    def saved_keys() -> list[str]:
        saved = DriftingGratingEnrichment.saved_keys()
        # TODO add labjack saved keys here
        return saved

    @staticmethod
    def descriptions() -> dict[str, str]:
        descs = DriftingGratingEnrichment.descriptions()
        # TODO add labjack descs here
        return descs
