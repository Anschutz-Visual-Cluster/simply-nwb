import numpy as np

from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.util.waves import startstop_of_squarewave
from simply_nwb.pipeline.value_mapping import EnrichmentReference
from simply_nwb.transforms import drifting_grating_metadata_read_from_filelist, labjack_concat_files

"""
Use me as a starter point to make your own enrichment
"""


class DriftingGratingEnrichment(Enrichment):
    def __init__(self, drifting_grating_metadata_filenames, drifting_kwargs={}):
        super().__init__(NWBValueMapping({
            "PredictSaccades": EnrichmentReference("PredictSaccades")  # Required that the saccades are already in file
        }))
        self.meta = drifting_grating_metadata_read_from_filelist(drifting_grating_metadata_filenames, **drifting_kwargs)

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

        # pynwb_obj.processing["Enrichment.PredictSaccades"].containers["saccades_predicted_temporal_epochs"].data[:]
        fps = pynwb_obj.processing["Enrichment.PredictSaccades"]["saccades_fps"].data[0]

        def process_saccade_epochs(saccade_epoch: np.ndarray):
            # saccade_epoch is a (numsaccade, 2) array for the start/stop of each saccade of a particular type
            norm = (saccade_epoch / fps)[:, 0]  # Grab the start of each epoch, divide by the fps to get the frame time in seconds
            bins_idxs = np.digitize(norm, video_windows[:, 0])  # Bin each saccade epoch time into indexes into the video windows for which frame section it is within
            # The start of each saccade in terms of it's framewindow
            epochstart_framewindows = video_windows[bins_idxs]

            # TODO use the framewindows for the epochs to find which grating windows they fell within, get those idxs,
            # then relay the relevant grating information for each saccade (using idxs?)
            tw = 2

        nasal = pynwb_obj.processing["Enrichment.PredictSaccades"]["saccades_predicted_nasal_epochs"].data[:]
        temporal = pynwb_obj.processing["Enrichment.PredictSaccades"]["saccades_predicted_temporal_epochs"].data[:]

        process_saccade_epochs(nasal)
        process_saccade_epochs(temporal)

        # TODO align the saccades with the video start/stops, then use that to determine which drifting grating the saccade was in
        tw = 2
        # px.line(np.digitize(saccade_epoch[:,0], video_windows[:, 0])).show()
        # get var from req
        # self._get_req_val("PutativeSaccades.saccades_putative_waveforms", pynwb_obj)
        # self._get_req_val("myvariable_i_need", pynwb_obj)
        # for k, v in self.data.items():
        #     if isinstance(v, str):
        #         v = [v]
        #     self._save_val(k, v, pynwb_obj)

    @staticmethod
    def get_name() -> str:
        return "DriftingGrating"

    @staticmethod
    def saved_keys() -> list[str]:
        # Keys may or may not be dynamically named, assuming they aren't
        return []  # TODO

    @staticmethod
    def descriptions() -> dict[str, str]:
        return {}  # TODO


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

        # import plotly.express as px
        # px.line(self.dats["y2"].T).show()
        tw = 2

    def get_video_startstop(self):
        # Get an array of (time, 2) for the start/stop of the frames
        return startstop_of_squarewave(self.dats[self.frames_channel])[:, :2]  # Chop off the state value, only want start/stop

    # def get_gratings_startstop(self):
    #     return startstop_of_squarewave(self.dats[self.grating_channel])[:, :2]  # Chop off state value

    # def _run(self, pynwb_obj):
    #     # get var from req
    #     # self._get_req_val("PutativeSaccades.saccades_putative_waveforms", pynwb_obj)
    #     # self._get_req_val("myvariable_i_need", pynwb_obj)
    #     # for k, v in self.data.items():
    #     #     if isinstance(v, str):
    #     #         v = [v]
    #     #     self._save_val(k, v, pynwb_obj)
    #     tw = 2

    @staticmethod
    def get_name() -> str:
        return "DriftingGratingLabjack"

    @staticmethod
    def saved_keys() -> list[str]:
        # Keys may or may not be dynamically named, assuming they aren't
        return [
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
