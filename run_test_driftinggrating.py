import logging
import os
import re
from pathlib import Path

import numpy as np
import pendulum
from pynwb import NWBHDF5IO
from pynwb.file import Subject
from simply_nwb import SimpleNWB
from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.enrichments.saccades import PutativeSaccadesEnrichment
from simply_nwb.pipeline.enrichments.saccades.drifting_grating import DriftingGratingLabjackEnrichment
from simply_nwb.pipeline.enrichments.saccades.predict_gui import PredictedSaccadeGUIEnrichment
import matplotlib.pyplot as plt

from simply_nwb.pipeline.enrichments.saccades.predict_ml_model import PredictSaccadeMLEnrichment
from simply_nwb.pipeline.enrichments.saccades.predicted_algorithm import PredictedSaccadeAlgoEnrichment
import plotly.express as px


def processing_drifting_grating(folderpath):
    if not os.path.exists("drift_predict.nwb"):
        raw_nwbfile = SimpleNWB.test_nwb()  # TODO Replace with your actual data's NWB!! See run_test_saccade_gui.py
        sess = NWBSession(raw_nwbfile)
        timestamp_filepath = str(list(Path().glob(f"{folderpath}/*timestamps*.txt"))[0])
        dlc_filepath = str(list(Path().glob(f"{folderpath}/*DLC*.csv"))[0])

        putative = PutativeSaccadesEnrichment.from_raw(
            raw_nwbfile, dlc_filepath, timestamp_filepath,
            # Need to give the units for the DLC file, if the number of columns are different than expected
            # For example if the DLC columns are like the below, the corresponding units will be
            # ['bodyparts_coords', 'center_x', 'center_y', 'center_likelihood', 'nasal_x', 'nasal_y', 'nasal_likelihood', 'temporal_x', 'temporal_y', 'temporal_likelihood', 'dorsal_x', 'dorsal_y', 'dorsal_likelihood', 'ventral_x', 'ventral_y', 'ventral_likelihood']
            units=["idx", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood"],
            # If the features tracked by DLC do not match the default, the names of the coordinates and the likelihoods will have to be overwritten
            x_center="center_x",
            y_center="center_y",
            likelihood="center_likelihood",
        )
        sess.enrich(putative)

        # predict_enrich = PredictedSaccadeGUIEnrichment(200, ["putative.nwb", "putative.nwb"], 40, putative_kwargs={
        #     # If the features tracked by DLC do not match the default, the names of the coordinates and the likelihoods will have to be overwritten
        #     "x_center": "center_x",
        #     "y_center": "center_y",
        #     "likelihood": "center_likelihood",
        # })
        predict_enrich = PredictSaccadeMLEnrichment()
        sess.enrich(predict_enrich)
        sess.save("drift_predict.nwb")

    if not os.path.exists("drift_labjack.nwb"):
        sess = NWBSession("drift_predict.nwb")
        drift = DriftingGratingLabjackEnrichment(Path().glob(f"{folderpath}/driftingGratingMetadata-*.txt"), Path().glob(f"{folderpath}/labjack/*.dat"))
        sess.enrich(drift)
        sess.save("drift_labjack.nwb")
    else:
        sess = NWBSession("drift_labjack.nwb")

    print("Description dict:")
    print(sess.description("DriftingGratingLabjack"))
    print("Available keys:")
    print(sess.available_keys("DriftingGratingLabjack"))
    # The long name still works
    eventdata = sess.pull("DriftingGratingLabjack.Event (1=Grating, 2=Motion, 3=Probe, 4=ITI)")
    timestamps = sess.pull("DriftingGratingLabjack.Timestamp")
    print("Available funcs:")
    sess.print_funclist("DriftingGratingLabjack")
    # eventdata and timestamps align
    num5 = sess.func("DriftingGratingLabjack.nasal_saccade_info")(5)  # Get info about nasal saccade number 5
    allsacc = sess.func("DriftingGratingLabjack.nasal_saccade_info")()  # Get info about all nasal saccades
    specific = sess.func("DriftingGratingLabjack.temporal_saccade_info")([1, 2, 3])  # Get info about a specific subset of saccades
    tw = 2


if __name__ == "__main__":
    # Point this function to a directory with driftingGrating.txts, labjack.dat's, timestamp.csvs and dlc.csvs
    # An example folder is available at: https://drive.google.com/file/d/1OhUccF4V8yYT-fOj54wyzanCHqGPyXSP/view?usp=sharing
    processing_drifting_grating("data/drifting_debug")

