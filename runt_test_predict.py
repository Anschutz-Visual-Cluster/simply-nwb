import glob
import os

from simply_nwb import SimpleNWB
from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.chain import PipelineChain
from simply_nwb.pipeline.enrichments.example import ExampleEnrichment
from simply_nwb.pipeline.enrichments.saccades import PutativeSaccadesEnrichment, DriftingGratingLabjackEnrichment
from simply_nwb.pipeline.enrichments.saccades.drifting_grating.ephys import DriftingGratingEPhysEnrichment
from simply_nwb.pipeline.enrichments.saccades.predict_ml_model import PredictSaccadeMLEnrichment
import matplotlib.pyplot as plt


def train():
    # PredictSaccadeMLEnrichment.retrain([
    #     (f"{prefix}/saccade_times.csv", f"{prefix}/20241023_unitR2_session004_rightCam_timestamps.txt", f"{prefix}/20241023_unitR2_session004_rightCam-0000DLC_resnet50_pupilsizeFeb6shuffle1_1030000.csv")
    # ], "direction_model.pickle", save_to_default_model=True)
    pass


def predict(folderpath):
    dlc_timestamps = glob.glob(f"{folderpath}/**/*rightCam*timestamps*.txt", recursive=True)
    assert len(dlc_timestamps) == 1, "Should only be 1 dlc timestamps txt! Found {} instead".format(len(dlc_timestamps))
    dlc_timestamps = dlc_timestamps[0]

    dlc_eyepos = glob.glob(f"{folderpath}/**/*rightCam*DLC*.csv", recursive=True)
    assert len(dlc_eyepos) == 1, "Should only be 1 rightCam DLC csv! Found {} instead".format(len(dlc_eyepos))
    dlc_eyepos = dlc_eyepos[0]

    # First step is to load data into an NWB
    raw_nwbfile = SimpleNWB.test_nwb()  # TODO Change me to an NWB with your experiment info!

    sess = PipelineChain([
        # Enrich our nwb into 'putative' saccades (what we think *might* be a saccade)
        PutativeSaccadesEnrichment.from_raw(
            raw_nwbfile, dlc_eyepos, dlc_timestamps,
            # Need to give the units for the DLC file, if the number of columns are different than expected
            # For example if the DLC columns are like:
            # ['bodyparts_coords', 'center_x', 'center_y', 'center_likelihood', 'nasal_x', 'nasal_y', 'nasal_likelihood', 'temporal_x', 'temporal_y', 'temporal_likelihood', 'dorsal_x', 'dorsal_y', 'dorsal_likelihood', 'ventral_x', 'ventral_y', 'ventral_likelihood']
            # Then the corresponding units will be
            units=["idx", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px",
                   "likelihood", "px", "px", "likelihood"],
            # If the features tracked by DLC do not match the default, the names of the coordinates and the likelihoods will have to be overwritten
            x_center="center_x",
            y_center="center_y",
            likelihood="center_likelihood",
        ),
        PredictSaccadeMLEnrichment(),  # Prebuilt models, see run_test_saccade_gui.py for example of GUI training
    ], "predict", save_checkpoints=True, skip_existing=True).run(NWBSession(raw_nwbfile))

    eyepos = sess.pull("PutativeSaccades.processed_eyepos")[:, 0]
    nasal = sess.pull("PredictSaccades.saccades_predicted_nasal_epochs")[:, 0]
    temporal = sess.pull("PredictSaccades.saccades_predicted_temporal_epochs")[:, 0]

    plt.plot(eyepos)
    for n in nasal:
        plt.vlines(n, min(eyepos), max(eyepos), color="red", label="nasal")

    for t in temporal:
        plt.vlines(t, min(eyepos), max(eyepos), color="green", label="temporal")
    plt.show()

    tw = 2


if __name__ == "__main__":
    predict("data/anna_ephys")

