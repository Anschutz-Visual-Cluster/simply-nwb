import glob
import os

from simply_nwb import SimpleNWB
from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.chain import PipelineChain
from simply_nwb.pipeline.enrichments.example import ExampleEnrichment
from simply_nwb.pipeline.enrichments.saccades import PutativeSaccadesEnrichment, DriftingGratingLabjackEnrichment
from simply_nwb.pipeline.enrichments.saccades.drifting_grating.ephys import DriftingGratingEPhysEnrichment
from simply_nwb.pipeline.enrichments.saccades.predict_ml_model import PredictSaccadeMLEnrichment


def ephys_align(folderpath):
    # Load the neuropixels event timestamps that were sent by labjack to get a reference for the np (neuropixels) clock
    # NOTE: This file was called timestamps.npy in GUI version 0.5.X!!!
    np_barcode = glob.glob(f"{folderpath}/**/*AP*/*TTL*/*sample_numbers*.npy", recursive=True)
    assert len(np_barcode) == 1, "Found multiple neuropixels barcode files! Manually specifying recommended!"
    np_barcode = np_barcode[0]

    np_spike_clusts = glob.glob("data/anna_ephys/**/spike_clusters.npy", recursive=True)
    assert len(np_spike_clusts) == 1, "Should only be 1 spike_clusters.npy from kilosort output"
    np_spike_clusts = np_spike_clusts[0]

    np_spike_times = glob.glob("data/anna_ephys/**/spike_times.npy", recursive=True)
    assert len(np_spike_times) == 1, "Should only be 1 spike_times.npy from kilosort output"
    np_spike_times = np_spike_times[0]

    labjack = glob.glob(f"{folderpath}/**/labjack/*.dat", recursive=True)
    assert len(labjack) > 0, "No labjack files found!"

    drifting = glob.glob(f"{folderpath}/**/driftingGratingMetadata*.txt", recursive=True)
    assert len(drifting) > 0, "No driftingGratingMetadata files found!"

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
        DriftingGratingLabjackEnrichment(drifting, labjack, skip_sparse_noise=True, sparse_noise_pulsecount_offset=340),
        DriftingGratingEPhysEnrichment(
            np_barcode,
            np_spike_clusts,
            np_spike_times,
            labjack_barcode_channel="y0"
        )
    ], "ephys", save_checkpoints=True, skip_existing=True).run(NWBSession(raw_nwbfile))


    tw = 2


if __name__ == "__main__":
    ephys_align("data/anna_ephys")

