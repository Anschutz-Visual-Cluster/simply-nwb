import logging
import os

import numpy as np
import pendulum
from pynwb.file import Subject
from simply_nwb import SimpleNWB
from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.enrichments.saccades import PutativeSaccadesEnrichment
from simply_nwb.pipeline.enrichments.saccades.predict_gui import PredictedSaccadeGUIEnrichment

import plotly.express as px

from simply_nwb.pipeline.enrichments.saccades.predict_ml_model import PredictSaccadeMLEnrichment


def create_nwb():
    # Create the NWB file, TODO Put data in here about mouse and experiment
    nwbfile = SimpleNWB.create_nwb(
        # Required
        session_description="Mouse cookie eating session",
        # Subtract 1 year so we don't run into the 'NWB start time is at a greater date than current' issue
        session_start_time=pendulum.now().subtract(years=1),
        experimenter=["Schmoe, Joe"],
        lab="Felsen Lab",
        experiment_description="Gave a mouse a cookie",
        # Optional
        identifier="cookie_0",
        subject=Subject(**{
            "subject_id": "1",
            "age": "P90D",  # ISO-8601 for 90 days duration
            "strain": "TypeOfMouseGoesHere",  # If no specific used, 'Wild Strain'
            "description": "Mouse#2 idk",
            "sex": "M",  # M - Male, F - Female, U - unknown, O - other
            # NCBI Taxonomy link or Latin Binomial (e.g.'Rattus norvegicus')
            "species": "http://purl.obolibrary.org/obo/NCBITaxon_10116",
        }),
        session_id="session0",
        institution="CU Anschutz",
        keywords=["mouse"],
        # related_publications="DOI::LINK GOES HERE FOR RELATED PUBLICATIONS"
    )
    # For creating a dummy test nwb you can do SimpleNWB.test_nwb() to get an nwb object in memory
    return nwbfile


def create_putative_nwb(dlc_filepath, timestamp_filepath):
    # Create a NWB file to put our data into
    print("Creating base NWB..")
    raw_nwbfile = create_nwb()  # This is the RAW nwbfile, direct output from a 'conversion script' in this example we just make a dummy one

    # Prepare an enrichment object to be run, and insert the raw data into our nwb in memory
    enrichment = PutativeSaccadesEnrichment.from_raw(
        raw_nwbfile, dlc_filepath, timestamp_filepath,
        # Need to give the units for the DLC file, if the number of columns are different than expected
        # For example if the DLC columns are like:
        # ['bodyparts_coords', 'center_x', 'center_y', 'center_likelihood', 'nasal_x', 'nasal_y', 'nasal_likelihood', 'temporal_x', 'temporal_y', 'temporal_likelihood', 'dorsal_x', 'dorsal_y', 'dorsal_likelihood', 'ventral_x', 'ventral_y', 'ventral_likelihood']
        # Then the corresponding units will be
        units=["idx", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood"],
        # If the features tracked by DLC do not match the default, the names of the coordinates and the likelihoods will have to be overwritten
        x_center="center_x",
        y_center="center_y",
        likelihood="center_likelihood",
    )

    sess = NWBSession(raw_nwbfile)  # Create a session using our in-memory NWB
    # Enrich our nwb into 'putative' saccades (what we think *might* be a saccade)
    print("Enriching to putative NWB..")
    sess.enrich(enrichment)

    sess.save("putative.nwb")  # Save to file


def graph_saccades(sess: NWBSession):
    print(sess.available_enrichments())
    print(sess.available_keys("PredictSaccades"))
    nasal = sess.pull("PredictSaccades.saccades_predicted_nasal_waveforms")[:, :, 0]
    temporal = sess.pull("PredictSaccades.saccades_predicted_temporal_waveforms")[:, :, 0]

    # plotly code
    colors = []
    colors.extend(["blue"]*len(nasal))
    colors.extend(["orange"]*len(temporal))
    px.line(np.vstack([nasal, temporal]).T, color_discrete_sequence=colors).show()

    # matplotlib code
    # [plt.plot(d, color="orange") for d in temporal]
    # [plt.plot(d, color="blue") for d in nasal]
    # plt.show()

    epochs_nasal = sess.pull("PredictSaccades.saccades_predicted_nasal_epochs")
    nasal_peaks = sess.pull("PredictSaccades.saccades_predicted_nasal_peak_indices")
    startstop_idxs = ((epochs_nasal - nasal_peaks[:, None]) + 40)  # 40 is half of the 80ms timewindow captured around the saccade, since they are centered

    # plotly code
    fig = px.line(nasal[:10].T, color_discrete_sequence=["blue"]*10)
    for i in range(10):
        fig.add_vline(startstop_idxs[i, 0])
        fig.add_vline(startstop_idxs[i, 1])
    fig.show()

    # matplotlib code
    # for i in range(10):
    #     plt.plot(nasal[i])
    #     plt.vlines(startstop_idxs[i, 0], np.min(nasal[i]), np.max(nasal[i]))
    #     plt.vlines(startstop_idxs[i, 1], np.min(nasal[i]), np.max(nasal[i]))
    # plt.show()

    tw = 2


def main(dlc_filepath, timestamp_filepath, drifting_grating_filepath):
    ### REMOVE THIS FOR ACTUAL TRAINING
    os.environ["NWB_DEBUG"] = "True"  # NOTE ONLY USE TO QUICKLY TRAIN A MODEL (not for real data)
    ####
    num_samples = 20  # YOU WILL WANT TO CHANGE THIS TO 100+ FOR ACTUAL USE!

    create_putative_nwb(dlc_filepath, timestamp_filepath)
    sess = NWBSession("putative.nwb")  # Load in the session we would like to enrich to predictive saccades

    print(f"Available funcs from PutativeSaccades:")
    sess.print_funclist('PutativeSaccades')

    dropped = sess.func("PutativeSaccades.dropped_frames")("rightCamTimestamps")
    print(f"Dropped frames: {dropped}")

    # Take our putative saccades and do the actual prediction for the start, end time, and time location
    print("Adding predictive data..")

    # For 'putative_kwargs' put the arguments in dict kwarg style for the PutativeSaccadesEnrichment() if it is non-default. Format like {"x_center": .., }
    # this example there are no extra kwargs, but if you name your DLC columns different, you will need to tell it which column names relate to your data
    # Columns will be concatenated with the above column, so something like
    # a,b,c
    # x,y,z
    # will be turned into the keys a_x, b_y, and z_c
    # so you would use {"x_center": "a_x", ..}

    # Code to use the interactive GUI for creating a model for determining saccade v noise and epochs
    # predict_enrich = PredictedSaccadeGUIEnrichment(200, ["putative.nwb", "putative.nwb"], num_samples, putative_kwargs={
    # If the features tracked by DLC do not match the default, the names of the coordinates and the likelihoods will have to be overwritten

    predict_enrich = PredictSaccadeMLEnrichment()  # Prebuilt models

    # This will open two guis, where you will identify which direction the saccade is, and what the start and stop is
    # when the gui data entry is done, it will begin training the classifier models. The models are saved so if
    # something breaks it can be re-started easily
    # Next time you enrich an NWB, if the model files are in the directory where this script is being run, it will use
    # those instead of training a new model
    # You will need to label at least 10 directional saccade, at least 2 of each direction (BARE MINIMUM, NOT GOOD FOR ACTUAL DATA)
    # and at least 10 epochs
    # To label direction, click the radio button (circle button) on the left and then click next
    # To label epoch timing, select start/stop and move the line with the arrow keys to the approximate start/stop of the saccade
    sess.enrich(predict_enrich)
    print("Saving predicted..")
    sess.save("predicted.nwb")  # Save as our finalized session, ready for analysis

    graph_saccades(sess)
    tw = 2


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    # drifting_grating_enrichment_testing()
    # DOWNLOAD EXAMPLE DATA: https://drive.google.com/file/d/1tQwGY4NG8EC37rE4gxCcxmIGIpxMpVxv/view?usp=sharing

    dlc_filepath = "data/extraction/nr1ko/20241023_unitR2_session004_rightCam-0000DLC_resnet50_pupilsizeFeb6shuffle1_1030000.csv"
    timestamp_filepath = "data/extraction/nr1ko/20241023_unitR2_session004_rightCam_timestamps.txt"
    drifting_grating_filepath = ""

    main(dlc_filepath, timestamp_filepath, drifting_grating_filepath)


