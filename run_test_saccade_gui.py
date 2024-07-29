import os

import pendulum
from pynwb import NWBHDF5IO
from pynwb.file import Subject
from simply_nwb import SimpleNWB
from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.enrichments.saccades import PutativeSaccadesEnrichment
from simply_nwb.pipeline.enrichments.saccades.predict_gui import PredictedSaccadeGUIEnrichment


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
    return nwbfile


def main():
    # Get the filenames for the timestamps.txt and dlc CSV
    dlc_filepath = f"data/20230420_unitME_session001_rightCam-0000DLC_resnet50_GazerMay24shuffle1_1030000.csv"
    timestamp_filepath = f"data/20230420_unitME_session001_rightCam_timestamps.txt"

    if os.path.exists("putative.nwb"):
        # Create a NWB file to put our data into
        print("Creating base NWB..")
        nwbfile = create_nwb()
        SimpleNWB.write(nwbfile, "base.nwb")
        del nwbfile

        # Load our newly created 'base' NWB and put in the 'putative' saccades (what we think *might* be a saccade)
        print("Enriching with putative NWB..")
        fp = NWBHDF5IO("base.nwb", "r")
        nwbfile = fp.read()

        enrichment = PutativeSaccadesEnrichment.from_raw(nwbfile, dlc_filepath, timestamp_filepath)
        SimpleNWB.write(nwbfile, "dlcdata.nwb")
        del nwbfile

        sess = NWBSession("dlcdata.nwb")  # Save to file
        sess.enrich(enrichment)
        sess.save("putative.nwb")  # Save to file
        del sess

    sess = NWBSession("putative.nwb")
    # Take our putative saccades and do the actual prediction for the start, end time, and time location
    print("Adding predictive data..")
    enrich = PredictedSaccadeGUIEnrichment(200, ["putative.nwb"], 20)
    sess.enrich(enrich)
    print("Saving to NWB")
    sess.save("my_session_fulldata.nwb")  # Save as our finalized session, ready for analysis
    tw = 2


if __name__ == "__main__":
    main()
