from simply_nwb import SimpleNWB
import pendulum
from pynwb.file import Subject


def nwb_gen():
    return SimpleNWB.create_nwb(
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
