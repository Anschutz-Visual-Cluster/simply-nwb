from pynwb.file import Subject

from simply_nwb.acquisition.tools import blackrock_load_data, blackrock_all_spiketrains
from simply_nwb import SimpleNWB
import pendulum


blackrock_nev_filename = "../data/wheel_4p3_lSC_2001.nev"
perg_filename = "../data/pg1_A_raw.TXT"


def blackrock_util_funcs():
    d = blackrock_load_data(blackrock_nev_filename)
    d2 = blackrock_all_spiketrains(blackrock_nev_filename)
    tw = 2


def gen_snwb():
    return SimpleNWB(
        # Required
        session_description="Poked mouse with a stick",
        session_start_time=pendulum.now(),
        experimenter=["Schmoe, Joe"],
        lab="Felsen Lab",
        experiment_description="Poked a mouse with sticks to see if they would react",

        # Optional
        identifier="Mouse stick poke experiment",
        subject=Subject(**{
            "subject_id": "1",
            "age": "P90D",  # ISO-8601 for 90 days duration
            "strain": "TypeOfMouseGoesHere",  # If no specific used, 'Wild Strain'
            "description": "Mouse#2 idk",
            "sex": "M",  # M - Male, F - Female, U - unknown, O - other
            # NCBI Taxonomy link or Latin Binomial (e.g.'Rattus norvegicus')
            "species": "http://purl.obolibrary.org/obo/NCBITaxon_10116",
        }),
        # subject={
        #     "subject_id": "1",
        #     "age": "P90D",  # ISO-8601 for 90 days duration
        #     "strain": "TypeOfMouseGoesHere",  # If no specific used, 'Wild Strain'
        #     "description": "Mouse#1 idk",
        #     "sex": "M",  # M - Male, F - Female, U - unknown, O - other
        #     # NCBI Taxonomy link or Latin Binomial (e.g.'Rattus norvegicus')
        #     "species": "http://purl.obolibrary.org/obo/NCBITaxon_10116",
        # },
        session_id="Session1",
        institution="CU Anschutz",
        keywords=["mouse"],

        # related_publications="DOI::LINK GOES HERE FOR RELATED PUBLICATIONS"
    )


def simple_nwb_nev():
    snwb = gen_snwb()
    snwb.blackrock_spiketrains_as_units(
        blackrock_filename=blackrock_nev_filename,
        device_description="BlackRock device hardward #123",
        electrode_description="Electrode desc",
        electrode_location_description="location description",
        electrode_position=(0.1, 0.2, 0.3),
        # stereotaxic position of this electrode group (x, y, z) (+y is inferior)(+x is posterior)(+z is right) (required)
        electrode_impedance=0.4,  # Impedance in ohms
        electrode_brain_region="brain region desc",
        electrode_filtering_description="filtering, thresholds descrpition",
        electrode_reference_description="stainless steel skull screw",
        # Description of the reference electrode and/or reference scheme used for this electrode, e.g.,"stainless steel skull screw" or "online common average referencing"

        # Optional args
        device_manufacturer="BlackRock",
        device_name="BlackRock#4",
        electrode_group_name="electrodegroup0"
    )
    return snwb


def nwb_perg():
    snwb = gen_snwb()
    snwb.add_p_erg_data(perg_filename, "perg_table")
    snwb.add_p_erg_data(perg_filename, "perg_table")

    # snwb.add_p_erg_data(perg_filename, "perg_table", reformat_column_names=False)
    return snwb


def nwb_perg_folder():
    snwb = gen_snwb()
    snwb.add_p_erg_folder(
        foldername="../data/pg_folder",
        file_pattern="*.txt",
        table_name="p_ergs",
    )
    return snwb


if __name__ == "__main__":
    blackrock_util_funcs()

    assert not gen_snwb().inspect()
    assert not simple_nwb_nev().inspect()
    assert not nwb_perg().inspect()
    assert not nwb_perg_folder().inspect()
