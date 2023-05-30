from pynwb.file import Subject

from simply_nwb.acquisition.tools import mp4_read_data
from simply_nwb.acquisition.tools import tif_read_image, tif_read_directory, tif_read_subfolder_directory
from simply_nwb.acquisition.tools import csv_load_dataframe, yaml_read_file
from simply_nwb.acquisition.tools import labjack_load_file
from simply_nwb.acquisition.tools import plaintext_metadata_read
from simply_nwb.acquisition.tools import blackrock_load_data, blackrock_all_spiketrains
from simply_nwb.util import panda_df_to_dyn_table
from simply_nwb import SimpleNWB
import pendulum
import os
import shutil
import numpy as np
import pandas as pd

from transferring import NWBTransfer

# Data is available on Google Drive, as Spencer for access
blackrock_nev_filename = "../data/wheel_4p3_lSC_2001.nev"
perg_filename = "../data/pg1_A_raw.TXT"
perg_folder = "../data/pg_folder"
labjack_filename = "../data/labjack_data.dat"
labjack_filename2 = "../data/labjack_data2.dat"
metadata_filename = "../data/metadata.txt"
yaml_filename = "../data/20230414_unitR2_session002_metadata.yaml"
mp4_filename = "../data/smallmp4.mp4"
tif_foldername_folder_fmt = "../data/tifs/folder_formatted"
tif_subfolder_kwargs = {"parent_folder": "../data/tifs/subfolder_formatted",
                        "subfolder_glob": "file*", "file_glob": "Image.tif"}
tif_single_filename = "../data/tifs/subfolder_formatted/file/Image.tif"
# Note: This CSV isn't formatted correctly so it will look weird when loaded
csv_filename = "../data/20230414_unitR2_session002_leftCam-0000DLC_resnet50_licksNov3shuffle1_1030000.csv"
dict_data = {"data1": [1, 2, 3, 4, 5], "data2": ["a", "b", "c", "d", "e"]}
dict_data_uneven_cols = {"data1": [1, 2, 3], "data2": ["a", "b", "c", "d", "e"], "aa": 5}
nwb_save_filename = "../data/nwb_test.nwb"


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


def nwb_nev():
    nwb = nwb_gen()
    SimpleNWB.blackrock_spiketrains_as_units(
        nwb,
        blackrock_filename=blackrock_nev_filename,
        device_description="BlackRock device hardware #123",
        electrode_description="Electrode desc",
        electrode_location_description="location description",
        electrode_position=(0.1, 0.2, 0.3),
        # stereotaxic position of this electrode group (x, y, z) (+y is inferior)(+x is posterior)(+z is right) (required)
        electrode_impedance=0.4,  # Impedance in ohms
        electrode_brain_region="brain region desc",
        electrode_filtering_description="filtering, thresholds description",
        electrode_reference_description="stainless steel skull screw",
        # Description of the reference electrode and/or reference scheme used for this electrode, e.g.,"stainless steel skull screw" or "online common average referencing"

        # Optional args
        device_manufacturer="BlackRock",
        device_name="BlackRock#4",
        electrode_group_name="electrodegroup0"
    )

    t = nwb.units["spike_times"][:] # List of spike times

    return nwb, []


def nwb_perg():
    nwb = nwb_gen()
    SimpleNWB.add_p_erg_data(nwb, perg_filename, "perg_table", description="test desc")
    SimpleNWB.add_p_erg_data(nwb, perg_filename, "perg_table", description="test desc")
    SimpleNWB.add_p_erg_folder(nwb, foldername=perg_folder, file_pattern="*.txt", table_name="p_ergs", description="test desc")

    t = nwb.acquisition["perg_table_data"]["average"][:]
    t = nwb.acquisition["perg_table_metadata"]["channel"][:]
    return nwb, []


def nwb_labjack():
    # Not a true .dat file, should be .txt but whatever
    r = labjack_load_file(labjack_filename)
    r2 = labjack_load_file(labjack_filename2)

    nwbfile = nwb_gen()

    SimpleNWB.labjack_as_behavioral_data(
        nwbfile,
        labjack_filename=labjack_filename,
        name="labjack_file_1",
        measured_unit_list=["idk units"]*9,  # 9 columns for data collected
        start_time=0.0,
        sampling_rate=1.0,
        description="Sampled at 1hz some data description here",
        behavior_module=None,
        behavior_module_name=None,
        comments="Labjack behavioral data"
    )
    t = nwbfile.processing["behavior"]["labjack_file_1_behavioral_events"]["Time"].data[:]
    t = nwbfile.processing["behavior"]["labjack_file_1_behavioral_events"]["v0"].data
    t = nwbfile.processing["behavior"]["labjack_file_1_metadata"]
    t = nwbfile.processing["behavior"]["labjack_file_1_metadata"]["CH+"]
    return nwbfile, []


def nwb_two_photon():
    nwb = nwb_gen()
    data = tif_read_directory(tif_foldername_folder_fmt, filename_glob="*ome.tif")

    SimpleNWB.two_photon_add_data(
        nwb,
        device_name="MyMicroscope",
        device_description="The coolest microscope ever",
        device_manufacturer="CoolMicroscopes Inc",
        optical_channel_description="an optical channel",
        optical_channel_emission_lambda=500.0,  # in nm
        imaging_name="my_data",
        imaging_rate=30.0,  # images acquired in Hz
        excitation_lambda=600.0,  # wavelength in nm
        indicator="GFP",
        location="V1",
        grid_spacing=[0.1, 0.1],
        grid_spacing_unit="meters",
        origin_coords=[0.1, 0.1],
        origin_coords_unit="meters",
        photon_series_name="MyTwoPhotonSeries",
        two_photon_unit="normalized amplitude",
        two_photon_rate=1.0,  # sampling rate in Hz
        image_data=data
    )
    # nwb.acquisition["TwoPhotonSeries"].data

    # Ignore the check_data_orientation check
    t = nwb.acquisition["MyTwoPhotonSeries"].data[:]
    return nwb, ["check_data_orientation"]


def nwb_processing_module_df():
    nwb = nwb_gen()

    d = pd.DataFrame.from_dict(dict_data)

    SimpleNWB.processing_add_dataframe(
        nwb,
        processed_name="ProcessedData",
        processed_description="Test processing",
        data=d
    )

    # Extra to test multiple adds
    SimpleNWB.processing_add_dataframe(
        nwb,
        processed_name="ProcessedData2",
        processed_description="Test processing",
        data=d
    )

    t = nwb.processing["misc"]["ProcessedData"]["data1"]
    return nwb, []


def nwb_processing_module_dict():
    nwb = nwb_gen()

    SimpleNWB.processing_add_dict(
        nwb,
        processed_name="ProcessedDictData1",
        processed_description="Test processing",
        data_dict=dict_data,
        uneven_columns=False
    )
    t = nwb.processing["misc"]["ProcessedDictData1"]["data1"][:]

    SimpleNWB.processing_add_dict(
        nwb,
        processed_name="ProcessedDictData2",
        processed_description="Test processing",
        data_dict=dict_data_uneven_cols,
        uneven_columns=True
    )
    # Because of uneven columns, each key is separate
    t = nwb.processing["misc"]["ProcessedDictData2_data1"]["data1"][:]
    t = nwb.processing["misc"]["ProcessedDictData2_data2"]["data2"][:]
    t = nwb.processing["misc"]["ProcessedDictData2_aa"]["aa"][:]

    # Add another dict
    SimpleNWB.processing_add_dict(
        nwb,
        processed_name="ProcessedDictData2",
        processed_description="Test processing",
        data_dict=dict_data,
        uneven_columns=False
    )
    t = nwb.processing["misc"]["ProcessedDictData2"]["data2"][:]
    contents = nwb.processing["misc"].containers

    # Ignore the check_single_row test since could be an edgecase
    return nwb, ["check_single_row"]


def nwb_mp4_test():
    data, frames = mp4_read_data(mp4_filename)
    # data, frames = mp4_read_data("../data/mp4_test.mp4")
    nwb = nwb_gen()
    SimpleNWB.mp4_add_as_behavioral(
        nwb,
        name="TestMovie",
        numpy_data=data,
        frame_count=frames,
        sampling_rate=30.0,  # Frames per second
        description="asdf description here"
    )
    tw = 2
    # nwb_to_inspect.processing["behavior"]["TestMovie"].data[0]
    return nwb, []


def blackrock_test():
    d = blackrock_load_data(blackrock_nev_filename)
    d2 = blackrock_all_spiketrains(blackrock_nev_filename)
    tw = 2


def plaintext_metadata_test():
    r = plaintext_metadata_read(metadata_filename)
    tw = 2


def csv_test():
    # Loads in well, but file isn't exactly in CSV format, still a test
    r = csv_load_dataframe(csv_filename)
    tw = 2


def yaml_test():
    r = yaml_read_file(yaml_filename)
    tw = 2


def tif_test():
    r = tif_read_image(tif_single_filename)
    rr = tif_read_subfolder_directory(**tif_subfolder_kwargs)
    rrr = tif_read_directory(tif_foldername_folder_fmt, filename_glob="*ome.tif")
    tw = 2


def util_test():
    # Note: This CSV isn't formatted correctly, so it will look weird when loaded
    r = panda_df_to_dyn_table(pd_df=csv_load_dataframe(csv_filename), table_name="test_table",
                              description="test description")
    tw = 2


def pkl_test():
    import pickle
    fn = "../data/output.pkl"
    fp = open(fn, "rb")
    data = pickle.load(fp)
    nwb = nwb_gen()
    SimpleNWB.processing_add_dict(
            nwb,
            processed_name="asdf",
            processed_description="asdf",
            data_dict=data,
            uneven_columns=True)
    # nwb.processing["misc"]["asdf_eyePositionUncorrected"]["eyePositionUncorrected"].data[:]
    tw = 2


def transfer_nwb_test():
    nwb = nwb_gen()
    SimpleNWB.write(nwb, nwb_save_filename)
    transfer0 = NWBTransfer(
        nwb_file_location=nwb_save_filename,
        raw_data_folder_location=perg_folder,
        transfer_location_root="../data/",
        lab_name="MyLabName",
        project_name="test_project",
        session_name="session0"
    )
    transfer0.upload()

    transfer1 = NWBTransfer(
        nwb_file_location=nwb_save_filename,
        raw_data_folder_location=perg_folder,
        transfer_location_root="../data/",
        lab_name="MyLabName",
        project_name="test_project",
        session_name="session1"
    )
    transfer1.upload(
        zip_location_override=transfer0.zip_file_location
    )
    print("Transfer 1 finished")

    try:
        NWBTransfer(
            nwb_file_location=nwb_save_filename,
            raw_data_folder_location=perg_folder,
            transfer_location_root="../data/",
            lab_name="MyLabName",
            project_name="test_project",
            session_name="session1"
        )
        raise Exception("Test NWBTransfer transfer_fail should fail, but didn't!")
    except ValueError as e:
        tw = 2
        # Test is supposed to fail
        pass

    print("Clearing test folder and related test things")
    shutil.rmtree("../data/MyLabName")
    os.mkdir("../data/MyLabName")


if __name__ == "__main__":
    # util_test()
    # blackrock_test()
    # csv_test()
    # plaintext_metadata_test()
    # yaml_test()
    # tif_test()
    # pkl_test()
    # transfer_nwb_test()


    funcs_to_assert = [
        # nwb_nev,
        # nwb_perg,
        # nwb_labjack,
        # nwb_two_photon,
        # nwb_processing_module_df,
        # nwb_processing_module_dict,
        nwb_mp4_test
    ]

    SimpleNWB.inspect(nwb_gen())

    for func in funcs_to_assert:
        nwb_to_inspect, ignore_error_names = func()
        results = SimpleNWB.inspect(nwb_to_inspect)

        # Remove ignored errors
        idxs_to_pop = []
        for idx, res in enumerate(results):
            for ignore in ignore_error_names:
                if res.check_function_name == ignore:
                    idxs_to_pop.append(idx)
        idxs_to_pop.reverse()
        [results.pop(i) for i in idxs_to_pop]

        # Should return '[]' so anything not [] will assert False
        assert not results
        print("Assert pass")
    print("All tests passed")
