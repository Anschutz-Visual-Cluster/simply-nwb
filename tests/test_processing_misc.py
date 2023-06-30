import numpy as np
from dict_plus.utils.simpleflatten import SimpleFlattener

from simply_nwb import SimpleNWB
from simply_nwb.transforms import plaintext_metadata_read, csv_load_dataframe, yaml_read_file
from simply_nwb.util import panda_df_to_dyn_table
import pandas as pd
from gen_nwb import nwb_gen


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
            data_dict=SimpleFlattener(simple_types=[np.ndarray, type(None)]).flatten(data),
            uneven_columns=True)
    # nwb.processing["misc"]["asdf_eyePositionUncorrected"]["eyePositionUncorrected"].data[:]
    tw = 2


def util_test():
    # Note: This CSV isn't formatted correctly, so it will look weird when loaded
    r = panda_df_to_dyn_table(pd_df=csv_load_dataframe("../data/20230414_unitR2_session002_leftCam-0000DLC_resnet50_licksNov3shuffle1_1030000.csv"), table_name="test_table",
                              description="test description")
    tw = 2


def csv_test():
    # Loads in well, but file isn't exactly in CSV format, still a test
    r = csv_load_dataframe("../data/20230414_unitR2_session002_leftCam-0000DLC_resnet50_licksNov3shuffle1_1030000.csv")
    tw = 2


def yaml_test():
    r = yaml_read_file("../data/20230414_unitR2_session002_metadata.yaml")
    tw = 2


def plaintext_metadata_test():
    r = plaintext_metadata_read("../data/metadata.txt")
    tw = 2


def nwb_processing_module_df():
    nwb = nwb_gen()

    d = pd.DataFrame.from_dict({"data1": [1, 2, 3, 4, 5], "data2": ["a", "b", "c", "d", "e"]})

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
        data_dict={"data1": [1, 2, 3, 4, 5], "data2": ["a", "b", "c", "d", "e"]},
        uneven_columns=False
    )
    t = nwb.processing["misc"]["ProcessedDictData1"]["data1"][:]

    SimpleNWB.processing_add_dict(
        nwb,
        processed_name="ProcessedDictData2",
        processed_description="Test processing",
        data_dict={"data1": [1, 2, 3], "data2": ["a", "b", "c", "d", "e"], "aa": 5},
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
        data_dict={"data1": [1, 2, 3, 4, 5], "data2": ["a", "b", "c", "d", "e"]},
        uneven_columns=False
    )
    t = nwb.processing["misc"]["ProcessedDictData2"]["data2"][:]
    contents = nwb.processing["misc"].containers

    # Ignore the check_single_row test since could be an edgecase
    return nwb, ["check_single_row"]
