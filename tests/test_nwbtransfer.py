import os
import shutil
from simply_nwb import SimpleNWB, NWBTransfer
from gen_nwb import nwb_gen


def transfer_nwb_test():
    nwb = nwb_gen()
    SimpleNWB.write(nwb, "../data/nwb_test.nwb")

    transfer0 = NWBTransfer(
        nwb_file_location="../data/nwb_test.nwb",
        raw_data_folder_location="../data/pg_folder",
        transfer_location_root="../data/",
        lab_name="MyLabName",
        project_name="test_project",
        session_name="session0"
    )
    transfer0.upload()

    transfer1 = NWBTransfer(
        nwb_file_location="../data/nwb_test.nwb",
        raw_data_folder_location="../data/pg_folder",
        transfer_location_root="../data/",
        lab_name="MyLabName",
        project_name="test_project",
        session_name="session1"
    )
    transfer1.upload(
        zip_location_override=transfer0.raw_zip_destination_filename
    )
    print("Transfer 1 finished")

    try:
        NWBTransfer(
            nwb_file_location="../data/nwb_test.nwb",
            raw_data_folder_location="../data/pg_folder",
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
    tw = 2


# transfer_nwb_test()
