from gen_nwb import nwb_gen
from simply_nwb.transforms import labjack_load_file
from simply_nwb import SimpleNWB


def nwb_labjack():
    # Not a true .dat file, should be .txt but whatever
    r = labjack_load_file("../data/labjack_data.dat")
    r2 = labjack_load_file("../data/labjack_data2.dat")

    nwbfile = nwb_gen()

    SimpleNWB.labjack_file_as_behavioral_data(
        nwbfile,
        labjack_filename="../data/labjack_data.dat",
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

def loading_labjack_testing():
    from simply_nwb.transforms.labjack import labjack_concat_files
    from pathlib import Path
    dats = list(Path("../data/anna/labjack").glob("*.dat"))
    d = labjack_concat_files(dats)
    tw = 2


# if __name__ == "__main__":
#     loading_labjack_testing()
