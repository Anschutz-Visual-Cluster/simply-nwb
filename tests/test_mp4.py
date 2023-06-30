from simply_nwb import SimpleNWB
from simply_nwb.transforms import mp4_read_data
from gen_nwb import nwb_gen


def nwb_mp4_test():
    data, frames = mp4_read_data("../data/smallmp4.mp4")
    nwb = nwb_gen()
    SimpleNWB.mp4_add_as_acquisition(
        nwb,
        name="TestMovie",
        numpy_data=data,
        frame_count=frames,
        sampling_rate=30.0,  # Frames per second
        description="asdf description here"
    )
    SimpleNWB.mp4_add_as_acquisition(
        nwb,
        name="TestMovie2",
        numpy_data=data,
        frame_count=frames,
        sampling_rate=30.0,  # Frames per second
        description="asdf description here"
    )
    tw = 2

    data = nwb.acquisition["TestMovie"].data
    data = nwb.acquisition["TestMovie2"].data[:]  # To Numpy array
    return nwb, []
