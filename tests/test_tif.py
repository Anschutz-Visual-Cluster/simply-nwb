from simply_nwb import SimpleNWB
from simply_nwb.transforms import tif_read_directory, tif_read_image, tif_read_subfolder_directory
from gen_nwb import nwb_gen


def tif_test():
    r = tif_read_image("../data/tifs/subfolder_formatted/file/Image.tif")

    rr = tif_read_subfolder_directory(
        parent_folder="../data/tifs/subfolder_formatted",
        subfolder_glob="file*",
        file_glob="Image.tif"
    )

    rrr = tif_read_directory("../data/tifs/folder_formatted", filename_glob="*ome.tif")
    tw = 2


def nwb_two_photon():
    nwb = nwb_gen()
    data = tif_read_directory("../data/tifs/folder_formatted", filename_glob="*ome.tif")

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
