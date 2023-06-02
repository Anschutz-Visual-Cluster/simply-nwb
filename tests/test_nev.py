from simply_nwb.transforms import blackrock_load_data, blackrock_all_spiketrains
from simply_nwb import SimpleNWB
from gen_nwb import nwb_gen


def nwb_nev():
    nwb = nwb_gen()
    SimpleNWB.blackrock_spiketrains_as_units(
        nwb,
        blackrock_filename="../data/wheel_4p3_lSC_2001.nev",
        device_description="BlackRock device hardware #123",
        electrode_description="Electrode desc",
        electrode_location_description="location description",
        electrode_position=(0.1, 0.2, 0.3),
        # stereotaxic position of this electrode group (x, y, z)
        # (+y is inferior)(+x is posterior)(+z is right) (required)
        electrode_impedance=0.4,  # Impedance in ohms
        electrode_brain_region="brain region desc",
        electrode_filtering_description="filtering, thresholds description",
        electrode_reference_description="stainless steel skull screw",
        # Description of the reference electrode and/or reference scheme used for this electrode,
        # e.g.,"stainless steel skull screw" or "online common average referencing"

        # Optional args
        device_manufacturer="BlackRock",
        device_name="BlackRock#4",
        electrode_group_name="electrodegroup0"
    )

    t = nwb.units["spike_times"][:]  # List of spike times

    return nwb, []


def nev_test():
    d = blackrock_load_data("../data/wheel_4p3_lSC_2001.nev")
    d2 = blackrock_all_spiketrains("../data/wheel_4p3_lSC_2001.nev")
    tw = 2
