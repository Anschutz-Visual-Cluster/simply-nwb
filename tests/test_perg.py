from simply_nwb import SimpleNWB
from gen_nwb import nwb_gen


def nwb_perg():
    nwb = nwb_gen()
    SimpleNWB.p_erg_add_data(nwb, "../data/pg1_A_raw.TXT", "perg_table", description="test desc")
    SimpleNWB.p_erg_add_data(nwb, "../data/pg1_A_raw.TXT", "perg_table", description="test desc")
    SimpleNWB.p_erg_add_folder(nwb, foldername="../data/pg_folder", file_pattern="*.txt", table_name="p_ergs", description="test desc")

    t = nwb.acquisition["perg_table_data"]["average"][:]
    t = nwb.acquisition["meta_channel"].data[:]
    return nwb, ["check_regular_timestamps"]
