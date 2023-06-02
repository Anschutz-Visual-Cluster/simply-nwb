from simply_nwb import SimpleNWB
from gen_nwb import nwb_gen


def nwb_perg():
    nwb = nwb_gen()
    SimpleNWB.add_p_erg_data(nwb, "../data/pg1_A_raw.TXT", "perg_table", description="test desc")
    SimpleNWB.add_p_erg_data(nwb, "../data/pg1_A_raw.TXT", "perg_table", description="test desc")
    SimpleNWB.add_p_erg_folder(nwb, foldername="../data/pg_folder", file_pattern="*.txt", table_name="p_ergs", description="test desc")

    t = nwb.acquisition["perg_table_data"]["average"][:]
    t = nwb.acquisition["perg_table_metadata"]["channel"][:]
    return nwb, []
