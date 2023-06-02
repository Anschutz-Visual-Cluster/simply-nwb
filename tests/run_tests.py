from simply_nwb import SimpleNWB
from test_nwbtransfer import transfer_nwb_test
from test_processing_misc import util_test, yaml_test, csv_test, pkl_test, plaintext_metadata_test
from test_processing_misc import nwb_processing_module_dict, nwb_processing_module_df
from test_nev import nev_test, nwb_nev
from test_tif import tif_test, nwb_two_photon
from test_perg import nwb_perg
from test_labjack import nwb_labjack
from test_mp4 import nwb_mp4_test
from gen_nwb import nwb_gen


if __name__ == "__main__":
    util_test()
    nev_test()
    csv_test()
    plaintext_metadata_test()
    yaml_test()
    tif_test()
    pkl_test()
    transfer_nwb_test()

    # Standalone test, will hang until killed
    # filesync_test()

    funcs_to_assert = [
        nwb_nev,
        nwb_perg,
        nwb_labjack,
        nwb_two_photon,
        nwb_processing_module_df,
        nwb_processing_module_dict,
        nwb_mp4_test
    ]

    SimpleNWB.inspect_obj(nwb_gen())

    for func in funcs_to_assert:
        nwb_to_inspect, ignore_error_names = func()
        results = SimpleNWB.inspect_obj(nwb_to_inspect)

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
