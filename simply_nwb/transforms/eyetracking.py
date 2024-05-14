import os

import pandas as pd
from pynwb import NWBFile

from simply_nwb.transforms import csv_load_dataframe_str
from simply_nwb.util import panda_df_to_list_of_timeseries


class _EyetrackingMixin(object):
    @staticmethod
    def eyetracking_add_to_processing(
            nwbfile: NWBFile,
            dlc_filepath: str,
            module_name: str,
            units: list[str] = None,
            sampling_rate: float = 200.0,
            comments: str = None,
            description: str = None
    ) -> NWBFile:

        if units is None:
            # units are the index, then pixels, then percent likelihood by default
            units = ["idx", "px", "px", "likelihood"]
        if description is None:
            description = "Processed eyetracking data for {}".format(module_name)
        if comments is None:
            comments = ""

        response_df = eyetracking_load_dlc(dlc_filepath)

        response_ts = panda_df_to_list_of_timeseries(
            response_df,
            measured_unit_list=units,
            start_time=0.0,
            sampling_rate=sampling_rate,
            description=description,
            comments=comments,
        )
        # Add the timeseries into the processing module
        from simply_nwb import SimpleNWB
        [
            SimpleNWB.add_to_processing_module(nwbfile, ts, module_name, description)
            for ts in response_ts
        ]

        return nwbfile


def eyetracking_load_dlc(dlc_filepath: str) -> pd.DataFrame:
    """
    Load eyetracking data from DLC into a Pandas Dataframe
    :param dlc_filepath:
    :return:
    """
    if not os.path.exists(dlc_filepath):
        raise ValueError(f"File '{dlc_filepath}' not found in current working path '{os.getcwd()}")

    io = open(dlc_filepath, "r")
    lines = io.readlines()

    lines.pop(0)  # First line is not important, just 'scorer,<resnet name>, ...'
    col_prefixes = lines.pop(0).split(",")  # Next line is the prefixes of the columns
    col_suffixes = lines.pop(0).split(",")

    # Combine the column names, insert back into data list
    headers = [f"{col_prefixes[i].strip()}_{col_suffixes[i].strip()}" for i in range(0, len(col_prefixes))]
    lines.insert(0, ",".join(headers))

    # Load CSV into a dataframe, convert to TimeSeries
    response_df = csv_load_dataframe_str("\n".join(lines))
    return response_df
