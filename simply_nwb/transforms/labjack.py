import os
from typing import Optional, Any

import numpy as np
import pendulum
import pandas as pd
from pynwb import NWBFile
from pynwb.behavior import BehavioralEvents

from simply_nwb.util import panda_df_to_list_of_timeseries, panda_df_to_dyn_table


class _LabjackMixin(object):

    @staticmethod
    def labjack_file_as_behavioral_data(
            nwbfile: NWBFile,
            labjack_filename: str,
            name: str,
            measured_unit_list: list[str],
            start_time: float,
            sampling_rate: float,
            description: str,
            behavior_module: Optional[Any] = None,
            behavior_module_name: Optional[str] = None,
            comments: str = "Labjack behavioral data"
    ) -> NWBFile:
        """
        Add LabJack data to the NWBFile as a behavioral entry from a filename

        :param nwbfile: NWBFile to add the data to
        :param labjack_filename: filename of the labjack file to load
        :param name: Name of this behavioral unit
        :param measured_unit_list: List of SI unit strings corresponding to the columns of the labjack data
        :param start_time: start time float in Hz
        :param sampling_rate: sampling rate in Hz
        :param description: description of the behavioral data
        :param behavior_module: Optional NWB behavior module to add this data to, otherwise will create a new one e.g. nwbfile.processing["behavior"]
        :param behavior_module_name: optional module name to add this behavior to, if exists will append. will ignore if behavior_module arg is supplied
        :param comments: additional comments about the data
        :return: NWBFile
        """
        return _LabjackMixin.labjack_as_behavioral_data(
            nwbfile,
            labjack_data=labjack_load_file(labjack_filename),
            name=name,
            measured_unit_list=measured_unit_list,
            start_time=start_time,
            sampling_rate=sampling_rate,
            description=description,
            behavior_module=behavior_module,
            behavior_module_name=behavior_module_name,
            comments=comments
        )

    @staticmethod
    def labjack_as_behavioral_data(
            nwbfile: NWBFile,
            labjack_data: dict,
            name: str,
            measured_unit_list: list[str],
            start_time: float,
            sampling_rate: float,
            description: str,
            behavior_module: Optional[Any],
            behavior_module_name: Optional[str],
            comments: str = "Labjack behavioral data"
    ) -> NWBFile:
        """
        Add LabJack data to the NWBFile as a behavioral entry, given the labjack data

        :param nwbfile: NWBFile to add the data to
        :param labjack_data: dict formatted like the return value from simply_nwb.transforms.labjack.labjack_load_file
        :param name: Name of this behavioral unit
        :param measured_unit_list: List of SI unit strings corresponding to the columns of the labjack data
        :param start_time: start time float in Hz
        :param sampling_rate: sampling rate in Hz
        :param description: description of the behavioral data
        :param behavior_module: Optional NWB behavior module to add this data to, otherwise will create a new one e.g. nwbfile.processing["behavior"]
        :param behavior_module_name: optional module name to add this behavior to, if exists will append. will ignore if behavior_module arg is supplied
        :param comments: additional comments about the data
        :return: NWBFile
        """
        if not isinstance(labjack_data, dict) or "metadata" not in labjack_data or "data" not in labjack_data:
            raise ValueError("Argument labjack_data should be a dict with keys 'metadata' and 'data'!")

        timeseries_list = panda_df_to_list_of_timeseries(
            pd_df=labjack_data["data"],
            measured_unit_list=measured_unit_list,
            start_time=start_time,
            sampling_rate=sampling_rate,
            description=description,
            comments=comments
        )

        behavior_events = BehavioralEvents(
            time_series=timeseries_list,
            name=f"{name}_behavioral_events"
        )

        if not behavior_module:
            if not behavior_module_name:
                behavior_module_name = "behavior"  # Default name

            if behavior_module_name in nwbfile.processing:
                behavior_module = nwbfile.processing[behavior_module_name]
            else:
                behavior_module = nwbfile.create_processing_module(
                    name=behavior_module_name,
                    description="Behavior processing module"
                )

        behavior_module.add(behavior_events)
        behavior_module.add(panda_df_to_dyn_table(
            pd_df=labjack_data["metadata"],
            table_name=f"{name}_metadata",
            description="Labjack metadata"
        ))

        return nwbfile


def _get_labjack_meta_lines(meta_lines: list[str]) -> pd.DataFrame:
    # Helper function to parse out the metadata from the labjack file, since they are in a different format than the
    # recorded data

    current_line = meta_lines.pop(0)
    col_headers = None
    rows = []

    while True:
        current_line = current_line.strip()
        if len(meta_lines) == 0:
            raise ValueError("Reached EOF during metadata scan")
        if current_line.lower().startswith("time"):
            break
        if not current_line:  # Blank line
            current_line = meta_lines.pop(0)
            continue
        if not col_headers:
            col_headers = [val.strip().split("=")[0] for val in current_line.split(",")[1:]]
            col_headers.insert(0, "channel_num")

        cols = current_line.split(",")
        # Skip first entry, as it has no header and won't .split()
        col_vals = [val.strip().split("=")[1] for val in cols[1:]]
        # Insert channel num
        col_vals.insert(0, cols[0])
        rows.append(col_vals)
        current_line = meta_lines.pop(0)

    # rows.insert(0, col_headers)
    meta_lines.insert(0, current_line)  # re-insert the data header line
    return pd.DataFrame.from_records(rows, columns=col_headers)


def labjack_load_file(filename: str) -> dict:
    """
    Returns labjack data and labjack metadata from a given filename

    :param filename: file to parse
    :return: data dict 'data' dataframe, 'metadata' dataframe, 'date' datetime
    """
    if not os.path.exists(filename):
        raise ValueError(f"File '{filename}' not found in current working path '{os.getcwd()}")

    print(f"Loading '{filename}..")

    with open(filename, "r") as f:
        lines = f.readlines()
        date = pendulum.parse(lines.pop(0).strip(), strict=False)  # First two lines are date and time
        time = pendulum.parse(lines.pop(0).strip(), strict=False, exact=False)
        date = date.set(
            hour=time.hour,
            minute=time.minute,
            second=time.second
        ).to_iso8601_string()

        # Read metadata about channels
        meta_data = _get_labjack_meta_lines(lines)
        lines = [line.strip().split("\t") for line in lines]
        data_headers = lines.pop(0)
        lines = [[float(v) for v in val] for val in lines]
        data = pd.DataFrame.from_records(lines, columns=data_headers)
        return {
            "data": data,
            "metadata": meta_data,
            "date": date
        }


def labjack_concat_files(file_list: list[str], alignment_key: str = "Time") -> dict:
    """
    Load a list of labjack files and concat them all, using a column as the alignment key (defaults to "Time")

    :returns: a dict like {col1: [data1], ...}
    """
    cols = None
    all_data = {}
    unsorted = []
    for filename in file_list:
        d = labjack_load_file(filename)
        if not cols:
            cols = d["data"].columns.tolist()
            for col in cols:
                all_data[col] = []

        unsorted.append(d)

    # Sort files by key (default is "Time") value
    print(f"Sorting loaded labjack files by alignment key: '{alignment_key}'..")
    fixed = list(sorted(unsorted, key=lambda x: x["data"][alignment_key][0]))

    for labjack_filedata in fixed:
        for col in cols:
            all_data[col].append(labjack_filedata["data"][col])

    # Turn the data into a numpy arr, I dont trust pandas/numpy so this is somewhat manual
    for col in cols:
        size = sum(len(v) for v in all_data[col])
        reshaped = np.full((size,), np.nan)
        cumsum = 0
        for idx in range(len(all_data[col])):
            series = all_data[col][idx].to_numpy()
            series_size = len(series)
            reshaped[cumsum:cumsum+series_size] = series
            cumsum = cumsum + series_size

        all_data[col] = reshaped

    print("Returning concatenated labjack array")
    return all_data  # Returns something like {"Time": <np.array>, "v0": ..., ...}
