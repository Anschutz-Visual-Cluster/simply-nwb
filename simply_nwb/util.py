from hdmf.common import DynamicTable, VectorData
from nwbinspector import inspect_nwbfile, inspect_nwbfile_object
from pynwb import NWBHDF5IO, TimeSeries
import warnings
import re


def read_nwb(filename):
    """
    Read a file from the filesystem into an NWB object

    :param filename: filename of an NWB file
    :return: file pointer ready to be .read() to get the nwb object
    """
    # Can't use context manager, will close file, return file pointer
    io = NWBHDF5IO(filename, mode="r")
    return io


def write_nwb(nwb_obj, filename):
    """
    Write an NWB object to a file on the local filesystem

    :param nwb_obj: pynwb.file.NWBFile object
    :param filename: path of a local file, doesn't need to exist
    :return: None
    """
    io = NWBHDF5IO(filename, mode="w")
    io.write(nwb_obj)
    io.close()


def warn_on_name_format(name_value, context_str=""):
    """
    Send a warning if the name format isn't in 'snake_case'

    :param name_value: value to check
    :param context_str: Extra string to put in warning message
    :return: True if passes, False otherwise
    """
    is_snake = True
    is_snake = name_value.lower() == name_value and is_snake
    # Check for any characters other than 'a-z' '_' and '0-9'
    is_snake = not bool(re.findall("[^a-z_0-9]", name_value)) and is_snake

    if not is_snake:
        warnings.warn(f"Name '{name_value} isn't in snake_case! {context_str}", UserWarning)
        return False
    return True


def inspect_nwb_file(filename):
    """
    Return the inspection list of a given NWB file

    :param filename: filename of the NWB to inspect
    :return: list of inspection objects for the given NWB, if empty, no issues found
    """
    return list(inspect_nwbfile(nwbfile_path=filename))


def inspect_nwb_obj(obj):
    """
    Return the inspection list of a given NWB object

    :param obj: NWBFile object to inspect
    :return: list of inspection objects, if empty no issues were found
    """
    return list(inspect_nwbfile_object(obj))


def panda_df_to_dyn_table(pd_df=None, table_name=None, description=None):
    column_names = list(pd_df.columns)
    v_columns = []
    for col_name in column_names:
        v_columns.append(VectorData(
            name=col_name,
            data=list(pd_df[col_name]),
            description=col_name
        ))

    return DynamicTable(
        name=table_name,
        description=description,
        columns=v_columns
    )


def panda_df_to_list_of_timeseries(pd_df=None, measured_unit_list=None, series_name_suffix="",
                                   start_time=None, sampling_rate=None, description=None, comments=None):
    timeseries_list = []
    if len(measured_unit_list) != len(pd_df.columns):
        raise ValueError(
            f"Invalid 'measured_unit_list' does not match number of columns '{len(measured_unit_list)}' != '{len(pd_df.columns)}' Units: '{measured_unit_list}' Cols: '{pd_df.columns}'")

    for idx, col_name in enumerate(pd_df.columns):
        timeseries_list.append(TimeSeries(
            name=f"{series_name_suffix}{col_name}",
            data=pd_df[col_name].to_numpy(),
            unit=measured_unit_list[idx],
            starting_time=start_time,
            rate=sampling_rate,
            description=f"column {col_name} {description}",
            comments=comments,
        ))

    return timeseries_list

