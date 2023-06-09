from hdmf.common import DynamicTable, VectorData
from nwbinspector import inspect_nwbfile, inspect_nwbfile_object
from pynwb import NWBHDF5IO, TimeSeries
import warnings
import re


def nwb_read(filename):
    """
    Read a file from the filesystem into an NWB object

    :param filename: filename of an NWB file
    :return: file pointer ready to be .read() to get the nwb object
    """
    # Can't use context manager, will close file, return file pointer
    io = NWBHDF5IO(filename, mode="r")
    return io


def nwb_write(nwb_obj, filename):
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


def dict_to_dyn_tables(dict_data=None, table_name=None, description=None, multiple_objs=True):
    """
    Util function to transform a python dict into a DynamicTable object
    If keys are not the same length, set multiple_objs=True

    :param dict_data: Dict to add
    :param table_name: name of the table
    :param description: description of the table
    :param multiple_objs: set to true if the columns are uneven
    :return: DynamicTable
    """
    if dict_data is None or not isinstance(dict_data, dict):
        raise ValueError("Must provide dict_data argument!")
    if table_name is None:
        raise ValueError("Must provide table_name argument!")
    if description is None:
        raise ValueError("Must provide description argument")

    # TODO Flatten tool here?
    for key, val in dict_data.items():
        if isinstance(val, dict):
            raise ValueError(f"Key '{key}' is a dict, cannot format! Pull the subkeys out to a top level!")

    column_names = list(dict_data.keys())
    v_datas = []
    for col_name in column_names:
        col_data = dict_data[col_name]
        if not isinstance(col_data, list):
            col_data = [col_data]

        v_data = VectorData(
            name=col_name,
            data=col_data,
            description=col_name
        )
        if multiple_objs:
            v_datas.append(DynamicTable(
                name=f"{table_name}_{col_name}",
                description=description,
                columns=[v_data]
            ))
        else:
            v_datas.append(v_data)

    if multiple_objs:
        return v_datas
    else:
        return DynamicTable(
            name=table_name,
            description=description,
            columns=v_datas
        )


def panda_df_to_dyn_table(pd_df=None, table_name=None, description=None):
    """
    Util function to transform a pandas DataFrame into a DynamicTable object

    :param pd_df: Pandas DataFrame
    :param table_name: name of the table
    :param description: description of the table
    :return: DynamicTable
    """
    if pd_df is None:
        raise ValueError("Must provide pd_df argument!")
    if table_name is None:
        raise ValueError("Must provide table_name argument!")
    if description is None:
        raise ValueError("Must provide description argument")

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


def panda_df_to_list_of_timeseries(pd_df=None, measured_unit_list=None, series_name_prefix="",
                                   start_time=None, sampling_rate=None, description=None, comments=None):
    """
    Turns a panda dataframe into a list of TimeSeries objects


    :param pd_df: dataframe to transform
    :param measured_unit_list: list of units for each column of the dataframe
    :param series_name_prefix: optional series prefix
    :param start_time: time the data started for each timeseries
    :param sampling_rate: sampling rate in Hz
    :param description: description of this dataframe
    :param comments: optional comments
    :return:
    """
    timeseries_list = []
    if len(measured_unit_list) != len(pd_df.columns):
        raise ValueError(
            f"Invalid 'measured_unit_list' does not match number of columns '{len(measured_unit_list)}' != '{len(pd_df.columns)}' Units: '{measured_unit_list}' Cols: '{pd_df.columns}'")

    for idx, col_name in enumerate(pd_df.columns):
        timeseries_list.append(TimeSeries(
            name=f"{series_name_prefix}{col_name}",
            data=pd_df[col_name].to_numpy(),
            unit=measured_unit_list[idx],
            starting_time=start_time,
            rate=sampling_rate,
            description=f"column {col_name} {description}",
            comments=comments,
        ))

    return timeseries_list


def _print(val, do_print=True):
    # Helper function
    if do_print:
        print(val, flush=True)


def is_camel_case(string, do_print=True):
    """
    Check if the given string is in CamelCase


    :param string: string to check
    :param do_print: if False, will not print out anything
    :return: bool if str is CamelCase
    """

    reg = re.compile("^[a-zA-Z]*$")
    matched = reg.match(string)
    if not matched:
        _print(f"String '{string}' does not match CamelCase regex!", do_print)
        return False
    if not string[0].isupper():
        _print(f"String '{string}' must start with a capital letter!", do_print)
        return False
    return True


def is_snake_case(string, do_print=True):
    """
    Checks if the given string is snake_case

    :param string: string to check if is snake_case
    :param do_print: if False, will not print anything
    :return: bool of if the string is snake_case
    """

    reg = re.compile("^[a-z_]*$")
    matched = reg.match(string)
    if not matched:
        _print(f"String '{string}' does not match snake_case regex!", do_print)
        return False
    return True


def is_filesystem_safe(string):
    """
    Generic check function for if a string is filesystem safe, limits to a-z A-Z 0-9 '_' '-'

    :param string: String to check
    :return: True if the given string matches the regex
    """
    reg = re.compile(r"^[a-zA-Z_\-0-9]*$")
    match = reg.match(string)
    if not match:
        return False
    return True
