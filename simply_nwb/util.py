from typing import Any, Optional

import pandas as pd
import pendulum
from hdmf.common import DynamicTable, VectorData
from nwbinspector import inspect_nwbfile, inspect_nwbfile_object
from pynwb import NWBHDF5IO, TimeSeries, NWBFile
import warnings
import re

from pynwb.file import Subject


def compare_nwbfiles(nwbfile1: NWBFile, nwbfile2: NWBFile, name1: str, name2: str):
    """
    Compare two nwbfiles, checking that they contain the same things, only check a handful of fields TODO?
    This is useful when writing a file, want to ensure all our fields that successfully wrote are read in correctly.
    If there is a difference, will throw a ValueError

    :param nwbfile1: NWBFile 1 to compare
    :param nwbfile2: NWBFile 2 to compare
    :param name1: String name of NWBFile 1, used in error message only
    :param name2: String name of NWBFile 2, used in error message only
    """

    fields_to_value_verify = [
        # "experimenter",  # TODO handle ("a, b") != ["a, b"]
        # "file_create_date", # TODO Add a flag to ignore timestamp? if comparing not on write
        "institution"
    ]

    # Verify field values are the same
    for f in fields_to_value_verify:
        v1 = getattr(nwbfile1, f)
        v2 = getattr(nwbfile2, f)
        if v1 != v2:
            raise ValueError(f"Difference '{name1}.{f}' != '{name2}.{f}' found! '{v1}' != '{v2}'")

    fields_to_count_verify = [
        "acquisition",
        "devices",
        "processing",
    ]
    # Verify that fields have the same count and names
    for f in fields_to_count_verify:

        fd1 = getattr(nwbfile1, f)  # fd1 - field data 1
        fd2 = getattr(nwbfile2, f)

        fds1 = set(list(fd1))  # fds1 - field data set 1
        fds2 = set(list(fd2))
        missing_entries = ",".join([str(v) for v in list(fds1.difference(fds2))])
        intro = f"Difference between '{name1}.{f}' and '{name2}.{f}' found!"

        if missing_entries:
            raise ValueError(f"{intro} Entries in '{name1}' that are missing in '{name2}': '{missing_entries}'")

        extra_entries = ",".join([str(v) for v in list(fds2.difference(fds1))])
        if len(fd1) != len(fd2):
            raise ValueError(f"{intro} Entries are not the same length! Extra entries in '{name2}' that don't exist in '{name1}': '{extra_entries}'")

    # Check processing containers
    nwb1_processing = getattr(nwbfile1, "processing")
    nwb2_processing = getattr(nwbfile2, "processing")

    for processing_module_name in list(nwb1_processing):
        for container_name in list(nwb1_processing[processing_module_name].containers):
            intro = f"Difference between '{name1}.processing.{processing_module_name}' and '{name2}.{processing_module_name}' found!"
            try:
                _ = nwb2_processing[processing_module_name][container_name]  # If value exists, call it good
                pass
            except KeyError:
                raise ValueError(f"{intro} Entry '{processing_module_name}.{container_name}' was found in '{name1}' but not in '{name2}'!")

    if nwbfile1.units is None:
        if nwbfile2.units is not None:
            raise ValueError(f"Error, '{name1}.units' is empty but '{name2}.units' is not!")
    else:
        if nwbfile2.units is None:
            raise ValueError(f"Error, '{name2}.units' is empty!")
        else:
            if len(nwbfile1.units) != len(nwbfile2.units):
                raise ValueError(f"Error, '{name1}.units' and '{name2}.units' Are not the same length!")


def nwb_write(nwb_obj: NWBFile, filename: str, verify: bool):
    """
    Write an NWB object to a file on the local filesystem, and verify the contents were written correctly and the file
    isn't corrupted

    :param nwb_obj: pynwb.file.NWBFile object
    :param filename: path of a local file, doesn't need to exist
    :param verify: Verify that *most* fields wrote correctly and the file didn't corrupt
    :return: None
    """
    io = NWBHDF5IO(filename, mode="w")
    try:
        src = nwb_obj.container_source
        if src == filename or src is None:  # Overwriting same NWB or writing a new one
            io.write(nwb_obj)
        else:
            src_io = nwb_obj.read_io
            nwb_obj.set_modified()
            io.export(nwbfile=nwb_obj, src_io=src_io)

        if verify:
            # Want to load file to check that it didn't corrupt
            tio = NWBHDF5IO(filename)
            try:
                test_nwb = tio.read()
                # Also note that your data can just 'be missing' because NWB decided not to write it 'for some reason'
            except Exception as e:
                warnings.warn(f"File is corrupted! NWB lets you write data that it won't read correctly, check your input data!")
                raise e
            finally:
                tio.close()

            compare_nwbfiles(nwb_obj, test_nwb, "InMemoryNWB", "WrittenFileNWB")
    finally:
        io.close()


def warn_on_name_format(name_value: str, context_str: str = "") -> bool:
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


def inspect_nwb_file(filename: str) -> list[Any]:
    """
    Return the inspection list of a given NWB file

    :param filename: filename of the NWB to inspect
    :return: list of inspection objects for the given NWB, if empty, no issues found
    """
    return list(inspect_nwbfile(nwbfile_path=filename))


def inspect_nwb_obj(obj: NWBFile) -> list[Any]:
    """
    Return the inspection list of a given NWB object

    :param obj: NWBFile object to inspect
    :return: list of inspection objects, if empty no issues were found
    """
    return list(inspect_nwbfile_object(obj))


def dict_to_dyn_tables(dict_data: dict, table_name: str, description: str, multiple_objs: bool = True) -> DynamicTable:
    """
    Util function to transform a python dict into a DynamicTable object
    If keys are not the same length, set multiple_objs=True

    :param dict_data: Dict to add
    :param table_name: name of the table
    :param description: description of the table
    :param multiple_objs: set to true if the columns are uneven
    :return: DynamicTable
    """

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


def panda_df_to_dyn_table(pd_df: pd.DataFrame, table_name: str, description: str) -> DynamicTable:
    """
    Util function to transform a pandas DataFrame into a DynamicTable object

    :param pd_df: Pandas DataFrame
    :param table_name: name of the table
    :param description: description of the table
    :return: DynamicTable
    """

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


def panda_df_to_list_of_timeseries(pd_df: pd.DataFrame, measured_unit_list: list[str], start_time: float,
                                   sampling_rate: float, description: str, series_name_prefix: str = "",
                                   comments: Optional[str] = None) -> list[TimeSeries]:
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


def _print(val: Any, do_print: bool = True):
    # Helper function
    if do_print:
        print(val, flush=True)


def is_camel_case(string: str, do_print: bool = True) -> bool:
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


def is_snake_case(string: str, do_print: bool = True) -> bool:
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


def is_filesystem_safe(string: str) -> bool:
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


def age_str_from_birthday(birthday_str: str) -> str:
    """
    Create a ISO-8601 Period date string from a birthday. Interprets most date formats correctly, to ensure
    correct behavior, format 'MM-DD-YYYY' is recommended

    :param birthday_str: String of birthday
    """
    birth = pendulum.parse(birthday_str, strict=False)
    now = pendulum.now()
    diff_in_days = now.diff(birth).days
    return f"P{diff_in_days}D"  # P90D is 90 days period


def create_mouse_subject(subject_id: str, birthday_str: str, strain: str, sex: str, desc: str) -> Subject:
    """
    Create a Subject object, simple wrapper

    :param subject_id: ID to use to uniquely identify a mouse
    :param birthday_str: String of birthday of the mouse
    :param strain: Strain of the mouse
    :param sex: Mouse sex
    :param desc: Description of the mouse
    """
    return Subject(
        subject_id=subject_id,
        age=age_str_from_birthday(birthday_str),
        strain=strain,
        sex=sex,
        description=desc
    )


def date_to_mouse_age(date_str: Optional[str]) -> Optional[str]:
    """
    Convert a date into an ISO-8601 period time value

    :param date_str: string of the date, should be pendulum parsable
    :return: string like 'P90D'
    """
    if date_str is None:
        return None

    if date_str.lower() != "unknown":
        mouse_age = "P" + str(pendulum.parse(date_str, strict=False).diff(
            pendulum.now()).in_days()) + "D"  # How many days since birthday
    else:
        mouse_age = None

    return mouse_age
