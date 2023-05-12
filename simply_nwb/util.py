from nwbinspector import inspect_nwbfile, inspect_nwbfile_object
from pynwb import NWBHDF5IO
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
