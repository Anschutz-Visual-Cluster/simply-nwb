import os
import re
from tabnanny import check
from typing import Union, Any

import numpy as np
from numpy import number


def plaintext_metadata_read(filename: str, sep: str = ":") -> {str: str}:
    """
    Read in data in a 'metadata' like format such as

    Key: val
    Key2: val2

    :param filename: str filename to read from
    :param sep: Separator for keys and values, defaults to ':'
    :return: dict of metadata data
    """
    if not os.path.exists(filename):
        raise ValueError(f"File '{filename}' not found in current working path '{os.getcwd()}")

    with open(filename, "r") as f:
        lines = f.readlines()
        return {line.split(sep)[0].strip(): line.split(sep)[1].strip() for line in lines}


def drifting_grating_metadata_read(filename: str, data_to_numpy: bool = True, columns_key: str = "Columns", max_line_len: int = 100000, filename_str: str = "filename") -> {str: Union[str, list]}:
    """
    Read in data for a drifting grating metadata like
    if data_to_numpy is True, will convert the numerical part of the file to a numpy array
    max_line_len is how far along a line we should parse for '(' and ')' if reached will error (number of chars)
    Dunno why I wrote a token parser for this lol

    Meta1: desc1
    Meta2: desc2
    Columns: col1, col2, col3, ...
    ...
    1,2,3,4
    6,7,8,9
    ...

    Where the top section is key: value and the bottom is just a csv
    Will turn it into a dict like {col1: [1, 6, ..], col2: [2, 7, ..]], Meta1: desc1, Meta1: desc2, ..., <columns_key>: [col1, col2, ...]}
    """

    if not os.path.exists(filename):
        raise ValueError(f"File {filename} not found in current working path '{os.getcwd()}'")
    print(f"Processing '{filename}'..")

    with open(filename, "r") as fp:
        filedata = fp.read()
    data = filedata.split("\n")
    processed = {}

    file_line_idx = 0
    while True:
        line = data[file_line_idx]
        # If line starts with a number, assume the rest is a numerical csv
        if line.startswith("------------") or len(line) < 1 or re.match(r"\d", line[0]):
            break
        sep_idx = line.find(":")
        key = line[:sep_idx].strip()
        val = line[sep_idx + 1:].strip()
        processed[key] = val
        file_line_idx = file_line_idx + 1
    if columns_key not in processed:
        raise ValueError(f"Could not process driftingGratingMetadata file '{filename}' Columns key '{columns_key}' wasn't found!")

    cols = []
    cols_str = processed[columns_key]
    starting_idx = 0
    range_len = 0
    str_idx = 0
    while True:
        char = cols_str[str_idx]
        if char == "(":  # Deal with pesky inline strs like key1 (1,2,3),key2
            while True:
                str_idx = str_idx + 1
                range_len = range_len + 1
                if cols_str[str_idx] == ")":
                    break
                if str_idx == max_line_len:
                    raise ValueError(f"String didn't have a terminating ')' or longer than {max_line_len} chars String: '{cols_str[str_idx]}'")
            str_idx = str_idx + 1  # Increment past the ')'
            range_len = range_len + 1

            if str_idx >= len(cols_str):  # ) is the end of the string
                char = ""
            else:
                char = cols_str[str_idx]

        if char == "," or str_idx + 1 >= len(cols_str):
            cols.append(cols_str[starting_idx:starting_idx + range_len])
            starting_idx = str_idx + 1
            range_len = 0
            if str_idx + 1 >= len(cols_str):
                break
        range_len = range_len + 1
        str_idx = str_idx + 1

    # Clean up the header strings by removing whitespace and removing trailing ','
    cols = [c.strip() for c in cols]
    cols = [c[:-1] if c.endswith(",") else c for c in cols]

    drift_data = data[file_line_idx:]

    processed.update({c: [] for c in cols})

    for drift_line in drift_data:
        if drift_line == "":
            break

        split = drift_line.split(",")
        if len(split) != len(cols):
            raise ValueError(f"Invalid number of columns for line '{split}' Doesnt match up with expected columns")
        for col_idx, col in enumerate(cols):
            processed[col].append(float(split[col_idx].strip()))

    processed[columns_key] = []
    for col in cols:
        processed[columns_key].append(col)

    if data_to_numpy:
        for col in cols:
            processed[col] = np.array(processed[col])
    processed[filename_str] = str(filename)

    return processed


def drifting_grating_metadata_read_from_filelist(files: list[str], data_to_numpy: bool = True, columns_key: str = "Columns", max_line_len: int = 100000, alignment_key: str = "Timestamp", filelen_str: str = "file_len", filename_str: str = "filename", expand_file_keys: bool = False) -> {str: Union [str, list]}:
    """
    Grab a list of labjack files and concat them together into a single data structure.
    Will arrange the data based on an alignment key, defaults to 'Timestamp'
    """
    # Keys to expand
    # "Baseline contrast", "Orientation", "Spatial frequency", "Velocity", "file_len", "filename"
    unsorted = []

    for file in files:
        data = drifting_grating_metadata_read(file, data_to_numpy, columns_key, max_line_len, filename_str=filename_str)
        file_len = len(data[alignment_key])
        data[filelen_str] = file_len
        unsorted.append(data)

    sorteddata = sorted(unsorted, key=lambda x: x[alignment_key][0])  # Sort on the first value of the alignment key, ie the first timestamp in the file

    alldata = {filelen_str: [], columns_key: sorteddata[0][columns_key]}
    current_len = 0  # Current length of the data, used to keep track of which files had which data
    for data in sorteddata:
        datalen = data[filelen_str]
        current_len = current_len + datalen
        alldata[filelen_str].append(current_len)

        for k, v in data.items():
            if k == filelen_str or k == columns_key:
                continue
            if k not in alldata:
                alldata[k] = []
            if isinstance(v, np.ndarray):
                v = list(v)

            if isinstance(v, list):
                alldata[k].extend(v)
            else:
                if expand_file_keys:
                    l = []
                    for _ in range(datalen):
                        l.append(v)
                    alldata[k].extend(l)
                else:
                    alldata[k].append(v)

    # Convert the numerical arrays to numpy
    for k, v in alldata.items():
        alldata[k] = np.array(v)

    return alldata
