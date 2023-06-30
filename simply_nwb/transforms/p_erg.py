import pandas as pd
from io import StringIO

from simply_nwb.util import warn_on_name_format


def _format_column_name(column_name: str) -> str:
    # Internal helper function to help rename columns
    # e.g. 'Data Pnt(ms):' -> 'data_pnt_ms'
    #
    # :param column_name: input column name
    # :return: str of reformatted column name

    replacements = [
        # Turn ')' into a '_'
        # e.g. 'Data Pnt(ms):' -> 'Data Pnt_ms):'
        ("(", "_"),
        (":", ""),
        (" ", "_"),
        (")", ""),
        ("\.", "")
    ]

    for from_val, to_val in replacements:
        column_name = column_name.replace(from_val, to_val)

    column_name = column_name.lower()  # To lowercase
    return column_name


def _parse_perg_metadata(filename: str) -> pd.DataFrame:
    # Helper func
    divider_count = 0
    line_datas = []

    with open(filename, "r") as f:
        data = f.readlines()
        for line in data:
            if divider_count == 0:
                if line.startswith("------"):
                    divider_count = 1
            elif divider_count == 1:
                if line.startswith("------"):
                    break
                else:
                    line_datas.append(line)
        line_datas.insert(0, "value")  # Insert dummy first entry
        return pd.read_csv(StringIO("\n".join(line_datas))).T  # Transpose since data is formatted sideways


def _reformat_column_names(panda_data: pd.DataFrame) -> pd.DataFrame:
    # Helper function to rename the columns in a pandas dataframe
    # :param panda_data: pd dataframe
    # :return: pd dataframe with different column names, formatted nicer

    rename_mapping = []
    for col_name in panda_data.columns:
        rename_mapping.append([col_name, _format_column_name(col_name)])

    for old_name, new_name in rename_mapping:
        panda_data[new_name] = panda_data[old_name]
        panda_data.pop(old_name)
    return panda_data


def _parse_perg_data(filename: str) -> pd.DataFrame:
    # Helper func to parse out only the data, a separate func will parse out metadata
    # :param filename: filename of the pERG file
    # :return: pandas dataframe

    last_header_idx = -1
    with open(filename, "r") as f:
        data = f.readlines()
        for idx, line in enumerate(data):
            if line.startswith("-------"):
                last_header_idx = idx
        if last_header_idx == -1:
            raise ValueError("Cannot find end of header!")

        data = data[last_header_idx + 1:]
        data = list(filter(lambda x: x.strip("\n") != '', data))

        return pd.read_csv(StringIO("\n".join(data)))


def perg_parse_to_table(filename: str, reformat_column_names: bool = True) -> (dict, dict):
    """
    Parse the output of a pERG reading into a dict

    :param filename: filename of the pERG data to parse
    :param reformat_column_names: reformat the column names, e.g 'Data Pnt(ms):' -> 'data_pnt_ms'
    :return: dict of data
    """
    if filename is None:
        raise ValueError("Filename cannot be none for pERG data parse!")

    panda_data = _parse_perg_data(filename)
    panda_metadata = _parse_perg_metadata(filename)

    if reformat_column_names:
        panda_data = _reformat_column_names(panda_data)
        panda_metadata = _reformat_column_names(panda_metadata)
    else:
        [warn_on_name_format(col, f"for pERG data '{filename}' consider using reformat_column_names=True") for col in panda_metadata.columns]
        [warn_on_name_format(col, f"for pERG metadata '{filename}' consider using reformat_column_names=True") for col in panda_data.columns]

    data_dict = {col: panda_data[col].to_numpy() for col in panda_data.columns}
    metadata_dict = {col: panda_metadata[col].to_numpy() for col in panda_metadata.columns}
    return data_dict, metadata_dict
