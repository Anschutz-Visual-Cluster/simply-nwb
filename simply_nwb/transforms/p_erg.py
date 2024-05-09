import glob
import os

import pandas as pd
from io import StringIO

from hdmf.common import DynamicTable, VectorData
from pynwb import NWBFile
from pynwb.misc import AnnotationSeries

from simply_nwb.util import warn_on_name_format


class _PergMixin(object):

    @staticmethod
    def p_erg_add_folder(nwbfile: NWBFile, foldername: str, file_pattern: str, table_name: str, description: str,
                         reformat_column_names: bool = True) -> NWBFile:
        """
        Add pERG data for each file into the NWB, from 'foldername' that matches 'file_pattern' into the NWB
        Example 'file_pattern' "\*txt"

        :param nwbfile: NWBFile object to add this data to
        :param foldername: folder where  the pERG datas are
        :param file_pattern: glob filepattern for selecting file e.g '\*.txt'
        :param table_name: name of new table to insert the data into in the NWB
        :param description: Description of the data to add
        :param reformat_column_names: Reformat column names to a nicer format from raw
        :return: None
        """

        if not os.path.exists(foldername):
            raise ValueError(
                f"Provided foldername '{foldername}' doesn't exist in current working directory: '{os.getcwd()}'!")
        if not os.path.isdir(foldername):
            raise ValueError(f"Provided foldername '{foldername}' isn't a folder!")

        pattern = os.path.join(foldername, file_pattern)
        files = glob.glob(pattern)
        if not files:
            raise ValueError(f"No files found matching pattern '{pattern}")
        for filename in files:
            _PergMixin.p_erg_add_data(
                nwbfile,
                filename=filename,
                table_name=table_name,
                reformat_column_names=reformat_column_names,
                description=description
            )
        return nwbfile

    @staticmethod
    def p_erg_add_data(nwbfile: NWBFile, filename: str, table_name: str , description: str, reformat_column_names: bool = True) -> NWBFile:
        """
        Add pERG data into the NWB, from file, formatting it

        :param nwbfile: NWBFile object to add this data to
        :param filename: filename of the pERG data to read
        :param table_name: name of new table to insert the data into in the NWB
        :param description: Description of the data to add
        :param reformat_column_names: Reformat column names to a nicer format from raw
        :return: NWBFile
        """
        warn_on_name_format(table_name)

        data_dict, metadata_dict = perg_parse_to_table(filename, reformat_column_names=reformat_column_names)
        data_dict_name = f"{table_name}_data"

        if data_dict_name in nwbfile.acquisition:
            nwbfile.acquisition[data_dict_name].add_row(data_dict)
        else:
            nwbfile.add_acquisition(DynamicTable(
                name=data_dict_name,
                description=description,
                columns=[
                    VectorData(
                        name=column,
                        data=[data_dict[column]],
                        description=column
                    )
                    for column in data_dict.keys()
                ]
            ))

        meta_key_format = "meta_{key}"
        format_key = lambda x: meta_key_format.format(key=x)

        # For each key, create an annotation series
        for meta_key, meta_value in metadata_dict.items():
            if isinstance(meta_value, list) and len(meta_value) > 1:
                raise ValueError(
                    "Metadata collected from pERG cannot be automatically formatted into NWB, nested list detected! Please flatten or manually enter data")

            nwb_key = format_key(meta_key)
            if nwb_key not in nwbfile.acquisition:
                nwbfile.add_acquisition(AnnotationSeries(
                    name=nwb_key,
                    description="pERG metadata for {}".format(meta_key),
                    data=[meta_value[0]],
                    timestamps=[0]
                ))
            else:
                nwbfile.acquisition[nwb_key].add_annotation(float(len(nwbfile.acquisition[nwb_key].data)),
                                                            meta_value[0])
        return nwbfile


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
