import pandas as pd
import os
from io import StringIO


def csv_load_dataframe(filename: str, **kwargs) -> pd.DataFrame:
    """
    Simple wrapper around pd.read_csv for file checks

    :param filename: str filename
    :param kwargs: extra args to pass to pandas.read_csv()
    :return: pandas.DataFrame
    """
    if filename is None:
        raise ValueError("Must provide filename to load a CSV file!")
    if not os.path.exists(filename):
        raise ValueError(f"File '{filename}' not found in current working path '{os.getcwd()}")

    return pd.read_csv(filename, **kwargs)


def csv_load_dataframe_str(strdata: str, **kwargs) -> pd.DataFrame:
    """
    Simple wrapper around pd.read_csv for a string rather than a file

    :param strdata: str to parse
    :param kwargs: extra args to pass to pandas.read_csv()
    :return: pandas.DataFrame
    """
    if strdata is None or not strdata:
        raise ValueError("Must provide strdata to load a CSV file!")

    return pd.read_csv(StringIO(strdata), **kwargs)
