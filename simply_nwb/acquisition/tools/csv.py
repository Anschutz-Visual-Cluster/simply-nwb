import pandas as pd
import os


def csv_load_dataframe(filename=None, **kwargs):
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
