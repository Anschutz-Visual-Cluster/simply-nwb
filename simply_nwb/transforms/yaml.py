import yaml
import os


def yaml_read_file(filename: str) -> dict:
    """
    Read a YAML file into a dict

    :param filename: filename of the yaml file
    :return: dict of data
    """
    if filename is None:
        raise ValueError("Must provide filename argument!")
    if not os.path.exists(filename):
        raise ValueError(f"File '{filename}' not found!")

    with open(filename, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)
