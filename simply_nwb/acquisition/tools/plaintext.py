import os


def plaintext_metadata_read(filename=None, sep=":"):
    """
    Read in data in a 'metadata' like format such as

    Key: val
    Key2: val2

    :param filename: str filename to read from
    :param sep: Separator for keys and values, defaults to ':'
    :return: dict of metadata data
    """
    if filename is None:
        raise ValueError("Must provide filename argument!")
    if not os.path.exists(filename):
        raise ValueError(f"File '{filename}' not found in current working path '{os.getcwd()}")

    with open(filename, "r") as f:
        lines = f.readlines()
        return {line.split(sep)[0].strip(): line.split(sep)[1].strip() for line in lines}
