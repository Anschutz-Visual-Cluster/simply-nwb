import os
import pendulum
import pandas as pd


def _get_labjack_meta_lines(meta_lines):
    # Helper function to parse out the metadata from the labjack file, since they are in a different format than the
    # recorded data

    current_line = meta_lines.pop(0)
    col_headers = None
    rows = []

    while True:
        current_line = current_line.strip()
        if len(meta_lines) == 0:
            raise ValueError("Reached EOF during metadata scan")
        if current_line.lower().startswith("time"):
            break
        if not current_line:  # Blank line
            current_line = meta_lines.pop(0)
            continue
        if not col_headers:
            col_headers = [val.strip().split("=")[0] for val in current_line.split(",")[1:]]
            col_headers.insert(0, "channel_num")

        cols = current_line.split(",")
        # Skip first entry, as it has no header and won't .split()
        col_vals = [val.strip().split("=")[1] for val in cols[1:]]
        # Insert channel num
        col_vals.insert(0, cols[0])
        rows.append(col_vals)
        current_line = meta_lines.pop(0)

    # rows.insert(0, col_headers)
    meta_lines.insert(0, current_line)  # re-insert the data header line
    return pd.DataFrame.from_records(rows, columns=col_headers)


def get_labjack_data(filename=None):
    """
    Returns labjack data and labjack metadata from a given filename
    :param filename: file to parse
    :return: data, metadata
    """
    if filename is None:
        raise ValueError("Must provide filename argument!")
    if not os.path.exists(filename):
        raise ValueError(f"File '{filename}' not found in current working path '{os.getcwd()}")

    with open(filename, "r") as f:
        lines = f.readlines()
        date = pendulum.parse(lines.pop(0).strip(), strict=False)
        time = pendulum.parse(lines.pop(0).strip(), strict=False, exact=False)
        date = date.set(
            hour=time.hour,
            minute=time.minute,
            second=time.second
        ).to_iso8601_string()

        # Read metadata about channels
        meta_data = _get_labjack_meta_lines(lines)
        lines = [line.strip().split("\t") for line in lines]
        data_headers = lines.pop(0)
        lines = [[float(v) for v in val] for val in lines]
        data = pd.DataFrame.from_records(lines, columns=data_headers)
        return {
            "data": data,
            "metadata": meta_data,
            "date": date
        }
