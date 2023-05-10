from neo.io.blackrockio import BlackrockIO
import numpy as np


def blackrock_load_data(filename, raw=False):
    """
    Load data from a blackrock file (nev, ns5 and nsX files) into a list of dictionaries
    Uses the NEO Library

    NEO Docs
    https://neo.readthedocs.io/en/stable/core.html

    NEO Github
    https://github.com/NeuralEnsemble/python-neo/tree/master



    :param filename: filename to load
    :param raw: If raw == True, don't process into a dict, return what NEO returns
    :return: list dicts of data
    """
    blio = BlackrockIO(filename=filename)
    blackrock_obj = blio.read()
    if raw:
        return blackrock_obj

    blackrock_dict = []
    for block in blackrock_obj:
        segment_datas = []
        for segment in block.segments:
            segment_datas.append({
                "annotations": segment.annotations,
                "description": segment.description,
                "t_start": float(segment.t_start),  # Returns a 'Quantity' object, float is easier to work with
                "t_stop": float(segment.t_stop),  # See above
                "t_units": segment.t_start.dimensionality.string,
                "size": segment.size,
                "irregularlysampledsignals": segment.irregularlysampledsignals,
                "name": segment.name,
                "rec_datetime": segment.rec_datetime,
                "analogsignals": segment.analogsignals,
                "spiketrains": [np.array(sp) for sp in segment.spiketrains],
                "spiketrains_channel_ids": segment.spiketrains.all_channel_ids,
                "spiketrains_multiplexed": segment.spiketrains.multiplexed,
                "spiketrains_t_start": segment.spiketrains.t_start,
                "spiketrains_t_stop": segment.spiketrains.t_stop
            })

        blackrock_dict.append({
            "annotations": block.annotations,
            "description": block.description,
            "file_datetime": block.file_datetime,
            "file_origin": block.file_origin,
            "rec_datetime": block.rec_datetime,
            "groups": block.groups,
            "segments": segment_datas
        })
    return blackrock_dict


def blackrock_all_spiketrains(filename):
    """
    Load data from a blackrock file (nev, ns5 and nsX files) and grab all the spike trains into a single list
    Uses the NEO Library

    NEO Docs
    https://neo.readthedocs.io/en/stable/core.html

    NEO Github
    https://github.com/NeuralEnsemble/python-neo/tree/master


    :param filename: file to load data from
    :return: list of numpy arrays for spiketrains
    """
    blio = BlackrockIO(filename=filename)
    blackrock_obj = blio.read()

    spike_trains = []
    for block in blackrock_obj:
        for segment in block.segments:
            spike_trains.extend([np.array(spt) for spt in segment.spiketrains])

    return spike_trains
