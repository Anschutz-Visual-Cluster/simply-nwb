import pynwb
from neo.io.blackrockio import BlackrockIO
import numpy as np


class _BlackrockMixin(object):
    @staticmethod
    def blackrock_spiketrains_as_units(
            nwbfile: pynwb.file.NWBFile,
            # Required args
            blackrock_filename: str,
            device_description: str,
            electrode_name: str,
            electrode_description: str,
            electrode_location_description: str,
            electrode_resistance: float,
            # Optional args
            device_manufacturer: str = None,
            device_name: str = None,
    ):
        """
        Automatically parse a blackrock NEV file from spike trains into an NWB file
        Code created from tutorial: https://pynwb.readthedocs.io/en/stable/tutorials/domain/plot_icephys.html#sphx-glr-tutorials-domain-plot-icephys-py

        :param nwbfile: NWBFile object to add this data to
        :param blackrock_filename: Filename for the nev or nsX file of blackrock data (required)
        :param device_description: description of device (required)
        :param electrode_name: Name of the electrode (required)
        :param electrode_description: description of electrode used (required)
        :param electrode_location_description: description of the electrode location (required)
        :param electrode_resistance: the impedance/resistance of the electrode, in ohms (required)
        :param device_name: Name of the device used (optional)
        :param device_manufacturer: device manufacturer, will default to "BlackRock" (optional)
        :return: NWBFile
        """
        if device_name is None:
            device_name = "BlackRock device"
        if device_manufacturer is None:
            device_manufacturer = "BlackRock"

        device = nwbfile.create_device(
            name=device_name,
            description=device_description,
            manufacturer=device_manufacturer
        )

        nwbfile.create_icephys_electrode(
            name=electrode_name,
            device=device,
            description=electrode_description,
            location=electrode_location_description,
            resistance=str(electrode_resistance)
        )

        blackrock_spiketrains = blackrock_all_spiketrains(blackrock_filename)
        # Add all spiketrains as units to the NWB

        [nwbfile.add_unit(spike_times=spike) for spike in blackrock_spiketrains]
        return nwbfile


def blackrock_load_data(filename: str, raw: bool = False) -> list[dict]:
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


def blackrock_all_spiketrains(filename: str) -> list:
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
