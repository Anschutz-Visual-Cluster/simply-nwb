from uuid import uuid4
import os
import sys
import glob

from hdmf.common.table import DynamicTable
from hdmf.common.table import VectorData
from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup
from pynwb.file import Subject
from .acquisition.tools import blackrock_all_spiketrains, perg_parse_to_table
from .util import warn_on_name_format, inspect_nwb_obj


class SimpleNWB(object):
    _REQUIRED_ARGS = [
        "session_description",
        "session_start_time",
        "experimenter",
        "lab",
        "experiment_description"
    ]

    def __init__(
            self,
            session_description=None,
            session_start_time=None,
            experimenter=None,
            lab=None,
            experiment_description=None,
            # Optional
            identifier=None,
            subject=None,
            session_id=None,
            institution=None,
            keywords=None,
            related_publications=None
    ):
        for arg in SimpleNWB._REQUIRED_ARGS:
            # Required args
            if locals()[arg] is None:
                raise ValueError("Did not provide '{}' to SimpleNWB()!".format(arg))

            # Optional args
            if identifier is None:
                identifier = str(uuid4())
            if session_id is None:
                session_id = str(uuid4())
            if institution is None:
                institution = "CU Anschutz"
            if isinstance(subject, dict):
                subject = Subject(**subject)
            elif not isinstance(subject, Subject):
                raise ValueError("'subject' argument must either be a dict or a pynwb.file.Subject type!")

            if keywords is None:
                keywords = []

            self.nwb_kwargs = {
                "identifier": identifier,
                "session_description": session_description,
                "session_start_time": session_start_time,
                "experimenter": experimenter,
                "lab": lab,
                "subject": subject,
                "experiment_description": experiment_description,
                "session_id": session_id,
                "institution": institution,
                "keywords": keywords,
                "related_publications": related_publications
            }

            self._nwbfile = None
            self._get_nwbfile()  # Generate nwbfile from args

    def inspect(self):
        results = inspect_nwb_obj(self.nwbfile)
        return results

    def blackrock_spiketrains_as_units(
            self,
            # Required args
            blackrock_filename=None,
            device_description=None,
            electrode_description=None,
            electrode_location_description=None,
            electrode_position=None,
            electrode_impedance=None,
            electrode_brain_region=None,
            electrode_filtering_description=None,
            electrode_reference_description=None,

            # Optional args
            device_manufacturer=None,
            device_name=None,
            electrode_group_name="electrodegroup0",
    ):
        """
        Automatically parse a blackrock NEV file from spike trains into an NWB file

        :param blackrock_filename: Filename for the nev or nsX file of blackrock data (required)
        :param device_description: description of device (required)
        :param electrode_description: description of electrode used (required)
        :param electrode_location_description: description of the electrode location (required)
        :param electrode_position: stereotaxic position of this electrode group (x, y, z) (+y is inferior)(+x is posterior)(+z is right) (required)
        :param electrode_impedance: the impedance of the electrode, in ohms (required)
        :param electrode_brain_region: the location of electrode within the subject, brain region (required)
        :param electrode_filtering_description: description of hardware filtering, including the filter name and frequency cutoffs (required)
        :param electrode_reference_description: Description of the reference electrode and/or reference scheme used for this electrode, e.g.,"stainless steel skull screw" or "online common average referencing" (required)
        :param device_name: Name of the device used (optional)
        :param device_manufacturer: device manufacturer, will default to "BlackRock" (optional)
        :param electrode_group_name: name for the group of this electrode (optional)

        :return: None, just a parsing function
        """

        if blackrock_filename is None:
            raise ValueError("Must provide 'blackrock_filename' argument!")

        # Device related arg checking and defaults
        if device_description is None:
            raise ValueError("Must provide a device description for the blackrock device used!")
        if device_name is None:
            device_name = "BlackRock device"
        if device_manufacturer is None:
            device_manufacturer = "BlackRock"

        # Electrode related arg checking
        if electrode_description is None:
            raise ValueError("Must provide a description for the electrode used!")
        if electrode_location_description is None:
            raise ValueError("Must provide an electrode location description")
        if electrode_position is None:
            raise ValueError("Must provide a (x, y, z) location for the electrode")
        if electrode_reference_description is None:
            raise ValueError("Must provide 'electrode_reference_description' argument!")
        if electrode_filtering_description is None:
            raise ValueError("Must provide 'electrode_filtering_description' argument!")
        if electrode_brain_region is None:
            raise ValueError("Must provide 'electrode_brain_region' argument!")
        if electrode_impedance is None:
            raise ValueError("Must provide 'electrode_impedance' (in ohms) argument!")

        device = Device(
            name=device_name,
            description=device_description,
            manufacturer=device_manufacturer
        )

        electrode_group = ElectrodeGroup(
            name=electrode_group_name,
            description=electrode_description,
            location=electrode_location_description,
            device=device,
            position=electrode_position
        )

        self._nwbfile.add_electrode(
            x=electrode_position[0],
            y=electrode_position[1],
            z=electrode_position[2],
            imp=electrode_impedance,
            location=electrode_brain_region,
            filtering=electrode_filtering_description,
            reference=electrode_reference_description,
            group=electrode_group
        )

        blackrock_spiketrains = blackrock_all_spiketrains(blackrock_filename)
        # Add all spiketrains as units to the NWB
        [self._nwbfile.add_unit(spike_times=spike) for spike in blackrock_spiketrains]

    def add_p_erg_folder(self, foldername=None, file_pattern=None, table_name=None, reformat_column_names=True):
        """
        Add pERG data for each file into the NWB, from 'foldername' that matches 'file_pattern' into the NWB
        Example 'file_pattern' "\*txt"


        :param foldername: folder where  the pERG datas are
        :param file_pattern: glob filepattern for selecting file e.g '\*.txt'
        :param table_name: name of new table to insert the data into in the NWB
        :param reformat_column_names: Reformat column names to a nicer format from raw
        :return: None
        """
        if foldername is None:
            raise ValueError("Must provide folder name argument!")
        if file_pattern is None:
            raise ValueError("Must provide file_pattern! Example: '*.txt'")
        if table_name is None:
            raise ValueError("Must provide 'table_name' to store data!")

        if not os.path.exists(foldername):
            raise ValueError(f"Provided foldername '{foldername}' doesn't exist in current working directory: '{os.getcwd()}'!")
        if not os.path.isdir(foldername):
            raise ValueError(f"Provided foldername '{foldername}' isn't a folder!")

        pattern = os.path.join(foldername, file_pattern)
        files = glob.glob(pattern)
        if not files:
            raise ValueError(f"No files found matching pattern '{pattern}")
        for filename in files:
            self.add_p_erg_data(
                filename=filename,
                table_name=table_name,
                reformat_column_names=reformat_column_names
            )

    def add_p_erg_data(self, filename=None, table_name=None, reformat_column_names=True):
        """
        Add pERG data into the NWB, from file, formatting it

        :param filename: filename of the pERG data to read
        :param table_name: name of new table to insert the data into in the NWB
        :param reformat_column_names: Reformat column names to a nicer format from raw
        :return:
        """
        if filename is None:
            raise ValueError("Invalid filename! Must provide argument")
        if table_name is None:
            raise ValueError("Must provide a name for the pERG table data!")
        warn_on_name_format(table_name)

        data_dict, metadata_dict = perg_parse_to_table(filename, reformat_column_names=reformat_column_names)
        data_dict_name = f"{table_name}_data"
        metadata_dict_name = f"{table_name}_metadata"

        if data_dict_name in self.nwbfile.acquisition:
            self.nwbfile.acquisition[data_dict_name].add_row(data_dict)
        else:
            self.nwbfile.add_acquisition(DynamicTable(
                name=data_dict_name,
                description="pERG data",
                columns=[
                    VectorData(
                        name=column,
                        data=[data_dict[column]],
                        description=column
                    )
                    for column in data_dict.keys()
                ]
            ))

        if metadata_dict_name in self.nwbfile.acquisition:
            self.nwbfile.acquisition[metadata_dict_name].add_row(metadata_dict)
        else:
            self.nwbfile.add_acquisition(DynamicTable(
                name=metadata_dict_name,
                description="pERG metadata",
                columns=[
                    VectorData(
                        name=column,
                        data=[metadata_dict[column]],
                        description=column
                    )
                    for column in metadata_dict.keys()
                ]
            ))

    def _get_nwbfile(self):
        if self._nwbfile is None:
            self._nwbfile = NWBFile(**self.nwb_kwargs)
        return self._nwbfile

    @property
    def nwbfile(self):
        return self._get_nwbfile()
