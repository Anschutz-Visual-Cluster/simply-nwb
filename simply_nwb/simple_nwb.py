from uuid import uuid4
import os
import sys
import glob

import numpy as np
import pandas as pd
import pendulum
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common.table import DynamicTable
from hdmf.common.table import VectorData
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import BehavioralEvents
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup
from pynwb.file import Subject
from pynwb.ophys import OpticalChannel, TwoPhotonSeries

from .acquisition.tools import labjack_load_file
from .acquisition.tools import blackrock_all_spiketrains, perg_parse_to_table
from .util import warn_on_name_format, inspect_nwb_obj, nwb_write, panda_df_to_dyn_table, \
    panda_df_to_list_of_timeseries, dict_to_dyn_tables


class SimpleNWB(object):
    _REQUIRED_ARGS = [
        "session_description",
        "session_start_time",
        "experimenter",
        "lab",
        "experiment_description"
    ]

    @staticmethod
    def create_nwb(
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
            if institution is None:
                raise ValueError("Must provide institution!")

            # Optional args
            if identifier is None:
                identifier = str(uuid4())
            if session_id is None:
                session_id = str(uuid4())
            if isinstance(subject, dict):
                subject = Subject(**subject)
            elif not isinstance(subject, Subject) and not subject is None:
                raise ValueError("'subject' argument must either be a dict or a pynwb.file.Subject type!")

            if keywords is None:
                keywords = []

            nwb_kwargs = {
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

            return NWBFile(**nwb_kwargs)

    @staticmethod
    def inspect(nwbfile):
        """
        Inspect the given NWBFile
        :param nwbfile: NWBFile object
        :return: List of issues with the file, if empty, inspection passed
        """

        results = inspect_nwb_obj(nwbfile)
        return results

    @staticmethod
    def write(nwbfile, filename=None):
        """
        Write the give NWBFile object to file
        :param nwbfile: NWBFile object to write
        :param filename: path to file to write, WILL OVERWRITE!
        :return: None
        """
        nwb_write(nwbfile, filename)

    @staticmethod
    def mp4_add_as_behavioral(
            nwbfile,
            name=None,
            numpy_data=None,
            frame_count=None,
            sampling_rate=None,
            description=None,
            chunking=True,
            compression=True
    ):
        """
        Add a numpy array as behavioral data, meant to work in conjunction with the mp4

        :param nwbfile: NWBFile to add the mp4 data to
        :param name: Name of the movie
        :param numpy_data: data, can be np.memmap
        :param frame_count: number of total frames
        :param sampling_rate: frames per second
        :param description: description of the movie
        :param chunking: Optional chunking for large files, defaults to True
        :param compression: Optional compression for large files, defaults to True
        :return: None
        """

        behavior_module = nwbfile.create_processing_module(name="behavior", description="test desc")
        mp4_timeseries = TimeSeries(
            name=name,
            data=H5DataIO(
                data=numpy_data[:frame_count],
                compression=compression,
                chunks=chunking
            ),
            description=description,
            unit="mp4 video frame images",
            rate=sampling_rate
        )

        behavior_module.add(mp4_timeseries)

    @staticmethod
    def processing_add_dict(
            nwbfile,
            processed_name=None,
            processed_description=None,
            data_dict=None,
            uneven_columns=False):
        """
        Add a processed dict into the NWB that doesn't fit in any other part of the NWB. MAKE SURE YOU CANT ADD IT ELSEWHERE BEFORE USING THIS FUNC!

        :param nwbfile: NWBFile to add data to
        :param processed_name: Name of the processing module
        :param processed_description: description of the processed data
        :param data_dict: dict data to add
        :param uneven_columns: Set this to True if the keys of the dict have different lengths
        """
        if processed_name is None:
            raise ValueError("Must supply processed_name argument!")
        if processed_description is None:
            raise ValueError("Must supply processed_description argument!")
        if data_dict is None or not isinstance(data_dict, dict):
            raise ValueError("Make sure to supply argument data_dict and it should be a dict type!")

        data_interfaces = dict_to_dyn_tables(
            dict_data=data_dict,
            table_name=processed_name,
            description=processed_description,
            multiple_objs=uneven_columns
        )
        if not uneven_columns:
            data_interfaces = [data_interfaces]
        if "misc" in nwbfile.processing:
            [nwbfile.processing["misc"].add_container(interface) for interface in data_interfaces]
        else:
            nwbfile.create_processing_module(
                name="misc",
                description=processed_description,
                data_interfaces=data_interfaces
            )

    @staticmethod
    def processing_add_dataframe(
            nwbfile,
            processed_name=None,
            processed_description=None,
            data=None
    ):
        """
        Add a processed pandas Dataframe into the NWB that doesn't fit in any other part of the NWB. MAKE SURE YOU CANT ADD IT ELSEWHERE BEFORE USING THIS FUNC!

        :param nwbfile: NWBFile to add data to
        :param processed_name: Name of the processing module
        :param processed_description: description of the processed data
        :param data: Pandas Dataframe data to add
        :return: None
        """
        if processed_name is None:
            raise ValueError("Must provide processed_name argument!")
        if processed_description is None:
            raise ValueError("Must provide processed_description argument!")
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("data argument must be Pandas Dataframe!")

        data_interface = panda_df_to_dyn_table(
            pd_df=data,
            table_name=processed_name,
            description=processed_description
        )

        if "misc" in nwbfile.processing:
            nwbfile.processing["misc"].add_container(data_interface)
        else:
            nwbfile.create_processing_module(
                name="misc",
                description=processed_description,
                data_interfaces=[data_interface]
            )

    @staticmethod
    def two_photon_add_data(
            nwbfile,
            device_name=None,
            device_description=None,
            device_manufacturer=None,
            optical_channel_description=None,
            optical_channel_emission_lambda=None,
            imaging_name=None,
            imaging_rate=None,
            excitation_lambda=None,
            indicator=None,
            location=None,
            grid_spacing=None,
            grid_spacing_unit=None,
            origin_coords=None,
            origin_coords_unit=None,
            two_photon_unit=None,
            two_photon_rate=None,
            photon_series_name=None,
            image_data=None,
    ):
        """
        Add images from a two photon microscope to an NWB with metadata.
        Load TIF images easily with the simply_nwb.acquisition.tools.tif module


        :param nwbfile: NWBFile to add the data to
        :param device_name: Name of the microscope e.g. MyMicroscope1
        :param device_description: Description of the microscope
        :param device_manufacturer: Manufacturer of the microscope
        :param optical_channel_description: Description of the optical channel, electrode name
        :param optical_channel_emission_lambda: Emission wavelength for optical channel, in nm
        :param imaging_name: Name of this imaging dataset e.g 'my_images'
        :param imaging_rate: Rate at which images were acquired, in Hz
        :param excitation_lambda: Excitation wavelength in nm
        :param indicator: Indicator, e.g. GFP
        :param location: Location, e.g. V1
        :param grid_spacing: Spacing of the grids used e.g. [0.1, 0.1]
        :param grid_spacing_unit: Unit of the grid spacing e.g. 'meters'
        :param origin_coords: Coords of the origin e.g. [0.1, 0.2] or for 3d data [0.1, 0.2, 0.3]
        :param origin_coords_unit: Unit of the origin coords
        :param two_photon_unit: Unit for the photon microscope, e.g. 'normalized amplitude'
        :param two_photon_rate: two photon sampling rate in Hz
        :param photon_series_name: Name of the two photon series for storage
        :param image_data: Numpy array of the data in shape (samples, xres, yres, channels) if only one channel, can omit
        :return:
        """

        if device_name is None:
            raise ValueError("Must supply device_name argument!")
        if device_description is None:
            raise ValueError("Must supply device_description argument!")
        if device_manufacturer is None:
            raise ValueError("Must supply device_manufacturer argument!")
        if optical_channel_description is None:
            raise ValueError("Must supply optical_channel_description argument!")
        if optical_channel_emission_lambda is None:
            raise ValueError("Must supply optical_channel_emission_lambda argument!")
        if imaging_name is None:
            raise ValueError("Must supply imaging_name argument!")
        if imaging_rate is None or not isinstance(imaging_rate, float):
            raise ValueError("Must supply imaging_rate, and as a float!")
        if excitation_lambda is None:
            raise ValueError("Must supply excitation_lambda wavelength in nm!")
        if indicator is None:
            raise ValueError("Must supply indicator argument! e.g 'GFP'")
        if location is None or not isinstance(location, str):
            raise ValueError("Must supply location string argument")
        if grid_spacing is None:
            raise ValueError("Must supply grid_spacing argument!")
        if grid_spacing_unit is None:
            raise ValueError("Must supply grid_spacing_unit argument!")
        if origin_coords is None:
            raise ValueError("Must supply origin_coords argument!")
        if origin_coords_unit is None:
            raise ValueError("Must supply origin_coords_unit argument!")
        if two_photon_unit is None:
            raise ValueError("Must supply two_photon_unit argument!")
        if two_photon_rate is None:
            raise ValueError("Must supply two_photon_rate argument!")
        if image_data is None or not isinstance(image_data, np.ndarray):
            raise ValueError("Must supply image_data as numpy array!")

        microscope = nwbfile.create_device(
            name=device_name,
            description=device_description,
            manufacturer=device_manufacturer,
        )

        optical_channel = OpticalChannel(
            name="OpticalChannel",
            description=optical_channel_description,
            emission_lambda=optical_channel_emission_lambda
        )

        imaging_plane = nwbfile.create_imaging_plane(
            name=imaging_name,
            optical_channel=optical_channel,
            imaging_rate=imaging_rate,
            description=f"Imaging plane of {location}",
            device=microscope,
            excitation_lambda=excitation_lambda,
            indicator=indicator,
            location=location,
            grid_spacing=grid_spacing,
            grid_spacing_unit=grid_spacing_unit,
            origin_coords=origin_coords,
            origin_coords_unit=origin_coords_unit,
        )

        two_photon_series = TwoPhotonSeries(
            name=photon_series_name,
            description="Images from a TwoPhoton microscope",
            data=image_data,
            imaging_plane=imaging_plane,
            rate=two_photon_rate,
            unit=two_photon_unit
        )

        nwbfile.add_acquisition(two_photon_series)

    @staticmethod
    def labjack_as_behavioral_data(
            nwbfile,
            labjack_filename=None,
            name=None,
            measured_unit_list=None,
            start_time=None,
            sampling_rate=None,
            description=None,
            behavior_module=None,
            behavior_module_name=None,
            comments="Labjack behavioral data"
    ):
        """
        Add LabJack data to the NWBFile as a behavioral entry

        :param nwbfile: NWBFile to add the data to
        :param labjack_filename: LabJack filename to read from
        :param name: Name of this behavioral unit
        :param measured_unit_list: List of SI unit strings corresponding to the columns of the labjack data
        :param start_time: start time float
        :param sampling_rate: sampling rate in Hz
        :param description: description of the behavioral data
        :param behavior_module: Optional NWB behavior module to add this data to, otherwise will create a new one e.g. nwbfile.processing["behavior"]
        :param behavior_module_name: optional module name to add this behavior to, if exists will append. will ignore if behavior_module arg is supplied
        :param comments: additional comments about the data
        :return:
        """
        if name is None:
            raise ValueError("Must provide name argument for labjack data!")
        if start_time is None:
            raise ValueError("Must provide start_time argument for labjack data!")
        if not isinstance(start_time, float):
            raise ValueError("start_time must be a float! For example, if using a whole number use 5.0 instead of 5")
        if sampling_rate is None:
            raise ValueError("Must provide sampling_rate argument for labjack data!")
        if not isinstance(sampling_rate, float):
            raise ValueError("start_time must be a float! For example, if using a whole number use 5.0 instead of 5")
        if description is None:
            raise ValueError("Must provide description argument for labjack data!")

        labjack_data = labjack_load_file(labjack_filename)

        timeseries_list = panda_df_to_list_of_timeseries(
            pd_df=labjack_data["data"],
            measured_unit_list=measured_unit_list,
            start_time=start_time,
            sampling_rate=sampling_rate,
            description=description,
            comments=comments
        )

        behavior_events = BehavioralEvents(
            time_series=timeseries_list,
            name=f"{name}_behavioral_events"
        )

        if not behavior_module:
            if not behavior_module_name:
                behavior_module_name = "behavior"  # Default name

            if behavior_module_name in nwbfile.processing:
                behavior_module = nwbfile.processing[behavior_module_name]
            else:
                behavior_module = nwbfile.create_processing_module(
                    name=behavior_module_name,
                    description="Behavior processing module"
                )

        behavior_module.add(behavior_events)
        behavior_module.add(panda_df_to_dyn_table(
            pd_df=labjack_data["metadata"],
            table_name=f"{name}_metadata",
            description="Labjack metadata"
        ))

        return nwbfile

    @staticmethod
    def blackrock_spiketrains_as_units(
            nwbfile,
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

        :param nwbfile: NWBFile object to add this data to
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

        nwbfile.add_electrode(
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
        [nwbfile.add_unit(spike_times=spike) for spike in blackrock_spiketrains]

    @staticmethod
    def add_p_erg_folder(nwbfile, foldername=None, file_pattern=None, table_name=None, description=None,
                         reformat_column_names=True):
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
        if foldername is None:
            raise ValueError("Must provide folder name argument!")
        if file_pattern is None:
            raise ValueError("Must provide file_pattern! Example: '*.txt'")
        if table_name is None:
            raise ValueError("Must provide 'table_name' to store data!")
        if description is None:
            raise ValueError("Must provide a description for the pERG data!")

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
            SimpleNWB.add_p_erg_data(
                nwbfile,
                filename=filename,
                table_name=table_name,
                reformat_column_names=reformat_column_names,
                description=description
            )

    @staticmethod
    def add_p_erg_data(nwbfile, filename=None, table_name=None, description=None, reformat_column_names=True):
        """
        Add pERG data into the NWB, from file, formatting it

        :param nwbfile: NWBFile object to add this data to
        :param filename: filename of the pERG data to read
        :param table_name: name of new table to insert the data into in the NWB
        :param description: Description of the data to add
        :param reformat_column_names: Reformat column names to a nicer format from raw
        :return:
        """
        if filename is None:
            raise ValueError("Invalid filename! Must provide argument")
        if table_name is None:
            raise ValueError("Must provide a name for the pERG table data!")
        if description is None:
            raise ValueError("Must provide a description for the pERG data!")

        warn_on_name_format(table_name)

        data_dict, metadata_dict = perg_parse_to_table(filename, reformat_column_names=reformat_column_names)
        data_dict_name = f"{table_name}_data"
        metadata_dict_name = f"{table_name}_metadata"

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

        if metadata_dict_name in nwbfile.acquisition:
            nwbfile.acquisition[metadata_dict_name].add_row(metadata_dict)
        else:
            nwbfile.add_acquisition(DynamicTable(
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
