from datetime import datetime
from typing import Optional, Any
from uuid import uuid4
import os
import glob

import numpy as np
import pandas as pd
import pynwb
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common.table import DynamicTable
from hdmf.common.table import VectorData
from pynwb import NWBFile
from pynwb.behavior import BehavioralEvents
from pynwb.file import Subject
from pynwb.image import ImageSeries
from pynwb.misc import AnnotationSeries
from pynwb.ophys import OpticalChannel, TwoPhotonSeries

from simply_nwb.transforms import labjack_load_file
from simply_nwb.transforms import blackrock_all_spiketrains, perg_parse_to_table
from simply_nwb.util import warn_on_name_format, inspect_nwb_obj, nwb_write, panda_df_to_dyn_table, \
    panda_df_to_list_of_timeseries, dict_to_dyn_tables, inspect_nwb_file


class SimpleNWB(object):

    @staticmethod
    def create_nwb(
            session_description: str,
            session_start_time: datetime,
            experimenter: [str],
            lab: str,
            experiment_description: str,
            institution: str,
            # Optional
            identifier: Optional[str] = None,
            subject: Optional[Subject] = None,
            session_id: Optional[str] = None,
            keywords: Optional[list[str]] = None,
            related_publications: Optional[str] = None
    ) -> NWBFile:
        """
        Create a new nwbfile from the given params. More infor here https://pynwb.readthedocs.io/en/stable/pynwb.file.html#pynwb.file.NWBFile

        :param session_description: description of the session
        :param session_start_time: start date and time of recording session
        :param experimenter: name of experimenter, list of form ["Lastname, Firstname"]
        :param lab: name of lab
        :param experiment_description: experiment description
        :param institution: institution
        :param identifier: Optional identifier for the file, if not supplied will be generated
        :param subject: Optional pynwb.file.Subject object for metadata
        :param session_id: Optional lab-specific session id, if not supplied will be generated
        :param keywords: Optional list of keywords e.g ["keyword1", "keyword2", ...]
        :param related_publications: Optional related publications in a list of the DOI, URL, PMID etc ["DOI:1234/asdf"]
        :return: NWBFile
        """
        # Optional args
        if identifier is None:
            identifier = str(uuid4())
        if session_id is None:
            session_id = str(uuid4())
        if isinstance(subject, dict):
            subject = Subject(**subject)
        elif not isinstance(subject, Subject) and subject is not None:
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
    def inspect_filename(nwbfilename: str) -> list[Any]:
        """
        Inspect the given NWBFile from filename

        :param nwbfilename: filename to the NWB
        :return: List of issues with the file, if empty, inspection passed
        """

        results = inspect_nwb_file(nwbfilename)
        return results

    @staticmethod
    def inspect_obj(nwbfile: NWBFile) -> list[Any]:
        """
        Inspect the given NWBFile object

        :param nwbfile: NWBFile object
        :return: List of issues with the file, if empty, inspection passed
        """

        results = inspect_nwb_obj(nwbfile)
        return results

    @staticmethod
    def write(nwbfile: NWBFile, filename: str, verify_on_write: Optional[bool] = True) -> NWBFile:
        """
        Write the give NWBFile object to file

        :param nwbfile: NWBFile object to write
        :param filename: path to file to write, WILL OVERWRITE!
        :param verify_on_write: Verify that *most* fields wrote correctly and the file didn't corrupt
        :return: NWBFile
        """
        nwb_write(nwbfile, filename, verify_on_write)
        return nwbfile

    @staticmethod
    def add_to_processing_module(nwbfile: NWBFile, data: Any, module_name: Optional[str] = None, module_description: Optional[str] = None) -> NWBFile:
        """
        Add data to a processing module, automatically creating it if it doesn't already exist

        :param nwbfile: NWBFile to add the data to
        :param module_name: Name of the processing module to create or use
        :param module_description: Description of the module, if not provided will be auto generated
        :param data: Data to be added to the processing module
        :return: NWBFile
        """
        if module_name is None:
            raise ValueError("Must provide module name argument!")
        if module_description is None:
            module_description = f"{module_name} processing module"
        if data is None:
            raise ValueError("Must provide data to add to the processing module!")

        if module_name in nwbfile.processing:
            pmodule = nwbfile.processing[module_name]
        else:
            pmodule = nwbfile.create_processing_module(name=module_name, description=module_description)

        pmodule.add(data)
        return nwbfile

    @staticmethod
    def tif_add_as_processing_imageseries(
            nwbfile: NWBFile,
            name: str,
            processing_module_name: str,
            numpy_data: Any,
            sampling_rate: float,
            description: str,
            starting_time: float = 0.0,
            chunking: bool = True,
            compression: bool = True
    ) -> NWBFile:
        """

        Add a numpy array as a processing module with image data, meant to work in conjunction with the tif reader utility

        :param nwbfile: NWBFile to add the tif data to
        :param name: Name of the movie
        :param numpy_data: data, can be np.memmap
        :param processing_module_name: Name of the processing module to append or create to add the images to
        :param sampling_rate: frames per second
        :param description: description of the movie
        :param starting_time: Starting time of this movie, relative to experiment start. Defaults to 0.0
        :param chunking: Optional chunking for large files, defaults to True
        :param compression: Optional compression for large files, defaults to True
        :return: NWBFile
        """
        images = ImageSeries(
            name=name,
            data=H5DataIO(
                data=numpy_data,
                compression=compression,
                chunks=chunking
            ),
            description=description,
            unit="n.a.",
            rate=sampling_rate,
            format="raw",
            starting_time=starting_time
        )

        SimpleNWB.add_to_processing_module(
            nwbfile,
            module_name=processing_module_name,
            module_description=description,
            data=images
        )
        return nwbfile

    @staticmethod
    def mp4_add_as_acquisition(
            nwbfile: NWBFile,
            name: str,
            numpy_data: Any,
            frame_count: float,
            sampling_rate: float,
            description: str,
            starting_time: float = 0.0,
            chunking: bool = True,
            compression: bool = True
    ) -> NWBFile:
        """
        Add a numpy array as acquisition data, meant to work in conjunction with the mp4 reader utility

        :param nwbfile: NWBFile to add the mp4 data to
        :param name: Name of the movie
        :param numpy_data: data, can be np.memmap
        :param frame_count: number of total frames
        :param sampling_rate: frames per second
        :param description: description of the movie
        :param starting_time: Starting time of this movie, relative to experiment start. Defaults to 0.0
        :param chunking: Optional chunking for large files, defaults to True
        :param compression: Optional compression for large files, defaults to True
        :return: NWBFile
        """
        if name is None:
            raise ValueError("Must supply name argument for the name of the ImageSeries!")
        if numpy_data is None:
            raise ValueError("Must supply numpy_data for insert into ImageSeries!")
        args = (sampling_rate, description, starting_time, chunking, compression)
        if None in args:
            raise ValueError(f"Missing argument, got None in args {str(args)}")

        mp4_timeseries = ImageSeries(
            name=name,
            data=H5DataIO(
                data=numpy_data[:frame_count],
                compression=compression,
                chunks=chunking
            ),
            description=description,
            unit="n.a.",
            rate=sampling_rate,
            format="raw",
            starting_time=starting_time
        )

        nwbfile.add_acquisition(mp4_timeseries)
        return nwbfile

    @staticmethod
    def processing_add_dict(
            nwbfile: NWBFile,
            processed_name: str,
            data_dict: dict,
            processed_description: str,
            processing_module_name: Optional[str] = None,
            uneven_columns: bool = False) -> NWBFile:
        """
        Add a processed dict into the NWB that doesn't fit in any other part of the NWB. MAKE SURE YOU CANNOT ADD IT ELSEWHERE BEFORE USING THIS FUNC!

        :param nwbfile: NWBFile to add data to
        :param processing_module_name: Name of the processing module to add the data to. If not set will default to 'misc'
        :param processed_name: Name of the dynamic table
        :param processed_description: description of the processed data
        :param data_dict: dict data to add
        :param uneven_columns: Set this to True if the keys of the dict have different lengths
        :return: NWBFile
        """
        if not processing_module_name:
            processing_module_name = "misc"

        data_interfaces = dict_to_dyn_tables(
            dict_data=data_dict,
            table_name=processed_name,
            description=processed_description,
            multiple_objs=uneven_columns
        )
        if not uneven_columns:
            data_interfaces = [data_interfaces]

        if processing_module_name in nwbfile.processing:
            [nwbfile.processing[processing_module_name].add_container(interface) for interface in data_interfaces]
        else:
            nwbfile.create_processing_module(
                name=processing_module_name,
                description=processed_description,
                data_interfaces=data_interfaces
            )
        return nwbfile

    @staticmethod
    def processing_add_dataframe(
            nwbfile: NWBFile,
            processed_name: str,
            processed_description: str,
            data: pd.DataFrame
    ) -> NWBFile:
        """
        Add a processed pandas Dataframe into the NWB that doesn't fit in any other part of the NWB. MAKE SURE YOU CANNOT ADD IT ELSEWHERE BEFORE USING THIS FUNC!

        :param nwbfile: NWBFile to add data to
        :param processed_name: Name of the processing module
        :param processed_description: description of the processed data
        :param data: Pandas Dataframe data to add
        :return: NWBFile
        """
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
        return nwbfile

    @staticmethod
    def two_photon_add_data(
            nwbfile: NWBFile,
            device_name: str,
            device_description: str,
            device_manufacturer: str,
            optical_channel_description: str,
            optical_channel_emission_lambda: float,
            imaging_name: str,
            imaging_rate: float,
            excitation_lambda: float,
            indicator: str,
            location: str,
            grid_spacing: list[float],
            grid_spacing_unit: str,
            origin_coords: list[float],
            origin_coords_unit: str,
            two_photon_unit: str,
            two_photon_rate: float,
            photon_series_name: str,
            image_data: Any,
    ) -> NWBFile:
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
        :return: NWBFile
        """
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
        return nwbfile

    @staticmethod
    def labjack_file_as_behavioral_data(
            nwbfile: NWBFile,
            labjack_filename: str,
            name: str,
            measured_unit_list: list[str],
            start_time: float,
            sampling_rate: float,
            description: str,
            behavior_module: Optional[Any] = None,
            behavior_module_name: Optional[str] = None,
            comments: str = "Labjack behavioral data"
    ) -> NWBFile:
        """
        Add LabJack data to the NWBFile as a behavioral entry from a filename

        :param nwbfile: NWBFile to add the data to
        :param labjack_filename: filename of the labjack file to load
        :param name: Name of this behavioral unit
        :param measured_unit_list: List of SI unit strings corresponding to the columns of the labjack data
        :param start_time: start time float in Hz
        :param sampling_rate: sampling rate in Hz
        :param description: description of the behavioral data
        :param behavior_module: Optional NWB behavior module to add this data to, otherwise will create a new one e.g. nwbfile.processing["behavior"]
        :param behavior_module_name: optional module name to add this behavior to, if exists will append. will ignore if behavior_module arg is supplied
        :param comments: additional comments about the data
        :return: NWBFile
        """
        return SimpleNWB.labjack_as_behavioral_data(
            nwbfile,
            labjack_data=labjack_load_file(labjack_filename),
            name=name,
            measured_unit_list=measured_unit_list,
            start_time=start_time,
            sampling_rate=sampling_rate,
            description=description,
            behavior_module=behavior_module,
            behavior_module_name=behavior_module_name,
            comments=comments
        )

    @staticmethod
    def labjack_as_behavioral_data(
            nwbfile: NWBFile,
            labjack_data: dict,
            name: str,
            measured_unit_list: list[str],
            start_time: float,
            sampling_rate: float,
            description: str,
            behavior_module: Optional[Any],
            behavior_module_name: Optional[str],
            comments: str = "Labjack behavioral data"
    ) -> NWBFile:
        """
        Add LabJack data to the NWBFile as a behavioral entry, given the labjack data

        :param nwbfile: NWBFile to add the data to
        :param labjack_data: dict formatted like the return value from simply_nwb.transforms.labjack.labjack_load_file
        :param name: Name of this behavioral unit
        :param measured_unit_list: List of SI unit strings corresponding to the columns of the labjack data
        :param start_time: start time float in Hz
        :param sampling_rate: sampling rate in Hz
        :param description: description of the behavioral data
        :param behavior_module: Optional NWB behavior module to add this data to, otherwise will create a new one e.g. nwbfile.processing["behavior"]
        :param behavior_module_name: optional module name to add this behavior to, if exists will append. will ignore if behavior_module arg is supplied
        :param comments: additional comments about the data
        :return: NWBFile
        """
        if not isinstance(labjack_data, dict) or "metadata" not in labjack_data or "data" not in labjack_data:
            raise ValueError("Argument labjack_data should be a dict with keys 'metadata' and 'data'!")

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
            SimpleNWB.p_erg_add_data(
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
