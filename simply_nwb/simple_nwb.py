from datetime import datetime
from typing import Optional, Any
from uuid import uuid4


import numpy as np
import pandas as pd
import pendulum
from pynwb import NWBFile
from pynwb.file import Subject
from pynwb.ophys import OpticalChannel, TwoPhotonSeries

from simply_nwb.util import inspect_nwb_obj, nwb_write, panda_df_to_dyn_table, dict_to_dyn_tables, inspect_nwb_file

from simply_nwb.transforms.tif import _TIFMixin
from simply_nwb.transforms.mp4 import _MP4Mixin
from simply_nwb.transforms.labjack import _LabjackMixin
from simply_nwb.transforms.blackrock import _BlackrockMixin
from simply_nwb.transforms.p_erg import _PergMixin
from simply_nwb.transforms.eyetracking import _EyetrackingMixin


class SimpleNWB(_TIFMixin, _MP4Mixin, _LabjackMixin, _BlackrockMixin, _PergMixin, _EyetrackingMixin, object):
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
    def test_nwb() -> NWBFile:
        """
        Generate a dummy nwb object to test with

        :return: test nwb object
        """

        return SimpleNWB.create_nwb(
            # Required
            session_description="Mouse cookie eating session",
            # Subtract 1 year so we don't run into the 'NWB start time is at a greater date than current' issue
            session_start_time=pendulum.now().subtract(years=1),
            experimenter=["Schmoe, Joe"],
            lab="Felsen Lab",
            experiment_description="Gave a mouse a cookie",
            # Optional
            identifier="cookie_0",
            subject=Subject(**{
                "subject_id": "1",
                "age": "P90D",  # ISO-8601 for 90 days duration
                "strain": "TypeOfMouseGoesHere",  # If no specific used, 'Wild Strain'
                "description": "Mouse#2 idk",
                "sex": "M",  # M - Male, F - Female, U - unknown, O - other
                # NCBI Taxonomy link or Latin Binomial (e.g.'Rattus norvegicus')
                "species": "http://purl.obolibrary.org/obo/NCBITaxon_10116",
            }),
            session_id="session0",
            institution="CU Anschutz",
            keywords=["mouse"],
            # related_publications="DOI::LINK GOES HERE FOR RELATED PUBLICATIONS"
        )

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
