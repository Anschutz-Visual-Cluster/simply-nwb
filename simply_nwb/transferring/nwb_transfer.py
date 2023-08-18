import os
import shutil
from typing import Optional

import pendulum
from simply_nwb.util import is_camel_case, is_snake_case, is_filesystem_safe, _print
import glob


class NWBTransfer(object):
    """
    Class to aid in transferring data around in a structured way
    """
    
    TIMESTAMP_SUFFIX = "{day}_{month}_{year}-{hour}_{minute}_{second}"

    def __init__(
            self,
            nwb_file_location: str,
            raw_data_folder_location: str,
            lab_name: str,
            project_name: str,
            session_name: str,
            transfer_location_root: str
    ):
        """
        Create a transfer helper object to copy the NWB and raw data over to the storage location

        :param nwb_file_location: String path location to the .nwb file
        :param raw_data_folder_location: Folder containing all the raw data relevant to the session, will be zipped
        :param lab_name: Name of the Lab, for example: 'FelsenLab', 'DenmanLab', 'PolegPolskyLab' in CamelCase. See https://en.wikipedia.org/wiki/Camel_case
        :param project_name: Project name, should be in snake_case, see https://simple.wikipedia.org/wiki/Snake_case
        :param session_name: Name for the session, will automatically add 'day_month_year_time' suffix. Should also be in snake_case
        :param transfer_location_root: Location of the storage system, if mounted locally can use Drive:\\ or network if network attached use \\domain\\folder
        """

        if not os.path.exists(raw_data_folder_location) or not os.path.isdir(raw_data_folder_location):
            raise ValueError("Make sure raw_data_folder_location exists and is a folder!")
        if not is_camel_case(lab_name):
            raise ValueError(f"Error '{lab_name}' is not in CamelCase!")
        if not is_snake_case(project_name):
            raise ValueError(f"Error '{project_name}' is not in snake_case!")
        if not is_filesystem_safe(session_name):
            raise ValueError(f"Error '{session_name}' is not in snake_case!")
        if transfer_location_root is None:
            raise ValueError("Must supply 'transfer_location_root' argument!")
        if not os.path.exists(transfer_location_root):
            raise ValueError(
                f"Given 'transfer_location_root' = '{transfer_location_root}' can't be found! (Try using an absolute path?)")
        if not os.path.isdir(transfer_location_root):
            raise ValueError(f"Given 'transfer_location_root' = '{transfer_location_root}' isn't a directory!")

        self.nwb_filename = NWBTransfer.make_nwb_filename(session_name)
        self.raw_data_folder_location = os.path.abspath(raw_data_folder_location)

        self.nwb_file_location = os.path.abspath(nwb_file_location)
        self.zip_file_name_no_extension = NWBTransfer.make_raw_zip_filename(session_name)
        self.zip_file_full_name = f"{self.zip_file_name_no_extension}.zip"

        self.lab_root = os.path.join(
            os.path.abspath(transfer_location_root),
            lab_name,
        )

        self.project_destination_path_root = os.path.join(
            self.lab_root,
            project_name
        )

        self.nwbs_folder = os.path.abspath(os.path.join(self.project_destination_path_root, "nwbs"))
        self.raw_folder = os.path.abspath(os.path.join(self.project_destination_path_root, "raw"))

        if not os.path.exists(self.project_destination_path_root):
            print(f"Project dir '{self.project_destination_path_root}' doesn't exist, creating")
            os.mkdir(self.lab_root)
            os.mkdir(self.project_destination_path_root)
            os.mkdir(self.nwbs_folder)
            os.mkdir(self.raw_folder)

        nwb_sessions = glob.glob(os.path.join(self.nwbs_folder, f"{session_name}*"))
        if nwb_sessions:
            raise ValueError(
                f"Error: nwb session '{session_name}' already exists in project '{project_name}'! Please make session names unique")

        raw_sessions = glob.glob(os.path.join(self.raw_folder, f"{session_name}*"))
        if raw_sessions:
            raise ValueError(
                f"Error: raw session '{session_name}' already exists in project '{project_name}'! Please make session names unique")

        self.nwb_destination_filename = os.path.join(
            self.project_destination_path_root,
            "nwbs",
            self.nwb_filename
        )

        self.raw_zip_destination_filename = os.path.join(
            self.project_destination_path_root,
            "raw",
            self.zip_file_full_name
        )

    def upload(self, zip_location_override: Optional[str] = None, do_print: bool = True) -> None:
        """
        Upload the NWB and raw data to the storage location, with an optional zip override. Can set do_print to False
        to silence this method

        :param zip_location_override: Optional override for the location of a zipfile, will skip zipping the raw data folder
        :param do_print: if False, method will not print anything
        :return: None
        """

        if zip_location_override:
            if os.path.exists(zip_location_override) and os.path.isfile(zip_location_override):
                _print(f"Copying '{zip_location_override}' to '{self.raw_zip_destination_filename}'..")
                shutil.copy(zip_location_override, self.raw_zip_destination_filename)
            else:
                raise ValueError("Given zip_location_override doesn't exist or isn't a file!")
        else:
            _print("Zipping up raw data..", do_print)
            old_cwd = os.getcwd()
            os.chdir(self.raw_folder)

            shutil.make_archive(
                self.zip_file_name_no_extension,
                "zip",
                self.raw_data_folder_location
            )
            os.chdir(old_cwd)
            _print("Raw data zip complete", do_print)

        _print(f"Copying '{self.nwb_file_location}' to '{self.nwb_destination_filename}'..")
        shutil.copy(self.nwb_file_location, self.nwb_destination_filename)

    @staticmethod
    def make_raw_zip_filename(session_name: str) -> str:
        return "{}_{}".format(
            session_name,
            NWBTransfer.make_timestamp_filename_suffix()
        )

    @staticmethod
    def make_nwb_filename(session_name: str) -> str:
        return "{}_{}.nwb".format(
            session_name,
            NWBTransfer.make_timestamp_filename_suffix()
        )

    @staticmethod
    def make_timestamp_filename_suffix() -> str:
        now = pendulum.now()
        timestamp_data = {
            "day": now.day,
            "month": now.month,
            "year": now.year,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second
        }

        return NWBTransfer.TIMESTAMP_SUFFIX.format_map(
            _EmptyDictHelper(**timestamp_data)
        )


class _EmptyDictHelper(dict):
    # Helper Class for formatting strings with dicts
    def __missing__(self, key):
        return ""
