import os
import signal
import shutil
import time
import glob
from threading import Thread
from typing import Callable, Union

from fastcrc import crc64


class CopyThread(Thread):
    def __init__(self, src_pth: str, dst: str, filename: str, do_del: bool):
        """
        Create a thread to copy an object to a destination. obj could be a file or directory

        :param src_pth: File or directory location path to copy to dst
        :param dst: Destination to copy the file to
        """
        super().__init__(target=self.run_method)
        self.src_pth = src_pth
        self.dst = os.path.join(dst, filename)
        self.do_del = do_del

    def run_method(self):
        if os.path.isdir(self.src_pth):
            print("Starting copy of directory '{}' to '{}'".format(self.src_pth, self.dst))
            shutil.copytree(self.src_pth, self.dst, dirs_exist_ok=True)
            print("Completed copy of directory '{}' to '{}'".format(self.src_pth, self.dst))
            if self.do_del:
                print("Removing directory '{}'".format(self.src_pth))
                shutil.rmtree(self.src_pth)
        else:
            print("Starting copy of file '{}' to '{}'".format(self.src_pth, self.dst))
            os.makedirs(os.path.dirname(self.dst), exist_ok=True)
            shutil.copy(self.src_pth, self.dst)
            print("Completed copy of file '{}' to '{}'".format(self.src_pth, self.dst))
            if self.do_del:
                print("Removing file '{}'".format(self.src_pth))
                os.remove(self.src_pth)


class OneWayFileSync(object):
    def __init__(self, source_directory: str, destination_directory: str,
                 watch_file_glob: Union[dict, str],
                 interval: int = 5, delete_on_copy: bool = False):
        """
        Create a new FileSync instance, to automatically copy files from src to dest, that match a file glob queried at an interval.
        Can provide a string for watch_file_glob or a dict. The dict will be in format::


            def split_filename(filename):  # Will split 'img_5.tif' and return '5'
                a = filename.split("_")
                b = a[1].split(".")
                return b[0]

            {
                # Will copy over files that match '*.tif' to 'image_{name}.tif'
                # Using the function above to reformat the filename
                # For example a file matched called 'img_5.tif' will be copied to 'image_5.tif'
                # since the above function will split off the '5' and then format to 'image_{name}.tif'
                # which is just 'image_5.tif'
                # See transferring_custom_one_way.py in the examples/ directory
                "*.tif": {
                    "filename": "image_{name}.tif",
                    "name_func": split_filename
                },
                "*.txt": {}  # Will copy all txt files without renaming them
                ...
            }

        :param source_directory: Directory to watch for new files
        :param destination_directory: Directory to copy files to
        :param watch_file_glob: Glob for selecting which files to copy from the source directory, can provide a dict for more customization
        :param interval: how often should the source directory be queried in seconds
        :param delete_on_copy: After copying a file to the destination, delete it from the source location
        """

        if not os.path.exists(source_directory):
            raise ValueError(f"Source directory '{source_directory}' not found!")
        if not os.path.exists(destination_directory):
            raise ValueError(f"Destination directory '{destination_directory}' not found!")
        if isinstance(watch_file_glob, str):
            watch_file_glob = {watch_file_glob: {}}

        self.src = source_directory
        self.dst = destination_directory
        self.interval = interval
        self.delete = delete_on_copy
        self._stopped = False
        # Register stopping function with sigint for ctrl+c killing
        signal.signal(signal.SIGINT, self.stop)

        def gen_process_func(fglob: str, data: dict) -> Callable[[str], str]:
            if not data:  # Data passed in was just {}
                def process(fn):
                    return fn
            else:
                if "name_func" not in data or "filename" not in data:
                    raise ValueError(
                        f"KeyData for file glob '{fglob}' is not in the expected format! See docs on usage")

                def process(fn):
                    name = data["name_func"](fn)
                    return data["filename"].format(name=name)

            return process

        self.fglobs = {
            fglob: gen_process_func(fglob, keydata) for fglob, keydata in watch_file_glob.items()
        }

    def _check_stop(self) -> bool:
        return self._stopped

    def stop(self, *args, **kwargs):
        """
        Stop the filesync watcher from code

        :param args: None
        :param kwargs: None
        :return: None
        """
        print("Shutting down FileSync..")
        print("Waiting for current copy operations to finish..")
        self._stopped = True

    def _calc_checksum(self, filename: str) -> str:
        io = open(filename, "rb")
        hsh = crc64.ecma_182(io.read())
        io.close()
        return hsh

    def start(self):
        """
        Start the filesync watcher, will watch for changes until killed/stopped
        :return:
        """
        print(f"Starting FileSync watch over directory: '{self.src}'")
        file_hashmap = {
            # checksum: filename
        }

        while not self._stopped:
            for file_glob, filename_process_func in self.fglobs.items():
                matches = glob.glob(os.path.join(self.src, file_glob))
                if not matches:  # Nothing matched this file glob, continue
                    continue

                for file_path in matches:
                    file = os.path.basename(file_path)
                    new_filename = filename_process_func(file)

                    if os.path.isfile(file_path):
                        fhash = self._calc_checksum(file_path)
                        if fhash in file_hashmap:
                            continue
                        else:
                            print(f"New file '{file}' detected")
                            CopyThread(file_path, self.dst, new_filename, self.delete).start()
                            if not self.delete:
                                file_hashmap[fhash] = file
                    else:
                        if file in file_hashmap:
                            continue
                        else:
                            print(f"New folder '{file}' detected")
                            CopyThread(file_path, self.dst, new_filename, self.delete).start()
                            if not self.delete:
                                file_hashmap[file] = file
            time.sleep(self.interval)
