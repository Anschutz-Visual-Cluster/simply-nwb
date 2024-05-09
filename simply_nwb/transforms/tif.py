import os
import warnings
from typing import Any

from PIL import Image
import numpy as np
import glob

from hdmf.backends.hdf5 import H5DataIO
from pynwb import NWBFile
from pynwb.image import ImageSeries


class _TIFMixin(object):
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
        if len(numpy_data.shape) < 3:
            raise ValueError("Invalid data shape! numpy_data must be at least 3 dimensional. If you only have 1 image, add an axis using myimgarr[None, :] to go from (a, b) to (1, a, b)")

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

        from simply_nwb import SimpleNWB
        SimpleNWB.add_to_processing_module(
            nwbfile,
            module_name=processing_module_name,
            module_description=description,
            data=images
        )
        return nwbfile


class ImageLoadError(Exception):
    pass


def tif_read_image(filename: str, show_error: bool = True) -> np.ndarray:
    """
    Read TIF Image into a numpy array

    :param filename: TIF file to read
    :param show_error: Print out errors or not, defaults to true
    :return: numpy array
    """
    if not os.path.exists(filename):
        raise ValueError(f"Filename '{filename}' not found!")
    print(f"Reading TIF: '{filename}'")

    try:
        img = Image.open(filename)
        arr = np.array(img)
        img.close()
    except Exception as e:
        if show_error:
            print("ERROR! Failed reading image! (Is the file corrupted?)")
            print(f"Error {str(e)}")
        raise ImageLoadError()

    return arr


def read_filelist(filelist: list[str], skip_on_error: bool = False, show_error: bool = True) -> list[np.ndarray]:
    """
    Load a list of files and return them

    :param filelist: list of str filenames
    :param skip_on_error: skip any images that fail
    :param show_error: show an error when images fail to load
    :return: list of loaded file data
    """
    data = []
    for file in filelist:
        try:
            data.append(tif_read_image(file, show_error=show_error))
        except ImageLoadError as e:
            if skip_on_error:
                print(f"Error reading file '{file}', skipping..")
                continue
            raise e
    return data


def tif_read_directory(foldername: str, filename_glob: str = "*.tif", skip_on_error: bool = False) -> np.ndarray:
    """
    Read a directory of TIF files, giving a filename glob for specific TIFs to grab

    :param foldername: Folder that contains the TIF images, directly inside
    :param skip_on_error: Skip any files that fail to load
    :param filename_glob: naming scheme for the TIF files to be collected. e.g. 'image_*.tif'
    :return: numpy array
    """
    if not os.path.exists(foldername):
        raise ValueError(f"Folder '{foldername}' not found!")

    files = glob.glob(os.path.join(foldername, filename_glob))
    if files is None or not files:
        raise ValueError(f"No files found with glob '{filename_glob}' in folder {foldername}!")
    if any([os.path.isdir(ff) for ff in files]):
        raise ValueError(f"Filename Glob '{filename_glob}' includes a directory!")

    print(f"Reading folder of TIFs: '{foldername}'")
    data = read_filelist(files, skip_on_error=skip_on_error, show_error=not skip_on_error)

    return np.array(data)  # Convert our list into a numpy array


def tif_read_subfolder_directory(parent_folder: str, subfolder_glob: str, file_glob: str, subfolder_glob_fail_on_file_found: bool = False, skip_on_error: bool = False) -> np.ndarray:
    """
    Read a directory that contains folders that contain TIFs, and read each TIF from each subfolder into memory

    :param parent_folder: main folder containing more folders
    :param subfolder_glob: glob to specify which subfolders to look into e.g. 'folder_num_*'
    :param subfolder_glob_fail_on_file_found: If the subfolder glob returns a file, fail the operation. Defaults to True and if a file is matched, it will ignore it
    :param file_glob: TIF file glob to specify which TIFs from the subfolders to read, e.g. 'image0*.tif'
    :param skip_on_error: Skip any files that fail to load
    :return: numpy array
    """
    if not os.path.exists(parent_folder):
        raise ValueError(f"Parent folder '{parent_folder}' not found!")

    subfolders = glob.glob(os.path.join(parent_folder, subfolder_glob))

    if subfolder_glob_fail_on_file_found and any([os.path.isfile(sf) for sf in subfolders]):
        raise ValueError(f"Subfolder glob '{subfolder_glob}' returned a file!")

    filenames = []
    print(f"Reading TIF Parent folder: '{parent_folder}'")
    for subfolder in subfolders:
        if os.path.isfile(subfolder):
            continue
        print(f"Reading Subfolder of TIFs: '{subfolder}'")
        files = glob.glob(os.path.join(subfolder, file_glob))
        if any([os.path.isdir(f) for f in files]):
            raise ValueError(f"File glob '{file_glob}' returned a directory!")
        filenames.extend(files)
    if not filenames:
        raise ValueError(f"No files found using subfolder glob '{subfolder_glob}' and file glob '{file_glob}'")

    print("Reading all files..")
    imgs = read_filelist(filenames, skip_on_error=skip_on_error, show_error=not skip_on_error)
    return np.array(imgs)
