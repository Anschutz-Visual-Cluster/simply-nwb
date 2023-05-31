import os
from PIL import Image
import numpy as np
import glob


def tif_read_image(filename=None):
    """
    Read TIF Image into a numpy array

    :param filename: TIF file to read
    :return: numpy array
    """

    if filename is None:
        raise ValueError("Must supply filename argument!")
    if not os.path.exists(filename):
        raise ValueError(f"Filename '{filename}' not found!")
    print(f"Reading TIF: '{filename}'")

    try:
        img = Image.open(filename)
        arr = np.array(img)
        img.close()
    except Exception as e:
        print("ERROR! Failed reading image! (Is the file corrupted?)")
        raise e
    return arr


def tif_read_directory(foldername=None, filename_glob="*.tif"):
    """
    Read a directory of TIF files, giving a filename glob for specific TIFs to grab

    :param foldername: Folder that contains the TIF images, directly inside
    :param filename_glob: naming scheme for the TIF files to be collected. e.g. 'image_*.tif'
    :return: numpy array
    """
    if foldername is None:
        raise ValueError("Must provide folder name argument!")
    if not os.path.exists(foldername):
        raise ValueError(f"Folder '{foldername}' not found!")

    files = glob.glob(os.path.join(foldername, filename_glob))
    if files is None:
        raise ValueError(f"No files found with glob '{filename_glob}' in folder {foldername}!")
    if any([os.path.isdir(ff) for ff in files]):
        raise ValueError(f"Filename Glob '{filename_glob}' includes a directory!")

    data = []
    print(f"Reading folder of TIFs: '{foldername}'")
    for file in files:
        data.append(tif_read_image(file))
    return np.array(data)  # Convert our list into a numpy array


def tif_read_subfolder_directory(parent_folder=None, subfolder_glob=None, subfolder_glob_fail_on_file_found=False, file_glob=None):
    """
    Read a directory that contains folders that contain TIFs, and read each TIF from each subfolder into memory

    :param parent_folder: main folder containing more folders
    :param subfolder_glob: glob to specify which subfolders to look into e.g. 'folder_num_*'
    :param subfolder_glob_fail_on_file_found: If the subfolder glob returns a file, fail the operation. Defaults to True and if a file is matched, it will ignore it
    :param file_glob: TIF file glob to specify which TIFs from the subfolders to read, e.g. 'image0*.tif'
    :return: numpy array
    """
    if parent_folder is None:
        raise ValueError("Must provide parent_folder argument!")
    if subfolder_glob is None:
        raise ValueError("Must provide subfolder_glob argument!")
    if file_glob is None:
        raise ValueError("Must provide file_glob argument!")

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
    imgs = []
    for fn in filenames:
        imgs.append(tif_read_image(fn))
    return np.array(imgs)
