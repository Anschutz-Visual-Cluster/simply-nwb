from pynwb import NWBHDF5IO


def read_nwb(filename):
    """
    Read a file from the filesystem into an NWB object
    :param filename: filename of an NWB file
    :return: file pointer ready to be .read() to get the nwb object
    """
    # Can't use context manager, will close file, return file pointer
    io = NWBHDF5IO(filename, mode="r")
    return io


def write_nwb(nwb_obj, filename):
    """
    Write an NWB object to a file on the local filesystem
    :param nwb_obj: pynwb.file.NWBFile object
    :param filename: path of a local file, doesn't need to exist
    :return: None
    """
    io = NWBHDF5IO(filename, mode="w")
    io.write(nwb_obj)
    io.close()
