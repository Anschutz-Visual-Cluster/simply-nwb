import atexit
from typing import Any

import numpy as np
import os
import cv2
import imageio.v3 as iio
import uuid

from hdmf.backends.hdf5 import H5DataIO
from pynwb import NWBFile
from pynwb.image import ImageSeries


class _MP4Mixin(object):

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


def _get_framecount(filename: str) -> int:
    try:
        movie = cv2.VideoCapture(filename)
        val = movie.get(cv2.CAP_PROP_FRAME_COUNT)
        if not val or val == 0:
            raise ValueError
        return int(val)
    except:
        # Return a really large number of frames, and reduce to actual later
        # only other way to get an accurate count of frames would be to read the entire file
        return 1000000


def mp4_read_data(filename: str) -> (np.ndarray, int):
    """
    Read the data from an MP4 file into a numpy array and get the framecount

    :param filename: mp4 file to read
    :return: (numpy arr, framecount)
    """
    if filename is None:
        raise ValueError("Must provide filename argument!")
    if not os.path.exists(filename):
        raise ValueError(f"Could not find file '{filename}'!")

    frame_count = _get_framecount(filename)

    frames = iio.imiter(filename, plugin="pyav")
    first_frame = next(frames)

    tmp_filename = f"tmpmemmap-{str(uuid.uuid4())}"
    try:
        mem_data = np.memmap(filename=tmp_filename, mode="w+", dtype=first_frame.dtype,
                             shape=(frame_count, *first_frame.shape))
    except OSError as e:
        raise Exception(f"Unable to create memmap, are you out of harddrive space? Error: {str(e)}")

    def clean_memmap(mmdata, tmp_fn):
        try:
            mmdata.flush()
            mmdata._mmap.close()
            del mmdata
            os.remove(tmp_fn)
        except KeyboardInterrupt as e:
            print(f"Program terminated before tempfile '{os.path.abspath(tmp_fn)}' could be removed")
        except Exception as e:
            print(f"Error deleting temporary file '{tmp_fn}'")
            raise e

    atexit.register(clean_memmap, mmdata=mem_data, tmp_fn=tmp_filename)

    mem_data[0] = first_frame
    count = 1
    for idx, frame in enumerate(frames):
        mem_data[idx + 1] = frame
        count = count + 1

    return mem_data, count
