import numpy as np
import os
import cv2
import atexit
import imageio.v3 as iio


def _get_framecount(filename):
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


def mp4_read_data(filename=None):
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

    tmp_filename = "tmp_memmap"
    mem_data = np.memmap(filename=tmp_filename, mode="w+", dtype=first_frame.dtype, shape=(frame_count, *first_frame.shape))

    def clean_memmap(mmdata, tmp_fn):
        try:
            mmdata.flush()
            mmdata._mmap.close()
            del mmdata
            os.remove(tmp_fn)
        except Exception as e:
            print(f"Error deleting temporary file '{tmp_fn}'")
            raise e

    atexit.register(clean_memmap, mmdata=mem_data, tmp_fn=tmp_filename)

    mem_data[0] = first_frame
    count = 1
    for idx, frame in enumerate(frames):
        mem_data[idx+1] = frame
        count = count + 1

    return mem_data, count
