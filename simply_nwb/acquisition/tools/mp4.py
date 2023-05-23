import warnings
import numpy as np
import pandas as pd
import os
import tempfile
import uuid
import cv2


def _get_framecount(movie):
    try:
        val = movie.get(cv2.CAP_PROP_FRAME_COUNT)
        if not val or val == 0:
            raise ValueError
        return val
    except:
        # Return a really large number of frames, and reduce to actual later
        # only other way to get an accurate count of frames would be to read the entire file
        return 1000000


def mp4_read_data(filename=None):
    if filename is None:
        raise ValueError("Must provide filename argument!")
    if not os.path.exists(filename):
        raise ValueError(f"Could not find file '{filename}'!")

    movie = cv2.VideoCapture(filename)
    frames = _get_framecount(movie)  # Note framecount could be overcounted, will have to truncate array later

    read_success, img_array = movie.read()
    if not read_success:
        raise ValueError(f"Cannot read first frame of mp4 file '{filename}'!")

    dtype = img_array.dtype
    shape = list(img_array.shape)
    shape.insert(0, int(frames))
    shape = tuple(shape)

    count = 0

    with tempfile.TemporaryDirectory() as tmp_dirname:
        print(tmp_dirname)

        fn = os.path.join(tmp_dirname, str(uuid.uuid4()))
        mem_data = np.memmap(filename=fn, dtype=dtype, mode="w+", shape=shape)
        while True:
            mem_data[count] = img_array
            read_success, img_array = movie.read()
            if not read_success:
                break
            else:
                count = count + 1

        # TODO Figure out what to return, np.memmap()?
        raise NotImplemented

        from simply_nwb.testing import gen_snwb
        from simply_nwb import SimpleNWB
        from pynwb import TimeSeries
        from hdmf.backends.hdf5.h5_utils import H5DataIO

        sn = gen_snwb()

        b = sn.create_processing_module(name="behavior", description="test desc")
        i = TimeSeries(
            name="testing",
            data=H5DataIO(
                data=mem_data[:5],
                compression=True,
                chunks=True
            ),
            unit="vals 0-255",
            rate=150.0
        )

        b.add(i)
        nwb_filename = f"ggmp4-{str(uuid.uuid4())}.nwb"
        SimpleNWB.write(sn, nwb_filename)

        try:
            mem_data.flush()
            mem_data._mmap.close()
            del mem_data
        except:
            warnings.warn("Couldn't close memory map!", ResourceWarning)

