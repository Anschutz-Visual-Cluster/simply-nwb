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
    except:
        # Return a really large number of frames, and reduce to actual later
        # only other way to get an accurate count of frames would be to read the entire file
        return 1000000


def mp4_read_data(filename=None):
    if not filename:
        raise ValueError("Must provide filename argument!")
    if not os.path.exists(filename):
        raise ValueError(f"Could not find file '{filename}'!")

    movie = cv2.VideoCapture(filename)
    frames = _get_framecount(movie)  # Note framecount could be overcounted, will have to truncate array later

    read_success, img_array = movie.read()
    if not read_success:
        raise ValueError(f"Cannot read first frame of mp4 file '{filename}'!")

    dtype = img_array.dtype
    shape = img_array.shape
    # TODO
    with tempfile.TemporaryDirectory() as tmp_dirname:
        fn = os.path.join(tmp_dirname, str(uuid.uuid4()))
        np.memmap(fn, dtype=dtype, mode="w+", shape=shape)

    # import cv2
    # from PIL import Image
    # import pickle
    #
    # a = cv2.VideoCapture("../data/mp4_test.mp4")
    # suc, img_arr = a.read()
    # done = False
    # count = 0
    # arr_data = []
    #
    # while not done:
    #     if count % 10000 == 0:
    #         print(count)
    #     suc, img_arr = a.read()
    #     if not suc:
    #         pickle.dump(arr_data, open("pkltest", "wb"))
    #         tw = 2
    #     else:
    #         arr_data.append(img_arr)
    #         count += 1
