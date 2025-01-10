import glob


def ephys_align(folderpath):
    # Load the neuropixels event timestamps that were sent by labjack to get a reference for the np (neuropixels) clock
    np_ts = glob.glob(f"{folderpath}/**/*TTL*/*timestamps*.npy", recursive=True)
    assert len(np_ts) == 1, "Found multiple neuropixels timestamp files! Manually specifying recommended!"
    np_ts = np_ts[0]



    labjack = glob.glob(f"{folderpath}/**/labjack/*.dat", recursive=True)
    assert len(labjack) > 0, "No labjack files found!"

    drifting = glob.glob(f"{folderpath}/**/driftingGratingMetadata*.txt", recursive=True)
    assert len(drifting) > 0, "No driftingGratingMetadata files found!"

    dlc_timestamps = glob.glob(f"{folderpath}/**/*rightCam*timestamps*.txt", recursive=True)
    assert len(dlc_timestamps) == 1, "Should only be 1 dlc timestamps txt! Found {} instead".format(len(dlc_timestamps))
    dlc_timestamps = dlc_timestamps[0]

    dlc_eyepos = glob.glob(f"{folderpath}/**/*rightCam*DLC*.csv", recursive=True)
    assert len(dlc_eyepos) == 1, "Should only be 1 rightCam DLC csv! Found {} instead".format(len(dlc_eyepos))
    dlc_eyepos = dlc_eyepos[0]



    pass


if __name__ == "__main__":
    ephys_align("data/anna_ephys")

