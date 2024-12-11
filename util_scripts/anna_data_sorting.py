from multiprocessing import Pool

import pendulum
import os
import re
import json
import shutil


def copy_over(argdata):
    # {path, gratings, dlc, timestamps, root, date, sessname, labjack, sesskey}
    sessdata, destination = argdata
    print(f"Copying session '{sessdata['sessfoldername']}'..")
    dst = str(os.path.join(destination, sessdata["sessfoldername"]))

    def cpy(fn,dstnm):
        fn2 = os.path.basename(fn)
        shutil.copyfile(fn, os.path.join(dstnm, fn2))

    if os.path.exists(dst):
        print(f"Session '{sessdata['sessfoldername']}' already exists, not copying..")
        return
    else:
        os.mkdir(dst)

    cpy(sessdata["timestamps"], dst)
    cpy(sessdata["dlc"], dst)
    [cpy(s, dst) for s in sessdata["gratings"]]
    shutil.copytree(sessdata["labjack"], str(os.path.join(dst, "labjack")))
    print(f"DONE COPYING SESSION '{sessdata['sessfoldername']}'..")

def find_files(curdir, pat):
    files = os.listdir(curdir)
    found = []
    for file in files:
        newpth = os.path.join(curdir, file)
        if os.path.isdir(newpth):
            found.extend(find_files(newpth, pat))
        else:
            if re.match(pat, file):
                found.append(os.path.join(curdir, file))
    return found


def search_for_sessions(root_path):
    print("Searching for dlc sessions..")
    ff = find_files(root_path, ".*rightCam_timestamps.*\.txt")
    sessions = {}
    for myfile in ff:
        parent = os.path.dirname(myfile)
        dlc = find_files(parent, ".*rightCam.*DLC.*\.csv")
        if len(dlc) == 0:
            continue
        assert len(dlc) == 1, f"{dlc} not len 1"
        dlc = dlc[0]
        gratings = find_files(parent, ".*driftingGratingMetadata.*\.txt")
        if len(gratings) == 0:
            print(f"No gratings found in directory '{parent}', skipping..")
            continue

        name = parent[len(root_path) + 1:]
        if os.environ.get("ANNA_DEBUG"):
            tt = name.split("/")[-1]
            date, _, sessname = tt.split("_")
            name = tt
        else:
            date, _, sessname = name.split("/")
            name = name.replace("/", "_")

        date = pendulum.parse(date)
        sesskey = (date.to_iso8601_string(), sessname[-3:])  # -3 is the last numbers 001, 002, etc

        print(f"Session '{sesskey}' found")
        sessions[sesskey] = {
            "path": parent,
            "gratings": gratings,
            "dlc": dlc,
            "timestamps": myfile,
            "root": root_path,
            "date": date,
            "sessname": sessname,
            "sesskey": sesskey,
            "sessfoldername": name
        }

    return sessions


def match_dlc_to_labjack(dlcsessions, labjackpath):
    # dlcsessions is a dict with k: v like
    # (sessdate, sessnum): {path, gratings, dlc, timestamps, root, date, sessname, sesskey}

    # Expects format to be yyyymmdd_<unitR2/etc>_session<num>
    print("Matching session to labjack..")
    labjackfolders = os.listdir(labjackpath)  # Expects format to be control<num>-mm-d-yy
    matched = {}
    all_labkeys = []

    for lab in labjackfolders:
        sepidx = lab.find("-") + 1
        labdate = lab[sepidx:]
        groupname = lab[:sepidx-1]
        labdate = pendulum.from_format(labdate, "MM-DD-YY")
        if not groupname.startswith("control"):
            continue
        groupnum = groupname[-3:]
        labkey = (labdate.to_iso8601_string(), groupnum)
        all_labkeys.append(labkey)

        assert labkey not in matched, f"Duplicate session found, '{labkey}'"

        if labkey in dlcsessions:
            print(f"Found labjack for session '{labkey}'")
            matched[labkey] = dlcsessions[labkey]
            matched[labkey]["labjack"] = os.path.join(labjackpath, lab)
        elif not os.environ.get("ANNA_DEBUG"):
            raise ValueError(f"No DLC data found for labjack session '{labkey}' Found DLC sessions '{list(dlcsessions.keys())}'")
    if os.environ.get("ANNA_DEBUG"):
        matched[labkey] = dlcsessions[list(dlcsessions.keys())[0]]
        matched[labkey]["labjack"] = os.path.join(labjackpath, lab)

    numfound = len(matched.keys())
    assert numfound > 0, f"No sessions matched! Labjacks: '{all_labkeys}' \n\nDLCs: '{list(dlcsessions.keys())}"

    print(f"Matched {numfound} sessions")
    return matched # (sessdate, sessnum): {path, gratings, dlc, timestamps, root, date, sessname, labjack}


def check_drifting_grating_anna_data():
    hdroot = "/media/felsenlab/Bard Brive"

    os.environ["ANNA_DEBUG"] = "True"

    labjackpath = f"{hdroot}/AnnaData/labjack"
    # dlc_root = "/media/retina2/AB-DATA-01/CompressedDataLocal"
    dlc_root = f"{hdroot}/AnnaData"

    destination = f"{hdroot}/AnnaData2"

    dlcsessions = search_for_sessions(dlc_root)
    sessions = match_dlc_to_labjack(dlcsessions, labjackpath)

    print("Starting copy pool..")
    with Pool(8) as p:
        args = list(zip(list(sessions.values()), [destination]*len(sessions.keys())))
        p.map(copy_over, args)
    # for s in list(sessions.values()):
    #     copy_over(destination)(s)
    tw = 2

if __name__ == "__main__":
    check_drifting_grating_anna_data()

