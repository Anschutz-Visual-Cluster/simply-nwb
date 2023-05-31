from simply_nwb.transforms import labjack_load_file, mp4_read_data
from simply_nwb import SimpleNWB
from simply_nwb.transforms import plaintext_metadata_read
from pynwb.file import Subject
import pendulum
import pickle
import os

# Simply-NWB Package Documentation
# https://simply-nwb.readthedocs.io/en/latest/index.html

INSTITUTION = "InstitutionHere"

EXPERIMENTERS = [
    "Lastname, Firstname"
]
LAB = "LabNameHere"

EXPERIMENT_DESCRIPTION = "Long description of experiment goes here"
EXPERIMENT_KEYWORDS = None  # optional
EXPERIMENT_RELATED_PUBLICATIONS = None  # optional

SESSION_IDENTIFIER = "session1"
SESSION_DESCRIPTION = "sess desc"
SUBJECT = Subject(
    subject_id="mouse1",
    age="P90D", # ISO-8601 90 days?
    strain="TypeOfMouse",  # Wild Strain ?
    description="Mouse desc goes here",
    sex="M"
)

SESSION_ROOT = "../data/mouse1"
METADATA_FILENAME = "metadata.txt"

# Need multiple labjack datas?
LABJACK_FILENAME = "labjack/lick1OKRvartest40-4-14-23/data_0.dat"
LABJACK_NAME = "Labjack Data"
LABJACK_SAMPLING_RATE = 20.0  # in Hz
LABJACK_DESCRIPTION = "labjack description here"
LABJACK_COMMENTS = "labjack comments here"

MP4_FILES = {
    "RightEye": "movies/righteye.mp4",
    "LeftEye": "movies/lefteye.mp4"
}
MP4_DESCRIPTION = "description of mp4 files here"
MP4_SAMPLING_RATE = 120.0

PICKLE_FILENAME = "pickledata.pkl"
PICKLE_DATA_NAME_PREFIX = "data"
PICKLE_DATA_DESCRIPTION = "desc of data here"


def main():
    metadata = plaintext_metadata_read(os.path.join(SESSION_ROOT, METADATA_FILENAME))
    start_date = pendulum.parse(metadata["Date"], tz="local")

    nwbfile = SimpleNWB.create_nwb(
        session_description=SESSION_DESCRIPTION,
        session_start_time=start_date,
        experimenter=EXPERIMENTERS,
        lab=LAB,
        experiment_description=EXPERIMENT_DESCRIPTION,
        session_id=SESSION_IDENTIFIER,
        institution=INSTITUTION,
        keywords=EXPERIMENT_KEYWORDS,  # TODO?
        related_publications=EXPERIMENT_RELATED_PUBLICATIONS
    )

    # Add labjack data to NWB
    labjack_filename_absolute = os.path.join(SESSION_ROOT, LABJACK_FILENAME)
    labjack_data = labjack_load_file(labjack_filename_absolute)

    SimpleNWB.labjack_as_behavioral_data(
        nwbfile,
        labjack_filename=labjack_filename_absolute,
        name=LABJACK_NAME,
        measured_unit_list=["unit for columns, Time", "v0", "v1", "v2", "v3", "y0", "y1", "y2", "y3"],
        start_time=pendulum.parse(labjack_data["date"]),
        sampling_rate=LABJACK_SAMPLING_RATE,
        description=LABJACK_DESCRIPTION,
        comments=LABJACK_COMMENTS
    )

    # Add pickle data to NWB
    pickle_file_obj = open(os.path.join(SESSION_ROOT, PICKLE_FILENAME), "rb")
    pickle_data = pickle.load(pickle_file_obj)
    SimpleNWB.processing_add_dict(
        nwbfile,
        processed_name=PICKLE_DATA_NAME_PREFIX,
        processed_description=PICKLE_DATA_DESCRIPTION,
        data=pickle_data,
        uneven_columns=True
    )

    # nwb.processing["misc"]["NAMEHERE_missingDataMask"]["missingDataMask"].data[0]["left"]
    # nwb.processing["misc"]["NAMEHERE_eyePositionUncorrected"]["eyePositionUncorrected"].data[:]

    # Add mp4 data to NWB
    # THIS WILL TAKE A LONG ASS TIME, ADD PRINTS?
    for mp4_name, mp4_filename in MP4_FILES.items():
        data, frames = mp4_read_data(os.path.join(SESSION_ROOT, mp4_filename))
        SimpleNWB.mp4_add_as_behavioral(
            nwbfile,
            name=mp4_name,
            numpy_data=data,
            frame_count=frames,
            sampling_rate=MP4_SAMPLING_RATE,
            description=MP4_DESCRIPTION
        )

    now = pendulum.now()
    SimpleNWB.write(nwbfile, "nwb-{}-{}_{}".format(now.month, now.day, now.hour))

    pass


if __name__ == "__main__":
    main()

