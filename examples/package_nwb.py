from simply_nwb.transforms import labjack_load_file, mp4_read_data
from simply_nwb import SimpleNWB
from simply_nwb.transforms import plaintext_metadata_read
from pynwb.file import Subject
import pendulum
import pickle
import os

# Simply-NWB Package Documentation
# https://simply-nwb.readthedocs.io/en/latest/index.html


# Constants at the top of the file for things you might want to change for
# different NWBs for flexibility
INSTITUTION: str = "InstitutionHere"

EXPERIMENTERS: [str] = [
    "Lastname, Firstname"
]
LAB: str = "LabNameHere"

EXPERIMENT_DESCRIPTION: str = "Long description of experiment goes here"
EXPERIMENT_KEYWORDS: [str] = ["mouse"]
EXPERIMENT_RELATED_PUBLICATIONS = None  # optional

SESSION_IDENTIFIER: str = "session1"
SESSION_DESCRIPTION: str = "sess desc"
SUBJECT: Subject = Subject(
    subject_id="mouse1",
    age="P90D", # ISO-8601 90 days
    strain="TypeOfMouse",  # if unknown, put Wild Strain
    description="Mouse desc goes here",
    sex="M"
)

SESSION_ROOT: str = "../data/mouse1"
METADATA_FILENAME: str = "metadata.txt"

LABJACK_FILENAME: str = "labjack/lick1OKRvartest40-4-14-23/data_0.dat"
LABJACK_NAME: str = "Labjack Data"
LABJACK_SAMPLING_RATE: float = 20.0  # in Hz
LABJACK_DESCRIPTION: str = "labjack description here"
LABJACK_COMMENTS: str = "labjack comments here"

MP4_FILES: {str: str} = {
    "RightEye": "movies/righteye.mp4",
    "LeftEye": "movies/lefteye.mp4"
}
MP4_DESCRIPTION: str = "description of mp4 files here"
MP4_SAMPLING_RATE: float = 120.0

PICKLE_FILENAME: str = "pickledata.pkl"
PICKLE_DATA_NAME_PREFIX: str = "data"
PICKLE_DATA_DESCRIPTION: str = "desc of data here"


def main():
    # Parse out the metadata.txt file
    metadata = plaintext_metadata_read(os.path.join(SESSION_ROOT, METADATA_FILENAME))
    start_date = pendulum.parse(metadata["Date"], tz="local")

    # Create the NWB object
    nwbfile = SimpleNWB.create_nwb(
        session_description=SESSION_DESCRIPTION,
        session_start_time=start_date,
        experimenter=EXPERIMENTERS,
        lab=LAB,
        experiment_description=EXPERIMENT_DESCRIPTION,
        session_id=SESSION_IDENTIFIER,
        institution=INSTITUTION,
        keywords=EXPERIMENT_KEYWORDS,
        related_publications=EXPERIMENT_RELATED_PUBLICATIONS
    )

    # Add labjack data to NWB
    labjack_filename_absolute = os.path.join(SESSION_ROOT, LABJACK_FILENAME)
    labjack_data = labjack_load_file(labjack_filename_absolute)

    SimpleNWB.labjack_file_as_behavioral_data(
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
        data_dict=pickle_data,
        uneven_columns=True
    )

    # Access the data like this: (where missingDataMask is part of the pickle file)
    # nwb.processing["misc"]["NAMEHERE_missingDataMask"]["missingDataMask"].data[0]["left"]
    # nwb.processing["misc"]["NAMEHERE_eyePositionUncorrected"]["eyePositionUncorrected"].data[:]

    # Add mp4 data to NWB
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


if __name__ == "__main__":
    main()
