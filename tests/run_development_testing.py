import numpy as np
import pendulum
from pynwb.file import Subject

from simply_nwb import SimpleNWB
from simply_nwb.pipeline.enrichments.saccades import PutativeSaccadeEnrichment
from simply_nwb.pipeline import Enrichment
# This file is used for testing things during development

from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.enrichments.example import ExampleEnrichment
from simply_nwb.transforms import tif_read_image


def main():
    class CustEnrich(Enrichment):
        @staticmethod
        def get_name():
            return "custom"

    nwbfile = SimpleNWB.test_nwb()

    sess = NWBSession("a", custom_enrichments=[CustEnrich])
    sess.nwb = nwbfile  # TODO debug remove me

    sess.enrich(ExampleEnrichment())
    prefix = "C:\\Users\\Matrix\\Downloads\\adsfasdf\\20240410\\unitME\\session001"
    dlc_filepath = f"{prefix}\\20240410_unitME_session001_rightCam-0000DLC_resnet50_GazerMay24shuffle1_1030000.csv"
    timestamp_filepath = f"{prefix}\\20240410_unitME_session001_rightCam_timestamps.txt"

    # filepath = "C:\\Users\\denma\\Documents\\GitHub\\simply-nwb\\data\\adsfasdf\\20240410\\unitME\\session001\\20240410_unitME_session001_rightCam-0000DLC_resnet50_GazerMay24shuffle1_1030000.csv"

    enrichment = PutativeSaccadeEnrichment.from_raw(nwbfile, dlc_filepath, timestamp_filepath)

    # TODO also test when already exists in NWB
    # enrichment = PutativeSaccadeEnrichment()
    sess.enrich(enrichment)


    tw = 2


if __name__ == "__main__":
    main()

