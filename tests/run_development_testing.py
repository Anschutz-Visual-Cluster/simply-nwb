import numpy as np
import pendulum
from pynwb.file import Subject

from simply_nwb import SimpleNWB
from simply_nwb.pipeline.enrichments.saccades import PutativeSaccadesEnrichment, PredictSaccadesEnrichment
from simply_nwb.pipeline import Enrichment, NWBValueMapping
# This file is used for testing things during development

from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.enrichments.example import ExampleEnrichment
from simply_nwb.transforms import tif_read_image


def putative():
    class CustEnrich(Enrichment):
        def __init__(self):
            super().__init__(NWBValueMapping({}))

        @staticmethod
        def get_name():
            return "MyCustom"

        def _run(self, pynwb_obj):
            pass

    # prefix = "C:\\Users\\denma\\Documents\\GitHub\\simply-nwb\\data\\adsfasdf\\20240410\\unitME\\session001\\"
    prefix = "C:\\Users\\Matrix\\Downloads\\adsfasdf\\20240410\\unitME\\session001"
    # prefix = "C:\\Users\\spenc\\Documents\\GitHub\\simply-nwb\\data\\adsfasdf\\20240410\\unitME\\session001"

    dlc_filepath = f"{prefix}\\20240410_unitME_session001_rightCam-0000DLC_resnet50_GazerMay24shuffle1_1030000.csv"
    timestamp_filepath = f"{prefix}\\20240410_unitME_session001_rightCam_timestamps.txt"

    sess = NWBSession("../data/test.nwb", custom_enrichments=[CustEnrich])
    sess.enrich(ExampleEnrichment())

    # enrichment = PutativeSaccadesEnrichment.from_raw(nwbfile, dlc_filepath, timestamp_filepath)
    enrichment = PutativeSaccadesEnrichment()
    sess.enrich(enrichment)

    sess.pull("PutativeSaccades.pose_corrected")
    print("Available enrichments: " + str(sess.available_enrichments()))
    print("Available keys for PutativeSaccades: " + str(sess.available_keys("PutativeSaccades")))

    sess.save("putative.nwb")


def predicting():
    p = PredictSaccadesEnrichment()
    sess = NWBSession("putative.nwb")
    sess.enrich(p)

    tw = 2
    pass


def main():
    # putative()
    predicting()
    tw = 2


if __name__ == "__main__":
    main()

