import glob
import os
import sys
import traceback
from io import StringIO

from build.lib.simply_nwb import SimpleNWB
from build.lib.simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.enrichments.saccades import PutativeSaccadesEnrichment, DriftingGratingLabjackEnrichment
from simply_nwb.pipeline.enrichments.saccades.predict_gui import PredictedSaccadeGUIEnrichment


def main():
    prefix = "/media/felsenlab/Bard Brive/AnnaData"
    os.chdir("../")  # So we can see models, required
    for folder in os.listdir(prefix):
        if os.path.exists(os.path.join(prefix, "nwbs", f"{folder}.nwb")):
            print(f"Session '{folder}' already processed, skipping..")
            continue
        try:
            print(f"Processing '{folder}'..")
            ts = glob.glob(os.path.join(prefix, folder, "*timestamps.txt"))
            assert len(ts) == 1, "Multiple timestamps found!"
            ts = ts[0]

            dlc = glob.glob(os.path.join(prefix, folder, "*.csv"))
            assert len(dlc) == 1, "Multiple DLC files found!"
            dlc = dlc[0]

            gratings = glob.glob(os.path.join(prefix, folder, "driftingGratingMetadata*.txt"))
            assert len(gratings) > 0, "No gratings found!"

            dats = glob.glob(os.path.join(prefix, folder, "labjack", "*.dat"))
            assert len(dats) > 0, "No dats found!"

            nwb = SimpleNWB.test_nwb()
            sess = NWBSession(nwb)

            print("Enriching with putative saccade data..")
            putat = PutativeSaccadesEnrichment.from_raw(nwb, dlc, ts, units=["idx", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood"], x_center="center_x", y_center="center_y", likelihood="center_likelihood")
            sess.enrich(putat)

            print("Enriching with predictive saccade data..")
            pred = PredictedSaccadeGUIEnrichment(200, [], 20, putative_kwargs={"x_center": "center_x", "y_center": "center_y", "likelihood": "center_likelihood"})
            sess.enrich(pred)

            print("Enriching with drifting grating labjack data..")
            grat = DriftingGratingLabjackEnrichment(gratings, dats)
            sess.enrich(grat)

            print("Saving session")
            sess.save(os.path.join(prefix, "nwbs", f"{folder}.nwb"))
            tw = 2
        except Exception as e:
            buffer = StringIO()
            exc_type, exc_obj, exc_trace = sys.exc_info()

            buffer.write(f"Exception: {str(exc_type)}\n")
            traceback.print_tb(exc_trace, file=buffer)
            buffer.seek(0)
            print(f"Error processing session '{folder}' writing error to file. Exception: '{exc_type}'")
            with open(os.path.join(prefix, "nwbs", f"{folder}-error.txt"), "w") as f:
                f.write(buffer.read())

    pass

if __name__ == "__main__":
    main()
