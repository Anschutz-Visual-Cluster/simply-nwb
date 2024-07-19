from simply_nwb import SimpleNWB
from simply_nwb.pipeline import NWBSession
from simply_nwb.pipeline.enrichments.saccades import PutativeSaccadesEnrichment
from simply_nwb.pipeline.enrichments.saccades.predict_gui import PredictedSaccadeGUIEnrichment


def gui_testing():
    # putatsess = NWBSession("../data/test.nwb")
    # putatsess.enrich(PutativeSaccadesEnrichment())
    # putatsess.save("putative.nwb")

    sess = NWBSession("putative.nwb")

    enrich = PredictedSaccadeGUIEnrichment(200, ["putative1.nwb", "putative2.nwb", "putative3.nwb"])
    sess.enrich(enrich)
    print("Saving to NWB")
    sess.save("gui_output.nwb")
    tw = 2


def main():
    gui_testing()


if __name__ == "__main__":
    main()
