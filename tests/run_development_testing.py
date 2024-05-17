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
    sess = NWBSession("putative.nwb")
    waveforms = sess.pull("PutativeSaccades.saccades_putative_waveforms")

    wav_x = waveforms[:, :, 0]
    wav_y = waveforms[:, :, 1]
    num_features = 30
    idxs = []
    for idx in range(wav_x.shape[0]):
        x_non_nan = np.all(np.invert(np.isnan(wav_x[idx])))
        y_non_nan = np.all(np.invert(np.isnan(wav_y[idx])))
        if x_non_nan and y_non_nan:  # Both entries are non-nan
            idxs.append(idx)

    wav_x = wav_x[idxs]
    wav_y = wav_y[idxs]

    # Train the classifier yourself and pass it into the enrichment
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.neural_network import MLPClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    #
    # grid = {
    #     'hidden_layer_sizes': [(int(n),) for n in np.arange(2, num_features, 1)],
    #     'max_iter': [
    #         1000000,
    #     ],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 0.05],
    #     'learning_rate': ['constant', 'adaptive'],
    # }
    # net = MLPClassifier()
    # search = GridSearchCV(net, grid)
    # search.fit(wav_x, wav_y)
    # clf = search.best_estimator_

    lda = LinearDiscriminantAnalysis()
    lda.fit(wav_x.ravel().reshape(1,-1), wav_y.ravel())
    p = PredictSaccadesEnrichment(lda)

    sess.enrich(p)

    tw = 2
    pass


def main():
    # putative()
    predicting()
    tw = 2


if __name__ == "__main__":
    main()

