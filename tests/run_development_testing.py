import random

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
    prefix = "C:\\Users\\denma\\Documents\\GitHub\\simply-nwb\\data\\adsfasdf\\20240410\\unitME\\session001\\"
    # prefix = "C:\\Users\\Matrix\\Downloads\\adsfasdf\\20240410\\unitME\\session001"
    # prefix = "C:\\Users\\spenc\\Documents\\GitHub\\simply-nwb\\data\\adsfasdf\\20240410\\unitME\\session001"

    dlc_filepath = f"{prefix}\\20240410_unitME_session001_rightCam-0000DLC_resnet50_GazerMay24shuffle1_1030000.csv"
    timestamp_filepath = f"{prefix}\\20240410_unitME_session001_rightCam_timestamps.txt"

    sess = NWBSession("../data/test.nwb")
    sess.enrich(ExampleEnrichment())

    # enrichment = PutativeSaccadesEnrichment.from_raw(nwbfile, dlc_filepath, timestamp_filepath)
    enrichment = PutativeSaccadesEnrichment()
    sess.enrich(enrichment)

    sess.pull("PutativeSaccades.pose_corrected")
    print("Available enrichments: " + str(sess.available_enrichments()))
    print("Available keys for PutativeSaccades: " + str(sess.available_keys("PutativeSaccades")))

    sess.save("putative.nwb")


def _get_pretrained_direction_model(sess):
    wv = sess.pull("PutativeSaccades.saccades_putative_waveforms")

    x_velocities, idxs = PredictSaccadesEnrichment._preformat_waveforms(wv)
    x_velocities = np.array(x_velocities)
    y_vals = [random.randint(0, 2) - 1 for _ in range(len(x_velocities))]  # Randomly generate -1, 0, 1 vals

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_velocities, y_vals)

    return lda


def _get_pretrained_epoch_models(sess):
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    num_features = 30
    # x vals of the waveforms for training on, handpicked and corresponding to the epoch_labels below
    wv = sess.pull("PutativeSaccades.saccades_putative_waveforms")

    training_x_waveforms, idxs = PredictSaccadesEnrichment._preformat_waveforms(wv)
    samps = np.random.normal(size=len(idxs)*3)
    epoch_labels = [[np.abs(samps[s*2-1])*-1, np.abs(samps[s*2])] for s in range(len(samps[2:len(idxs)*2+1:2]))]  # List of pairs of offsets from peak for each waveform to train on TODO divide by fps?

    # Transformer
    transformer = StandardScaler().fit(epoch_labels)
    standardized_epoch_labels = transformer.transform(epoch_labels)

    # Regressor
    # hidden_layer_sizes = [(int(n),) for n in np.arange(2, num_features, 1)]
    hidden_layer_sizes = [(4,)]
    grid = {
        'estimator__hidden_layer_sizes': hidden_layer_sizes,
        'estimator__max_iter': [
            1000,
        ],
        'estimator__activation': ['tanh'],  # , 'relu'],
        'estimator__solver': ['sgd'],  # , 'adam'],
        'estimator__alpha': [0.0001],  # , 0.05],
        'estimator__learning_rate': ['constant']  # , 'adaptive'],
    }

    reg = MultiOutputRegressor(MLPRegressor(verbose=True))
    search = GridSearchCV(reg, grid)

    search.fit(training_x_waveforms, standardized_epoch_labels)
    regressor = search.best_estimator_

    return regressor, transformer


def predicting():
    sess = NWBSession("putative.nwb")

    direct = _get_pretrained_direction_model(sess)
    temp_reg, temp_tran = _get_pretrained_epoch_models(sess)
    nasal_reg, nasal_tran = _get_pretrained_epoch_models(sess)

    p = PredictSaccadesEnrichment(
        direction_classifier=direct,
        temporal_epoch_regressor=temp_reg,
        temporal_epoch_transformer=temp_tran,
        nasal_epoch_regressor=nasal_reg,
        nasal_epoch_transformer=nasal_tran
    )
    
    sess.enrich(p)
    tw = 2


def main():
    # putative()
    predicting()
    tw = 2


if __name__ == "__main__":
    main()

