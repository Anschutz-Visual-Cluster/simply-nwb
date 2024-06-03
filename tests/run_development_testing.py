import glob
import os
import random

import h5py
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


def _get_pretrained_direction_model(sess, wv=None, y_vals=None):
    if wv is None:
        wv = sess.pull("PutativeSaccades.saccades_putative_waveforms")

    tmp_wv = np.broadcast_to(wv[:, :, None], shape=(*wv.shape, 2))
    x_velocities, idxs = PredictSaccadesEnrichment._preformat_waveforms(tmp_wv)

    x_velocities = np.array(x_velocities)
    if y_vals is None:
        y_vals = [random.randint(0, 2) - 1 for _ in range(len(x_velocities))]  # Randomly generate -1, 0, 1 vals

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_velocities, y_vals)

    return lda


def _get_pretrained_epoch_models(sess, wv=None, epoch_labels=None):
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    num_features = 30
    # x vals of the waveforms for training on, handpicked and corresponding to the epoch_labels below
    if wv is None:
        wv = sess.pull("PutativeSaccades.saccades_putative_waveforms")

    tmp_wv = np.broadcast_to(wv[:, :, None], shape=(*wv.shape, 2))  # pretend this epoch waveform is a direction to use the same preprocessing func
    training_x_waveforms, idxs = PredictSaccadesEnrichment._preformat_waveforms(tmp_wv, num_features=num_features)
    if epoch_labels is None:
        samps = np.random.normal(size=len(idxs) * 3)
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


def _get_direction_training_data():
    source_folder = "E:\\AnnaTrainingData"  # Scan for '*output.hdf'
    # want prediction/saccades/direction/<X/y>

    train_x = []
    train_y = []
    for file in os.listdir(source_folder):
        if file.endswith("output.hdf"):
            data = h5py.File(os.path.join(source_folder, file))
            train_x.append(np.array(data["prediction"]["saccades"]["direction"]["X"]))  # x (needs to be downsampled)
            train_y.append(np.array(data["prediction"]["saccades"]["direction"]["y"]).reshape(-1))  # y

    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)
    return train_x, train_y


def _get_epoch_training_data(direction):
    source_folder = "E:\\AnnaTrainingData"  # Scan for '*output.hdf'
    recording_fps = 200  # TODO set me
    # want prediction/saccades/epochs/<X/y/z>

    train_x = []
    train_y = []
    train_z = []

    # X = x value, velocity and resampled
    # y = (start offset time, end offset time)  <saccadestart>----y[0]--<saccadepeak/center>---y[1]--<saccadeend>
    # z = (direction of saccade, -1 or 1)

    for file in os.listdir(source_folder):
        if file.endswith("output.hdf"):
            data = h5py.File(os.path.join(source_folder, file))
            train_x.append(np.array(data["prediction"]["saccades"]["epochs"]["X"]))  # x (needs to be downsampled)
            train_y.append(np.array(data["prediction"]["saccades"]["epochs"]["y"]))  # y
            train_z.append(np.array(data["prediction"]["saccades"]["epochs"]["z"]).reshape(-1))  # z

    train_x = np.vstack(train_x)
    train_y = np.vstack(train_y)
    train_z = np.hstack(train_z)

    direction_idxs = np.where(train_z == direction)[0]
    train_x = train_x[direction_idxs]
    train_y = train_y[direction_idxs] / recording_fps  # Divide by recording fps to get epochs in units of frames
    train_z = train_z[direction_idxs]

    train_z = train_z.reshape(-1, 1)  # Reshape so each 'label' is it's own array [1,1,1,..] -> [[1],[1],..]

    return train_x, train_y, train_z


def _find_joshdata(root_directory):
    # Find sessions and return data in a list
    # like [["session_name", "path/dlc.csv", "path/timestamps.txt", "path/output.hdf"], ..]
    found_sessions = []
    for folder in os.listdir(root_directory):
        subfolder = os.path.join(root_directory, folder)
        if os.path.isdir(subfolder):
            dlc = glob.glob(os.path.join(subfolder, "*rightCam*csv"), recursive=True)[0]
            timestamps = glob.glob(os.path.join(subfolder, "*timestamps*txt"), recursive=True)[0]
            output = glob.glob(os.path.join(subfolder, "*output*hdf"), recursive=True)[0]
            name = os.path.basename(subfolder)

            found_sessions.append([name, dlc, timestamps, output])

    return found_sessions


def dictify(data):
    import h5py
    if isinstance(data, h5py.Dataset):
        try:
            return list(data[:])
        except Exception as e:
            print(f"Errorrrrrr {str(e)}")
            return "BROKEN!!!!!!!!!!!!!!!!!!!!!!"
    else:
        dd = dict(data)
        d = {}
        for k, v in dd.items():
            d[k] = dictify(v)
        return d


def validation():
    # 1 is nasal, -1 is temporal
    nasal_x, nasal_y, nasal_z = _get_epoch_training_data(1)
    temporal_x, temporal_y, temporal_z = _get_epoch_training_data(-1)
    dx, dy = _get_direction_training_data()

    # list of folderpaths with a dlc.csv, timestamps.txt and output.hdf
    sessions_to_validate = _find_joshdata("E:\\AnnaTrainingData\\joshdata")

    for name, dlc_csv, timestamps, output_hdf in sessions_to_validate:
        temp_nwbfile_name = "test_nwb.nwb"
        h5data = h5py.File(output_hdf)

        SimpleNWB.write(SimpleNWB.test_nwb(), temp_nwbfile_name)
        sess = NWBSession(temp_nwbfile_name)

        # Putative
        putative_enrichment = PutativeSaccadesEnrichment.from_raw(sess.nwb, dlc_csv, timestamps)
        sess.enrich(putative_enrichment)

        # Predictive
        nasal_reg, nasal_tran = _get_pretrained_epoch_models(sess, wv=nasal_x, epoch_labels=nasal_y)
        temp_reg, temp_tran = _get_pretrained_epoch_models(sess, wv=temporal_x, epoch_labels=temporal_y)
        direct = _get_pretrained_direction_model(sess, wv=dx, y_vals=dy)

        predictive_enrichment = PredictSaccadesEnrichment(
            direction_classifier=direct,
            temporal_epoch_regressor=temp_reg,
            temporal_epoch_transformer=temp_tran,
            nasal_epoch_regressor=nasal_reg,
            nasal_epoch_transformer=nasal_tran
        )

        sess.enrich(predictive_enrichment)
        sess.save(f"{name}_enriched.nwb")
        print("Dictifying hdf and sess")
        orig = dictify(h5data)
        modif = sess.to_dict()

        tw = 2
        break

    tw = 2

    pass


def main():
    # putative()
    # predicting()
    validation()
    tw = 2


if __name__ == "__main__":
    main()

