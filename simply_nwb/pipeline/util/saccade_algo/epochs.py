import pickle

# NOT CURRENTLY USED OR FULLY IMPLEMENTED!!

# Only have an epoch regressor and transformer for backwards compatibility with older code
class EpochRegressor(object):
    def predict(self, resampled_waveform):
        return resampled_waveform  # Pass along data for the transformer


class EpochTransformer(object):
    def __init__(self):
        # TODO
        transformer_fn = "predict_gui_nasal_epoch_transformer.pickle"
        regressor_fn = "predict_gui_nasal_epoch_regressor.pickle"
        self.reg = self._load_pickle(regressor_fn)
        self.transf = self._load_pickle(transformer_fn)

    def _load_pickle(self, filename): # TODO Remove me
        with open(filename, "rb") as fp:
            return pickle.load(fp)

    def inverse_transform(self, resampled_waveform):
        # TODO create algorithm for this, for now using pretrained
        return self.transf.inverse_transform(self.reg.predict(resampled_waveform))
