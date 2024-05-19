import numpy as np

from simply_nwb.pipeline.util import resample_interp
from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.value_mapping import EnrichmentReference


class PredictSaccadesEnrichment(Enrichment):
    def __init__(self, direction_classifier, temporal_epoch_regressor, temporal_epoch_transformer, nasal_epoch_regressor, nasal_epoch_transformer):
        """
        Create a new enrichment for predicting saccades, requires the putative saccades enrichment to run

        :param direction_classifier: Classifier to predict saccade direction, expects input data to be (N, features) where N is num samples, and features is _resample_waveform_to_velocity (default 30)
        :param temporal_epoch_regressor: Regressor for 'increasing' temporal waveforms
        :param temporal_epoch_transformer: Transformer for 'increasing' temporal waveforms
        :param nasal_epoch_regressor: Regressor for 'decreasing' temporal waveforms
        :param nasal_epoch_transformer: Transformer for 'decreasing' temporal waveforms
        """
        super().__init__(NWBValueMapping({
            "PutativeSaccades": EnrichmentReference("PutativeSaccades")
        }))
        self._direction_cls = direction_classifier
        self._temporal_epoch_regressor = temporal_epoch_regressor
        self._temporal_epoch_transformer = temporal_epoch_transformer
        self._nasal_epoch_regressor = nasal_epoch_regressor
        self._nasal_epoch_transformer = nasal_epoch_transformer

    @staticmethod
    def get_name() -> str:
        return "PredictSaccades"

    @staticmethod
    def saved_keys() -> list[str]:
        return [
            "saccades_predicted_indices",
            "saccades_predicted_waveforms",
            "saccades_predicted_labels"
        ]

    @staticmethod
    def _preformat_waveforms(waveforms: np.ndarray):
        # Helper func to format the waveforms as velocities, sampled, with no NaNs
        # Expects waveforms to be a (N, t, 2) arr where N is the number of samples, t is the time length, and 2 is x,y
        num_features = 30  # Number of features to use when resampling the data
        wav_x = waveforms[:, :, 0]
        wav_y = waveforms[:, :, 1]
        x_velocities = []
        idxs = []

        for idx in range(wav_x.shape[0]):
            is_x_non_nan = np.all(np.invert(np.isnan(wav_x[idx])))
            is_y_non_nan = np.all(np.invert(np.isnan(wav_y[idx])))
            if is_x_non_nan and is_y_non_nan:  # Both entries are non-nan
                # Forward difference (discrete derivative/velocity)
                # Resample to match the number of 'features'
                resampd = PredictSaccadesEnrichment._resample_waveform_to_velocity(wav_x[idx])
                x_velocities.append(resampd)
                idxs.append(idx)

        """
        y == 0 -> waveform is noise

        y == -1 -> waveform is pos, increasing, called 'temporal' (see ascii art below)
                                       ---- 
                                      |
                                  ----
        y == 1 -> waveform is neg, decreasing, called 'nasal'
                             ----  
                                 |
                                  ----
        """
        return x_velocities, idxs

    @staticmethod
    def _resample_waveform_to_velocity(single_waveform: np.ndarray, sampling_size=30):
        # Resample waveform and take the velocity/forward difference/derivative
        _, resampd = resample_interp(np.diff(single_waveform), sampling_size)
        return resampd

    def _run(self, pynwb_obj):
        self._predict_saccade_direction(pynwb_obj)
        self._predict_saccade_epochs(pynwb_obj)

    def _predict_saccade_direction(self, pynwb_obj):
        self.logger.info("Predicting saccade waveform labels (direction)..")
        # Pull waveforms and indices from putative saccades
        waveforms = self._get_req_val("PutativeSaccades.saccades_putative_waveforms", pynwb_obj)
        indices = self._get_req_val("PutativeSaccades.saccades_putative_indices", pynwb_obj)
        # Format the waveforms by getting velocities, also get the indexes of the waveforms that are used
        x_velocities, idxs = self._preformat_waveforms(waveforms)

        # Reindex to only include the waveforms that were formatted from above
        waveforms = waveforms[idxs]
        indices = indices[idxs]

        # Resample the x values before running through prediction model
        resamps = []
        for idx in range(len(waveforms)):
            resamps.append(self._resample_waveform_to_velocity(waveforms[idx, :, 0]))  # Resample the x's
        resamps = np.array(resamps)

        # Predict -1, 0, or 1
        preds = self._direction_cls.predict(resamps)

        self._save_val("saccades_predicted_indices", indices, pynwb_obj)
        self._save_val("saccades_predicted_waveforms", waveforms, pynwb_obj)
        self._save_val("saccades_predicted_labels", preds, pynwb_obj)

    def _predict_saccade_epochs(self, pynwb_obj):
        self.logger.info("Predicting saccade epochs..")

        # X = x value, velocity and resampled
        # y = (start offset time, end offset time)  <saccadestart>----y[0]--<saccadepeak/center>---y[1]--<saccadeend>
        # z = (direction of saccade, -1 or 1)
        # use the nasal and temporal regressor and transformers to take the current data and save predicted values
        # will need to resample X, possibly divide by fps on y (z is direction)

        pass

