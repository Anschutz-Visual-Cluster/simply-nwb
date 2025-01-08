import numpy as np

from simply_nwb.pipeline.util import resample_interp
from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.value_mapping import EnrichmentReference


class PredictSaccadesEnrichment(Enrichment):
    def __init__(self, direction_classifier, temporal_epoch_regressor, temporal_epoch_transformer, nasal_epoch_regressor, nasal_epoch_transformer, use_mlp_input=False):
        """
        Create a new enrichment for predicting saccades, requires the putative saccades enrichment to run

        :param direction_classifier: Classifier to predict saccade direction, expects input data to be (N, features) where N is num samples, and features is _resample_waveform_to_velocity (default 30)
        :param temporal_epoch_regressor: Regressor for 'increasing' temporal waveforms
        :param temporal_epoch_transformer: Transformer for 'increasing' temporal waveforms
        :param nasal_epoch_regressor: Regressor for 'decreasing' temporal waveforms
        :param nasal_epoch_transformer: Transformer for 'decreasing' temporal waveforms
        :param use_mlp_input: Use the input length for the multilayer perceptron, size 80, uses eye position instead of velocity to determine direction
        """
        super().__init__(NWBValueMapping({
            "PutativeSaccades": EnrichmentReference("PutativeSaccades")
        }))
        self._direction_cls = direction_classifier
        self._temporal_epoch_regressor = temporal_epoch_regressor
        self._temporal_epoch_transformer = temporal_epoch_transformer
        self._nasal_epoch_regressor = nasal_epoch_regressor
        self._nasal_epoch_transformer = nasal_epoch_transformer
        self._use_mlp_input = use_mlp_input

    @staticmethod
    def get_name() -> str:
        return "PredictSaccades"

    @staticmethod
    def saved_keys() -> list[str]:
        return [
            "saccades_fps",  # [number]
            "saccades_predicted_nasal_epochs",  # (saccade_num, (start,end))
            "saccades_predicted_temporal_epochs",  # same as above
            "saccades_predicted_nasal_waveforms",  # (saccade_num, waveform_time, (x,y))
            "saccades_predicted_temporal_waveforms",  # same as above
            "saccades_predicted_noise_waveforms",  # Noise waveforms (saccadenum, waveform_time, (x,y))
            "saccades_predicted_temporal_peak_indices",  # Peaks (centers) of each, in indexes (saccadenum,)
            "saccades_predicted_nasal_peak_indices",  # same as above
        ]

    @staticmethod
    def descriptions() -> dict[str, str]:
        return {
            "saccades_fps": "fps of the video, form is [fps]",
            "saccades_predicted_nasal_epochs": "Start and end of the nasal saccades (saccadenum, start/end) in absolute frames",
            "saccades_predicted_temporal_epochs": "Start and end of the temporal saccades (saccadenum, start/end) in absolute frames",
            "saccades_predicted_noise_waveforms": "Noise waveforms (saccadenum, time, x/y)",
            "saccades_predicted_nasal_waveforms": "Waveforms for only nasal saccades (saccadenum, time, x/y) in pixels",
            "saccades_predicted_temporal_waveforms": "Waveforms for only temporal saccades (saccadenum, time, x/y) in pixels",
            "saccades_predicted_temporal_peak_indices": "Indexes of peaks for temporal labeled waveforms (saccadenum,) absolute frames",  # same as above
            "saccades_predicted_nasal_peak_indices": "Indexes of peaks for nasal labeled waveforms (saccadenum,) absolute frames",  # same as above
        }

    @staticmethod
    def preformat_waveforms(waveforms: np.ndarray, num_features=30, single_dim=False):
        # Helper func to format the waveforms as velocities, sampled, with no NaNs
        # Expects waveforms to be a (N, t, 2) arr where N is the number of samples, t is the time length, and 2 is x,y
        if single_dim:
            waveforms = waveforms[:, :, None]
            waveforms = np.pad(waveforms, ((0, 0), (0, 0), (0, 1)), constant_values=-1)
            # add new axis and pad with -1 vals to mimic the extra missing axis

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
                resampd = PredictSaccadesEnrichment._resample_waveform_to_velocity(wav_x[idx], num_features)
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
        return np.array(x_velocities), np.array(idxs)

    @staticmethod
    def _resample_waveform_to_velocity(single_waveform: np.ndarray, sampling_size=30):
        # Resample waveform and take the velocity/forward difference/derivative
        _, resampd = resample_interp(np.diff(single_waveform), sampling_size)
        return resampd

    def _run(self, pynwb_obj):
        pred_labels, pred_waveforms, pred_sacc_indices = self._predict_saccade_direction(pynwb_obj)
        self._predict_saccade_epochs(pynwb_obj, pred_labels, pred_waveforms, pred_sacc_indices)

    def _predict_saccade_direction(self, pynwb_obj):
        self.logger.info("Predicting saccade waveform labels (direction)..")
        # Pull waveforms and indices from putative saccades
        waveforms = self._get_req_val("PutativeSaccades.saccades_putative_waveforms", pynwb_obj)
        indices = self._get_req_val("PutativeSaccades.saccades_putative_peak_indices", pynwb_obj)
        # Format the waveforms by getting velocities, also get the indexes of the waveforms that are used
        x_velocities, idxs = self.preformat_waveforms(waveforms)

        # Reindex to only include the waveforms that were formatted from above
        waveforms = waveforms[idxs]
        indices = indices[idxs]

        # Predict -1, 0, or 1
        if self._use_mlp_input:
            pred_labels = self._direction_cls.predict(waveforms[:, :, 0])
        else:
            pred_labels = self._direction_cls.predict(x_velocities)  # Used for LDA

        return pred_labels, waveforms, indices

    def _predict_saccade_epochs(self, pynwb_obj, pred_labels, pred_waveforms, pred_sacc_indices):
        self.logger.info("Predicting saccade epochs..")

        # X = x value, velocity and resampled
        # y = (start offset time, end offset time)  <saccadestart>----y[0]--<saccadepeak/center>---y[1]--<saccadeend>
        # z = (direction of saccade, -1 or 1)
        # use the nasal and temporal regressor and transformers to take the current data and save predicted values
        # will need to resample X, possibly divide by fps on y (z is direction)

        sacc_fps = self._get_req_val("PutativeSaccades.saccades_fps", pynwb_obj)[0]
        self._save_val("saccades_fps", [sacc_fps], pynwb_obj)

        noise_idxs = np.where(pred_labels == 0)
        self._save_val("saccades_predicted_noise_waveforms", pred_waveforms[noise_idxs], pynwb_obj)

        for regressor, transformer, saccade_direction, name in [
                (self._temporal_epoch_regressor, self._temporal_epoch_transformer, -1, "temporal"),
                (self._nasal_epoch_regressor, self._nasal_epoch_transformer, 1, "nasal")]:

            dir_idxs = np.where(pred_labels == saccade_direction)[0]
            resamp, nonnan_idxs = self.preformat_waveforms(pred_waveforms[dir_idxs])
            assert len(nonnan_idxs) == dir_idxs.shape[0], "Epoch waveform formatting changed array size, additional NaN values were detected between direction prediction and epoch prediction, will not line up!! (this shouldn't happen!)"

            reg_pred = regressor.predict(resamp)
            pred = transformer.inverse_transform(reg_pred)
            # Broadcast the indices so we can add them to the predicted relative offsets easily
            reshaped_sacc_indices = np.broadcast_to(pred_sacc_indices[dir_idxs][nonnan_idxs].reshape(-1, 1), (*pred_sacc_indices[dir_idxs][nonnan_idxs].shape, 2))
            up_pred = pred * sacc_fps + reshaped_sacc_indices  # Convert from seconds to frames using the fps

            self._save_val(f"saccades_predicted_{name}_waveforms", pred_waveforms[dir_idxs][nonnan_idxs], pynwb_obj)
            self._save_val(f"saccades_predicted_{name}_epochs", up_pred, pynwb_obj)
            self._save_val(f"saccades_predicted_{name}_peak_indices", pred_sacc_indices[dir_idxs], pynwb_obj)

