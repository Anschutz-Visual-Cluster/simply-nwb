import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
from pynwb import NWBFile, TimeSeries
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from simply_nwb import SimpleNWB
from simply_nwb.pipeline import Enrichment
from simply_nwb.pipeline.util import interpolate_flat_arr, smooth_flat_arr
from simply_nwb.pipeline.value_mapping import NWBValueMapping
from simply_nwb.transforms import csv_load_dataframe_str


"""
Code adapted from https://github.com/jbhunt/myphdlib/blob/5c5fe627507046e888eabcd963c47906dfbea7b1/myphdlib/pipeline/saccades.py#L614
"""


class PutativeSaccadesEnrichment(Enrichment):
    def __init__(self, stim_name="RightCamStim", timestamp_name="rightCamTimestamps", likelihood_threshold=0.99, fps=200, x_center="pupilCenter_x", y_center="pupilCenter_y", likelihood="pupilCenter_likelihood"):
        """
        Create a new PutativeSaccadesEnrichment

        :param stim_name: Name of the stimulus, defaults to RightCamStim
        :param likelihood_threshold: threshold to use for the likelihood for eye positions
        :param fps: frames per second of the video used, defaults to 200
        """

        # Give the superclass a mapping of required values for this enrichment to run
        super().__init__(NWBValueMapping({
            "x": [lambda x: x.processing, stim_name, x_center, lambda y: y.data[:]],
            "y": [lambda x: x.processing, stim_name, y_center, lambda y: y.data[:]],
            "likelihood": [lambda x: x.processing, stim_name, likelihood, lambda y: y.data[:]],
            "timestamps": [lambda x: x.stimulus, timestamp_name, lambda y: y.data[:]]
        }))

        self._stim_name = stim_name
        self.likelihood_threshold = likelihood_threshold
        self.fps = fps

    @staticmethod
    def from_raw(
            nwbfile: NWBFile,
            dlc_filename: str,
            timestamps_filename: str,
            fps: int = 200,
            stim_name: str = "RightCamStim",
            timestamp_name: str = "rightCamTimestamps",
            units: list[str] = None,
            sampling_rate: float = 200.0,
            comments: str = None,
            description: str = None
    ) -> 'PutativeSaccadesEnrichment':
        """
        Create a PutativeSaccadeEnrichment from raw files rather than automagically from an NWB file with existing data

        :param nwbfile: NWBFile object to add the raw data to as this is enriched
        :param dlc_filename: filepath to the .csv file formatted in DLC format
        :param timestamps_filename: filepath to the \*_timestamps.txt file
        :param fps: frames per second of the video recording, defaults to 200
        :param stim_name: Name of the stimulus to label as it's being inserted
        :param timestamp_name: Name of the container in the stimulus section where the timestamps are located
        :param units: List of units of the columns of DLC if not set has a default
        :param sampling_rate: Sampling rate of the video recording that was run through DLC
        :param comments: Comments
        :param units: units for the columns of the dlc file, units are the index, then pixels, then percent likelihood by default
        :param description: Description, will be autogenerated if not supplied
        :return: Enrichment object
        """

        enr = PutativeSaccadesEnrichment(stim_name=stim_name, fps=fps)

        # Add DLC
        SimpleNWB.eyetracking_add_to_processing(
            nwbfile,
            dlc_filename,
            module_name=stim_name,
            sampling_rate=sampling_rate,
            units=units,
            comments=comments,
            description=description
        )
        # Add timestamps.txt
        csv_fp = open(timestamps_filename, "r")
        csv_data = csv_load_dataframe_str("Timestamps\n" + csv_fp.read())
        csv_fp.close()

        nwbfile.add_stimulus(TimeSeries(
            name=timestamp_name,
            data=list(csv_data["Timestamps"]),
            rate=1.0,
            unit="s"
        ))

        return enr

    @staticmethod
    def get_name() -> str:
        return "PutativeSaccades"

    @staticmethod
    def saved_keys() -> list[str]:
        return [
            "pose_corrected",
            "pose_interpolated",
            "pose_decomposed",
            "pose_missing",
            "pose_reoriented",
            "pose_filtered",
            "saccades_putative_indices",
            "saccades_putative_waveforms",
            "saccades_fps",
        ]

    def _run(self, pynwb_obj):
        """
        Enrich the nwb

        :param pynwb_obj: NWB object to enrich
        """

        # Extract eye position
        self.logger.info("Extracting eye position..")
        x = self._get_req_val("x", pynwb_obj)
        y = self._get_req_val("y", pynwb_obj)
        likelihood = self._get_req_val("likelihood", pynwb_obj)
        x[likelihood < self.likelihood_threshold] = np.nan  # Set eye pos values to nan if they dont meet the threshold
        y[likelihood < self.likelihood_threshold] = np.nan

        corrected = self._correct_eye_position(x, y, pynwb_obj)  # pose/corrected
        interpolated = self._interpolate_eye_position(corrected)  # pose/interpolated
        decomposed, missing_data_mask = self._decompose_eye_position(interpolated)  # pose/decomposed and pose/missing/<eye>
        reoriented = self._reorient_eye_position(decomposed, corrected)  # pose/reoriented
        filtered = self._filter_eye_position(reoriented, missing_data_mask)  # pose/filtered
        saccade_waveforms, saccade_indices = self._detect_putative_saccades(filtered)  # saccades/putative/{eye}/<indices and waveform>

        self.logger.info("Saving to NWB..")
        self._save_val("pose_corrected", corrected, pynwb_obj)
        self._save_val("pose_interpolated", interpolated, pynwb_obj)
        self._save_val("pose_decomposed", decomposed, pynwb_obj)
        self._save_val("pose_missing", missing_data_mask, pynwb_obj)
        self._save_val("pose_reoriented", reoriented, pynwb_obj)
        self._save_val("pose_filtered", filtered, pynwb_obj)
        self._save_val("saccades_putative_indices", saccade_indices, pynwb_obj)
        self._save_val("saccades_putative_waveforms", saccade_waveforms, pynwb_obj)
        self._save_val("saccades_fps", [self.fps], pynwb_obj)
        self.logger.info("Done")

    def _correct_eye_position(self, x, y, pynwb_obj):
        self.logger.info("Correcting eye position..")
        corrected = np.full([x.shape[0] + int(1e6), 2], np.nan)
        timestamps = self._get_req_val("timestamps", pynwb_obj)
        factor = np.median(timestamps)

        frame_offset = 0
        frame_idx = 0
        missing_frames = 0

        for frame in timestamps:
            frame_offset = frame_offset + (round(frame / factor) - 1)  # Increment frame
            if frame_idx >= x.shape[0]:
                missing_frames = missing_frames + 1
            else:
                corrected[frame_idx + frame_offset] = np.array([x[frame_idx], y[frame_idx]])
            frame_idx = frame_idx + 1

        corrected = corrected[:frame_idx + frame_offset, :]
        return corrected

    def _interpolate_eye_position(self, corrected):
        self.logger.info("Interpolating eye position..")
        interpolated = np.copy(corrected)
        for col_idx in [0, 1]:  # Loop over the two columns (x,y) and interpolate both, one at at time
            to_interpl = interpolated[:, col_idx]
            dropped = np.isnan(to_interpl)
            windows = []

            row_idx = 0
            while True:  # Find windows of dropped frames, append to list of idxs
                if row_idx >= dropped.size:
                    break
                row_dropped = dropped[row_idx]

                if row_dropped:
                    num_dropped = 0
                    for rdropped in dropped[row_idx:]:  # Go over until we find a frame that wasn't dropped
                        if not rdropped:
                            break
                        num_dropped = num_dropped + 1

                    if num_dropped <= 4:  # If we drop more than 4 frames in a row
                        if row_idx + num_dropped + 1 >= to_interpl.size:
                            row_idx = row_idx + num_dropped
                            continue  # Skip over these dropped rows and don't interpolate them
                        else:
                            windows.append([row_idx - 1, row_idx + num_dropped + 1])
                    row_idx = row_idx + num_dropped
                else:  # Our row was not dropped
                    row_idx = row_idx + 1

        for start, stop in windows:
            xframes = np.arange(start + 1, stop - 1, 1)
            xvals = np.array([start, stop - 1])
            yvals = np.array([to_interpl[start], to_interpl[stop - 1]])
            interp_y = np.interp(xframes, xvals, yvals)
            interpolated[start + 1: stop - 1, col_idx] = interp_y

        return interpolated

    def _decompose_eye_position(self, interpolated):
        self.logger.info("Decomposing eye position..")
        # Fill in nan values with an imputer
        imputer = SimpleImputer(missing_values=np.nan).fit_transform(interpolated)
        missing_data_mask = np.isnan(interpolated).any(1)  # Remember which values were imputed

        decomposed = PCA(n_components=2).fit_transform(imputer)  # PCA the eye positions from imputed data
        decomposed[missing_data_mask] = np.nan  # Re-mark missing values with nan

        return decomposed, missing_data_mask

    def _reorient_eye_position(self, decomposed, corrected):
        self.logger.info("Reorienting eye position..")

        reoriented = np.full_like(corrected, np.nan)

        for col_idx in range(2):  # loop over x, y cols
            decomposed_column_copy = np.copy(decomposed[:, col_idx])
            decomposed_column = decomposed[:, col_idx]

            corrected_column = corrected[:, col_idx]
            corrected_nan_idxs = np.where(np.isnan(corrected_column))[0]  # idxs where vals are nan

            # delete nan values from column
            corrected_column = np.delete(corrected_column, corrected_nan_idxs)
            decomposed_column = np.delete(decomposed_column, corrected_nan_idxs)

            # Break down into pearson correlation coefficients and a (two tailed) p value
            corr_coeff, p_val = scipy.stats.pearsonr(corrected_column, decomposed_column)

            if corr_coeff > 0.05 and p_val < 0.05:  # Signal is already oriented the correct direction
                pass
            elif corr_coeff < -0.05 and p_val < 0.05:
                decomposed_column_copy = decomposed_column_copy * -1  # Flip signal sign
            else:
                raise ValueError(f"Could not determine correlation between raw and decomposed eye position. Column '{col_idx}' corr_coeff '{corr_coeff}' p_val '{p_val}'")

            reoriented[:, col_idx] = decomposed_column_copy  # set reoriented to (possibly) flipped signal

            # TODO: Check that left and right eye position is anti-correlated

        return reoriented

    def _filter_eye_position(self, reoriented, missing_data_mask):
        self.logger.info("Filtering eye position..")

        filtered = np.full_like(reoriented, np.nan)

        smoothing_time_window_size = 25

        smoothing_window_size = 1 / self.fps * 1000
        smoothing_window_size = smoothing_time_window_size / smoothing_window_size
        smoothing_window_size = round(smoothing_window_size)
        if smoothing_window_size % 2 == 0:
            smoothing_window_size = smoothing_window_size + 1  # Make sure that the window size is odd

        for col_idx in range(2):  # Iterate over the x,y cols of the (n, 2) arr
            interp_reori = interpolate_flat_arr(reoriented[:, col_idx])  # Fill out nan values
            smoothed_reori = smooth_flat_arr(interp_reori, window_size=smoothing_window_size)  # convolve array
            smoothed_reori[missing_data_mask] = np.nan  # re-set nan vals
            filtered[:, col_idx] = smoothed_reori

        return filtered

    def _detect_putative_saccades(self, filtered):
        amplitude_threshold = 0.99
        min_inter_peak_interval = 0.075
        perisacc_window = (-0.2, 0.2)
        center_sacc_waveforms = False
        smoothing_window_size = 0.025

        self.logger.info("Extracting putative saccades..")
        saccade_dist_threshold = self.fps * min_inter_peak_interval  # Minimum inter-saccade interval (in seconds)
        peak_offsets = np.array([  # Sample offset added to each peak sample index
            round(perisacc_window[0] * self.fps),
            round(perisacc_window[1] * self.fps)
        ])

        num_features = peak_offsets[1] - peak_offsets[0]  # N samples across saccades waveforms

        window_len = round(smoothing_window_size * self.fps)
        if window_len % 2 == 0:
            window_len += 1  # ensure window length is odd

        # Impute over filtered data
        imputed = np.full_like(filtered, np.nan)
        for col_idx in range(2):  # Iterate over cols, (x, y)
            col = filtered[:, col_idx]

            imputed[:, col_idx] = np.interp(
                np.arange(col.size),
                np.arange(col.size)[np.isfinite(col)],  # Only where values are non-nan and non-inf
                col[np.isfinite(col)]
            )

        saccade_indicies = []
        saccade_waveforms = []

        velocity = np.abs(
            smooth_flat_arr(
                np.diff(imputed[:, 0]),  # forward difference of the x vals
                window_len
            )
        )
        height_threshold = np.percentile(velocity, amplitude_threshold * 100)  # Percentile of the velocity at the thres

        # Get saccade waveforms, filtering out waveforms with an invalid length
        peak_idxs, peak_props = scipy.signal.find_peaks(velocity, height=height_threshold, distance=saccade_dist_threshold)
        for peak_idx in peak_idxs:
            # If aligning center, offset by 1
            if center_sacc_waveforms:
                peak_offsets[1] = peak_offsets[1] + 1

            # Extract saccade waveform
            start_idx = peak_idx + peak_offsets[0]
            stop_idx = peak_idx + peak_offsets[1]

            saccade_waveform = filtered[start_idx: stop_idx, :]
            if saccade_waveform.shape[0] != num_features:  # Exclude incomplete waveforms
                continue

            saccade_waveforms.append(saccade_waveform)
            saccade_indicies.append(peak_idx)

        # Sort the putative saccades by chronological order
        sorted_idxs = np.argsort(saccade_indicies)
        saccade_waveforms = np.array(saccade_waveforms)[sorted_idxs]
        saccade_indicies = np.array(saccade_indicies)[sorted_idxs]
        self.logger.info(f"Detected '{saccade_waveforms.shape[0]}' putative saccade waveforms under '{self._stim_name}'")

        return saccade_waveforms, saccade_indicies

