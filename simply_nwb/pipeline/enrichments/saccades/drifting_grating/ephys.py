import os
import types
import warnings

import numpy as np
from population_analysis.processors.kilosort import KilosortProcessor
from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.funcinfo import FuncInfo
from simply_nwb.pipeline.value_mapping import EnrichmentReference


class DriftingGratingEPhysEnrichment(Enrichment):
    """
    Enrich the saccade data with metadata about the drifting grating using labjack as the global clock, including ephys data

    Default labjack signal mapping is as follows
    y0 'barcode' of a count
    y1 default drifting grating event happened, align driftingGrating-0.txt, starts at zero, goes to 1
    Assumes that the first 'pulse' in the labjack data corresponds to the first event in the driftingGrating-0.txt

    y2 video camera timing acquisition frame, timestamps for video frames
    a frame is 0 or 1, each time it flips is a new frame, 0 to 1, 1 to 0 etc..
    y3 misc analogue signal, per usecase

    Requires the Neuropixels timestamps for labjack signals aka 'barcode' to sync to labjack time

    the 'barcode' is a signal encoding an integer value, of which is the same between the labjack and neuropixels,
    which allows alignment between the two time spaces

    """
    NEUROPIXELS_SAMPLING_RATE = 30000
    LABJACK_SAMPLING_RATE = 2000

    def __init__(self, np_barcode_fn, np_spike_clusts_fn, np_spike_times_fn, labjack_barcode_channel="y0", lj_timestamps_colname="Time"):
        super().__init__(NWBValueMapping({
            "DriftingGratingLabjack": EnrichmentReference("DriftingGratingLabjack")  # Required that the saccades, labjack and driftingGrating are already in file
        }))

        self.lj_timestamps_colname = lj_timestamps_colname
        self.np_barcode_fn = np_barcode_fn
        self.np_spike_clusts_fn = np_spike_clusts_fn
        self.np_spike_times_fn = np_spike_times_fn
        self.labjack_barcode_channel = labjack_barcode_channel

        for fn in [np_barcode_fn, np_spike_clusts_fn, np_spike_times_fn]:
            assert os.path.exists(fn)

        self._spike_clusts = None
        self._spike_times = None
        self._np_barcode = None

    @property
    def spike_clusts(self):
        if self._spike_clusts is None:
            self._spike_clusts = np.load(self.np_spike_clusts_fn)
        return self._spike_clusts

    @property
    def spike_times(self):
        if self._spike_times is None:
            self._spike_times = np.load(self.np_spike_times_fn)
        return self._spike_times

    @property
    def np_barcode(self):
        if self._np_barcode is None:
            self._np_barcode = np.load(self.np_barcode_fn)
        return self._np_barcode

    def _run(self, pynwb_obj):
        """
        Grab the barcode signal, which (after decoding) is a series of pulses encoding an integer value that is then used to align the
        labjack with the neuropixels.

        """

        self.logger.info("Extracting and decoding neuropixels barcode..")
        np_signal = self.extract_barcode_signals(self.np_barcode, DriftingGratingEPhysEnrichment.NEUROPIXELS_SAMPLING_RATE)
        # indices are the idxs of the first value in the 'pulsetrain' of the neuropixels barcode signal
        # vals is the integer values
        np_barcode_indices, np_barcode_vals = self.decode_barcode_signals(np_signal, DriftingGratingEPhysEnrichment.NEUROPIXELS_SAMPLING_RATE)

        self.logger.info("Extracting, converting and decoding labjack barcode..")
        lj_barcode = self._get_req_val(f"DriftingGratingLabjack.{self.labjack_barcode_channel}", pynwb_obj)
        # Need to convert the signal into transition idxs 'states'
        lj_states = np.where(np.logical_or(np.diff(lj_barcode) > +0.5, np.diff(lj_barcode) < -0.5))[0]
        lj_signal = self.extract_barcode_signals(lj_states, DriftingGratingEPhysEnrichment.LABJACK_SAMPLING_RATE)
        lj_barcode_indices, lj_barcode_vals = self.decode_barcode_signals(lj_signal, DriftingGratingEPhysEnrichment.LABJACK_SAMPLING_RATE)

        # Align the two integers, and grab the common ones (sometimes the recording devices don't start/stop at the same time)
        matched_vals, common_lj, common_np = np.intersect1d(lj_barcode_vals, np_barcode_vals, return_indices=True)

        # Align the spike times with the neuropixels integer values
        spike_times_in_counter_time = np.interp(self.spike_times, np.array(np_barcode_indices)[common_np], matched_vals)
        # Take the aligned spike times (in neuropixel integers) to the labjack integers, to the labjack indices
        spikes_times_in_labjack_indices = np.round(np.interp(spike_times_in_counter_time, matched_vals, np.array(lj_barcode_indices)[common_lj])).astype(int)

        labjack_time = self._get_req_val(f"DriftingGratingLabjack.{self.lj_timestamps_colname}", pynwb_obj)
        spike_times_in_labjack_time = labjack_time[spikes_times_in_labjack_indices]

        self._save_val("spike_times_in_neuropixels_time", self.spike_times, pynwb_obj)
        self._save_val("spike_times_in_labjack_time", spike_times_in_labjack_time, pynwb_obj)
        self._save_val("spike_clusters", self.spike_clusts, pynwb_obj)

        # kilosort processor broken for low-memory machines/not optimized
        # kp = KilosortProcessor(self.spike_clusts, spike_times_in_labjack_time)
        # kp.calculate_firingrates(1.0, False)

        # Things that might be interesting?
        # Get spikes for a given unit optional labjack idx time window
        # Get unit spikes around saccade (trials, unitnum, t) allow multiple input
        # Get trial times for a saccade in labjack idxs (trials,)
        # Normalize firing rate?
        tw = 2
        pass

    @staticmethod
    def get_name() -> str:
        return "DriftingGratingEPhys"

    @staticmethod
    def saved_keys() -> list[str]:
        return [
            "spike_times_in_labjack_time",
            "spike_times_in_neuropixels_time",
            "spike_clusters"
        ]

    @staticmethod
    def descriptions() -> dict[str, str]:
        return {
            "spike_times_in_labjack_time": "Kilosort spike times in terms of labjack time",
            "spike_times_in_neuropixels_time": "Original Kilosort (not aligned) spike times in neuropixels time",
            "spike_clusters": "Unit ID associated with the spike times"
        }

    @staticmethod
    def func_list() -> list[FuncInfo]:
        return [
            #(self, funcname: str, funcdescription: str, arg_and_description_list: dict[str, str], example_str: str):
            # Form of functions should be f(nwbobj, args, kwargs) -> Any
            # FuncInfo(
            #     "test",
            #     "test function",
            #     {
            #         "args": "list of args to pass to the function",
            #         "kwargs": "dict of keyword arguments to pass to the function"
            #     },
            #     "test(['myarg'], {'mykwarg': 8})"
            # )
        ]

    # Code adapted from: https://github.com/jbhunt/myphdlib/blob/668e138548d6344a7a0c9b4873f4ab7491013f0d/myphdlib/pipeline/events.py#L104
    def extract_barcode_signals(self, stateTransitionIndices, samplingRate, maximumWrapperPulseDuration=0.011, minimumBarcodeInterval=3, pad=100):
        self.logger.info('Extracting barcode signals')

        # Parse individual barcode pulse trains
        longIntervalIndices = np.where(np.diff(stateTransitionIndices) >= minimumBarcodeInterval * samplingRate)[0]
        pulseTrains = np.split(stateTransitionIndices, longIntervalIndices + 1)

        # Filter out incomplete pulse trains
        pulseDurationThreshold = round(maximumWrapperPulseDuration * samplingRate)
        pulseTrainsFiltered = list()
        for pulseTrain in pulseTrains:

            # Need at least 1 pulse on each side for the wrapper
            if pulseTrain.size < 4:
                continue

            # Wrapper pulses should be smaller than the encoding pulses
            firstPulseDuration = pulseTrain[1] - pulseTrain[0]
            finalPulseDuration = pulseTrain[-1] - pulseTrain[-2]
            if firstPulseDuration > pulseDurationThreshold:
                continue
            elif finalPulseDuration > pulseDurationThreshold:
                continue

            # Complete pulses
            pulseTrainsFiltered.append(pulseTrain)

        padded = list()
        for pulseTrainFiltered in pulseTrainsFiltered:
            row = list()
            for index in pulseTrainFiltered:
                row.append(index)
            nRight = pad - len(row)
            for value in np.full(nRight, np.nan):
                row.append(value)
                padded.append(row)
        padded = np.array(padded)
        return padded

    # Code adapted from: https://github.com/jbhunt/myphdlib/blob/668e138548d6344a7a0c9b4873f4ab7491013f0d/myphdlib/pipeline/events.py#L197
    def decode_barcode_signals(self, pulseTrainsPadded, samplingRate, barcodeBitSize=0.03, wrapperBitSize=0.01):
        self.logger.info('Decoding barcode signals')

        if pulseTrainsPadded.size == 0:
            raise ValueError("Invalid pulseTrainPadded passed as arg! Len is 0!")

        pulseTrains = [
            row[np.invert(np.isnan(row))].astype(np.int32).tolist()
            for row in pulseTrainsPadded
        ]

        barcodeValues, barcodeIndices = list(), list()

        offset = 0
        for pulseTrain in pulseTrains:

            wrapperFallingEdge = pulseTrain[1]
            wrapperRisingEdge = pulseTrain[-2]
            barcodeLeftEdge = wrapperFallingEdge + round(wrapperBitSize * samplingRate)
            barcodeRightEdge = wrapperRisingEdge - round(wrapperBitSize * samplingRate)

            # Determine the state at the beginning and end of the data window
            firstStateTransition = pulseTrain[2]
            if (firstStateTransition - barcodeLeftEdge) / samplingRate < 0.001:
                initialSignalState = currentSignalState = True
            else:
                initialSignalState = currentSignalState = False
            finalStateTransition = pulseTrain[-3]
            if (barcodeRightEdge - finalStateTransition) / samplingRate < 0.001:
                finalSignalState = True
            else:
                finalSignalState = False

            # Determine what indices to use for computing time intervals between state transitions
            if initialSignalState == True and finalSignalState == True:
                iterable = pulseTrain[2: -2]
            elif initialSignalState == True and finalSignalState == False:
                iterable = np.concatenate([pulseTrain[2:-2], np.array([barcodeRightEdge])])
            elif initialSignalState == False and finalSignalState == False:
                iterable = np.concatenate(
                    [np.array([barcodeLeftEdge]), pulseTrain[2:-2], np.array([barcodeRightEdge])])
            elif initialSignalState == False and finalSignalState == True:
                iterable = np.concatenate([np.array([barcodeLeftEdge]), pulseTrain[2:-2]])

            # Determine how many bits are stored in each time interval and keep track of the signal state
            bitList = list()
            for nSamples in np.diff(iterable):
                nBits = int(round(nSamples / (barcodeBitSize * samplingRate)))
                for iBit in range(nBits):
                    bitList.append(1 if currentSignalState else 0)
                currentSignalState = not currentSignalState

            # Decode the strings of bits
            bitString = ''.join(map(str, bitList[::-1]))
            if len(bitString) != 32:
                raise Exception(f'More or less that 32 bits decoded')
            value = int(bitString, 2) + offset

            # 32-bit integer overflow
            if value == 2 ** 32 - 1:
                offset = 2 ** 32

            #
            barcodeValues.append(value)
            barcodeIndices.append(pulseTrain[0])


        return barcodeIndices, barcodeValues
