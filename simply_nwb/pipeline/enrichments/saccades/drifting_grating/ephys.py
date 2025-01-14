import os
import types
import warnings

import numpy as np
from population_analysis.processors.kilosort import KilosortProcessor

from simply_nwb.pipeline import Enrichment, NWBValueMapping
from simply_nwb.pipeline.funcinfo import FuncInfo
from simply_nwb.pipeline.util.waves import startstop_of_squarewave
from simply_nwb.pipeline.value_mapping import EnrichmentReference
from simply_nwb.transforms import drifting_grating_metadata_read_from_filelist, labjack_concat_files


# TODO Create graph code for analyzing the labjack data?


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

    """
    def __init__(self, np_barcode_fn, np_spike_clusts_fn, np_spike_times_fn, labjack_barcode_channel="y3", squarewave_args={}):
        super().__init__(NWBValueMapping({
            "DriftingGratingLabjack": EnrichmentReference("DriftingGratingLabjack")  # Required that the saccades, labjack and driftingGrating are already in file
        }))

        self.np_barcode_fn = np_barcode_fn
        self.np_spike_clusts_fn = np_spike_clusts_fn
        self.np_spike_times_fn = np_spike_times_fn
        self.squarewave_args = squarewave_args
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
        # kp = KilosortProcessor(self.spike_clusters, self.spike_timings)
        # raw_spike_times = kp.calculate_spikes(load_precalculated)
        # raw_firing_rates, fr_bins = kp.calculate_firingrates(SPIKE_BIN_MS, load_precalculated)

        self._save_val("asdf", [8], pynwb_obj)
        tw = 2
        pass

    @staticmethod
    def get_name() -> str:
        return "DriftingGratingEPhys"

    @staticmethod
    def saved_keys() -> list[str]:
        return ["asdf"]

    @staticmethod
    def descriptions() -> dict[str, str]:
        return {"asdf": "asdf"}

    @staticmethod
    def func_list() -> list[FuncInfo]:
        return [
            #(self, funcname: str, funcdescription: str, arg_and_description_list: dict[str, str], example_str: str):
            # Form of functions should be f(nwbobj, args, kwargs) -> Any
            FuncInfo(
                "test",
                "test function",
                {
                    "args": "list of args to pass to the function",
                    "kwargs": "dict of keyword arguments to pass to the function"
                },
                "test(['myarg'], {'mykwarg': 8})"
            )
        ]

    @staticmethod
    def test(pynwb_obj, args, kwargs):
        # Called by NWBSession(..).func("ExampleEnrichment.test", args, kwargs)
        print(f"test func being called with obj {pynwb_obj} args {args} kwargs {kwargs}")

    # Code adapted from: https://github.com/jbhunt/myphdlib/blob/668e138548d6344a7a0c9b4873f4ab7491013f0d/myphdlib/pipeline/events.py#L104
    def _extractBarcodeSignals(self, maximumWrapperPulseDuration=0.011, minimumBarcodeInterval=3, pad=100, samplingRateNeuropixels = 30000, ):
        self.logger.info('Extracting barcode signals')

        #
        barcodes = {
            'labjack': {
                'pulses': None,
                'indices': None,
                'values': None,
            },
            'neuropixels': {
                'pulses': None,
                'indices': None,
                'values': None
            }
        }

        # Load labjack data matrix
        M = self.load('labjack/matrix')

        #
        barcode_trains = {
            "labjack": None,
            "neuropixels": None
        }

        for device in ('labjack', 'neuropixels'):

            # Identify pulse trains recorded by Neuropixels
            if device == 'neuropixels':
                stateTransitionIndices = self.eventSampleNumbers
                if stateTransitionIndices is None:
                    self.save(f'barcodes/neuropixels/trains', np.array([]).astype(float))
                    continue
                global samplingRateNeuropoixels
                samplingRate = samplingRateNeuropixels
                #

            # Identify pulse trains recorded by labjack
            elif device == 'labjack':
                pass
                # Grab the starting index of the state transition
                # channelIndex = self.labjackChannelMapping['barcode']
                # signal = M[:, channelIndex]
                # stateTransitionIndices = np.where(
                #     np.logical_or(
                #         np.diff(signal) > +0.5,
                #         np.diff(signal) < -0.5
                #     )
                # )[0]
                # samplingRate = self.labjackSamplingRate

            # Parse individual barcode pulse trains
            longIntervalIndices = np.where(
                np.diff(stateTransitionIndices) >= minimumBarcodeInterval * samplingRate
            )[0]
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

            #
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

            barcode_trains[device] = padded
        return


    def _decodeBarcodeSignals(self, barcodeBitSize=0.03, wrapperBitSize=0.01):
        """
        """

        self.log('Decoding barcode signals')

        for device in ('labjack', 'neuropixels'):

            #
            if device == 'labjack':
                samplingRate = self.labjackSamplingRate
            elif device == 'neuropixels':
                global samplingRateNeuropixels
                samplingRate = samplingRateNeuropixels

            #
            pulseTrainsPadded = self.load(f'barcodes/{device}/trains')
            if pulseTrainsPadded.size == 0:
                self.save(f'barcodes/{device}/indices', np.array([]).astype(float))
                self.save(f'barcodes/{device}/values', np.array([]).astype(float))
                continue
            pulseTrains = [
                row[np.invert(np.isnan(row))].astype(np.int32).tolist()
                for row in pulseTrainsPadded
            ]
            barcodeValues, barcodeIndices = list(), list()

            #
            offset = 0
            for pulseTrain in pulseTrains:

                #
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

            #
            self.save(f'barcodes/{device}/indices', np.array(barcodeIndices))
            self.save(f'barcodes/{device}/values', np.array(barcodeValues))

        return