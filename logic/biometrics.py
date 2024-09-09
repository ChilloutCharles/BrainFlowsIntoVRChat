from logic.base_logic import OptionalBaseLogic

from brainflow.board_shim import BoardShim, BrainFlowPresets
from brainflow.data_filter import DataFilter, WaveletTypes
from scipy.signal import find_peaks, butter, filtfilt

import numpy as np
import utils

class Biometrics(OptionalBaseLogic):
    OXYGEN_PERCENT = "OxygenPercent"
    HEART_FREQ = "HeartBeatsPerSecond"
    HEART_BPM = "HeartBeatsPerMinute"
    RESP_FREQ = "BreathsPerSecond"
    RESP_BPM = "BreathsPerMinute"

    def __init__(self, board, supported=True, window_seconds=10, ema_decay=0.025):
        super().__init__(board, supported)

        if supported:
            board_id = board.get_board_id()
        
            self.ppg_channels = BoardShim.get_ppg_channels(
                board_id, BrainFlowPresets.ANCILLARY_PRESET)
            self.ppg_sampling_rate = BoardShim.get_sampling_rate(
                board_id, BrainFlowPresets.ANCILLARY_PRESET)

            self.window_seconds = window_seconds
            self.max_sample_size = self.ppg_sampling_rate * self.window_seconds

            # heart rate filter params
            lowcut = 30 / 60
            highcut = 150 / 60
            order = 2
            b, a = butter(order, (lowcut, highcut), btype="bandpass", fs=self.ppg_sampling_rate)
            self.hr_filter = lambda data: filtfilt(b, a, data)
            self.min_distance = self.ppg_sampling_rate / highcut

            # ema smoothing variables
            self.current_values = None
            self.ema_decay = ema_decay

    def estimate_heart_rate(self, ppg_ir, ppg_red, ppg_ambient):
        # do not modify data
        ppg_ir, ppg_red, ppg_ambient = np.copy(ppg_ir), np.copy(ppg_red), np.copy(ppg_ambient)

        # remove ambient light
        ppg_ir -= ppg_ambient
        ppg_red -= ppg_ambient

        # Denoise and Filter to possible heart rates
        DataFilter.perform_wavelet_denoising(ppg_ir, WaveletTypes.DB4, 4)
        DataFilter.perform_wavelet_denoising(ppg_red, WaveletTypes.DB4, 4)
        ppg_ir = self.hr_filter(ppg_ir)
        ppg_red = self.hr_filter(ppg_red)

        # find peaks in signal
        red_peaks, _ = find_peaks(ppg_red, distance=self.min_distance)
        ir_peaks, _ = find_peaks(ppg_ir, distance=self.min_distance)

        # get inter-peak sample intervals
        sample_ipis = np.concatenate((np.diff(red_peaks), np.diff(ir_peaks)))
        
        # get bpm from mean inter-peak sample interval
        average_ipi = np.mean(sample_ipis) / self.ppg_sampling_rate
        heart_bpm = 60 / average_ipi
        
        return heart_bpm
    
    def calculate_data_dict(self):
        ret_dict = {}

        # get current data from board
        ppg_data = self.board.get_current_board_data(
            self.max_sample_size, BrainFlowPresets.ANCILLARY_PRESET)
        
        # get ambient, ir, red channels, and clean the channels with ambient
        ppg_ambient = ppg_data[self.ppg_channels[2]]
        ppg_ir = ppg_data[self.ppg_channels[1]]
        ppg_red = ppg_data[self.ppg_channels[0]]

        # calculate oxygen level
        oxygen_level = DataFilter.get_oxygen_level(ppg_ir, ppg_red, self.ppg_sampling_rate) * 0.01

        # calculate heartrate
        heart_bpm = self.estimate_heart_rate(ppg_ir, ppg_red, ppg_ambient)

        # calculate respiration
        resp_bpm = heart_bpm / 4

        # create data dictionary
        ppg_dict = {
            Biometrics.OXYGEN_PERCENT : oxygen_level,
            Biometrics.HEART_FREQ : heart_bpm / 60,
            Biometrics.HEART_BPM : heart_bpm,
            Biometrics.RESP_FREQ : resp_bpm / 60,
            Biometrics.RESP_BPM : resp_bpm
        }

        # smooth using exponential moving average
        target_values = np.array(list(ppg_dict.values()))
        if not isinstance(self.current_values, np.ndarray):
            self.current_values = target_values
        else:
            self.current_values = utils.smooth(self.current_values, target_values, self.ema_decay)
        
        # add smooth values and round bpms
        ppg_dict = {k:v for k,v in zip(ppg_dict.keys(), self.current_values.tolist())}
        for k in (Biometrics.HEART_BPM, Biometrics.RESP_BPM):
            ppg_dict[k] = int(ppg_dict[k] + 0.5)
        
        ret_dict.update(ppg_dict)

        return ret_dict

    def get_data_dict(self):
        ret_dict = super().get_data_dict()
        if self.supported:
            ret_dict |= self.calculate_data_dict()
        return ret_dict