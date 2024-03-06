from logic.base_logic import OptionalBaseLogic

from brainflow.board_shim import BoardShim, BrainFlowPresets
from brainflow.data_filter import DataFilter, AggOperations, NoiseTypes, FilterTypes, DetrendOperations, WindowOperations
from scipy.signal import find_peaks

import numpy as np
import utils

class Biometrics(OptionalBaseLogic):
    OXYGEN_PERCENT = "OxygenPercent"
    HEART_FREQ = "HeartBeatsPerSecond"
    HEART_BPM = "HeartBeatsPerMinute"
    RESP_FREQ = "BreathsPerSecond"
    RESP_BPM = "BreathsPerMinute"

    def __init__(self, board, supported=True, fft_size=1024, ema_decay=0.025):
        super().__init__(board, supported)

        if supported:
            board_id = board.get_board_id()
        
            self.ppg_channels = BoardShim.get_ppg_channels(
                board_id, BrainFlowPresets.ANCILLARY_PRESET)
            self.ppg_sampling_rate = BoardShim.get_sampling_rate(
                board_id, BrainFlowPresets.ANCILLARY_PRESET)

            self.window_seconds = int(fft_size / self.ppg_sampling_rate) + 1
            self.max_sample_size = self.ppg_sampling_rate * self.window_seconds
            self.fft_size = fft_size

            # ema smoothing variables
            self.current_values = None
            self.ema_decay = ema_decay

    def estimate_heart_rate(self, hr_ir, hr_red, ppg_ambient):
        # do not modify data
        hr_ir, hr_red, hr_ambient = np.copy(hr_ir), np.copy(hr_red), np.copy(ppg_ambient)

        # Possible min and max heart rate in hz
        lowcut = 0.5
        highcut = 4.25
        order = 4

        # remove ambient light
        hr_ir = np.clip(hr_ir - hr_ambient, 0, None)
        hr_red = np.clip(hr_red - hr_ambient, 0, None)

        # detrend and filter down to possible heart rates
        DataFilter.detrend(hr_red, DetrendOperations.LINEAR)
        DataFilter.detrend(hr_ir, DetrendOperations.LINEAR)
        DataFilter.perform_bandpass(hr_red, self.ppg_sampling_rate, lowcut, highcut, order, FilterTypes.BUTTERWORTH, 0)
        DataFilter.perform_bandpass(hr_ir, self.ppg_sampling_rate, lowcut, highcut, order, FilterTypes.BUTTERWORTH, 0)
        
        # find peaks in signal
        red_peaks, _ = find_peaks(hr_red, distance=self.ppg_sampling_rate/2)
        ir_peaks, _ = find_peaks(hr_ir, distance=self.ppg_sampling_rate/2)

        # get inter-peak intervals
        red_ipis = np.diff(red_peaks) / self.ppg_sampling_rate
        ir_ipis = np.diff(ir_peaks) / self.ppg_sampling_rate
        ipis = np.concatenate((red_ipis, ir_ipis))
        
        # get bpm from mean inter-peak interval
        average_ipi = np.mean(ipis)
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