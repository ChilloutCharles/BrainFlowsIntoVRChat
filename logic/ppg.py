from logic.base_logic import OptionalBaseLogic

from brainflow.board_shim import BoardShim, BrainFlowPresets
from brainflow.data_filter import DataFilter, AggOperations, NoiseTypes, FilterTypes, DetrendOperations, WindowOperations

import numpy as np
import utils

class Ppg(OptionalBaseLogic):
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

    def estimate_respiration(self, resp_signal, ppg_ambient):
        # do not modify data
        resp_signal, resp_ambient = np.copy(resp_signal), np.copy(ppg_ambient)

        # Possible min and max respiration in hz
        lowcut = 0.1
        highcut = 0.5

        # Detrend the signal to remove linear trends
        # DataFilter.detrend(resp_signal, DetrendOperations.LINEAR.value)

        # filter down to possible respiration rates
        DataFilter.perform_bandpass(resp_signal, self.ppg_sampling_rate, lowcut, highcut, 3, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandpass(resp_ambient, self.ppg_sampling_rate, lowcut, highcut, 3, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        resp_signal -= resp_ambient

        # Perform FFT
        fft_data = DataFilter.perform_fft(resp_signal, WindowOperations.NO_WINDOW.value)
        fft_freq = np.linspace(0, self.ppg_sampling_rate / 2, len(fft_data) // 2)

        # Find the peak frequency in the respiratory range
        idx = np.where((fft_freq >= lowcut) & (fft_freq <= highcut))
        peak_freq = fft_freq[idx][np.argmax(np.abs(fft_data[idx]))]

        # Return breathing rate in BPM
        return peak_freq

    def estimate_heart_rate(self, hr_ir, hr_red, ppg_ambient):
        # do not modify data
        hr_ir, hr_red, hr_ambient = np.copy(hr_ir), np.copy(hr_red), np.copy(ppg_ambient)

        # Possible min and max heart rate in hz
        lowcut = 0.1
        highcut = 4.25

        # filter down to possible heart rates
        DataFilter.perform_bandpass(hr_ir, self.ppg_sampling_rate, lowcut, highcut, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandpass(hr_red, self.ppg_sampling_rate, lowcut, highcut, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandpass(hr_ambient, self.ppg_sampling_rate, lowcut, highcut, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        hr_ir -= hr_ambient
        hr_red -= hr_ambient

        ### Brainflow Heart Example ###
        ### https://github.com/brainflow-dev/brainflow/blob/master/python_package/examples/tests/muse_ppg.py ###
        heart_bpm = DataFilter.get_heart_rate(hr_ir, hr_red, self.ppg_sampling_rate, self.fft_size)
        return heart_bpm

    def get_data_dict(self):
        ret_dict = super().get_data_dict()

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
        resp_ir = self.estimate_respiration(ppg_ir, ppg_ambient)
        resp_red = self.estimate_respiration(ppg_red, ppg_ambient)
        resp_avg = np.mean((resp_ir, resp_red))

        # create data dictionary
        ppg_dict = {
            Ppg.OXYGEN_PERCENT : oxygen_level,
            Ppg.HEART_FREQ : heart_bpm / 60,
            Ppg.HEART_BPM : heart_bpm,
            Ppg.RESP_FREQ : resp_avg,
            Ppg.RESP_BPM : resp_avg * 60
        }

        # smooth using exponential moving average
        target_values = np.array(list(ppg_dict.values()))
        if not isinstance(self.current_values, np.ndarray):
            self.current_values = target_values
        else:
            self.current_values = utils.smooth(self.current_values, target_values, self.ema_decay)
        
        # add smooth values and round bpms
        ppg_dict = {k:v for k,v in zip(ppg_dict.keys(), self.current_values.tolist())}
        for k in (Ppg.HEART_BPM, Ppg.RESP_BPM):
            ppg_dict[k] = int(ppg_dict[k] + 0.5)
        
        ret_dict.update(ppg_dict)
        return ret_dict

class HeartRate(Ppg):
    def __init__(self, board, supported=True, fft_size=1024, ema_decay=0.025):
        super().__init__(board, supported, fft_size, ema_decay)
    
    def get_data_dict(self):
        ret_dict = super(Ppg, self).get_data_dict()
        if self.supported:
            ret_dict = super().get_data_dict()
            keys = (Ppg.SUPPORTED, Ppg.HEART_BPM, Ppg.HEART_FREQ)
            ret_dict = {k: ret_dict[k] for k in keys}
        return ret_dict

class Respiration(Ppg):
    def __init__(self, board, supported=True, fft_size=1024, ema_decay=0.025):
        super().__init__(board, supported, fft_size, ema_decay)
    
    def get_data_dict(self):
        ret_dict = super(Ppg, self).get_data_dict()
        if self.supported:
            ret_dict = super().get_data_dict()
            keys = (Ppg.SUPPORTED, Ppg.RESP_BPM, Ppg.RESP_FREQ, Ppg.OXYGEN_PERCENT)
            ret_dict = {k: ret_dict[k] for k in keys}
        return ret_dict