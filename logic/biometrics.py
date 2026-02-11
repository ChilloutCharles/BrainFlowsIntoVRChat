from logic.base_logic import OptionalBaseLogic

from brainflow.board_shim import BoardShim, BrainFlowPresets
from brainflow.data_filter import DataFilter
from scipy.signal import find_peaks, iirnotch, butter, filtfilt

import numpy as np
import utils

class Biometrics(OptionalBaseLogic):
    OXYGEN_PERCENT = "OxygenPercent"
    HEART_FREQ = "HeartBeatsPerSecond"
    HEART_BPM = "HeartBeatsPerMinute"
    RESP_FREQ = "BreathsPerSecond"
    RESP_BPM = "BreathsPerMinute"

    VRCHAT_HEART_FREQ_DIVISOR = 4
    HEART_FREQ_UPDATE_THRESHOLD = 0.01

    def __init__(self, board, supported=True, window_size=5, ema_decay=0.025):
        super().__init__(board, supported)

        self.last_hr = None
        self.hr_threshold = Biometrics.HEART_FREQ_UPDATE_THRESHOLD

        if supported:
            board_id = board.get_board_id()
        
            self.ppg_channels = BoardShim.get_ppg_channels(
                board_id, BrainFlowPresets.ANCILLARY_PRESET)
            self.ppg_sampling_rate = BoardShim.get_sampling_rate(
                board_id, BrainFlowPresets.ANCILLARY_PRESET)

            self.window_seconds = window_size
            self.max_sample_size = self.ppg_sampling_rate * self.window_seconds

            # ema smoothing variables
            self.current_values = None
            self.ema_decay = ema_decay

            # hr filters
            lowcut =  40.0 / 60.0 
            highcut = 180 / 60.0
            order = 3
            self.notch_params = iirnotch(0.05, 0.005, self.ppg_sampling_rate) # baseline wander filter
            self.bp_params = butter(order, (2 * lowcut, 2 * highcut), 'bandpass', fs=self.ppg_sampling_rate) # multiplied by 2 due to double peaks

    def estimate_heart_rate(self, hr_ir, hr_red):
        # do not modify data
        hr_ir, hr_red = np.copy(hr_ir), np.copy(hr_red)

        def hr_preprocess(hr_arr):
            filted = filtfilt(*self.notch_params, x=hr_arr)
            filted = filted - np.mean(filted)
            filted = filted ** 2 # doubles peak count
            filted = filtfilt(*self.bp_params, x=filted)
            return filted
        
        hr_red = hr_preprocess(hr_red)
        hr_ir = hr_preprocess(hr_ir)

        ## peak count approach

        # find peaks
        red_peaks, _ = find_peaks(hr_red, height=0)
        ir_peaks, _ = find_peaks(hr_ir, height=0)

        # discard anomalous peaks
        def clean_peaks(peaks, data):
            peak_vals = data[peaks]
            med = np.median(peak_vals)
            diffs = np.abs(peak_vals - med)
            mad = np.median(diffs)
            good_idx = np.argwhere(diffs < 3.0 * mad)
            return peaks[good_idx]
        
        red_peaks = clean_peaks(red_peaks, hr_red)
        ir_peaks = clean_peaks(ir_peaks, hr_ir)
        
        peak_count = (len(red_peaks) + len(ir_peaks)) * 0.5 # analyzing same time period twice
        heart_bpm = 60.0 * peak_count / (2.0 * self.window_seconds) # divide by 2 due to double peaks

        return heart_bpm
    
    def calculate_data_dict(self):
        ret_dict = {}

        # get current data from board
        ppg_data = self.board.get_current_board_data(
            self.max_sample_size, BrainFlowPresets.ANCILLARY_PRESET)
        
        # get ir, red channels
        ppg_ir = ppg_data[self.ppg_channels[1]]
        ppg_red = ppg_data[self.ppg_channels[0]]

        # calculate oxygen level
        oxygen_level = DataFilter.get_oxygen_level(ppg_ir, ppg_red, self.ppg_sampling_rate) * 0.01

        # calculate heartrate
        heart_bpm = self.estimate_heart_rate(ppg_ir, ppg_red)

        # calculate respiration
        resp_bpm = heart_bpm / 4

        # create data dictionary
        ppg_dict = {
            Biometrics.OXYGEN_PERCENT : oxygen_level,
            Biometrics.HEART_FREQ : heart_bpm / 60 / Biometrics.VRCHAT_HEART_FREQ_DIVISOR,
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

        current_hr = ppg_dict.pop(Biometrics.HEART_FREQ)
        if self.last_hr is None or abs(current_hr - self.last_hr) > self.hr_threshold:
            self.last_hr = current_hr
            ret_dict[Biometrics.HEART_FREQ] = current_hr
        
        ret_dict.update(ppg_dict)

        return ret_dict

    def get_data_dict(self):
        ret_dict = super().get_data_dict()
        if self.supported:
            ret_dict |= self.calculate_data_dict()
        return ret_dict