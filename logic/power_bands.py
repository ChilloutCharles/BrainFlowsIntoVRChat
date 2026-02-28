from logic.base_logic import BaseLogic
from constants import BAND_POWERS

import utils

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter

from scipy.signal import butter, iirnotch, filtfilt
from meegkit import detrend
import pywt

import re
import numpy as np

from sklearn.preprocessing import StandardScaler

class PwrBands(BaseLogic):
    LEFT = 'Left'
    RIGHT = 'Right'
    AVERAGE = 'Avg'

    def __init__(self, board, window_seconds=2, ema_decay=0.025):
        super().__init__(board)
        
        board_id = board.get_board_id()
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        eeg_names = BoardShim.get_eeg_names(board_id)

        self.window_seconds = window_seconds
        self.max_sample_size = self.sampling_rate * window_seconds

        # sort left and right channels
        eeg_nums = map(lambda eeg_name: int(''.join(re.findall(r'\d+', eeg_name))), eeg_names)
        chan_num_pairs = list(zip(self.eeg_channels, eeg_nums))
        self.left_chans = [eeg_chan for eeg_chan, eeg_num in chan_num_pairs if eeg_num % 2 != 0]
        self.right_chans = [eeg_chan for eeg_chan, eeg_num in chan_num_pairs if eeg_num % 2 == 0]

        # ema smoothing variables
        self.current_dict = {}
        self.ema_decay = ema_decay

        # window of a possible blink
        self.smooth_window = int(.3 * self.sampling_rate)

        # filter params
        self.filt_params = [
            iirnotch(50.0, 10, fs=self.sampling_rate), # line noise 50hz
            iirnotch(60.0, 10, fs=self.sampling_rate), # line noise 60hz
            butter(4, (2.0, 45.0), 'bandpass', fs=self.sampling_rate) # target range
        ]

        # wavelet params
        self.wlt = 'bior1.3'
        self.blink_level = 1

        # normalization
        self.scaler = StandardScaler()

    
    def get_data_dict(self):
        # get current data from board
        data = self.board.get_current_board_data(self.max_sample_size)
        eeg_data = data[self.eeg_channels]
        
        # detrend
        eeg_data = eeg_data.T
        eeg_data, _ , _ = detrend.detrend(eeg_data, 1)
        eeg_data = eeg_data.T

        # filter data
        for b, a in self.filt_params:
            eeg_data = filtfilt(b, a, eeg_data)

        # normalize
        eeg_data = self.scaler.fit_transform(eeg_data.T).T

        # blink removal by dwt level
        coeffs = pywt.wavedec(eeg_data, self.wlt)
        coeffs[0] *= 0
        coeffs[1] *= 0
        coeffs[2] *= 0
        eeg_data = pywt.waverec(coeffs, self.wlt)

        # y = np.arange(self.max_sample_size)
        # for i, eeg in enumerate(eeg_data):
        #     plt.plot(y, eeg, label=i)
        # plt.ylim((-6, 6))
        # plt.legend()
        # plt.grid()
        # plt.show()



        # calculate band features for left, right, and overall
        use_filters = False
        data[self.eeg_channels] = eeg_data
        left_powers, _ = DataFilter.get_avg_band_powers(data, self.left_chans, self.sampling_rate, use_filters)
        right_powers, _ = DataFilter.get_avg_band_powers(data, self.right_chans, self.sampling_rate, use_filters)
        avg_powers, _ = DataFilter.get_avg_band_powers(data, self.eeg_channels, self.sampling_rate, use_filters)

        # create location dict
        location_dict = {
            PwrBands.LEFT     : left_powers,
            PwrBands.RIGHT    : right_powers,
            PwrBands.AVERAGE  : avg_powers
        }

        # smooth out powers
        location_dict = {loc : self.location_smooth(loc, powers) for loc, powers in location_dict.items()}

        # create power dicts per location
        def make_power_dict(powers):
            return {bp.name : powers[bp] for bp in BAND_POWERS}
        ret_dict = {loc: make_power_dict(powers) for loc, powers in location_dict.items()}

        return ret_dict
    
    def location_smooth(self, loc_name, target_values, has_artifact = False):
        current_values, old_target_values = self.current_dict.get(loc_name, (None, None))

        # pause target update on artifact window
        if has_artifact and isinstance(old_target_values, np.ndarray):
            target_values = old_target_values

        # ema to target
        if isinstance(current_values, np.ndarray):
            current_values = utils.smooth(current_values, target_values, self.ema_decay)
        else:
            current_values = target_values
            
        self.current_dict[loc_name] = (current_values, target_values)
        return current_values