from logic.base_logic import BaseLogic
from constants import BAND_POWERS

import utils

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, NoiseTypes

from meegkit import star, detrend

import re
import numpy as np

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
        self.smooth_window = int(.3 * .5 * self.sampling_rate)
    
    def get_data_dict(self):
        # get current data from board
        data = self.board.get_current_board_data(self.max_sample_size)

        # remove line noise
        for eeg_chan in self.eeg_channels:
            DataFilter.remove_environmental_noise(data[eeg_chan], self.sampling_rate, NoiseTypes.FIFTY_AND_SIXTY)

        # filter data
        x = data[self.eeg_channels].T
        x, _, _= detrend.detrend(x, 3)
        x, _, _ = star.star(x, 2, depth=2, n_smooth=self.smooth_window, verbose=False)
        data[self.eeg_channels] = x.T

        # calculate band features for left, right, and overall
        use_filters = True
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