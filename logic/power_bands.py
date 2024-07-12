from logic.base_logic import BaseLogic
from constants import BAND_POWERS
import utils

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes, WaveletTypes

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

        # adaptive filter
        self.mu = 0.001  # Step size
        self.n_order = 20  # Filter order
        self.filter_weights = [np.zeros(self.n_order) for _ in range(len(self.eeg_channels))]

    def get_data_dict(self):
        # get current data from board
        buffer_samples = self.board.get_board_data_count()
        buffer_data = self.board.get_current_board_data(buffer_samples)
        data = self.board.get_current_board_data(self.max_sample_size)

        # denoise, detrend, adapt filter data
        for i, eeg_chan in enumerate(self.eeg_channels):
            DataFilter.remove_environmental_noise(buffer_data[eeg_chan], self.sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
            DataFilter.detrend(buffer_data[eeg_chan], DetrendOperations.LINEAR)

            buffer = buffer_data[eeg_chan]
            input_signal = buffer[-self.max_sample_size:]
            filtered_sample, self.filter_weights[i] = utils.lms_filter(input_signal, self.mu, self.n_order, self.filter_weights[i], buffer)
            data[eeg_chan] = filtered_sample

        
        # calculate band features for left, right, and overall
        left_powers, _ = DataFilter.get_avg_band_powers(data, self.left_chans, self.sampling_rate, True)
        right_powers, _ = DataFilter.get_avg_band_powers(data, self.right_chans, self.sampling_rate, True)
        avg_powers, _ = DataFilter.get_avg_band_powers(data, self.eeg_channels, self.sampling_rate, True)

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
    
    def location_smooth(self, loc_name, target_values):
        current_values = self.current_dict.get(loc_name, None)

        if isinstance(current_values, np.ndarray):
            current_values = utils.smooth(current_values, target_values, self.ema_decay)
        else:
            current_values = target_values
            
        self.current_dict[loc_name] = current_values
        return current_values