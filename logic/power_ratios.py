from logic.base_logic import Base_Logic
from constants import BAND_POWERS

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes, AggOperations

import re

class Power_Ratios(Base_Logic):
    def __init__(self, board, window_seconds = 2):
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

    def get_data_dict(self):
        # get current data from board
        data = self.board.get_current_board_data(self.max_sample_size)

        # denoise and smooth data
        for eeg_chan in self.eeg_channels:
            DataFilter.remove_environmental_noise(data[eeg_chan], self.sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
            DataFilter.perform_rolling_filter(data[eeg_chan], 3, AggOperations.MEDIAN.value)
            DataFilter.detrend(data[eeg_chan], DetrendOperations.LINEAR)
        
        # calculate band features for left, right, and overall
        left_powers, _ = DataFilter.get_avg_band_powers(data, self.left_chans, self.sampling_rate, True)
        right_powers, _ = DataFilter.get_avg_band_powers(data, self.right_chans, self.sampling_rate, True)
        avg_powers, _ = DataFilter.get_avg_band_powers(data, self.eeg_channels, self.sampling_rate, True)

        # format powers to be returned in a dictionary        
        def make_power_dict(prefix, powers):
            return {prefix + bp.name.lower(): powers[bp] for bp in BAND_POWERS}

        left_dict = make_power_dict("osc_band_power_left_", left_powers)
        right_dict = make_power_dict("osc_band_power_right_", right_powers)
        avg_dict = make_power_dict("osc_band_power_avg_", avg_powers)
        ret_dict = left_dict | right_dict | avg_dict

        return ret_dict