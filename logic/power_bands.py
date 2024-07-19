from logic.base_logic import BaseLogic
from constants import BAND_POWERS

import utils
from utils import RealTimeLMSFilter

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter

import re
import numpy as np

import mne

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

        # mne params
        mne.set_log_level('ERROR')
        self.info = mne.create_info(eeg_names, self.sampling_rate, 'eeg')
        self.montage = mne.channels.make_standard_montage('standard_1020')

        # ema smoothing variables
        self.current_dict = {}
        self.ema_decay = ema_decay

        # adaptive filter
        self.lms_adaptive = RealTimeLMSFilter(num_taps=20)

    def get_data_dict(self):
        # get current data from board
        data = self.board.get_current_board_data(self.max_sample_size)

        mne_data = data[self.eeg_channels]
        raw = mne.io.RawArray(mne_data, self.info, verbose=None)
        raw.set_montage(self.montage)
        raw.notch_filter(freqs=(50, 60), fir_design='firwin')
        raw.filter(l_freq=2., h_freq=45., fir_design='firwin')
        data[self.eeg_channels] = self.lms_adaptive.process_signal(raw.get_data())
        
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