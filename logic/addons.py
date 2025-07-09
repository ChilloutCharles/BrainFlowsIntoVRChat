from logic.base_logic import BaseLogic
from logic.neuro_feedback import NeuroFB
import utils

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, NoiseTypes, WaveletTypes, ThresholdTypes

import re
import math
import numpy as np

class Addons(BaseLogic):
    def __init__(self, board, window_seconds=2, normalize_scale=1.1, ema_decay=0.025):
        super().__init__(board)
        self.neuro_feedback_logic = NeuroFB(board, window_seconds=window_seconds,
            normalize_scale=normalize_scale, ema_decay=ema_decay)
    
    def get_data_dict(self):
        # get neurofeedback scores 
        nf_dict = self.neuro_feedback_logic.get_data_dict()

        # get average scores
        focus = nf_dict[NeuroFB.FOCUS + NeuroFB.AVERAGE + NeuroFB.SIGNED]
        relax = nf_dict[NeuroFB.RELAX + NeuroFB.AVERAGE + NeuroFB.SIGNED]

        # remap focus and relax to 1D
        # convert to polar, discard magnitude
        angle = math.atan2(focus, relax)
        hueshift = angle / (2.0 * math.pi) + 0.5

        return {
            "HueShift": hueshift
        }
    
class BlinkDetect(BaseLogic):
    LEFT = 'Left'
    RIGHT = 'Right'

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

        # bigger threshold to only detect blinks
        self.art_thresh = 200

    def get_data_dict(self):
        # get current data from board
        data = self.board.get_current_board_data(self.max_sample_size)

        # denoise and filter data
        for eeg_chan in self.eeg_channels:
            DataFilter.perform_wavelet_denoising(data[eeg_chan], WaveletTypes.DB4, 5, threshold=ThresholdTypes.SOFT)
            DataFilter.remove_environmental_noise(data[eeg_chan], self.sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
        
        # get artifact masks for left and right sides
        # do not use absolute difference to only detect blinks once
        left_mask = utils.get_artifact_mask(data[self.left_chans], self.sampling_rate, self.art_thresh, is_absolute=False)
        right_mask = utils.get_artifact_mask(data[self.right_chans], self.sampling_rate, self.art_thresh, is_absolute=False)

        # get latest sample and check if blinked
        left_blink = 1.0 if np.any(left_mask[:, -1]) else 0.0
        right_blink = 1.0 if np.any(right_mask[:, -1]) else 0.0

        # create location dict
        location_dict = {
            BlinkDetect.LEFT    : left_blink,
            BlinkDetect.RIGHT   : right_blink
        }

        ret_dict = {loc: self.location_smooth(loc, value) for loc, value in location_dict.items()}

        return ret_dict

    def location_smooth(self, loc_name, target_values):
        current_values = self.current_dict.get(loc_name, None)

        # ema to target
        if current_values:
            current_values = utils.smooth(current_values, target_values, self.ema_decay)
        else:
            current_values = target_values
        
        self.current_dict[loc_name] = current_values
        return current_values