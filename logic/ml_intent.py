from model.intent.model import EnsembleModel
from logic.base_logic import BaseLogic
import utils

from brainflow.board_shim import BoardShim
import numpy as np


class MLIntent(BaseLogic):
    def __init__(self, board, ema_decay=1/60):
        super().__init__(board)

        board_id = board.get_board_id()
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)

        self.model = EnsembleModel()
        self.ema_decay = ema_decay
        self.current_value = 0

        self.sample_size = int(1.0 * self.sampling_rate)
    
    def get_data_dict(self):
        ret_dict = super().get_data_dict()

        # get current data from board
        data = self.board.get_current_board_data(self.sample_size)
        eeg_data = data[self.eeg_channels]
        
        # predict binary thought, round
        target_value = self.model.predict(eeg_data, self.sampling_rate)
        target_value = np.round(target_value)[0]

        # smooth
        self.current_value = utils.smooth(self.current_value, target_value, self.ema_decay)

        # return as dictionary
        ret_dict |= {'Action' : self.current_value}
        return ret_dict