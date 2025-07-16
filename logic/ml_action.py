from model.intent.pipeline import Pipeline
from logic.base_logic import BaseLogic
import utils
import numpy as np

from brainflow.board_shim import BoardShim

# imported so decorator can recognize loaded model
import model.intent.model as model

class MLAction(BaseLogic):
    def __init__(self, board, ema_decay=1/60):
        super().__init__(board)

        board_id = board.get_board_id()
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)

        self.pipeline = Pipeline()
        self.ema_decay = ema_decay
        self.current_value = 0

        self.sample_size = int(1.0 * self.sampling_rate)
    
    def get_data_dict(self):
        ret_dict = super().get_data_dict()

        # get current data from board
        data = self.board.get_current_board_data(self.sample_size)
        eeg_data = data[self.eeg_channels]
        
        # predict binary thought
        target_value = self.pipeline.predict(eeg_data, self.sampling_rate)

        # smooth
        self.current_value = utils.smooth(self.current_value, target_value, self.ema_decay)
        count = len(self.current_value)

        # generate angles from output count
        angles = (2.0 * np.pi) * (np.arange(0, count) / count)

        # treat outputs as magnitudes and create vectors
        xs = self.current_value * np.cos(angles)
        ys = self.current_value * np.sin(angles)

        # sum vectors
        x = np.sum(xs)
        y = np.sum(ys)

        # get action index with highest score
        action_idx = np.argmax(self.current_value)

        # return as dictionary
        ret_dict['Action'] = action_idx.item()
        ret_dict |= {'Action{}'.format(i): value for i, value in enumerate(self.current_value.tolist())}
        ret_dict |= {
            'ActionH' : x,
            'ActionV' : y
        }
        return ret_dict