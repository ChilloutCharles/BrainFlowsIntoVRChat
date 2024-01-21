from logic.base_logic import Base_Logic

from brainflow.board_shim import BoardShim, BrainFlowPresets
from brainflow.data_filter import DataFilter, DetrendOperations, AggOperations

class HeartRate(Base_Logic):
    def __init__(self, board):
        super().__init__(board)

        board_id = board.get_board_id()
        
        self.ppg_channels = BoardShim.get_ppg_channels(
            board_id, BrainFlowPresets.ANCILLARY_PRESET)
        self.ppg_sampling_rate = BoardShim.get_sampling_rate(
            board_id, BrainFlowPresets.ANCILLARY_PRESET)

        self.window_seconds = int(1024 / self.ppg_sampling_rate) + 1
        self.max_sample_size = self.ppg_sampling_rate * self.window_seconds

    def get_data_dict(self):
        # get current data from board
        ppg_data = self.board.get_current_board_data(
            self.max_sample_size, BrainFlowPresets.ANCILLARY_PRESET)
        
        # denoise and detrend
        # for ppg_chan in self.ppg_channels:
            # DataFilter.perform_rolling_filter(ppg_data[ppg_chan], 3, AggOperations.MEDIAN.value)
            # DataFilter.detrend(ppg_data[ppg_chan], DetrendOperations.LINEAR)
        
        # get ir and red channels
        ppg_ir = ppg_data[self.ppg_channels[1]]
        ppg_red = ppg_data[self.ppg_channels[0]]

        # calculate heartrate and oxygen level from channels
        oxygen_level = DataFilter.get_oxygen_level(ppg_ir, ppg_red, self.ppg_sampling_rate) * 0.01
        ### Brainflow Heart Example ###
        ### https://github.com/brainflow-dev/brainflow/blob/master/python_package/examples/tests/muse_ppg.py ###
        heart_bpm = DataFilter.get_heart_rate(ppg_ir, ppg_red, self.ppg_sampling_rate, 1024)
        heart_bps = heart_bpm / 60

        # format as dictionary
        return {
            "osc_oxygen_percent" : oxygen_level,
            "osc_heart_bps" : heart_bps,
            "osc_heart_bpm" : int(heart_bpm + 0.5)
        }