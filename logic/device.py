from logic.base_logic import BaseLogic
from brainflow.board_shim import BoardShim
import time

class Device(BaseLogic):
    CONNECTED = "Connected"
    TIME_DIFF = "TimeSinceLastSample"
    BATTERY = "BatteryPercent"

    def __init__(self, board, window_seconds=2, board_timeout=5):
        super().__init__(board)
        
        board_id = board.get_board_id()
        self.time_channel = BoardShim.get_timestamp_channel(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)

        if 'battery_channel' in BoardShim.get_board_descr(board_id):
            self.battery_channel = BoardShim.get_battery_channel(board_id)
        else:
            self.battery_channel = None

        self.window_seconds = window_seconds
        self.max_sample_size = self.sampling_rate * window_seconds
        self.board_timeout = board_timeout

    def get_data_dict(self):
        data = self.board.get_current_board_data(self.max_sample_size)
        ret_dict = {}

        # timeout check
        time_data = data[self.time_channel]
        last_sample_time = time_data[-1]
        current_time = time.time()
        time_diff = current_time - last_sample_time

        if time_diff > self.board_timeout:
            ret_dict[Device.CONNECTED] = False
            raise TimeoutError("Biosensor board timed out")
        ret_dict[Device.TIME_DIFF] = time_diff
        ret_dict[Device.CONNECTED] = True

        # battery channel (if available)
        if self.battery_channel:
            ret_dict[Device.BATTERY] = data[self.battery_channel][-1]
        
        return ret_dict
