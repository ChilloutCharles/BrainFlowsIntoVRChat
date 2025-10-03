from logic.base_logic import BaseLogic
from brainflow.board_shim import BoardShim
import time
import constants

class Meta(BaseLogic):
    VMAJOR = "VersionMajor"
    VMINOR = "VersionMinor"
    
    def __init__(self, board):
        super().__init__(board)
    
    def get_data_dict(self):
        return {
            Meta.VMAJOR : constants.VERSION_MAJOR,
            Meta.VMINOR : constants.VERSION_MINOR
        }

class Info(Meta):
    CONNECTED = "DeviceConnected"
    TIME_DIFF = "SecondsSinceLastUpdate"
    BATTERYLEVEL = "BatteryLevel"
    BATTERYSUPPORT = "BatterySupported"

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
        ret_dict = super().get_data_dict()
        time_data = data[self.time_channel]

        # no data check
        data_size = len(time_data)
        if data_size == 0:
            raise EOFError("No data samples available")

        # timeout check
        last_sample_time = time_data[-1]
        current_time = time.time()
        time_diff = current_time - last_sample_time

        if time_diff > self.board_timeout:
            ret_dict[Info.CONNECTED] = False
            raise TimeoutError("Biosensor board timed out")
        ret_dict[Info.TIME_DIFF] = time_diff
        ret_dict[Info.CONNECTED] = True

        # battery channel
        ret_dict[Info.BATTERYLEVEL] = data[self.battery_channel][-1] if self.battery_channel else -1.0
        ret_dict[Info.BATTERYSUPPORT] = bool(self.battery_channel)
        
        return ret_dict
