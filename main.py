import argparse
import time
from collections import ChainMap

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter

from pythonosc.udp_client import SimpleUDPClient

from constants import OSC_Path, OSC_BASE_PATH

from logic.telemetry import Telemetry
from logic.focus_relax import Focus_Relax
from logic.heartrate import HeartRate

def tryFunc(func, val):
    try:
        return func(val)
    except:
        return None


def main():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()

    ### Uncomment this to see debug messages ###
    # BoardShim.set_log_level(LogLevels.LEVEL_DEBUG.value)

    ### Paramater Setting ###
    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int,
                        help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str,
                        help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str,
                        help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str,
                        help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str,
                        help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str,
                        help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str,
                        help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file',
                        required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    ### OSC Setup ###
    ip = "127.0.0.1"
    send_port = 9000
    osc_client = SimpleUDPClient(ip, send_port)

    ### Biosensor board setup ###
    board = BoardShim(args.board_id, params)
    board.prepare_session()
    master_board_id = board.get_board_id()

    ### Streaming Params ###
    window_seconds = 5
    startup_time = window_seconds

    ### Logic Modules ###
    logics = [
        Telemetry(board, window_seconds),
        Focus_Relax(board, window_seconds)
    ]

    ### Muse 2/S heartbeat support ###
    if master_board_id in (BoardIds.MUSE_2_BOARD, BoardIds.MUSE_S_BOARD):
        board.config_board('p52')
        heart_rate_logic = HeartRate(board)
        heart_window_seconds = heart_rate_logic.window_seconds
        startup_time = max(startup_time, heart_window_seconds)
        logics.append(heart_rate_logic)

    try:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Intializing (wait {}s)'.format(startup_time))
        board.start_stream(streamer_params=args.streamer_params)
        time.sleep(startup_time)

        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Main Loop Started')
        while True:
            # Execute all logic
            data_dicts = list(map(lambda logic: logic.get_data_dict(), logics))
            full_dict = dict(ChainMap(*data_dicts))

            # Send messages from executed logic
            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sending")
            osc_client.send_message(OSC_Path.ConnectionStatus, True)
            for osc_name in full_dict:
                BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "{}:\t{:.3f}".format(osc_name, full_dict[osc_name]))
                osc_client.send_message(OSC_BASE_PATH + osc_name, full_dict[osc_name])

    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Shutting down')
    except TimeoutError:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value,
                              'Biosensor board timed out')
    finally:
        osc_client.send_message(OSC_Path.ConnectionStatus, False)
        ### Cleanup ###
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()