import argparse
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.exit_codes import BrainFlowError

from logic.device import Device
from logic.power_bands import PowerBands
from logic.neuro_feedback import NeuroFeedback
from logic.respiration import Respiration
from logic.addons import Addons

from reporters.osc_reporter import OSC_Reporter
from reporters.deprecated_osc_reporter import Old_OSC_Reporter

import pprint

def tryFunc(func, val):
    try:
        return func(val)
    except:
        return None


def main():
    pp = pprint.PrettyPrinter()

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
    
    # custom command line arguments
    parser.add_argument('--window-seconds', type=int,
                        help='data window in seconds into the past to do calculations on', required=False, default=2)
    parser.add_argument('--refresh-rate', type=int,
                        help='refresh rate for the main loop to run at', required=False, default=60)
    parser.add_argument('--ema-decay', type=float,
                        help='exponential moving average constant to smooth outputs', required=False, default=1)

    # osc command line arguments
    parser.add_argument('--osc-ip-address', type=str,
                        help='ip address of the osc listener', required=False, default="127.0.0.1")
    parser.add_argument('--osc-port', type=int,
                        help='port the osc listener', required=False, default=9000)
    
    # choose which reporter to use
    parser.add_argument("--use-old-reporter", type=bool, action=argparse.BooleanOptionalAction, 
                        help='add this argument to use the old osc reporter')
    
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
    use_old_reporter = args.use_old_reporter
    ip = args.osc_ip_address
    send_port = args.osc_port
    osc_reporter = Old_OSC_Reporter(ip, send_port) if use_old_reporter else OSC_Reporter(ip, send_port)

    def BoardInit(args):
        ### Streaming Params ###
        refresh_rate_hz = args.refresh_rate
        window_seconds = args.window_seconds
        ema_decay = args.ema_decay / args.refresh_rate
        startup_time = window_seconds

        ### Biosensor board setup ###
        board = BoardShim(args.board_id, params)
        board.prepare_session()
        master_board_id = board.get_board_id()

        ### Logic Modules ###
        logics = [
            Device(board, window_seconds=window_seconds),
            PowerBands(board, window_seconds=window_seconds, ema_decay=ema_decay),
            NeuroFeedback(board, window_seconds=window_seconds, ema_decay=ema_decay),
            Addons(board, window_seconds=window_seconds, ema_decay=ema_decay)
        ]

        ### Muse 2/S heartbeat support ###
        if master_board_id in (BoardIds.MUSE_2_BOARD, BoardIds.MUSE_S_BOARD):
            board.config_board('p52')
            heart_rate_logic = Respiration(board, fft_size=2048, ema_decay=ema_decay)
            heart_window_seconds = heart_rate_logic.window_seconds
            startup_time = max(startup_time, heart_window_seconds)
            logics.append(heart_rate_logic)

        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Intializing (wait {}s)'.format(startup_time))
        board.start_stream(streamer_params=args.streamer_params)
        time.sleep(startup_time)
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Tracking Started')

        return board, logics, refresh_rate_hz

    try:
        # Initialize board and logics
        board, logics, refresh_rate_hz = BoardInit(args)

        while True:
            try:
                # get execution start time for time delay
                start_time = time.time()
                
                # Execute all logic
                BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Execute all Logic")
                data_dict = {type(logic).__name__ : logic.get_data_dict() for logic in logics}

                # Send messages from executed logic
                BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sending")
                send_pairs = osc_reporter.send(data_dict)
                for param_path, param_value in send_pairs:
                    BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "{}:\t{:.3f}".format(param_path, param_value))
                
                # sleep based on refresh_rate
                BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sleeping")
                execution_time = time.time() - start_time
                sleep_time = 1.0 / refresh_rate_hz - execution_time
                sleep_time = sleep_time if sleep_time > 0 else 0
                time.sleep(sleep_time)

            except TimeoutError as e:
                # display disconnect and release old session
                osc_reporter.send({Device.__name__ : {Device.CONNECTED:False}})
                board.release_session()

                BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Biosensor board error: ' + str(e))

                # attempt reinitialize 3 times
                for i in range(3):
                    try: 
                        board, logics, refresh_rate_hz = BoardInit(args)
                        break
                    except BrainFlowError as e:
                        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Retry {} Biosensor board error: {}'.format(i, str(e)))

    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Shutting down')
        board.stop_stream()
    finally:
        osc_reporter.send({Device.__name__ : {Device.CONNECTED:False}})
        board.release_session()


if __name__ == "__main__":
    main()