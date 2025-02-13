import argparse
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.exit_codes import BrainFlowError, BrainFlowExitCodes

from logic.base_logic import BaseLogic
from logic.telemetry import Info
from logic.power_bands import PwrBands
from logic.neuro_feedback import NeuroFB
from logic.biometrics import Biometrics
from logic.addons import Addons

from reporters.osc_reporter import OSC_Reporter
from reporters.debug_osc_reporter import Debug_Reporter
from reporters.deprecated_osc_reporter import Old_OSC_Reporter
from reporters.reporter import Reporter
from reporters.log_reporter import Log_Reporter

def enable_loggers():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()

def parse_args() -> argparse.Namespace:
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
    parser.add_argument('--file', type=str, help='file',
                        required=False, default='')
    
    # board id by name or id
    parser.add_argument('--board-id', type=str, help='board id or name, check docs to get a list of supported boards',
                        required=True)

    # custom command line arguments
    parser.add_argument('--window-seconds', type=int,
                        help='data window in seconds into the past to do calculations on', required=False, default=2)
    parser.add_argument('--refresh-rate', type=int,
                        help='refresh rate for the main loop to run at', required=False, default=60)
    parser.add_argument('--ema-decay', type=float,
                        help='exponential moving average constant to smooth outputs', required=False, default=1)
    parser.add_argument('--retry-count', type=int,
                        help='sets the amount of times to reconnect before giving up', required=False, default=3)

    # osc command line arguments
    parser.add_argument('--osc-ip-address', type=str,
                        help='ip address of the osc listener', required=False, default="127.0.0.1")
    parser.add_argument('--osc-port', type=int,
                        help='port the osc listener', required=False, default=9000)
    
    # choose which reporter to use
    parser.add_argument("--use-old-reporter", type=bool, action=argparse.BooleanOptionalAction, 
                        help='add this argument to use the old osc reporter')

    # toggle debug mode
    parser.add_argument("--debug", type=bool, action=argparse.BooleanOptionalAction, 
                        help='add this argument to toggle debug mode on')
    
    # toggle value logging
    parser.add_argument("--enable-logs", type=bool, action=argparse.BooleanOptionalAction, 
                        help='add this argument to toggle value logging on')

    # arguments to configure MLAction
    parser.add_argument("--enable-action", type=bool, action=argparse.BooleanOptionalAction, 
                        help='add this argument to enable ml action logic')
    parser.add_argument("--action-ema-multiplier", type=float, required=False, default=5.0,
                        help='multiplier to speed up or slow down the reactiveness of ml action logic')
    
    return parser.parse_args()

def configure_brainflow_params(args: argparse.Namespace) -> BrainFlowInputParams:
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

    return params

def BoardInit(args: argparse.Namespace) -> tuple[BoardShim, list[BaseLogic], int]:
    ### Only Import MLAction if activated ###
    if args.enable_action:
        from logic.ml_action import MLAction

    ### Board Id selection ###
    try:
        master_board_id = int(args.board_id)
    except ValueError:
        master_board_id = BoardIds[args.board_id.upper()]

    ### Streaming Params ###
    refresh_rate_hz = args.refresh_rate
    window_seconds = args.window_seconds
    ema_decay = args.ema_decay / args.refresh_rate
    startup_time = window_seconds

    ### Parse params ###
    params = configure_brainflow_params(args)


    ### Biosensor board setup ###
    board = BoardShim(master_board_id, params)
    board.prepare_session()

    ### Logic Modules ###
    has_muse_ppg = master_board_id in (BoardIds.MUSE_2_BOARD, BoardIds.MUSE_S_BOARD)
    
    fft_size= 64 * 10 # TODO: Make this configurable
    biometrics_logic = Biometrics(board, has_muse_ppg, fft_size=fft_size, ema_decay=ema_decay)

    logics = [
        Info(board, window_seconds=window_seconds),
        PwrBands(board, window_seconds=window_seconds, ema_decay=ema_decay),
        NeuroFB(board, window_seconds=window_seconds, ema_decay=ema_decay),
        Addons(board, window_seconds=window_seconds, ema_decay=ema_decay),
        biometrics_logic
    ]

    ### Muse 2/S heartbeat support ###
    if has_muse_ppg:
        board.config_board('p52')
        heart_window_seconds = biometrics_logic.window_seconds
        startup_time = max(startup_time, heart_window_seconds)
    
    ### Add ml action to logics if enabled
    if args.enable_action:
        logics.append(MLAction(board, ema_decay = ema_decay * args.action_ema_multiplier))

    ### Adding one second to startup time for adaptive filters ###
    startup_time += 1

    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Intializing (wait {}s)'.format(startup_time))
    board.start_stream(streamer_params=args.streamer_params)
    time.sleep(startup_time)
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Tracking Started')

    return board, logics, refresh_rate_hz

def setup_reporter(args: argparse.Namespace) -> Reporter:
    ip = args.osc_ip_address
    send_port = args.osc_port
    use_old_reporter = args.use_old_reporter
    reporters = [Old_OSC_Reporter(ip, send_port) if use_old_reporter else OSC_Reporter(ip, send_port)]
    
    if args.debug:
        reporters.append(Debug_Reporter(ip, send_port))
    if args.enable_logs:
        reporters.append(Log_Reporter(ip, send_port))

    reporter_dict = {type(rp).__name__:rp for rp in reporters}

    return Reporter(reporter_dict)

def main():
    enable_loggers()

    #region Configure
    ### Parse input ###
    args = parse_args()

    ### Debug message toggle ###
    if args.debug:
        BoardShim.set_log_level(LogLevels.LEVEL_DEBUG.value)
    
    ### Reporter Setup ###
    reporter = setup_reporter(args)
    #endregion Configure

    # start main loop with all retries
    main_loop(args, reporter, args.retry_count)


def main_loop(args: argparse.Namespace, reporter: Reporter, retries: int):
    #region Init
    while True:
        try:
            # Initialize board and logics
            board, logics, refresh_rate_hz = BoardInit(args)
            retries = args.retry_count # reset retry count on success
            break
        except KeyboardInterrupt:
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Received interrupt signal! Shutting down...')
            return
        except BrainFlowError as board_init_error:
            if board_init_error.exit_code == BrainFlowExitCodes.BOARD_NOT_READY_ERROR and retries > 0:
                BoardShim.log_message(LogLevels.LEVEL_WARN.value, f'Biosensor board not ready! (Retries left: {retries})')
                return main_loop(args, reporter, retries - 1) # Reconnect, resume operation, decrement retries
            else:
                BoardShim.log_message(LogLevels.LEVEL_ERROR.value, f'Biosensor board error: {board_init_error}')
                return
    #endregion init

    #region Main loop
    try:
        while True:
            # get execution start time for time delay
            start_time = time.time()
            
            # Execute all logic
            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Execute all Logic")
            data_dict = {type(logic).__name__ : logic.get_data_dict() for logic in logics}

            # Send messages from executed logic
            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sending")
            send_pairs = reporter.send(data_dict)
            for param_path, param_value in send_pairs:
                BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "{}:\t{:.3f}".format(param_path, param_value))
            
            # sleep based on refresh_rate
            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sleeping")
            execution_time = time.time() - start_time
            sleep_time = 1.0 / refresh_rate_hz - execution_time
            sleep_time = max(sleep_time, 0)
            time.sleep(sleep_time)
            
    except TimeoutError as e:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, f'Biosensor board error: {e}')
        reporter.send({Info.__name__ : {Info.CONNECTED:False}})
        board.release_session()
        return main_loop(args, reporter, retries) # Reconnect and start retry loop
    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Received interrupt signal! Shutting down...')
        board.stop_stream()
    
    # display disconnect and release old session
    reporter.send({Info.__name__ : {Info.CONNECTED:False}})
    board.release_session()
    #endregion Main loop


if __name__ == "__main__":
    main()
