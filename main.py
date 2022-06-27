import argparse
import time
import enum
import numpy as np
import scipy.stats as st

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels
from brainflow.data_filter import DataFilter, DetrendOperations

from pythonosc.udp_client import SimpleUDPClient


class BAND_POWERS(enum.IntEnum):
    Gamma = 4
    Beta = 3
    Alpha = 2
    Theta = 1
    Delta = 0


class OSC_Path:
    Relax = '/avatar/parameters/osc_relax_avg'
    Focus = '/avatar/parameters/osc_focus_avg'


def tanh_normalize(data, scale, offset):
    return np.tanh(scale * (data + offset))


def smooth(current_value, target_value, weight):
    current_value = (1.0 - weight) * current_value + weight * target_value
    return current_value


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

    ### EEG Band Calculation Params ###
    # normalize ratios between -1 and 1.
    # Ratios are centered around 1.0. Tune scale to taste
    offset = -1
    relax_scale = 1.3
    focus_scale = 1.3

    ### Smoothing params ###
    relax_weight = 0.05
    focus_weight = 0.05

    ### EEG board setup ###
    board = BoardShim(args.board_id, params)
    master_board_id = board.get_board_id()
    eeg_channels = BoardShim.get_eeg_channels(master_board_id)
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    board.prepare_session()

    ### EEG Streaming Params ###
    window_size = 2
    update_speed = (250 - 3) * 0.001  # 4Hz update rate for VRChat OSC
    num_points = window_size * sampling_rate
    current_focus = 0
    current_relax = 0

    try:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Intializing')
        board.start_stream(num_points, args.streamer_params)
        time.sleep(5)

        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Main Loop Started')
        while True:
            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sampling")
            data = board.get_current_board_data(num_points)
            for eeg_channel in eeg_channels:
                DataFilter.detrend(data[eeg_channel],
                                   DetrendOperations.LINEAR)
            bands = DataFilter.get_avg_band_powers(
                data, eeg_channels, sampling_rate, True)
            feature_vector, _ = bands

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Calculating")
            target_focus = feature_vector[BAND_POWERS.Beta] / \
                feature_vector[BAND_POWERS.Theta]
            target_focus = tanh_normalize(target_focus, focus_scale, offset)
            current_focus = smooth(current_focus, target_focus, focus_weight)

            target_relax = feature_vector[BAND_POWERS.Alpha] / \
                feature_vector[BAND_POWERS.Theta]
            target_relax = tanh_normalize(target_relax, relax_scale, offset)
            current_relax = smooth(current_relax, target_relax, relax_weight)

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Focus: {:.3f}\tRelax: {:.3f}".format(
                current_focus, current_relax))

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sending")
            osc_client.send_message(OSC_Path.Focus, current_focus)
            osc_client.send_message(OSC_Path.Relax, current_relax)

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sleeping")
            time.sleep(update_speed)

    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Shutting down')
    finally:
        ### Cleanup ###
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
