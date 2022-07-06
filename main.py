import argparse
import time
import enum
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations

from pythonosc.udp_client import SimpleUDPClient
from scipy.signal import find_peaks


class BAND_POWERS(enum.IntEnum):
    Gamma = 4
    Beta = 3
    Alpha = 2
    Theta = 1
    Delta = 0


class OSC_Path:
    Relax = '/avatar/parameters/osc_relax_avg'
    Focus = '/avatar/parameters/osc_focus_avg'
    Battery = '/avatar/parameters/osc_battery_lvl'
    HeartBps = '/avatar/parameters/osc_heart_bps'
    HeartBpm = '/avatar/parameters/osc_heart_bpm'


def tanh_normalize(data, scale, offset):
    return np.tanh(scale * (data + offset))


def smooth(current_value, target_value, weight):
    current_value = (1.0 - weight) * current_value + weight * target_value
    return current_value


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

    ### EEG Band Calculation Params ###
    current_focus = 0
    current_relax = 0
    current_value = np.array([current_focus, current_relax])

    # normalize ratios between -1 and 1.
    # Ratios are centered around 1.0. Tune scale to taste
    normalize_offset = -1
    normalize_scale = 1.3

    # Smoothing params
    smoothing_weight = 0.05
    detrend_eeg = True

    ### EEG board setup ###
    board = BoardShim(args.board_id, params)
    master_board_id = board.get_board_id()
    eeg_channels = tryFunc(BoardShim.get_eeg_channels, master_board_id)
    sampling_rate = tryFunc(BoardShim.get_sampling_rate, master_board_id)
    battery_channel = tryFunc(BoardShim.get_battery_channel, master_board_id)
    ppg_channels = tryFunc(BoardShim.get_ppg_channels, master_board_id)
    board.prepare_session()

    ### Device specific commands ###
    if master_board_id == BoardIds.MUSE_2_BOARD or master_board_id == BoardIds.MUSE_S_BOARD:
        board.config_board('p50')

    ### EEG Streaming Params ###
    eeg_window_size = 2
    heart_window_size = 5
    update_speed = (250 - 3) * 0.001  # 4Hz update rate for VRChat OSC

    try:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Intializing')
        board.start_stream(450000, args.streamer_params)
        time.sleep(5)

        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Main Loop Started')
        while True:
            BoardShim.log_message(
                LogLevels.LEVEL_DEBUG.value, "Getting Board Data")
            data = board.get_current_board_data(
                eeg_window_size * sampling_rate)
            battery_level = None if not battery_channel else data[battery_channel][-1]

            ### START EEG SECTION ###
            BoardShim.log_message(
                LogLevels.LEVEL_DEBUG.value, "Calculating Power Bands")
            if detrend_eeg:
                for eeg_channel in eeg_channels:
                    DataFilter.detrend(data[eeg_channel],
                                       DetrendOperations.LINEAR)
            bands = DataFilter.get_avg_band_powers(
                data, eeg_channels, sampling_rate, True)
            feature_vector, _ = bands

            BoardShim.log_message(
                LogLevels.LEVEL_DEBUG.value, "Calculating Metrics")
            numerator = np.array(
                [feature_vector[BAND_POWERS.Beta], feature_vector[BAND_POWERS.Alpha]])
            denominator = np.array(
                [feature_vector[BAND_POWERS.Theta], feature_vector[BAND_POWERS.Theta]])
            target_value = np.divide(numerator, denominator)
            target_value = tanh_normalize(
                target_value, normalize_scale, normalize_offset)
            current_value = smooth(
                current_value, target_value, smoothing_weight)

            current_focus = current_value[0]
            current_relax = current_value[1]

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Focus: {:.3f}\tRelax: {:.3f}".format(
                current_focus, current_relax))
            ### END EEG SECTION ###

            ### START PPG SECTION ###
            if ppg_channels:
                BoardShim.log_message(
                    LogLevels.LEVEL_DEBUG.value, "Calculating BPM")
                data = board.get_current_board_data(
                    heart_window_size * sampling_rate)
                ir_data_channel = ppg_channels[1]
                ir_data = data[ir_data_channel]
                peaks, _ = find_peaks(ir_data)
                # divide by magic number 4, not sure why this works
                heart_bps = len(ir_data) / len(peaks) / 4
                heart_bpm = int(heart_bps * 60 + 0.5)
                BoardShim.log_message(
                    LogLevels.LEVEL_DEBUG.value, "BPS: {:.3f}\tBPM: {}".format(heart_bps, heart_bpm))
            ### END PPG SECTION ###

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sending")
            osc_client.send_message(OSC_Path.Focus, current_focus)
            osc_client.send_message(OSC_Path.Relax, current_relax)
            if battery_level:
                osc_client.send_message(OSC_Path.Battery, battery_level)
            if ppg_channels:
                osc_client.send_message(OSC_Path.HeartBps, heart_bps)
                osc_client.send_message(OSC_Path.HeartBpm, heart_bpm)

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
