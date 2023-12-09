import argparse
from cmath import pi
import time
import enum
import re
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes

from pythonosc.udp_client import SimpleUDPClient
from scipy.signal import find_peaks

import utils


class BAND_POWERS(enum.IntEnum):
    Gamma = 4
    Beta = 3
    Alpha = 2
    Theta = 1
    Delta = 0


OSC_BASE_PATH = '/avatar/parameters/'


class OSC_Path:
    RelaxAvg = OSC_BASE_PATH + 'osc_relax_avg'
    RelaxLeft = OSC_BASE_PATH + 'osc_relax_left'
    RelaxRight = OSC_BASE_PATH + 'osc_relax_right'
    FocusAvg = OSC_BASE_PATH + 'osc_focus_avg'
    FocusLeft = OSC_BASE_PATH + 'osc_focus_left'
    FocusRight = OSC_BASE_PATH + 'osc_focus_right'
    Battery = OSC_BASE_PATH + 'osc_battery_lvl'
    ConnectionStatus = OSC_BASE_PATH + 'osc_is_connected'
    HeartBps = OSC_BASE_PATH + 'osc_heart_bps'
    HeartBpm = OSC_BASE_PATH + 'osc_heart_bpm'
    OxygenPercent = OSC_BASE_PATH + 'osc_oxygen_percent'
    HueShift = OSC_BASE_PATH + 'osc_hueshift'


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
    master_board_id = board.get_board_id()
    eeg_channels = tryFunc(BoardShim.get_eeg_channels, master_board_id)
    eeg_names = tryFunc(BoardShim.get_eeg_names, master_board_id)
    sampling_rate = tryFunc(BoardShim.get_sampling_rate, master_board_id)
    battery_channel = tryFunc(BoardShim.get_battery_channel, master_board_id)
    time_channel = tryFunc(BoardShim.get_timestamp_channel, master_board_id)
    board.prepare_session()

    ### Board info ###
    BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, str(
        BoardShim.get_board_descr(master_board_id)))

    ### EEG Band Calculation Params ###
    current_value = np.zeros(6)
    eeg_window_size = 2
    ppg_window_size = eeg_window_size
    eeg_sample_size = eeg_window_size * sampling_rate

    ### Global Sample Size ###
    max_sample_size = eeg_sample_size

    # normalize ratios between -1 and 1.
    # Ratios are centered around 1.0. Tune scale to taste
    normalize_offset = -1
    normalize_scale = 1.3

    # Smoothing params
    smoothing_weight = 0.05

    ### Sort left and right eeg channels for left-right brain anaylsis ###
    idx_name_pairs = zip(eeg_channels, eeg_names)

    def right_check(tup):
        (eeg_chan, eeg_name) = tup
        eeg_num = int(''.join(re.findall(r'\d+', eeg_name)))
        return (eeg_chan, eeg_num % 2 == 0)
    idx_isright_pairs = list(map(right_check, idx_name_pairs))
    right_chans = [eeg_chan for (eeg_chan, isright)
                   in idx_isright_pairs if isright]
    left_chans = [eeg_chan for (eeg_chan, isright)
                  in idx_isright_pairs if not isright]

    ### Muse 2/S heartbeat support ###
    is_ppg = False
    if master_board_id in (BoardIds.MUSE_2_BOARD, BoardIds.MUSE_S_BOARD):
        board.config_board('p52')
        ppg_channels = BoardShim.get_ppg_channels(
            master_board_id, BrainFlowPresets.ANCILLARY_PRESET)
        ppg_sampling_rate = BoardShim.get_sampling_rate(
            master_board_id, BrainFlowPresets.ANCILLARY_PRESET)
        ppg_window_size = int(1024 / ppg_sampling_rate) + 1
        ppg_sample_size = ppg_window_size * ppg_sampling_rate
        max_sample_size = max(ppg_sample_size, eeg_sample_size)
        heart_bps = 80 / 60
        is_ppg = True

    ### Streaming Params ###
    update_speed = 1 / 4  # 4Hz update rate for VRChat OSC
    startup_time = max(eeg_window_size, ppg_window_size)
    board_timeout = 5

    try:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Intializing')
        board.start_stream(max_sample_size, args.streamer_params)
        time.sleep(startup_time)

        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Main Loop Started')
        while True:
            BoardShim.log_message(
                LogLevels.LEVEL_DEBUG.value, "Getting Board Data")
            data = board.get_current_board_data(max_sample_size)

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Timeout Check")
            time_data = data[time_channel]
            last_sample_time = time_data[-1]
            current_time = time.time()
            if current_time - last_sample_time > board_timeout:
                raise TimeoutError("Biosensor board timed out")

            battery_level = None if not battery_channel else data[battery_channel][-1]
            if battery_level:
                BoardShim.log_message(
                    LogLevels.LEVEL_DEBUG.value, "Battery: {}".format(battery_level))

            ### START EEG SECTION ###
            BoardShim.log_message(
                LogLevels.LEVEL_DEBUG.value, "Extract and get latest {} eeg samples".format(eeg_sample_size))

            def getlastn(arr):
                idx = min(len(arr), eeg_sample_size)
                return arr[-idx:]
            left_data = np.array(list(map(getlastn, data[left_chans])))
            right_data = np.array(list(map(getlastn, data[right_chans])))

            BoardShim.log_message(
                LogLevels.LEVEL_DEBUG.value, "Clean eeg samples")
            for eeg_row in left_data + right_data:
                DataFilter.remove_environmental_noise(eeg_row,
                                                      BoardShim.get_sampling_rate(
                                                          master_board_id),
                                                      NoiseTypes.FIFTY_AND_SIXTY.value)
                DataFilter.detrend(eeg_row, DetrendOperations.LINEAR)

            BoardShim.log_message(
                LogLevels.LEVEL_DEBUG.value, "Calculating Power Bands")
            left_ftv, _ = DataFilter.get_avg_band_powers(
                left_data, list(range(len(left_data))), sampling_rate, True)
            right_ftv, _ = DataFilter.get_avg_band_powers(
                right_data, list(range(len(right_data))), sampling_rate, True)

            BoardShim.log_message(
                LogLevels.LEVEL_DEBUG.value, "Calculating Metrics")
            alpha_avg = \
                0.5 * left_ftv[BAND_POWERS.Alpha] + \
                0.5 * right_ftv[BAND_POWERS.Alpha]
            beta_avg = \
                0.5 * left_ftv[BAND_POWERS.Beta] + \
                0.5 * right_ftv[BAND_POWERS.Beta]
            theta_avg = \
                0.5 * left_ftv[BAND_POWERS.Theta] + \
                0.5 * right_ftv[BAND_POWERS.Theta]

            paths = [
                OSC_Path.FocusLeft,
                OSC_Path.RelaxLeft,
                OSC_Path.FocusRight,
                OSC_Path.RelaxRight,
                OSC_Path.FocusAvg,
                OSC_Path.RelaxAvg
            ]
            numerator = np.array([
                left_ftv[BAND_POWERS.Beta],
                left_ftv[BAND_POWERS.Alpha],
                right_ftv[BAND_POWERS.Beta],
                right_ftv[BAND_POWERS.Alpha],
                beta_avg,
                alpha_avg,
            ])
            denominator = np.array([
                left_ftv[BAND_POWERS.Theta],
                left_ftv[BAND_POWERS.Theta],
                right_ftv[BAND_POWERS.Theta],
                right_ftv[BAND_POWERS.Theta],
                theta_avg,
                theta_avg,
            ])

            target_value = np.divide(numerator, denominator)
            target_value = utils.tanh_normalize(
                target_value, normalize_scale, normalize_offset)
            current_value = utils.smooth(
                current_value, target_value, smoothing_weight)

            hue_shift = utils.remap_cantor_pair(
                current_value[-2], current_value[-1])

            def map_band_power_tups(band_power):
                band_value = \
                    0.5 * left_ftv[band_power.value] + \
                    0.5 * right_ftv[band_power.value]
                osc_path = OSC_BASE_PATH + "osc_band_power_" + band_power.name.lower()
                return (osc_path, band_value)

            path_value_pairs = \
                list(zip(paths, current_value)) + \
                list(map(map_band_power_tups, BAND_POWERS)) + \
                [(OSC_Path.HueShift, hue_shift)]
            for (osc_path, osc_value) in path_value_pairs:
                BoardShim.log_message(
                    LogLevels.LEVEL_DEBUG.value, "{}:\t{:.3f}".format(osc_path, osc_value))

            ### END EEG SECTION ###

            ### START PPG SECTION ###
            if is_ppg:
                ppg_data = board.get_current_board_data(
                    max_sample_size, BrainFlowPresets.ANCILLARY_PRESET)

                def get_ppg_n(arr):
                    idx = min(len(arr), ppg_sample_size)
                    return arr[-idx:]
                ppg_ir = np.array(get_ppg_n(ppg_data[ppg_channels[1]]))
                ppg_red = np.array(get_ppg_n(ppg_data[ppg_channels[0]]))

                oxygen_level = DataFilter.get_oxygen_level(
                    ppg_ir, ppg_red, ppg_sampling_rate) * 0.01

                ### Brainflow Heart Example ###
                ### https://github.com/brainflow-dev/brainflow/blob/master/python_package/examples/tests/muse_ppg.py ###
                heart_rate = DataFilter.get_heart_rate(
                    ppg_ir, ppg_red, ppg_sampling_rate, 1024)
                target_bps = heart_rate / 60
                heart_bps = utils.smooth(
                    heart_bps, target_bps, smoothing_weight)
                heart_bpm = int(heart_bps * 60 + 0.5)

                BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "{}:\t{:.3f}".format(
                    OSC_Path.HeartBpm, heart_bpm))
                BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "{}:\t{:.3f}".format(
                    OSC_Path.HeartBps, heart_bps))
                BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "{}:\t{:.3f}".format(
                    OSC_Path.OxygenPercent, oxygen_level))
            ### END PPG SECTION ###

            ### OSC SECTION ###
            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sending")
            osc_client.send_message(OSC_Path.ConnectionStatus, True)

            for (osc_path, osc_value) in path_value_pairs:
                osc_client.send_message(osc_path, osc_value)

            if battery_level:
                osc_client.send_message(OSC_Path.Battery, battery_level)

            if is_ppg:
                osc_client.send_message(OSC_Path.HeartBpm, heart_bpm)
                osc_client.send_message(OSC_Path.HeartBps, heart_bps)
                osc_client.send_message(OSC_Path.OxygenPercent, oxygen_level)
            ### OSC SECTION ###

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sleeping")
            time.sleep(update_speed)

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
