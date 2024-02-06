import argparse
import time
import pickle

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes, WaveletTypes


from sklearn import svm
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import numpy as np

import pickle

window_seconds = 1

def main():
    with open("model.ml", "rb") as f:
        clf = pickle.load(f)
    print(clf.classes_)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
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
    params.master_board = args.master_board

    board = BoardShim(args.board_id, params)

    sampling_rate = BoardShim.get_sampling_rate(args.board_id)
    eeg_channels = BoardShim.get_eeg_channels(args.board_id)
    sampling_size = sampling_rate * window_seconds

    board.prepare_session()
    board.start_stream()

    # 1. wait 5 seconds before starting
    print("Get ready in {} seconds".format(window_seconds))
    time.sleep(2)

    current_value = 0

    while True:
        data = board.get_current_board_data(sampling_size)

        # detrend and denoise
        for eeg_chan in eeg_channels:
            DataFilter.remove_environmental_noise(data[eeg_chan], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
            DataFilter.detrend(data[eeg_chan], DetrendOperations.LINEAR)
        
        # Independent Component Analysis and filter through kurtosis threshold
        ica = FastICA(3)
        signals = ica.fit_transform(data[eeg_channels])
        mix_matrix = ica.mixing_
        kurt = kurtosis(signals, axis=0, fisher=True)
        remove_indexes = np.where(kurt > 3)[0]
        signals[:, remove_indexes] = 0
        data[eeg_channels] = np.dot(signals, mix_matrix.T) + ica.mean_

        intent_wavelets = []
        for eeg_channel in eeg_channels:
        # eeg_channel = eeg_channels[0]
        # Wavelet Transform on signal
            intent_eeg = data[eeg_channel]
            intent_wavelet_coeffs, intent_lengths = DataFilter.perform_wavelet_transform(intent_eeg, WaveletTypes.DB4, 5)

            # only look at detailed parts which will contain the higher frequencies
            intent_wavelet_coeffs = intent_wavelet_coeffs[intent_lengths[0] : ]

            intent_wavelets.append(intent_wavelet_coeffs)

        intent_wavelets = np.array(intent_wavelets).flatten()

        pred_string = clf.predict([intent_wavelets])
        target_value = 1.0 if pred_string[0] == 'button' else 0.0
        ema_value = 0.05
        current_value = current_value * (1 - ema_value) + target_value * ema_value

        string = "^" if current_value > 0.5 else "*"
        visual = string * int(50 * current_value)
        print("{}\t{}".format(pred_string, visual))

        time.sleep(1/60)

if __name__ == "__main__":
    main()
