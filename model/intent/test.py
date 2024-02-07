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

from train import extract_features, preprocess_data

window_seconds = 1

def main():
    ## Load model dictionary and extract models
    with open("models.ml", "rb") as f:
        model_dict = pickle.load(f)
    feature_scaler = model_dict["feature_scaler"]
    feature_pca = model_dict["feature_pca"]
    classifier = model_dict["svm"]

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
    time_channel = BoardShim.get_timestamp_channel(args.board_id)
    sampling_size = sampling_rate * window_seconds

    board.prepare_session()
    board.start_stream()

    # 1. wait 5 seconds before starting
    print("Get ready in {} seconds".format(window_seconds*2))
    time.sleep(window_seconds*2)

    current_value = 0
    while True:
        data = board.get_current_board_data(sampling_size)

        ## timeout check
        time_data = data[time_channel]
        if time_data[0] == time_data[-1]:
            raise TimeoutError("Board Timed Out")

        eeg_data = data[eeg_channels]
        pp_data = preprocess_data(eeg_data, sampling_rate)
        ft_data = extract_features(pp_data)
        scaled_features = feature_scaler.transform([ft_data])
        fitted_features = feature_pca.transform(scaled_features)
        pred_string = classifier.predict(fitted_features)[0]

        target_value = 1.0 if pred_string == 'button' else 0.0
        ema_value = 0.05
        current_value = current_value * (1 - ema_value) + target_value * ema_value

        string = "^" if current_value > 0.5 else "*"
        visual = string * int(50 * current_value)
        print("{:<10}{}".format(pred_string, visual))
        

        time.sleep(1/60)

if __name__ == "__main__":
    main()
