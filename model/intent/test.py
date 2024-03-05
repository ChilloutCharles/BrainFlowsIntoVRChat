import argparse
import time
import keras
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from train import extract_features, preprocess_data

window_seconds = 1.0

def main():
    ## Load CNN model
    model = keras.models.load_model("shallow.keras")

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
    sampling_size = int(sampling_rate * window_seconds)

    ema_value = 1/60 * 2

    board.prepare_session()
    board.start_stream()

    # 1. wait 2 seconds before starting
    print("Get ready in {} seconds".format(2))
    time.sleep(2)

    current_value = 0
    while True:
        data = board.get_current_board_data(sampling_size)

        eeg_data = data[eeg_channels]
        pp_data = preprocess_data(eeg_data, sampling_rate)
        ft_data = np.array(extract_features(pp_data))
        
        w_coeff_rows = ft_data.shape[0]
        w_coeff_size = ft_data.shape[1]
        ft_data = ft_data.reshape((1, w_coeff_rows, w_coeff_size, 1))

        prediction_probs = model.predict(ft_data, verbose=0)[0]
        target_value = prediction_probs[0].item()
        target_value = np.round(target_value, 3)

        current_value = current_value * (1 - ema_value) + target_value * ema_value

        string = "^" if current_value > 0.5 else "*"
        visual = string * int(50 * current_value)
        print("{:<10}{}".format(target_value, visual))

        time.sleep(1/60)

if __name__ == "__main__":
    main()
