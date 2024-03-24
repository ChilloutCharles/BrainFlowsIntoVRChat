import argparse
import time
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from pipeline import Pipeline

window_seconds = 1.0

def main():
    ## Load pipeline
    pipeline = Pipeline()

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
        target_value = pipeline.predict(eeg_data, sampling_rate)
        current_value = current_value * (1 - ema_value) + target_value * ema_value
        print(np.round(current_value, 3))

        time.sleep(1/60)

if __name__ == "__main__":
    main()
