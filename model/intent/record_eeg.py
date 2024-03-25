import argparse
import time
import pickle

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

window_seconds = 10

def main():
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
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--actions', type=int, help='number of actions to record',
                        required=True)
    parser.add_argument('--sessions', type=int, help='number of sessions per action to record',
                    required=False, default=2)
    # board id by name or id
    parser.add_argument('--board-id', type=str, help='board id or name, check docs to get a list of supported boards',
                        required=True)
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

    action_count = args.actions
    sesesion_count = args.sessions

    ### Board Id selection ###
    try:
        master_board_id = int(args.board_id)
    except ValueError:
        master_board_id = BoardIds[args.board_id.upper()]

    board = BoardShim(master_board_id, params)

    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    sampling_size = sampling_rate * window_seconds

    action_dict = {action_idx:[] for action_idx in range(action_count)}
    record_data = {
        "board_id" : master_board_id,
        "window_seconds" : window_seconds
    }

    board.prepare_session()
    board.start_stream()

    # 1. wait 5 seconds before starting
    wait_seconds = 2
    print("Get ready in {} seconds".format(wait_seconds))
    time.sleep(wait_seconds)

    for i in action_dict:
        for _ in range(sesesion_count):
            input("Ready to record action {}. Press enter to continue".format(i))
            print("Think Action {} for {} seconds".format(i, window_seconds))
            time.sleep(window_seconds + 1)
            data = board.get_current_board_data(sampling_size)
            action_dict[i].append(data)
    record_data["action_dict"] = action_dict

    # save record data
    print("Saving Data")
    with open('recorded_eeg.pkl', 'wb') as f:
        pickle.dump(record_data, f)
    
    board.stop_stream()
    board.release_session()

if __name__ == "__main__":
    main()