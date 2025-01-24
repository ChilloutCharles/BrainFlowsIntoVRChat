import argparse
import time
import pickle
import os

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from sound_helper import SoundHelper

SAVE_FILENAME = 'recorded_eeg'
SAVE_EXTENSION = '.pkl'

def main():
    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, required=False, default=0, help='timeout for device discovery or connection')
    parser.add_argument('--ip-port', type=int, required=False, default=0, help='ip port')
    parser.add_argument('--ip-protocol', type=int, required=False, default=0, help='ip protocol, check IpProtocolType enum')
    parser.add_argument('--ip-address', type=str, required=False, default='', help='ip address')
    parser.add_argument('--serial-port', type=str, required=False, default='', help='serial port')
    parser.add_argument('--mac-address', type=str, required=False, default='', help='mac address')
    parser.add_argument('--other-info', type=str, required=False, default='', help='other info')
    parser.add_argument('--serial-number', type=str, required=False, default='', help='serial number')
    parser.add_argument('--file', type=str, required=False, default='', help='file',)
    parser.add_argument('--actions', type=int, required=True, help='number of actions to record')
    parser.add_argument('--sessions', type=int, required=False, default=2, help='number of sessions per action to record')
    parser.add_argument('--window-length', type=int, required=False, default=10, help='length in seconds of eeg data pulled per session')
    parser.add_argument('--window-buffer', type=int, required=False, default=2, help='time in seconds before eeg data is recorded each session (delay after hitting enter)')
    parser.add_argument('--overwrite', type=int, required=False, default=0, help='1 to overwrite/remove old recordings, 0 to add results as an additional data file')
    parser.add_argument('--board-id', type=str, required=True, help='board id or name, check docs to get a list of supported boards')
    parser.add_argument('--start-delay', type=int, required=False, help='delay between pressing enter and recording start', default=3)
    parser.add_argument('--enable-sounds', type=bool, required=False, help='enables sound indicators for starting / stopping a recording', default=True)
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
    session_count = args.sessions
    window_length = args.window_length
    window_buffer = args.window_buffer

    recording_delay = args.start_delay
    sounds_enabled = args.enable_sounds

    sound_helper = SoundHelper(sounds_enabled)
    
    doOverwrite = args.overwrite == 1

    ### Board Id selection ###
    try:
        master_board_id = int(args.board_id)
    except ValueError:
        master_board_id = BoardIds[args.board_id.upper()]

    board = BoardShim(master_board_id, params)

    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    sampling_size = sampling_rate * window_length

    action_dict = {action_idx:[] for action_idx in range(action_count)}
    record_data = {
        "board_id" : master_board_id,
        "window_seconds" : window_length
    }

    board.prepare_session()
    board.start_stream()

    # Wait 5 seconds before starting
    wait_seconds = 2
    print("Get ready in {} seconds".format(wait_seconds))
    time.sleep(wait_seconds)

    for i in action_dict:
        for _ in range(session_count):
            input("Ready to record action {}. Press enter to continue".format(i))
            
            # Wait j seconds before starting
            j = recording_delay
            while j > 0:
                sound_helper.play_sound(u"sounds/boop.wav")
                print(f"Recording in {j}...", end="\r")
                time.sleep(1)
                j -= 1
            sound_helper.play_sound(u"sounds/start.wav")
                
            print("Think Action {} for {} seconds".format(i, window_length + window_buffer))
            time.sleep(window_length + window_buffer)
            data = board.get_current_board_data(sampling_size)
            action_dict[i].append(data)

            sound_helper.play_sound(u"sounds/done.wav")
    record_data["action_dict"] = action_dict

    # Save record data
    print("Saving Data")
    
    # If overwriting, delete any recording file with a .pkl extension
    if(doOverwrite):
        for filename in os.listdir():
            if filename.startswith(SAVE_FILENAME) and filename.endswith(SAVE_EXTENSION):
                os.remove(filename)
        filename_target = SAVE_FILENAME + SAVE_EXTENSION
    else:  
        # If not overwriting, find the next available filename
        current_number = 0;  
        while(True):
            
            # Create the filenames that may exist in the directory, starting with record_eeg.pkl, then record_eeg1.pkl, etc.
            current_filename = create_filename(current_number)
                
            if(not os.path.isfile(current_filename)):
                break
            current_number += 1
        filename_target = current_filename
    
    with open(filename_target, 'wb') as f:
        pickle.dump(record_data, f)
    
    board.stop_stream()
    board.release_session()

def create_filename(number):
    filename = SAVE_FILENAME
    if(number != 0):
        filename += str(number)
    return filename + SAVE_EXTENSION

if __name__ == "__main__":
    main()