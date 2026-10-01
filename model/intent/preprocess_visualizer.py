import argparse
import time
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, NoiseTypes, WaveletTypes, ThresholdTypes, FilterTypes, DetrendOperations

from scipy.signal import resample

import pywt

# from pipeline import Pipeline

window_seconds = 1.0

def main():
    ## Load pipeline
    # pipeline = Pipeline()

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

    ### Board Id selection ###
    try:
        master_board_id = int(args.board_id)
    except ValueError:
        master_board_id = BoardIds[args.board_id.upper()]

    board = BoardShim(master_board_id, params)

    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    eeg_channels = BoardShim.get_eeg_channels(master_board_id)
    sampling_size = int(sampling_rate * window_seconds)

    ema_value = 1/60 * 2

    board.prepare_session()
    board.start_stream()

    # 1. wait 2 seconds before starting
    print("Get ready in {} seconds".format(2))
    time.sleep(2)

    import matplotlib.pyplot as plt

    plt.ion()
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Create the image once
    img = ax.imshow(
        np.zeros((80, len(eeg_channels))),
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )

    ax.set_ylabel("Time / Sample")
    ax.set_xlabel("EEG Channel")
    ax.set_title("Live MRA Level 3 (approx discarded)")

    from scipy.ndimage import gaussian_filter1d

    
    # plt.colorbar(img, ax=ax)

    while True:
        data = board.get_current_board_data(sampling_size)

        for eeg_chan in eeg_channels:
            DataFilter.detrend(data[eeg_chan], DetrendOperations.LINEAR)
            DataFilter.remove_environmental_noise(data[eeg_chan], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
            DataFilter.perform_bandpass(data[eeg_chan], sampling_rate, 0.5, 40, 4, FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)

        eeg_data = data[eeg_channels]
        eeg_data = np.array(eeg_data)
        eeg_data = resample(eeg_data, 80, axis=1)
        eeg_data = np.array(pywt.mra(eeg_data, 'db4', 3, transform='dwt', axis=1)[1:]) 
        eeg_data = np.square(eeg_data)
        eeg_data = eeg_data / 100 
        eeg_data = eeg_data.transpose(2, 1, 0)
        eeg_data = np.flip(eeg_data, axis=0)

        eeg_data_smooth = gaussian_filter1d(
            eeg_data,
            sigma=2,
            axis=0
        )
    
        np.clip(eeg_data_smooth, 0, 1, out=eeg_data_smooth)
        img.set_data(eeg_data_smooth)

        fig.canvas.draw()
        fig.canvas.flush_events()

        # Give matplotlib time to update
        plt.pause(1/60)


        
        # time.sleep(1/60)

if __name__ == "__main__":
    main()
