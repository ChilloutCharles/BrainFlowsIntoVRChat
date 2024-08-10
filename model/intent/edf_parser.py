import os
import mne
import numpy as np
import time

from multiprocess import Pool

if __name__ == '__main__':
    start_time = time.time()

    p = Pool(20)

    def find_edf_files(directory):
        edf_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".edf"):
                    edf_files.append(os.path.join(root, file))
        return edf_files

    datadir = "dataset"
    paths = find_edf_files(datadir)
    raw_list = list(p.map(lambda p: mne.io.read_raw_edf(p, preload=True), paths))

    def get_windows(raw):
        raw.notch_filter((50, 60))
        raw.filter(l_freq=2.0, h_freq=45.0)

        events, event_id = mne.events_from_annotations(raw)
        if len(event_id) != 3:
            return None

        # Identify T0, T1, T2
        selected_events = events[(events[:, 2] == event_id['T1']) | (events[:, 2] == event_id['T2']) | (events[:, 2] == event_id['T0'])]

        # Create epochs around these events
        epochs = mne.Epochs(raw, selected_events, tmin=0, tmax=1.0, preload=True, baseline=None)

        # Convert epochs to NumPy arrays
        return epochs.get_data()

    data = list(p.map(get_windows, raw_list))
    data = list(filter(lambda x: x is not None, data))

    # filter out bad data
    window_count = data[0].shape[0]
    sample_count = data[0].shape[-1]
    data = list(filter(lambda d: d.shape[0] == window_count and d.shape[-1] == sample_count, data))
    arr = np.array(data)

    # reshape data
    print(arr.shape)
    arr = arr.reshape(-1, 64, 161)

    with open("dataset.pkl", "wb") as f:
        np.save(f, arr)

    end_time = time.time() - start_time

    print(arr.shape, end_time/60)