import os, gc
import mne
import numpy as np
import time
import pywt
import joblib

from multiprocess import Pool
from sklearn.preprocessing import StandardScaler as Scaler

from constants import LOW_CUT, HIGH_CUT

if __name__ == '__main__':
    start_time = time.time()

    with Pool(20) as p:
        def find_edf_files(directory):
            edf_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".edf"):
                        edf_files.append(os.path.join(root, file))
            return edf_files

        datadir = "dataset"
        paths = find_edf_files(datadir)
        raw_list = list(p.map(mne.io.read_raw_edf, paths))

        def get_windows(raw):
            raw.load_data()
            
            # preprocessing
            raw.notch_filter(freqs=50)
            raw.notch_filter(freqs=60)
            raw.filter(l_freq=LOW_CUT, h_freq=HIGH_CUT, method='iir') # using iir to match brainflow filter
            
            events, event_id = mne.events_from_annotations(raw)
            if len(event_id) != 3:
                return None
            
            sfreq = raw.info['sfreq'] 

            # Identify T0, T1, T2
            selected_events = events[(events[:, 2] == event_id['T1']) | (events[:, 2] == event_id['T2']) | (events[:, 2] == event_id['T0'])]

            # Create Synthetic Events to get the whole minute
            start_event_sample = selected_events[0, 0]
            synthetic_events = np.array([
                [int(start_event_sample + i * sfreq), 0, 1]  # Each event 1 second apart
                for i in range(60 * 2)
            ])

            # Create epochs around these events
            epochs = mne.Epochs(raw, synthetic_events, tmin=0, tmax=(1.0-1/sfreq), preload=True, baseline=None)

            # Convert epochs to NumPy arrays
            data = epochs.get_data()

            # Convert to NumPy array (optional: depends on downstream needs)
            return np.array(data)

        data = list(p.map(get_windows, raw_list))
        data = list(filter(lambda x: x is not None, data))

    # filter out bad data
    window_count = data[0].shape[0]
    sample_count = data[0].shape[-1]
    data = list(filter(lambda d: d.shape[0] == window_count and d.shape[-1] == sample_count, data))
    data = np.array(data)

    # normalize
    print('Normalizing...')
    scaler = Scaler()
    entries = data.shape[0]
    for batch in data:
        scaler.partial_fit(batch.reshape(-1, 1))
    for i in range(entries):
        data[i] = scaler.transform(data[i].reshape(-1, 1)).reshape(data[i].shape)

    # multi-resolution analysis
    print('MRA...', data.shape)
    with Pool(16) as p:
        level = 2
        d_shape = (data.shape[0], level + 1, *data.shape[1:])
        d = np.memmap('large_arr.tmp', dtype='float32', mode='w+', shape=d_shape)

        def create_mra_func(level):
            return lambda row: np.array(pywt.mra(row, 'db4', level, transform='dwt'))
        generator = p.imap_unordered(create_mra_func(level), data)
        
        for i, result in enumerate(generator):
            print(f"Appending result {i/d_shape[0]}")
            d[i] = result
            
            if i % 100 == 0:
                d.flush()
        d.flush()
        data = d

    # reshape data
    print('Reshaping', data.shape)
    data = data.transpose((0, 2, 4, 3, 1))
    data = data.reshape(-1, *data.shape[-3:])
    print('Reshaped', data.shape)

    # serialize
    print('Saving...')
    joblib.dump(scaler, 'scaler.gz')
    joblib.dump(data, 'dataset.pkl')

    # cleanup
    del data
    gc.collect()
    os.remove('large_arr.tmp')

    end_time = time.time() - start_time
    print('Runtime (mins)', end_time/60)