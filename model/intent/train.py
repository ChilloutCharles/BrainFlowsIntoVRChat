'''
1. denoise and detrend data
2. perform wavelet transform
3. flatten output
4. train against svm
'''
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes, WaveletTypes

from sklearn import svm

import pickle
import random
import numpy as np

with open("recorded_eeg.pkl", "rb") as f:
    recorded_data = pickle.load(f)

board_id = recorded_data['board_id']
record_seconds = recorded_data['window_seconds']
intent_data = recorded_data['intent_data']
baseline_data = recorded_data['baseline_data']

sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)

for i, data in enumerate(intent_data + baseline_data):
    for eeg_chan in eeg_channels:
        assert len(data[eeg_chan]) == sampling_rate * record_seconds
        DataFilter.remove_environmental_noise(data[eeg_chan], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
        DataFilter.detrend(data[eeg_chan], DetrendOperations.LINEAR)


window_seconds = 2
train_size = 500
test_size = 50

slice_indexes = list(range(len(intent_data)))
random.shuffle(slice_indexes)
train_slice_indexes = slice_indexes[:-1]
test_slice_index = slice_indexes[-1]

intent_indexes = list(range(0, sampling_rate * (record_seconds - window_seconds)))

train_indexes = []
for intent_index in intent_indexes:
    for slice_index in train_slice_indexes:
        pair = (slice_index, intent_index)
        train_indexes.append(pair)
train_indexes = random.sample(train_indexes, train_size)

test_indexes = []
for intent_index in intent_indexes:
    pair = (test_slice_index, intent_index)
    test_indexes.append(pair)
test_indexes = random.sample(test_indexes, test_size)

def create_data(indexes):
    pairs = [] # (X, y)
    for slice_index, sample_index in indexes:
        intent_slice = intent_data[slice_index]
        baseline_slice = baseline_data[slice_index]

        i, j = sample_index, sample_index + sampling_rate * window_seconds

        intent_wavelets = []
        baseline_wavelets = []
        for eeg_chan in eeg_channels:
            intent_eeg = intent_slice[eeg_chan][i:j]
            intent_wavelet_coeffs, intent_lengths = DataFilter.perform_wavelet_transform(intent_eeg, WaveletTypes.DB3, 6)
            # intent_wavelet_coeffs = intent_wavelet_coeffs[intent_lengths[0]:]

            baseline_eeg = baseline_slice[eeg_chan][i:j]
            baseline_wavelet_coeffs, baseline_lengths = DataFilter.perform_wavelet_transform(baseline_eeg, WaveletTypes.DB3, 6)
            # baseline_wavelet_coeffs = baseline_wavelet_coeffs[baseline_lengths[0]:]

            intent_wavelets.append(intent_wavelet_coeffs)
            baseline_wavelets.append(baseline_wavelet_coeffs)

        intent_wavelets = np.array(intent_wavelets).flatten()
        baseline_wavelets = np.array(baseline_wavelets).flatten()

        pairs.append((intent_wavelets, "button"))
        pairs.append((baseline_wavelets, "baseline"))

    random.shuffle(pairs)

    X, y = zip(*pairs)
    return X, y

X, y = create_data(train_indexes)

clf = svm.SVC()
clf.fit(X,y)

test_X, test_y = create_data(test_indexes)
test_score = clf.score(test_X, test_y)
print(test_score)



