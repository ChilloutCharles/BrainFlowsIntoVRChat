'''
1. denoise and detrend data
2. perform wavelet transform
3. flatten output
4. train against svm
'''
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes, WaveletTypes

from sklearn import svm
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report, confusion_matrix 
from scipy.stats import kurtosis

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
    # detrend and denoise
    for eeg_chan in eeg_channels:
        assert len(data[eeg_chan]) == sampling_rate * record_seconds
        DataFilter.remove_environmental_noise(data[eeg_chan], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
        DataFilter.detrend(data[eeg_chan], DetrendOperations.LINEAR)
    
    # Independent Component Analysis and selection through kurtosis threshold
    _, _, _, source_signals = DataFilter.perform_ica(data, 3, eeg_channels)
    kurtosis_threshold = 3
    kurtoses = [np.abs(kurtosis(component)) for component in source_signals]
    kurtoses = [k if k < kurtosis_threshold else 0 for k in kurtoses]
    comp_idx = np.argmax(kurtoses)
    data[eeg_channels] = source_signals[comp_idx]
    
window_seconds = 1
data_size_multiplier = 0.7

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

train_size = int(len(train_indexes) * data_size_multiplier)
train_indexes = random.sample(train_indexes, train_size)

test_indexes = []
for intent_index in intent_indexes:
    pair = (test_slice_index, intent_index)
    test_indexes.append(pair)
test_size = int(len(test_indexes) * data_size_multiplier)
test_indexes = random.sample(test_indexes, test_size)

def create_data(indexes):
    pairs = [] # (X, y)
    for slice_index, sample_index in indexes:
        intent_slice = intent_data[slice_index]
        baseline_slice = baseline_data[slice_index]

        i, j = sample_index, sample_index + sampling_rate * window_seconds

        intent_wavelets = []
        baseline_wavelets = []
        # for eeg_chan in eeg_channels:
        eeg_chan = eeg_channels[0]
        intent_eeg = intent_slice[eeg_chan][i:j]
        intent_wavelet_coeffs, intent_lengths = DataFilter.perform_wavelet_transform(intent_eeg, WaveletTypes.DB4, 5)

        baseline_eeg = baseline_slice[eeg_chan][i:j]
        baseline_wavelet_coeffs, baseline_lengths = DataFilter.perform_wavelet_transform(baseline_eeg, WaveletTypes.DB4, 5)

        # only look at detailed parts which will contain the higher frequencies
        intent_wavelet_coeffs = intent_wavelet_coeffs[intent_lengths[0] : ]
        baseline_wavelet_coeffs = baseline_wavelet_coeffs[baseline_lengths[0] : ]

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

param_grid = {'C': [0.1, 1, 10, 100],   
              'gamma': [1, 0.1, 0.01, 'auto', 'scale']}
              # 'probability': [True]}  
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3, n_jobs=5) 
grid.fit(X, y) 
print(grid.best_params_) 
print(grid.best_estimator_) 
clf = grid.best_estimator_


# clf = svm.SVC(C=1, gamma='auto', probability=True)
# clf = svm.SVC(C=10, probability=True)
# clf.fit(X, y)

# test_X, test_y = create_data(test_indexes)
# preds = clf.predict(test_X)
# print(classification_report(test_y, preds))
print("Best cross-validation score:", grid.best_score_)
with open('model.ml', 'wb') as f:
    pickle.dump(clf, f)