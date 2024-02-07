import argparse
import time
import pickle

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes, WaveletTypes, FilterTypes, WindowOperations


from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np

import pickle

## preprocess and extract features to be shared between train and test

def preprocess_data(session_data, sampling_rate):
    for eeg_chan in range(len(session_data)):
        DataFilter.remove_environmental_noise(session_data[eeg_chan], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
        DataFilter.perform_bandpass(session_data[eeg_chan], sampling_rate, 30, 100, 6, FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0) # only beta and gamma
        DataFilter.detrend(session_data[eeg_chan], DetrendOperations.LINEAR)
    return session_data

def extract_features(preprocessed_data):
    features  = []
    for eeg_row in preprocessed_data:
        # intent_wavelet_coeffs, intent_lengths = DataFilter.perform_wavelet_transform(eeg_row, WaveletTypes.DB4, 5)
        # features.extend(intent_wavelet_coeffs)
        fft_data = DataFilter.perform_fft(eeg_row, WindowOperations.NO_WINDOW.value)
        features.extend(np.abs(fft_data))
    return np.array(features)

## helper function to generate windows
def segment_data(eeg_data, samples_per_window, overlap=0):
    _, total_samples = eeg_data.shape
    step_size = samples_per_window - overlap
    windows = []

    for start in range(0, total_samples - samples_per_window + 1, step_size):
        end = start + samples_per_window
        window = eeg_data[:, start:end]
        windows.append(window)

    return np.array(windows)


def main():
    ## define models to be saved for later
    param_grid = {'C': [0.1, 1, 10, 100],   
              'gamma': [1, 0.1, 0.01, 'auto', 'scale']}
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3, n_jobs=5) 
    feature_scaler = StandardScaler()
    feature_pca = PCA(n_components=0.95)

    ## load recorded data details
    with open("recorded_eeg.pkl", "rb") as f:
        recorded_data = pickle.load(f)

    board_id = recorded_data['board_id']
    intent_sessions = recorded_data['intent_data']
    baseline_sessions = recorded_data['baseline_data']
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    ## generate sample windows from recorded data
    window_size = 1 * sampling_rate
    intent_sessions = [data[eeg_channels] for data in intent_sessions]
    baseline_sessions = [data[eeg_channels] for data in baseline_sessions]

    overlap = int(window_size * 0.93)
    intent_windows = np.concatenate([segment_data(session, window_size, overlap) for session in intent_sessions])
    baseline_windows = np.concatenate([segment_data(session, window_size, overlap) for session in baseline_sessions])

    ## extract the features from the windows
    intent_feature_windows = []
    baseline_feature_windows = []

    for session_data in intent_windows:
        preprocessed_data = preprocess_data(session_data, sampling_rate)
        feature_windows = extract_features(preprocessed_data)
        intent_feature_windows.append(feature_windows)

    for session_data in baseline_windows:
        preprocessed_data = preprocess_data(session_data, sampling_rate)
        feature_windows = extract_features(preprocessed_data)
        baseline_feature_windows.append(feature_windows)

    ## Combine features from all sessions and create labels
    feature_windows = np.concatenate((intent_feature_windows, baseline_feature_windows))
    labels = np.array(["button"] * len(intent_feature_windows) + ["baseline"] * len(baseline_feature_windows))

    ## fit scaler and pca models
    scaled_feature_windows = feature_scaler.fit_transform(feature_windows)
    fitted_feature_windows = feature_pca.fit_transform(scaled_feature_windows)

    ## create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(fitted_feature_windows, labels, test_size=0.25, shuffle=True)

    ## scan for optimal hyperparams for svm model
    grid.fit(X_train, y_train)

    ## Extract svm and print results
    best_model = grid.best_estimator_
    print("Best Estimator:", best_model)

    y_pred = best_model.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    print(class_report)

    ## Save models for realtime use
    model_dict = {
        "feature_scaler" : feature_scaler,
        "feature_pca" : feature_pca,
        "svm" : best_model
    }
    with open('models.ml', 'wb') as f:
        pickle.dump(model_dict, f)


if __name__ == "__main__":
    main()
