import pickle
import random

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes, WaveletTypes, FilterTypes

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.optimizers import Adam

from sklearn.metrics import classification_report

## preprocess and extract features to be shared between train and test
def preprocess_data(session_data, sampling_rate):
    for eeg_chan in range(len(session_data)):
        DataFilter.detrend(session_data[eeg_chan], DetrendOperations.LINEAR)
        DataFilter.remove_environmental_noise(session_data[eeg_chan], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
        DataFilter.perform_bandpass(session_data[eeg_chan], sampling_rate, 30, 100, 6, FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0) # only gamma and high gamma
    return session_data

def extract_features(preprocessed_data):
    features  = []
    for eeg_row in preprocessed_data:
        intent_wavelet_coeffs, _ = DataFilter.perform_wavelet_transform(eeg_row, WaveletTypes.DB4, 5)
        features.append(intent_wavelet_coeffs)
    return features

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
    ## load recorded data details
    with open("recorded_eeg.pkl", "rb") as f:
        recorded_data = pickle.load(f)

    board_id = recorded_data['board_id']
    intent_sessions = recorded_data['intent_data']
    baseline_sessions = recorded_data['baseline_data']
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    ## get recording session data
    intent_sessions = [data[eeg_channels] for data in intent_sessions]
    baseline_sessions = [data[eeg_channels] for data in baseline_sessions]
    
    ## Seperate train and validation sets
    random.shuffle(intent_sessions)
    random.shuffle(baseline_sessions)

    ## generate sample windows from recorded data
    intent_train, intent_valid = intent_sessions[:-1], [intent_sessions[-1]]
    baseline_train, baseline_valid = baseline_sessions[:-1], [baseline_sessions[-1]]

    window_size = int(1.0 * sampling_rate)
    overlap = int(window_size * 0.93)
    intent_train_windows = np.concatenate([segment_data(session, window_size, overlap) for session in intent_train])
    intent_valid_windows = np.concatenate([segment_data(session, window_size, overlap) for session in intent_valid])
    baseline_train_windows = np.concatenate([segment_data(session, window_size, overlap) for session in baseline_train])
    baseline_valid_windows = np.concatenate([segment_data(session, window_size, overlap) for session in baseline_valid])

    ## extract the features from the windows
    def process_windows(windows):
        feature_windows = []
        for session_data in windows:
            preprocessed_data = preprocess_data(session_data, sampling_rate)
            features = extract_features(preprocessed_data)
            feature_windows.append(features)
        return feature_windows
    
    intent_train_features = process_windows(intent_train_windows)
    baseline_train_features = process_windows(baseline_train_windows)
    intent_valid_features = process_windows(intent_valid_windows)
    baseline_valid_features = process_windows(baseline_valid_windows)

    ## Combine features from all sessions and create labels
    feature_windows =  np.array(
        baseline_train_features + 
        intent_train_features + 
        baseline_valid_features + 
        intent_valid_features )
    base_label = [0, 1]
    intent_label = [1, 0]
    labels = np.array( 
        [base_label] * len(baseline_train_features) +
        [intent_label] * len(intent_train_features) +
        [base_label] * len(baseline_valid_features) +
        [intent_label] * len(intent_valid_features) )
    split_idx = len(intent_train_features) + len(baseline_train_features)

    ## scale train and test sets 
    X_train = feature_windows[:split_idx]
    X_test = feature_windows[split_idx:]

    ## split labels
    y_train = labels[:split_idx]
    y_test = labels[split_idx:]

    ## shuffle train and test sets
    train_indexes = list(range(len(X_train)))
    random.shuffle(train_indexes)
    X_train = X_train[train_indexes]
    y_train = y_train[train_indexes]

    test_indexes = list(range(len(X_test)))
    random.shuffle(test_indexes)
    X_test = X_test[test_indexes]
    y_test = y_test[test_indexes]

    ## reshape data for Conv2D layer
    w_coeff_rows = X_train.shape[1]
    w_coeff_size = X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], w_coeff_rows, w_coeff_size, 1))
    X_test = X_test.reshape((X_test.shape[0], w_coeff_rows, w_coeff_size, 1))

    ## Define the model
    conv_input_shape = (w_coeff_rows, w_coeff_size, 1)
    temporal_kernel = (w_coeff_rows//2, w_coeff_size//w_coeff_rows)

    model = Sequential()
    model.add(Conv2D(32, temporal_kernel, activation='relu', input_shape=conv_input_shape))
    model.add(Dropout(0.5))  # prevent overfitting
    model.add(Flatten()) # flatten cnn output for Dense
    model.add(Dense(2, activation='softmax'))  # Output layer

    ## Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    ## Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

    ## Evaluate the model on the test set
    predictions_prob = model.predict(X_test)
    predictions = np.argmax(predictions_prob, axis=1)
    y_test_idxs = np.argmax(y_test, axis=1)
    print(classification_report(y_test_idxs, predictions))

    ## Save models for realtime use
    model.save('shallow.keras')


if __name__ == "__main__":
    main()
