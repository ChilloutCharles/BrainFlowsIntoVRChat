import pickle
import random

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes, WaveletTypes
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, SeparableConv1D, BatchNormalization, GlobalAveragePooling1D
from keras.optimizers import Adam

from sklearn.metrics import classification_report

## preprocess and extract features to be shared between train and test
def preprocess_data(session_data, sampling_rate):
    for eeg_chan in range(len(session_data)):
        DataFilter.detrend(session_data[eeg_chan], DetrendOperations.LINEAR)
        DataFilter.remove_environmental_noise(session_data[eeg_chan], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
    return session_data

def extract_features(preprocessed_data):
    features  = []
    for eeg_row in preprocessed_data:
        coeffs, _ = DataFilter.perform_wavelet_transform(eeg_row, WaveletTypes.DB4, 5)
        features.append(coeffs)
    return np.stack(features, axis=-1)

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

    ## Define the model
    # model inspired by thin mobilenet 
    # https://scholarworks.iupui.edu/server/api/core/bitstreams/a7fbc815-0f25-480a-bce1-0cb231238b66/content
    model = Sequential()
    model.add(SeparableConv1D(2, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(SeparableConv1D(4, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(2, activation='softmax'))

    ## Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    ## Train the model
    epochs = X_train.shape[0] * 2 // 10
    model.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=1, validation_data=(X_test, y_test))

    ## Evaluate the model on the test set
    predictions_prob = model.predict(X_test)
    predictions = np.argmax(predictions_prob, axis=1)
    y_test_idxs = np.argmax(y_test, axis=1)
    print(classification_report(y_test_idxs, predictions))

    ## Save models for realtime use
    model.save('shallow.keras')


if __name__ == "__main__":
    main()
    