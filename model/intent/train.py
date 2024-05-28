import pickle

from brainflow.board_shim import BoardShim
import numpy as np
import matplotlib.pyplot as plt
import random

import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import classification_report

from model import create_first_layer, create_last_layer
from pipeline import preprocess_data, extract_features

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
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    action_dict = recorded_data['action_dict']
    window_size = int(1.0 * sampling_rate)
    overlap = int(window_size * 0.93)

    # train and test windows seperated out to prevent overlap
    def windows_from_datas(datas):
        eegs = [data[eeg_channels] for data in datas]
        windows_per_session = [segment_data(eeg, window_size, overlap) for eeg in eegs]
        random.shuffle(windows_per_session)
        windows, windows_test = windows_per_session[1:], [windows_per_session[0]]
        return np.concatenate(windows), np.concatenate(windows_test)
    
    # dictionary of tuples containing train and test windows
    action_windows = {k:windows_from_datas(datas) for k, datas in action_dict.items()}

    ## extract the features from the windows
    def process_windows(windows):
        feature_windows = []
        for session_data in windows:
            preprocessed_data = preprocess_data(session_data, sampling_rate)
            features = extract_features(preprocessed_data)
            feature_windows.append(features)
        return feature_windows
    
    processed_windows = {k:(process_windows(train),process_windows(test)) for k, (train, test) in action_windows.items()}
    
    ## create train and test sets and labels
    i_train = np.concatenate([[k] * len(windows_train) for k, (windows_train, _) in processed_windows.items()])
    shuffle_indexes = list(range(len(i_train)))
    random.shuffle(shuffle_indexes)
    X_train = np.concatenate([windows_train for windows_train, _ in processed_windows.values()])[shuffle_indexes]
    y_train = to_categorical(i_train, num_classes=len(processed_windows))[shuffle_indexes]

    i_test = np.concatenate([[k] * len(windows_test) for k, (_, windows_test) in processed_windows.items()])
    X_test = np.concatenate([windows_test for windows_test, _ in processed_windows.values()])
    y_test = to_categorical(i_test, num_classes=len(processed_windows))

    print(X_train.shape)

    ## Compile the model
    classes = len(processed_windows)
    channels = len(eeg_channels)

    # load pretrained encoder and keep it static
    pretrained_encoder = keras.models.load_model("physionet_encoder.keras")
    pretrained_encoder.trainable = False
    
    model = Sequential([
        create_first_layer(channels),
        pretrained_encoder,
        create_last_layer(classes)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

    ## Set up EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=16, restore_best_weights=True, verbose=0)

    ## Train the model
    batch_size = 128
    epochs = X_train.shape[0] * 2
    fit_history = model.fit(
        X_train, y_train, 
        epochs=epochs, batch_size=batch_size, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping], 
        verbose=1
    )

    # print out model summary
    model.summary()

    ## Evaluate the model on the test set
    predictions_prob = model.predict(X_test)
    predictions = np.argmax(predictions_prob, axis=1)
    y_test_idxs = np.argmax(y_test, axis=1)
    print(classification_report(y_test_idxs, predictions))

    ## Plot history accuracy from model
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    ## Save models for realtime use
    model.save('shallow.keras')


if __name__ == "__main__":
    main()
    