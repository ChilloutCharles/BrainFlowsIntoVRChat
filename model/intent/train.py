import pickle
import os
import argparse

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

SAVE_FILENAME = "recorded_eeg"
SAVE_EXTENSION = ".pkl"

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
    ## Parse arguments for test and sample sizes
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=float, required=False, default=1.0, help='random sample proportion of recorded data to use')
    parser.add_argument('--test_size', type=float, required=False, default=0.1, help='proportion of sampled data to reserve for validation')
    args = parser.parse_args()
    
    # Load and merge recorded data details of all present data files
    # .pkl file merging code based off https://github.com/open-mmlab/mmaction2/issues/1431
    # This is unoptimized for repeated trainings of large filesets. But that is rare.
    
    # Start off by getting data from the first file
    with open(SAVE_FILENAME + SAVE_EXTENSION, 'rb') as f:
        initial_data = pickle.load(f)
        recorded_data = {
            'board_id' : initial_data['board_id'],
            'window_seconds' : initial_data['window_seconds'],
            'action_dict' : initial_data['action_dict']
        }
        action_count = len(initial_data['action_dict'])
    
    # Then get the action_dict from all of them
    print("Finding data files...")
    for d in os.listdir():
        if d.startswith(SAVE_FILENAME) and d.endswith(SAVE_EXTENSION):
            print("Opening " + d + "...")
            
            # Get data from file
            current_data = {}
            with open(d, 'rb') as f:
                current_data = pickle.load(f)
            action_dict = current_data['action_dict']

            # Check the number of actions recorded, and give a warning and option to continue if they are different than the first file
            current_actions = len(action_dict)
            if(current_actions !=  action_count):
                warning_option = input("WARNING! The amount of current actions ({}) is different than actions in {} ({}). Would you like to continue including this data? (Y/n)".format(action_count, d, current_actions))
                if warning_option != 'Y':
                    exit()

            # Go action by action and combine the data
            
            # Ignore the first file, it was added already
            if(d == SAVE_FILENAME + SAVE_EXTENSION):
                continue
            
            for i in range(action_count):
                # This creates a new entry in the action_dict. This should coincide with a warning in the console
                if(not action_dict.get(i)):
                    action_dict[i] = []
                    
                for action in action_dict[i]:
                    recorded_data['action_dict'][i].append(action)
        
    board_id = recorded_data['board_id']
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    action_dict = recorded_data['action_dict']
    window_size = int(1.0 * sampling_rate)
    overlap = window_size - 1 # maximum overlap!

    ## Segment time series data and split for train test sets
    def windows_from_datas(datas, test_size, sample_size):
        eegs = [data[eeg_channels] for data in datas]
        windows_per_session = [segment_data(eeg, window_size, overlap) for eeg in eegs]
        all_windows = np.concatenate(windows_per_session)

        # time based split: last windows used for validation 
        split_idx = int(len(all_windows) * (1 - test_size))
        windows_train = list(all_windows[:split_idx - overlap])
        windows_test = list(all_windows[split_idx:])

        # random sample windows for faster training
        windows_train = random.sample(windows_train, k=int(len(windows_train) * sample_size))
        windows_test = random.sample(windows_test, k=int(len(windows_test) * sample_size))

        return windows_train, windows_test

    action_windows = {action_label:windows_from_datas(datas, test_size=args.test_size, sample_size=args.sample_size) for action_label, datas in action_dict.items()}

    ## extract the features from the windows
    def process_windows(windows):
        feature_windows = []
        for session_data in windows:
            preprocessed_data = preprocess_data(session_data, sampling_rate)
            features = extract_features(preprocessed_data)
            feature_windows.append(features)
        return feature_windows
    
    processed_windows = {action_label:(process_windows(windows_train),process_windows(windows_test)) for action_label, (windows_train, windows_test) in action_windows.items()}
    
    ## create train and test sets and labels
    i_train = np.concatenate([[action_label] * len(windows_train) for action_label, (windows_train, _) in processed_windows.items()])
    shuffle_indexes = list(range(len(i_train)))
    random.shuffle(shuffle_indexes)
    X_train = np.concatenate([windows_train for windows_train, _ in processed_windows.values()])[shuffle_indexes]
    y_train = to_categorical(i_train, num_classes=len(processed_windows))[shuffle_indexes]

    i_test = np.concatenate([[action_label] * len(windows_test) for action_label, (_, windows_test) in processed_windows.items()])
    X_test = np.concatenate([windows_test for _ , windows_test in processed_windows.values()])
    y_test = to_categorical(i_test, num_classes=len(processed_windows))

    ## load pretrained encoder and keep it static
    pretrained_encoder = keras.models.load_model("physionet_encoder.keras")
    pretrained_encoder.trainable = False

    ## create channel expander/normalizer and classification layer
    classes = len(processed_windows)
    user_channels = len(eeg_channels)
    encoder_channels = pretrained_encoder.input_shape[-1]

    expandalizer = create_first_layer(user_channels, encoder_channels)
    classifier = create_last_layer(classes)
    
    ## Create Model
    model = Sequential([
        expandalizer,
        pretrained_encoder,
        classifier
    ])

    ## Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

    ## Set up EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=2*3, restore_best_weights=True, verbose=0)

    ## Train the model
    batch_size = 128
    epochs = 128
    fit_history = model.fit(
        X_train, y_train, 
        epochs=epochs, batch_size=batch_size, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping], 
        verbose=1
    )

    ## Print out model summary
    model.summary()
    
    ## Save models for realtime use
    model.save('shallow.keras')

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


if __name__ == "__main__":
    main()
    