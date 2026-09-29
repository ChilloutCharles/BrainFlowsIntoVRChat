import tensorflow as tf

## Limit GPU usage
MB_LIMIT = 12000
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set virtual device configuration for the first GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MB_LIMIT)]
        )
        print(f"Virtual GPU with {MB_LIMIT} MB memory limit created.")
    except RuntimeError as e:
        print("Error while creating virtual GPU:", e)
else:
    print("No GPU found.")

import pickle
import os
import argparse

from brainflow.board_shim import BoardShim
import numpy as np
import matplotlib.pyplot as plt
import random

import keras
from keras.models import Sequential
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

import tensorflow as tf

from model import create_classifier
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
    parser.add_argument('--test_size', type=float, required=False, default=0.2, help='proportion of sampled data to reserve for validation')
    args = parser.parse_args()
    
    # Load and merge recorded data details of all present data files
    # .pkl file merging code based off https://github.com/open-mmlab/mmaction2/issues/1431
    # This is unoptimized for repeated trainings of large filesets. But that is rare.
    
    # get all files with correct name and extension
    print("Finding data files...")
    file_names = [d for d in os.listdir() if d.startswith(SAVE_FILENAME) and d.endswith(SAVE_EXTENSION)]
    first = file_names[0]
    rest = file_names[1:]

    # Start off by getting data from the first file
    with open(first, 'rb') as f:
        print("Opening " + first + "...")
        initial_data = pickle.load(f)
        recorded_data = {
            'board_id' : initial_data['board_id'],
            'window_seconds' : initial_data['window_seconds'],
            'action_dict' : initial_data['action_dict']
        }
        action_count = len(initial_data['action_dict'])
    
    # Then get the action_dict from all of them
    for d in rest:
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

    ## extract the features from the windows
    def process_windows(windows):
        feature_windows = []
        for session_data in windows:
            preprocessed_data = preprocess_data(session_data, sampling_rate)
            features = extract_features(preprocessed_data)
            feature_windows.append(features)
        return feature_windows

    def windows_from_datas(datas):
        eegs = [data[eeg_channels] for data in datas]
        windows_per_session = [segment_data(eeg, window_size, overlap) for eeg in eegs]
        all_windows = np.concatenate(windows_per_session)
        all_windows = np.array(process_windows(all_windows))
        return all_windows

    action_dict = {action_label:windows_from_datas(datas) for action_label, datas in action_dict.items()}

    def dynamic_dataset_generator(test_ratio):
        # 1. Start with empty Python lists (fast to append to)
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        for action_label, windows in action_dict.items():
            item_count = len(windows)
            val_size = int(item_count * test_ratio)
            
            # Enforce safe bounds for selection
            max_split = item_count - val_size - 2 * window_size
            
            # Absolute zero safety check
            if max_split <= 0 or val_size == 0:
                left_chunk, middle_chunk, right_chunk = windows, windows[:0], windows[:0]
            else:
                split_idx = random.randrange(max_split)
                val_start = split_idx + window_size
                val_end = val_start + val_size
                right_start = val_end + window_size

                left_chunk = windows[:split_idx]
                middle_chunk = windows[val_start:val_end]
                right_chunk = windows[right_start:]

            X_train_list.extend([left_chunk, right_chunk])
            X_test_list.append(middle_chunk)

            # DYNAMIC LABEL GENERATION - Never mismatches or crashes!
            actual_train_count = len(left_chunk) + len(right_chunk)
            y_train_list.append(np.full(actual_train_count, action_label))
            y_test_list.append(np.full(len(middle_chunk), action_label))

        # 4. Concatenate ONCE at the end (highly optimized)
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        # 2. Shuffle X_train and y_train together using a random index permutation
        shuffled_indices = np.random.permutation(len(X_train))
        X_train = X_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        return X_train, y_train, X_test, y_test

    ## load the dataset first to get sizes    
    _, _, X_test, y_test = dynamic_dataset_generator(args.test_size)
    X_full, y_full, _, _ = dynamic_dataset_generator(0)

    ## load pretrained encoder freeze it for use in perceptual loss
    pretrained_encoder = keras.models.load_model("physionet_encoder.keras")

    ## get class count and input shape from training data
    classes = len(action_dict)
    input_shape = X_full.shape[1:]

    ## Epoch Finding Starting 
    batch_size = 256
    epochs = 15
    train_speed = 0.0005

    # Train on fresh model
    model = create_classifier(pretrained_encoder, classes, input_shape)
    model.compile(optimizer=AdamW(train_speed), loss='sparse_categorical_crossentropy')

    never_stopping = EarlyStopping(monitor='loss', patience=np.inf, restore_best_weights=True, verbose=0)
    fit_history = model.fit(
        X_full, y_full, 
        epochs=epochs, batch_size=batch_size,
        callbacks=[never_stopping],
        verbose=1
    )

    ## Print out model summary
    model.summary()
    
    ## Save models for realtime use
    model.save('shallow.keras')

    ## Evaluate the model on the test set
    X_eval = X_test
    y_eval = y_test
    
    
    print("Model evaluation:")
    model.evaluate(X_eval, y_eval)

    predictions_prob = model.predict(X_eval)
    predictions = np.argmax(predictions_prob, axis=1)
    print(classification_report(y_eval, predictions))

    # Use the dark background style
    plt.style.use('dark_background')

    ## Plot history accuracy from model
    plt.plot(fit_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.ylim(0, 1)
    plt.savefig('loss.png')

    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE

    # Assuming `latent` has shape (samples, timesteps, channels, features)
    seq_model = Sequential(model.layers[:-1])
    latent = seq_model(X_eval)
    
    # Step 1: Reshape to 2D by flattening the last three dimensions
    samples = latent.shape[0]  # Number of samples
    # Flatten the timesteps, channels, and features dimensions into a single dimension
    latent_flat = tf.reshape(latent, (samples, -1)).numpy()

    # Step 2: Standardize the flattened data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(latent_flat)

    # Step 3: Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    # Step 5: Plot the t-SNE result
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_eval, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Labels')
    plt.title('t-SNE Visualization of Labeled Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

    # Set the scatter plot aspect to be square
    plt.axis('square')
    plt.savefig('tsne.png')


if __name__ == "__main__":
    main()
    
