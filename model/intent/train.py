import pickle
import os
import argparse

from brainflow.board_shim import BoardShim
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy.lib.stride_tricks as stride_tricks

import keras
from keras.models import Sequential
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import classification_report

import tensorflow as tf

from model import create_classifier
from pipeline import preprocess_data, extract_features

import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import get_artifact_mask

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

    # get class count
    classes = len(action_dict)

    # unpack action data
    action_items = action_dict.items()

    # duplicate indices for each piece of data
    action_items = [([action_index]*len(a_datas), a_datas) for action_index, a_datas in action_items]
    
    # aggregate indices and datas in parallel
    a_index, a_data = zip(*action_items)
    a_index = sum(a_index, [])
    a_data = sum(a_data, [])

    # extract only eeg data
    a_data = np.stack([a_d[eeg_channels] for a_d in a_data])

    # split a_data to train and test
    split_idx = int(a_data.shape[-1] * args.test_size)
    train_data = a_data[:, :, split_idx:]
    test_data = a_data[:, :, :split_idx]
    
    # create sliding windows
    train_data = stride_tricks.sliding_window_view(train_data, axis=-1, window_shape=window_size)
    test_data = stride_tricks.sliding_window_view(test_data, axis=-1, window_shape=window_size)

    # dupe indices to align with flattening
    train_index = np.repeat(a_index, train_data.shape[2])
    test_index = np.repeat(a_index, test_data.shape[2])

    # transpose and flatten windows
    train_data = train_data.transpose(0, 2, 1, 3)
    train_data = train_data.reshape(-1, *train_data.shape[2:])
    test_data = test_data.transpose(0, 2, 1, 3)
    test_data = test_data.reshape(-1, *test_data.shape[2:])

    # shuffle training data
    shuffle_indices = np.arange(len(train_data))
    np.random.shuffle(shuffle_indices)
    train_data = train_data[shuffle_indices]
    train_index = train_index[shuffle_indices]

    # rename to X_data
    X_train = train_data
    X_test = test_data

    # preproccess
    X_train = np.stack([preprocess_data(d, sampling_rate) for d in X_train])
    X_test = np.stack([preprocess_data(d, sampling_rate) for d in X_test])

    # artifact rejection, remove corresponding indicies as well
    X_train_has_artifact = np.logical_not([np.sum(get_artifact_mask(d, sampling_rate)) for d in X_train])
    X_test_has_artifact = np.logical_not([np.sum(get_artifact_mask(d, sampling_rate)) for d in X_test])
    
    X_train = X_train[X_train_has_artifact]
    X_test = X_test[X_test_has_artifact]

    train_index = train_index[X_train_has_artifact]
    test_index = test_index[X_test_has_artifact]

    # extract features
    X_train = np.stack([extract_features(d) for d in X_train])
    X_test = np.stack([extract_features(d) for d in X_test])

    # create y and i data
    y_train = to_categorical(train_index, num_classes=classes)
    y_test = to_categorical(test_index, num_classes=classes)
    i_test = test_index

    # ## load pretrained encoder freeze it for use in perceptual loss
    # pretrained_encoder = keras.models.load_model("physionet_encoder.keras")

    # ## get class count and input shape from training data
    # input_shape = X_train.shape[1:]

    ## Create Model
    # model = create_classifier(pretrained_encoder, classes, input_shape)
    model = create_classifier(classes)

    ## Compile the model
    model.compile(optimizer='adamw', loss='categorical_crossentropy')

    ## Set up EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=0)

    ## Train the model
    batch_size = 256
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
    print("Model evaluation:")
    model.evaluate(X_test, y_test)
    print(classification_report(y_test_idxs, predictions))

    # Use the dark background style
    plt.style.use('dark_background')

    ## Plot history accuracy from model
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylim(0, 1)
    plt.savefig('loss.png')

    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE

    # Assuming `latent` has shape (samples, timesteps, channels, features)
    seq_model = Sequential(model.layers[:-1])
    latent = seq_model(X_test)
    
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
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=i_test, cmap='viridis', alpha=0.7)
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
    
