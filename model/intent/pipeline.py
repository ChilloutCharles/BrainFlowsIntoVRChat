import keras
import os
import numpy as np
from scipy import signal

from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes, FilterTypes
import pickle

abs_script_path = os.path.abspath(__file__)
abs_script_dir = os.path.dirname(abs_script_path)

with open(os.path.join(abs_script_dir, 'physionet_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

## preprocess and extract features to be shared between train and test
def preprocess_data(session_data, sampling_rate):
    for eeg_chan in range(len(session_data)):
        DataFilter.detrend(session_data[eeg_chan], DetrendOperations.LINEAR)
        DataFilter.remove_environmental_noise(session_data[eeg_chan], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
        DataFilter.perform_lowpass(session_data[eeg_chan], sampling_rate, 80, 4, FilterTypes.BUTTERWORTH.value, 0) # resample effect mitigation
    return session_data

def extract_features(preprocessed_data):
    features  = []
    for eeg_row in preprocessed_data:
        # resample and reshape to match physionet dataset
        feature = signal.resample(eeg_row, 160)
        feature = scaler.transform(feature.reshape(-1, 1)).reshape(feature.shape)
        features.append(feature)
    return np.stack(features, axis=-1)

class Pipeline:
    def __init__(self):
        file_name = "shallow.keras"
        model_path = os.path.join(abs_script_dir, file_name)
        self.classifier = keras.models.load_model(model_path)

    def predict(self, eeg_data, sampling_rate):
        pp_data = preprocess_data(eeg_data, sampling_rate)
        ft_data = extract_features(pp_data)
        prediction_probs = self.classifier.predict(ft_data[None, ...], verbose=0)[0]
        return prediction_probs
