import keras
import os
import numpy as np
import pywt
import joblib
from scipy import signal

from brainflow.data_filter import DataFilter, NoiseTypes, FilterTypes

abs_script_path = os.path.abspath(__file__)
abs_script_dir = os.path.dirname(abs_script_path)

## preprocess and extract features to be shared between train and test
def preprocess_data(session_data, sampling_rate):
    for eeg_chan in range(len(session_data)):
        # remove line noise
        DataFilter.remove_environmental_noise(session_data[eeg_chan], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
        # bandpass to alpha, beta, gamma, 80 for resample effect mitigation
        DataFilter.perform_bandpass(session_data[eeg_chan], sampling_rate, 8, 80, 4, FilterTypes.BUTTERWORTH.value, 0)
    return session_data

def extract_features(preprocessed_data):
    features  = []
    for eeg_row in preprocessed_data:
        # resample to match physionet dataset
        eeg_row = signal.resample(eeg_row, 160)
        features.append(eeg_row)
    features = np.stack(features, axis=-1)
    return features

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
