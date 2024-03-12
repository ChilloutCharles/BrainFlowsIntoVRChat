import keras
import os
import numpy as np

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes, WaveletTypes

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

class Pipeline:
    def __init__(self):
        # get path for models no matter where this is run
        abs_script_path = os.path.abspath(__file__)
        abs_script_dir = os.path.dirname(abs_script_path)
        file_name = "shallow.keras"
        model_path = os.path.join(abs_script_dir, file_name)
        
        self.classifier = keras.models.load_model(model_path)

    def predict(self, eeg_data, sampling_rate):
        pp_data = preprocess_data(eeg_data, sampling_rate)
        ft_data = extract_features(pp_data)
        prediction_probs = self.classifier.predict(ft_data[None, ...], verbose=0)[0]
        action_prob = prediction_probs[0].item()
        return action_prob
