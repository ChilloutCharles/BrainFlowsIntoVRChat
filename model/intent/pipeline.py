import keras
import os
import numpy as np
import pywt
import threading
from scipy import signal
from brainflow.data_filter import DataFilter, NoiseTypes, FilterTypes

try:
    from .constants import LOW_CUT, HIGH_CUT
except ImportError:
    from constants import LOW_CUT, HIGH_CUT


abs_script_path = os.path.abspath(__file__)
abs_script_dir = os.path.dirname(abs_script_path)

## preprocess and extract features to be shared between train and test
def preprocess_data(session_data, sampling_rate):
    for eeg_chan in range(len(session_data)):
        # remove line noise
        DataFilter.remove_environmental_noise(session_data[eeg_chan], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
        # bandpass to LOW_CUT, HIGH_CUT
        DataFilter.perform_bandpass(session_data[eeg_chan], sampling_rate, LOW_CUT, HIGH_CUT, 1, FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
    return session_data

def extract_features(preprocessed_data):
    # resample to expected 160hz sampling rate
    features = signal.resample(preprocessed_data, 160, axis=-1)
    # do multi resolution analysis
    features = np.array(pywt.mra(features, 'db4', level=2, transform='dwt'))
    # transpose to correct axis order
    features = features.transpose((2, 1, 0))
    return features

class Pipeline:
    def __init__(self):
        file_name = "shallow.keras"
        model_path = os.path.join(abs_script_dir, file_name)
        self.classifier = keras.models.load_model(model_path)
        
        self.latest_eeg_data = None
        self.latest_sampling_rate = None
        self.data_ready = threading.Event()  # Event to new data to be processed
        
        self.prediction = None
        self.prediction_ready = threading.Event()  # Event to signal first prediction

        self.lock = threading.Lock()
        
        self.prediction_thread = threading.Thread(target=self._predict_loop, daemon=True)
        self.prediction_thread.start()

    def _predict_loop(self):
        while True:
            # Get latest EEG data
            self.data_ready.wait()  # Wait till data is ready
            eeg_data = self.latest_eeg_data
            sampling_rate = self.latest_sampling_rate
            
            # Process data
            pp_data = preprocess_data(eeg_data, sampling_rate)
            ft_data = extract_features(pp_data)
            prediction_probs = self.classifier.predict(ft_data[None, ...], verbose=0)[0]

            # Store latest prediction
            with self.lock:
                self.prediction = prediction_probs
                self.prediction_ready.set()  # Signal that at least one prediction is available

    def predict(self, eeg_data, sampling_rate):
        # Overwrites the latest EEG data without blocking
        with self.lock:
            self.latest_eeg_data = eeg_data
            self.latest_sampling_rate = sampling_rate
            self.data_ready.set()
        
        self.prediction_ready.wait()  # Wait till first prediction is available
        with self.lock:
            self.data_ready.clear()  # Reset so predict loop doesn't cycle when there's no new data
            return self.prediction
