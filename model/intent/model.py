import keras
import os
import numpy as np

from model.intent.train import extract_features, preprocess_data

class EnsembleModel:
    def __init__(self):
        # get path for models no matter where this is run
        abs_script_path = os.path.abspath(__file__)
        abs_script_dir = os.path.dirname(abs_script_path)
        file_name = "shallow.keras"
        model_path = os.path.join(abs_script_dir, file_name)
        
        self.classifier = keras.models.load_model(model_path)

    def predict(self, eeg_data, sampling_rate):
        pp_data = preprocess_data(eeg_data, sampling_rate)
        ft_data = np.array(extract_features(pp_data))
        
        w_coeff_rows = ft_data.shape[0]
        w_coeff_size = ft_data.shape[1]
        ft_data = ft_data.reshape((1, w_coeff_rows, w_coeff_size, 1))

        prediction_probs = self.classifier.predict(ft_data, verbose=0)[0]
        action_prob = prediction_probs[0].item()

        return action_prob
