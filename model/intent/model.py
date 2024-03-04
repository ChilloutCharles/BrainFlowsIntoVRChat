import pickle
import os

from model.intent.train import extract_features, preprocess_data

class EnsembleModel:
    def __init__(self):
        # get path for models no matter where this is run
        abs_script_path = os.path.abspath(__file__)
        abs_script_dir = os.path.dirname(abs_script_path)
        file_name = "models.ml"
        models_path = os.path.join(abs_script_dir, file_name)

        with open(models_path, "rb") as f:
            model_dict = pickle.load(f)
        self.feature_scaler = model_dict["feature_scaler"]
        self.classifier = model_dict["svm"]

    def predict(self, eeg_data, sampling_rate):
        pp_data = preprocess_data(eeg_data, sampling_rate)
        ft_data = extract_features(pp_data)

        # partial update scaler based on sampling rate to account for drift
        self.feature_scaler.partial_fit([ft_data], sample_weight=[1/sampling_rate])
        scaled_features = self.feature_scaler.transform([ft_data])

        action_idx = self.classifier.predict(scaled_features)
        
        return action_idx
