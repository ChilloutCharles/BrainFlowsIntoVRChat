import pickle
import numpy as np
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
        self.feature_pca = model_dict["feature_pca"]
        self.classifier = model_dict["svm"]
        self.action_idx = np.argwhere(self.classifier.classes_ == "button")[0] 

    def predict(self, eeg_data, sampling_rate):
        pp_data = preprocess_data(eeg_data, sampling_rate)
        ft_data = extract_features(pp_data)

        scaled_features = self.feature_scaler.transform([ft_data])
        fitted_features = self.feature_pca.transform(scaled_features)

        probabilities = self.classifier.predict_proba(fitted_features)[0]
        action_probability = probabilities[self.action_idx]
        
        return action_probability
