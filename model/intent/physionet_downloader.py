import os
import wget
import zipfile

# create the dataset folder
os.makedirs('dataset', exist_ok=True)

# download dataset zip into folder
url = 'https://www.physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip'
filename = wget.download(url, out='dataset/')

# unzip downloaded zip file
with zipfile.ZipFile("dataset//eeg-motor-movementimagery-dataset-1.0.0.zip", 'r') as zip_ref:
    zip_ref.extractall('dataset/')