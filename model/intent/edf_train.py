import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as Scaler

from model import encoder, decoder

import pickle

# Load the data
data = np.load('dataset.pkl')

# Normalize the data
scaler = Scaler()
data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
with open('physionet_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Remove the last sample from each sequence, 161 -> 160
data = data[:, :, :-1]

# reshape array for use with model
data = data.transpose(0, 2, 1)
print(data.shape)

# Split the data into training and validation sets
X_train, X_val = train_test_split(data, test_size=0.2)

# Build the autoencoder
autoencoder = Sequential([
    encoder,
    decoder
])
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='huber')

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=0)

# Train the autoencoder with early stopping
batch_size = 128
epochs = 128
fit_history = autoencoder.fit(
    X_train, X_train, 
    epochs=epochs, batch_size=batch_size, 
    validation_data=(X_val, X_val), 
    callbacks=[early_stopping], verbose=1
)

#Save the model
print("Saving Model")
encoder = autoencoder.layers[0]
decoder = autoencoder.layers[1]

encoder.save('physionet_encoder.keras')
decoder.save('physionet_decoder.keras')

autoencoder.summary()

# Evaluate the model
print("Model evaluation:")
autoencoder.evaluate(X_val, X_val)


# View Reconstruction
import matplotlib.pyplot as plt
import random

reconstructed = autoencoder.predict(X_val)

i = random.randint(0, len(X_val) - 1)
j = random.randint(0, 64 - 1)
original = X_val[i][j].flatten()
reconstructed_sample = reconstructed[i][j].flatten()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(original)
plt.title('Original Data')
plt.subplot(1, 2, 2)
plt.plot(reconstructed_sample)
plt.title('Reconstructed Data')
plt.savefig('autoencoder_reconstruct.png')