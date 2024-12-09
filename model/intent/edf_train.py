import numpy as np
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as Scaler

from model import auto_encoder

# Load the data
data = np.load('dataset.pkl')

# Normalize the data
scaler = Scaler()
data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

# Remove the last sample from each sequence, 161 -> 160
data = data[:, :, :-1]

# reshape array for use with model
data = data.transpose(0, 2, 1)
print(data.shape)

# Split the data into training and validation sets
X_train, X_val = train_test_split(data, test_size=0.2)

# Build the autoencoder
autoencoder = auto_encoder
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)

# Train the autoencoder with early stopping
batch_size = 256 * 2
epochs = 128
fit_history = autoencoder.fit(
    X_train, X_train, 
    epochs=epochs, batch_size=batch_size, 
    validation_data=(X_val, X_val), 
    callbacks=[early_stopping], 
    verbose=1
)

#Save the model
print("Saving Model")
encoder = autoencoder.encoder
decoder = autoencoder.decoder

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

X_val = X_val.transpose(0, 2, 1)
reconstructed = reconstructed.transpose(0, 2, 1)

i = random.randint(0, len(X_val) - 1)
js = list(range(0, 64))
random.shuffle(js)
js = js[:4]  # Select 4 random channels
original = X_val[i][js]
reconstructed_sample = reconstructed[i][js]

# Use the dark background style
plt.style.use('dark_background')

# Create subplots for each selected channel
fig, axs = plt.subplots(len(js), 1, figsize=(9, 16))

# Plot the original and reconstructed signals for each channel
for idx, j in enumerate(js):
    axs[idx].plot(original[idx], label='original')
    axs[idx].plot(reconstructed_sample[idx], label='reconstructed')
    axs[idx].set_title(f'Channel {j} Reconstruction Comparison')
    axs[idx].legend(loc='upper left')

plt.tight_layout()
plt.savefig('autoencoder_reconstruct.png')