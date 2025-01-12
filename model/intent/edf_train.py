import numpy as np
import joblib

from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping
from keras.losses import Huber, MeanSquaredError as MSE

from model import MaskedAutoEncoder

# Load the data
print('Loading data...')
data = joblib.load('dataset.pkl')

# Split the data into training and validation sets
print('Data Loaded. Processing:', data.shape)
np.random.shuffle(data)

print('Shuffled. Splitting...')
pivot = int(data.shape[0] * 0.2)
X_val = data[:pivot]
X_train = data[pivot:]
del data

# Setup variables for batch generation and training
print('Data Processed. Setting up...')
batch_size = 512
epochs = 256
train_steps = X_train.shape[0] // batch_size
test_steps = X_val.shape[0] // batch_size

# set up train and val generators
def batch_generator(X, batch_size):
    x_count = len(X)
    while True:
        for i in range(0, x_count, batch_size):
            x = X[i:i + batch_size]
            yield x, x

train_generator = batch_generator(X_train, batch_size)
val_generator = batch_generator(X_val, batch_size)

# Build the autoencoder
input_shape = X_train.shape[1:]
autoencoder = MaskedAutoEncoder(mask_ratio=0.8, input_shape=input_shape, patch_shape=(5, 4))
autoencoder.compile(optimizer=AdamW(0.001), loss='mse')

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)

# Train the autoencoder with early stopping
print('Setup Complete. Training...')
fit_history = autoencoder.fit(
    x = train_generator, y = None,
    epochs=epochs,
    validation_data=val_generator, 
    callbacks=[early_stopping],
    steps_per_epoch=train_steps,
    validation_steps=test_steps,
    verbose=1
)

#Save the model
print("Saving Model")
encoder = autoencoder.assemble_feature_extractor()
encoder.save('physionet_encoder.keras')
encoder.summary()

# Evaluate the model
print("Model evaluation:")
autoencoder.evaluate(
    x=val_generator, y=None,
    steps=test_steps
)

# View Reconstruction
import matplotlib.pyplot as plt
import random

r_index = np.random.choice(X_val.shape[0])
X_val = X_val[[r_index]]
reconstructed = autoencoder.predict(X_val)

# reconstruct original signal shape by summing the mra levels
X_val = np.sum(X_val, axis=-1)
reconstructed = np.sum(reconstructed, axis=-1)

# transpose back into time last
X_val = X_val.transpose(0, 2, 1)
reconstructed = reconstructed.transpose(0, 2, 1)

i = 0
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