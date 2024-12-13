import numpy as np
import joblib

from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as Scaler

from model import MaskedAutoEncoder

# Load the data
data = np.load('dataset.pkl')

# Normalize the data
scaler = Scaler()
data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
joblib.dump(scaler, 'scaler.gz')
print("scaled")

# Split the data into training and validation sets
X_train, X_val = train_test_split(data, test_size=0.2)

# Setup variables for batch generation and training
batch_size = 512
epochs = 128
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
autoencoder = MaskedAutoEncoder(times=160, out_dim=64, patch_shape=(10, 4))
autoencoder.compile(optimizer=AdamW(0.001), loss='huber')

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=0)

# Train the autoencoder with early stopping
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