import tensorflow as tf
import itertools

## Limit GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set virtual device configuration for the first GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)]  # 12GB memory limit
        )
        print("Virtual GPU with 12 GB memory limit created.")
    except RuntimeError as e:
        print("Error while creating virtual GPU:", e)
else:
    print("No GPU found.")

import numpy as np
import joblib

from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping
from keras.layers import Activation

from model import MaskedAutoEncoder

# Load the data
print('Loading data...')
data = joblib.load('dataset.pkl')

# Split the data into training and validation sets
print('Data Loaded. Processing:', data.shape)
sample_count = data.shape[0]
np.random.shuffle(data)

print('Shuffled. Splitting...')
pivot = int(data.shape[0] * 0.2)
X_val_orig = data[:pivot]
X_train = data[pivot:]
del data

# Setup variables for batch generation and training
print('Data Processed. Setting up...')
batch_size = 512
epochs = 256
batch_count = sample_count // batch_size
X_train = np.array_split(X_train, batch_count, axis=0)
X_val = np.array_split(X_val_orig, batch_count, axis=0)

train_steps = len(X_train)
test_steps = len(X_val)

# set up train and val generators
def batch_generator(splits):
    iterator = itertools.cycle(splits)
    for x in iterator:
        yield x, x
train_generator = batch_generator(X_train)
val_generator = batch_generator(X_val)

# Build the autoencoder
input_shape = X_train[0].shape[1:]
autoencoder = MaskedAutoEncoder(
    input_shape=input_shape, 
    patch_shape=(10, 4), 
    mask_ratio=0.75,
    num_heads=8,
    ae_size=(5, 1)
)
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

# Number of random windows to select
num_windows = 16

# Select random windows
r_indices = np.random.choice(X_val_orig.shape[0], size=num_windows, replace=False)
X_val_subset = X_val_orig[r_indices]

# Get the reconstructed outputs
reconstructed_subset = autoencoder.predict(X_val_subset)

# Transpose to time last
X_val_subset = X_val_subset.transpose(0, 2, 1, 3)
reconstructed_subset = reconstructed_subset.transpose(0, 2, 1, 3)

# Scale to [0, 1]
sigmoid = Activation('sigmoid')
X_val_rgb = np.array(sigmoid(X_val_subset))
reconstructed_rgb = np.array(sigmoid(reconstructed_subset))

# Use the dark background style
plt.style.use('dark_background')

# Dynamically calculate grid size based on figsize
figsize = (12, 12)  # Adjustable variable for figure size
aspect_ratio = figsize[0] / figsize[1]
cols = int(np.ceil(np.sqrt(num_windows * aspect_ratio)))
rows = int(np.ceil(num_windows / cols)) * 2

# Create subplots dynamically
fig, axs = plt.subplots(rows, cols, figsize=figsize)

# Plot the data
for i in range(num_windows):
    row, col = divmod(i, cols)
    row *= 2  # Each pair occupies two rows

    # get entry number to display
    j = r_indices[i]

    # Plot original RGB
    axs_original = axs[row, col]
    axs_original.imshow(X_val_rgb[i])
    axs_original.set_title(f"Original {j}")
    axs_original.axis("off")

    # Plot reconstructed RGB
    axs_reconstruct = axs[row + 1, col]
    axs_reconstruct.imshow(reconstructed_rgb[i])
    axs_reconstruct.set_title(f"Reconstructed {j}")
    axs_reconstruct.axis("off")

# Remove unused subplots
for ax in axs.flat:
    if not ax.images:
        ax.axis("off")

# Save visual
plt.tight_layout()
plt.savefig('autoencoder_reconstruct.png')