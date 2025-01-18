import numpy as np
import joblib

from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping

from model import MaskedAutoEncoder

from sklearn.preprocessing import MinMaxScaler

# Load the data
print('Loading data...')
data = joblib.load('dataset.pkl')
viz_scaler = MinMaxScaler()
viz_scaler.fit(data.reshape(-1, 1))

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
autoencoder = MaskedAutoEncoder(
    input_shape=input_shape, 
    patch_shape=(10, 4), 
    mask_ratio=0.9,
    num_heads=8
)
autoencoder.compile(optimizer=AdamW(0.001), loss='huber')

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
from matplotlib.colors import rgb_to_hsv

# Number of random windows to select
num_windows = 16

# Select random windows
r_indices = np.random.choice(X_val.shape[0], size=num_windows, replace=False)
X_val_subset = X_val[r_indices]

# Get the reconstructed outputs
reconstructed_subset = autoencoder.predict(X_val_subset)

# Transpose to time last
X_val_subset = X_val_subset.transpose(0, 2, 1, 3)
reconstructed_subset = reconstructed_subset.transpose(0, 2, 1, 3)

# Scale to [0, 1]
X_val_subset = viz_scaler.transform(X_val_subset.reshape(-1, 1)).reshape(X_val_subset.shape)
reconstructed_subset = viz_scaler.transform(reconstructed_subset.reshape(-1, 1)).reshape(reconstructed_subset.shape)

# Use the dark background style
plt.style.use('dark_background')

# Apply MRA-to-HSV conversion
X_val_rgb = np.array([rgb_to_hsv(x) for x in X_val_subset])
reconstructed_rgb = np.array([rgb_to_hsv(x) for x in reconstructed_subset])

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
    axs[row, col].imshow(X_val_rgb[i])
    axs[row, col].set_title(f"Original {j}")
    axs[row, col].axis("off")

    # Plot reconstructed RGB
    axs[row + 1, col].imshow(reconstructed_rgb[i])
    axs[row + 1, col].set_title(f"Reconstructed {j}")
    axs[row + 1, col].axis("off")

# Remove unused subplots
for ax in axs.flat:
    if not ax.images:
        ax.axis("off")

# Save visual
plt.tight_layout()
plt.savefig('autoencoder_reconstruct.png')