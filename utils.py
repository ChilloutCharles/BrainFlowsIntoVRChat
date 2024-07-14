import numpy as np
from pprint import pprint

def tanh_normalize(data, scale, offset):
    return np.tanh(scale * (data + offset))


def smooth(current_value, target_value, weight):
    current_value = (1.0 - weight) * current_value + weight * target_value
    return current_value


def map2dto1d(x, y, n):
    return x * n + y


def lms_filter(input_signal, mu, n_order, weights, buffer, clip=1):
    # Ensure buffer length is at least n_order + len(input_signal)
    assert len(buffer) >= n_order + len(input_signal), "Buffer length is insufficient"
    
    # Extract the relevant part of the buffer for processing
    buffer_segment = np.array(buffer[-(n_order + len(input_signal)):])

    # Matrix form of the buffer segments using sliding window view
    X = np.lib.stride_tricks.sliding_window_view(buffer_segment, window_shape=n_order)

    # Only take the first len(input_signal) rows
    X = X[:len(input_signal)]
    
    # Predicted noise
    Y = np.dot(X, weights)
    
    # Error signal
    E = input_signal - Y

    # Gradient clipping to prevent excessively large updates
    gradient = 2 * mu * np.dot(E, X)
    gradient = np.clip(gradient, -clip, clip)  # Clip gradients to prevent large updates

    # Update weights
    weights += gradient
    
    # Filtered signal
    filtered_output = input_signal - Y

    return filtered_output, weights