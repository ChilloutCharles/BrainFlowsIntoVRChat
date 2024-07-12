import numpy as np


def tanh_normalize(data, scale, offset):
    return np.tanh(scale * (data + offset))


def smooth(current_value, target_value, weight):
    current_value = (1.0 - weight) * current_value + weight * target_value
    return current_value


def map2dto1d(x, y, n):
    return x * n + y


def lms_filter(input_signal, mu, n_order, weights, buffer):
    filtered_output = np.zeros_like(input_signal)
    for i, sample in enumerate(input_signal):
        x = np.array(buffer[-n_order:])     # Get the last n_order samples from the buffer
        y = np.dot(weights, x)              # Filter output (predict noise)
        filtered_sample = sample - y        # Remove predicted noise from the input signal
        e = sample - filtered_sample        # Error signal
        weights += 2 * mu * e * x           # Update filter weights
        filtered_output[i] = filtered_sample
    return filtered_output, weights