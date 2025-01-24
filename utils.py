import numpy as np

def tanh_normalize(data, scale, offset):
    return np.tanh(scale * (data + offset))


def smooth(current_value, target_value, weight):
    current_value = (1.0 - weight) * current_value + weight * target_value
    return current_value


def map2dto1d(x, y, n):
    return x * n + y


def compute_snr(original_signal, filtered_signal):
    signal_power = np.var(filtered_signal)
    noise_power = np.var(original_signal - filtered_signal)
    snr = 10 * np.log10(signal_power / noise_power)
    return np.round(snr, 4)