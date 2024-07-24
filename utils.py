import numpy as np
from padasip.filters import FilterGMCC as filter

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

class AdaptiveFilter:
    def __init__(self, window_size=4, mu=0.1):
        self.window_size = window_size
        self.mu = mu
        self.filter = filter(n=window_size, mu=mu)
        
    def create_sliding_window(self, data):
        shape = (data.size - self.window_size + 1, self.window_size)
        strides = (data.strides[0], data.strides[0])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    
    def filter_signal(self, eeg_signal, desired_signal):
        # Pad the EEG signal before creating the sliding window
        eeg_signal_padded = np.pad(eeg_signal, (self.window_size - 1, 0), 'constant')
        
        # Create sliding window input matrix
        input_matrix = self.create_sliding_window(eeg_signal_padded)
        
        # Apply the filter to the EEG data
        filtered_signal = np.zeros(desired_signal.shape)
        for i in range(input_matrix.shape[0]):
            output = self.filter.predict(input_matrix[i, :])
            self.filter.adapt(desired_signal[i], input_matrix[i, :])
            filtered_signal[i] = output
        
        return filtered_signal



