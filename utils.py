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

class RealTimeLMSFilter:
    def __init__(self, num_taps=5, mu=0.01):
        self.num_taps = num_taps
        self.mu = mu
        self.weights = np.zeros(num_taps)

    def filter_sample(self, sample):
        if len(self.signal_buffer) < self.num_taps:
            self.signal_buffer.append(sample)
            return sample
        else:
            x = np.array(self.signal_buffer[-self.num_taps:][::-1])
            y = np.dot(self.weights, x)
            e = sample - y
            
            # Calculate the gradient and apply clipping
            gradient = 2 * self.mu * e * x
            
            # Update weights
            self.weights += gradient
            self.signal_buffer.append(sample)
            return e

    def process_signal(self, signal):
        self.signal_buffer = []
        filtered_signal = np.zeros_like(signal)
        for i, sample in enumerate(signal):
            filtered_signal[i] = self.filter_sample(sample)
        return filtered_signal