import numpy as np
from scipy.signal import butter, filtfilt

epsilon = np.finfo(float).eps

def tanh_log(data, scale):
    return np.tanh(scale * np.log(data))


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


## artifact detection by openbci algorithm
## https://openbci.com/community/automated-eye-blink-detection-online-2/
def get_artifact_mask(data, sampling_rate, threshold=100, is_absolute=True):
    # b, a = butter(2, 10 / (sampling_rate / 2), btype='low')  # 10 Hz lowpass filter
    
    # lowpass filter to blink range
    # filtered = filtfilt(b, a, data)
    filtered = data

    # calculate the mean by channel
    mean = np.mean(filtered, keepdims=True, axis=-1)

    # find difference between filtered and mean
    diff = filtered - mean
    
    # absolute difference if specified
    if is_absolute:
        diff = np.abs(diff)
    
    # create mask where 1 if artifact detected
    mask = diff > threshold

    return mask