import numpy as np
from scipy.signal import butter, filtfilt

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


## artifact detection inspired by openbci algorithm
## threshold is defaulted to 4 times standard deviation of the data window
## https://openbci.com/community/automated-eye-blink-detection-online-2/
def get_artifact_mask(data, sampling_rate, std_mult=4, is_absolute=True):
    b, a = butter(2, 10 / (sampling_rate / 2), btype='low')  # 10 Hz lowpass filter
    
    # lowpass filter to blink range
    filtered = filtfilt(b, a, data)

    # calculate the mean
    mean = np.mean(filtered, keepdims=True)

    # create a threshold from standard deviation
    std = np.std(filtered, keepdims=True)
    threshold = std_mult * std

    # find difference between filtered and mean
    diff = filtered - mean
    
    # absolute difference if specified
    if is_absolute:
        diff = np.abs(diff)
    
    # create mask where 1 if artifact detected
    mask = diff > threshold

    return mask