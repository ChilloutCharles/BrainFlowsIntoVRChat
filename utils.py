import numpy as np
from scipy.stats import kurtosis


def tanh_normalize(data, scale, offset):
    return np.tanh(scale * (data + offset))

def smooth(current_value, target_value, weight):
    current_value = (1.0 - weight) * current_value + weight * target_value
    return current_value

def map2dto1d(x, y, n):
    return x * n + y # TODO: find a better continous function to map 2D to 1D

def ica_kurtosis_threshold(data, ica, threshold=3):
    components = ica.fit_transform(data)
    kurtoses = kurtosis(components, axis=0)
    remove_idxs = np.where(np.abs(kurtoses) > threshold)[0]
    components[:, remove_idxs] = 0
    data = ica.inverse_transform(components)
    return data