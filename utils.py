import numpy as np


def tanh_normalize(data, scale, offset):
    return np.tanh(scale * (data + offset))


def smooth(current_value, target_value, weight):
    current_value = (1.0 - weight) * current_value + weight * target_value
    return current_value


def cantor_pair(x, y):
    return 0.5 * (x + y) * (x + y + 1) + y


def remap_cantor_pair(x, y):
    # Calculate the Cantor Pairing value.
    cantor_value = cantor_pair(x, y)

    # Remap the Cantor value from the range [-1, 1] to [0, 1].
    remapped_value = (cantor_value + 1) / 2

    return remapped_value


print(remap_cantor_pair(0.573, -0.549))
