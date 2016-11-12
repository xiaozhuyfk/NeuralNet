from math import sqrt
import numpy as np

def uniform_initializer(shape):
    assert(len(shape) == 2)
    input_dim, output_dim = shape
    low = - sqrt(6) / sqrt(input_dim + output_dim)
    high = sqrt(6) / sqrt(input_dim + output_dim)
    return np.random.uniform(low=low, high=high, size=shape)