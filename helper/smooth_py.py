import numpy as np
from scipy.signal import lfilter


def smooth_py(y, width):
    """
    Realize Matlab smooth() func.
    :param y: list
    :param width:
    :return: list
    """
    n = len(y)
    b1 = np.ones(width) / width
    a1 = [1]
    c = lfilter(b1, a1, y, axis=0)
    cbegin = np.cumsum(y[0:width - 2])
    cbegin = [x / y for x, y in zip(cbegin[::2], list(range(1, width - 1, 2)))]
    cend = np.cumsum(y[n - width + 2:n][::-1])
    cend = [x / y for x, y in zip(cend[::-2], list(range(1, width - 1))[::-2])]
    c_new = []
    c_new.extend(cbegin)
    c_new.extend(c[width - 1:].tolist())
    c_new.extend(cend)
    return c_new
