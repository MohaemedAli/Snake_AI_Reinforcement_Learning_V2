import numpy as np


def moving_average(values, window=100):
    """Moving average with a growing window at the start."""

    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return values

    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.empty(len(values))

    for i in range(len(values)):
        start = max(0, i + 1 - window)
        result[i] = (cumulative[i + 1] - cumulative[start]) / (i + 1 - start)

    return result
