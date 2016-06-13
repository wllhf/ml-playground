import numpy as np


def gaussian(data):
    """ Maximum likelihood estimator for gaussian distribution.

    Parameters:
    -----------
    data: numpy array (n, d)
      n samples with d feature dimensions.

    Returns:
    --------
    m: numpy array (d,)
      Empirical mean of gaussian.
    s: numpy array (d, d)
      Empirical sigma**2 of gaussian.
    """

    m = np.average(data, axis=0)
    d = data-m
    s = np.dot(d.T, d)/data.shape[0]

    return m, s
