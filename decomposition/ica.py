import numpy as np

from .utils.data import whiten


# sigmoid function
def g0(x):
    return 1/(1+np.exp(-x))


def g1(x):
    return -np.exp(-x) / np.power(1+np.exp(-x), 2)


def g2(x):
    return np.exp(-x)*(1-2*np.exp(-x)) / np.power(1+np.exp(-x), 3)


class ICA(object):

    def __init__(self):
        pass

    def fit(self, data):
        """
        """
        X = data - np.mean(data, axis=0)
        X = whiten(X)
        W = np.random.rand(X.shape[0], X.shape[0])
