import numpy as np


def whiten(X, fudge=1E-18):
    """ """
    d, V = np.linalg.eigh(np.dot(X.T, X))
    D = np.diag(1. / np.sqrt(d+fudge))
    W = np.dot(np.dot(V, D), V.T)
    X_white = np.dot(X, W)
    return X_white, W
