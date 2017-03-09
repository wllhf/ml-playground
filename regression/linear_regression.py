import numpy as np


class LinRegression(object):

    def __init__(self):
        self.alpha = None
        self.beta = None

    def fit(self, Y, X):
        """ Ordinary least squares regression.

        Parameters:
        -----------
        X: numpy array (nxd)
        Y: numpy array (n)
        """
        X = X if len(X.shape) > 1 else np.expand_dims(X, axis=1)
        Y = Y if len(Y.shape) > 1 else np.expand_dims(Y, axis=1)

        # center data
        Yc = Y - Y.mean()
        Xc = X - X.mean()

        # beta = (XcT*Xc)^-1 *XcT*Yc
        self.beta = np.linalg.inv((Xc.T).dot(Xc)).dot(Xc.T).dot(Yc)
        self.alpha = np.sum(Y - X.dot(self.beta), axis=1)/float(X.shape[0])

    def predict(self, X):
        X = X if len(X.shape) > 1 else np.expand_dims(X, axis=1)
        return X.dot(self.beta) + self.alpha
