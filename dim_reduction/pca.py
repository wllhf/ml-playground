import numpy as np


class PCA(object):

    def __init__(self):
        self.v = None

    def fit(self, data):
        """
        Parameters:
        -----------
        data: numpy array (n, d)
        """
        data = data - np.mean(data, axis=0)  # center data
        covar = np.transpose(data).dot(data)/data.shape[0]  # covariance
        w, self.v = np.linalg.eig(covar)  # eigenvalues and -vectors
        idx = w.argsort()[::-1]  # sort by eigenvalue largest -> smallest
        w = w[idx]
        self.v = self.v[:, idx]
        return w, self.v

    def transform(self, data):
        data = data - np.mean(data, axis=0)  # center data
        return data.dot(self.v)

    def reduce(self, data, dims):
        v_inv = np.linalg.inv(v)
        pass
