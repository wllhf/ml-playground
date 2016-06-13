import numpy as np


class PCA(object):

    def __init__(self):
        self.mean = None
        self.v = None

    def fit(self, data):
        """
        Parameters:
        -----------
        data: numpy array (n, d)
        """
        self.mean = np.mean(data, axis=0)  # center data
        data = data - self.mean
        covar = np.transpose(data).dot(data)/data.shape[0]  # covariance
        w, self.v = np.linalg.eig(covar)  # eigenvalues and -vectors
        idx = w.argsort()[::-1]  # sort by eigenvalue largest -> smallest
        w = w[idx]
        self.v = self.v[:, idx]
        return w, self.v

    def transform(self, data, center=True):
        if center:
            data_t = (data - self.mean).dot(self.v)
        else:
            data_t = (data - self.mean).dot(self.v) + self.mean
        return data_t

    def inverse(self, data, centered=True):
        if centered:
            data_i = data.dot(np.linalg.inv(self.v)) + self.mean
        else:
            data_i = (data - self.mean).dot(np.linalg.inv(self.v)) + self.mean
        return data_i
