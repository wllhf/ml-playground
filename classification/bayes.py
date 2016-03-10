import numpy as np
from scipy import stats
from tools import mle


class NaiveBayes(object):

    def __init__(self, n_classes):
        self.n = n_classes
        self.priors = None
        self.thetas = None
        self.model = stats.multivariate_normal.pdf

    def fit(self, data, labels):
        self.thetas = []
        for l in np.unique(labels):
            m, s = mle.gaussian(data[labels == l])
            self.thetas.append((m, s))

        self.priors = np.bincount(np.squeeze(labels))/float(labels.shape[0])

    def predict(self, x):
        ps = np.empty((x.shape[0], self.priors.shape[0]))
        for i in range(self.n):
            ps[:, i] = self.model(x, self.thetas[i][0], self.thetas[i][1])*self.priors[i]

        return np.argmax(ps, axis=1)
