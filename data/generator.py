import numpy as np


class Generator():
    """ """

    def __init__(self, n_features, n_classes):
        self.d = n_features
        self.C = range(1, n_classes+1)
        self.density = 5
        self.X = None
        self.Y = None

    def gauss(self, a=-5, b=5, sigma=1):
        mu = (b-a)*np.random.random(self.d) + a
        cov = np.random.random((self.d, self.d))*sigma
        return mu, cov

    def generate(self, n_samples):
        a = np.sqrt(float(n_samples*len(self.C))/self.density)/2.0
        dists = [self.gauss(-a, a) for c in self.C]
        samples = [np.random.multivariate_normal(d[0], d[1], n_samples) for d in dists]
        self.X = np.vstack(samples)
        self.Y = np.vstack([np.ones((n_samples, 1))*c for c in self.C])

    def shuffle(self):
        indices = np.array(range(len(self.Y)))
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.Y = self.Y[indices]
