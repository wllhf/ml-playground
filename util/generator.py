import pickle
import numpy as np


class Generator():
    """ """

    def __init__(self, n_dims, n_classes, density=2.0):
        self.d = n_dims
        self.C = range(n_classes)
        self.density = density
        self.X = None
        self.Y = None

    def gauss(self, a=-5, b=5, sigma=1):
        mu = (b-a)*np.random.random(self.d) + a
        cov = np.random.random((self.d, self.d))*sigma
        cov = np.dot(cov, cov.T)
        return mu, cov

    def generate_gaussian(self, n_samples):
        a = np.sqrt(float(n_samples*len(self.C))/self.density)/2.0
        dists = [self.gauss(-a, a) for c in self.C]
        samples = [np.random.multivariate_normal(d[0], d[1], n_samples) for d in dists]
        self.X = np.vstack(samples)
        self.Y = np.vstack([np.ones((n_samples, 1), dtype=np.uint8)*c for c in self.C])
        return dists

    def shuffle(self):
        indices = np.array(range(len(self.Y)))
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.Y = self.Y[indices]

    def save(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
