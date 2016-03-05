import numpy as np
import matplotlib.pyplot as plt


def best_split(X, Y):
    d = X.shape[1]
    n = X.shape[0]
    h = np.zeros((n, d))
    indices = X.argsort(axis=0)
    labels = np.hstack([Y[indices[:, i]] for i in range(d)])  # sorted class labels per feature dim

    for fi in range(d):  # walk over feature dimensions
        for si in range(0, n):  # walk over samples
            hl = entropy(labels[:si, fi])
            hr = entropy(labels[si:, fi])
            h[si, fi] = si*hl + (n-si)*hr  # resulting entropy (to be minimized)

    best = np.unravel_index(h.argmin(), h.shape)
    return best[1], X[indices[best], best[1]]


def entropy(Y):
    # jep, it has to be a single line of code ;)
    probabilities = [float(np.sum(Y == c))/Y.shape[0] for c in np.unique(Y)]
    return -1*sum([p*np.log(p) for p in probabilities])


# generate toy data ###########################################################
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

# main ########################################################################
if __name__ == '__main__':
    g = Generator(2, 4)
    g.generate(50)
    split = best_split(g.X, g.Y)

    plt.scatter(g.X[:, 0], g.X[:, 1], c=g.Y, s=50)
    if split[0] == 0:
        plt.axvline(split[1])
    if split[0] == 1:
        plt.axhline(split[1])
    plt.show()
