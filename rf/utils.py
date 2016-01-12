import numpy as np


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
    split = best[1], X[indices[best], best[1]]

    # sort input by best split column
    X = X[indices[:, best[1]], :]
    Y = labels[:, best[1]]

    # return split and splitted data set
    return split, (X[:best[0], :],  Y[:best[0]]), (X[best[0]:, :], Y[best[0]:])


def entropy(Y):
    # jep, it has to be a single line of code ;)
    probabilities = [float(np.sum(Y == c))/Y.shape[0] for c in np.unique(Y)]
    return -1*sum([p*np.log(p) for p in probabilities])
