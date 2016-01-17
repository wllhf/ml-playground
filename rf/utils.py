import numpy as np


def entropy(Y):
    """ Negative entropy of Y. """
    probabilities = np.bincount(Y)/float(Y.shape[0])
    return sum([p*np.log(p) for p in probabilities if p > 0])


def best_split(X, Y, criterion=entropy, indices=None):
    """ Splits a data set by impurity criterion.

    Parameters:
    -----------
        X: numpy array, shape: (n_samples, n_feature_dim), dtype: float
            Data points.
        Y: numpy array, shape: (n_samples, 1), dtype: uint
            Class labels.
        criterion: function
            Impurity criterion function.
        indices: list of int
            Feature dimensions to use.

    Result:
    -------
        (feature, value), (Xl, Yl), (Xr, Yr)
    """
    indices = range(X.shape[1]) if indices is None else indices
    d = len(indices)
    n = X.shape[0]
    h = np.zeros((n-1, d))

    sorting = X[:, indices].argsort(axis=0)
    labels = np.hstack([Y[sorting[:, i]] for i in range(d)])  # sorted class labels per feature dim

    for i in range(d):  # walk over feature dimensions
        for si in range(1, n):  # walk over samples
            hl = criterion(labels[:si, i])
            hr = criterion(labels[si:, i])
            h[si-1, i] = si*hl + (n-si)*hr  # resulting impurity (negative, to be maximized)

    best = np.unravel_index(h.argmax(), h.shape)
    fi = indices[best[1]]
    split = fi, X[sorting[best], fi]

    # sort input by best split column
    X = X[sorting[:, best[1]], :]
    Y = labels[:, best[1], np.newaxis]

    # return split and splitted data set
    return split, (X[:best[0]+1, :],  Y[:best[0]+1]), (X[best[0]+1:, :], Y[best[0]+1:])
