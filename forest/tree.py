import numpy as np

import utils


# worker functions for multiprocessing ########################################
def fit_worker(args):
    args[0].fit(args[1], args[2])
    return args[0]


def predict_worker(args):
    return args[0].predict(args[1])
###############################################################################


class Node():
    def __init__(self, depth):
        self.is_leaf = False
        self.depth = depth
        self.fi = None
        self.th = None
        self.l_child = None
        self.r_child = None

    def append_l(self):
        self.l_child = Node(self.depth+1)
        return self.l_child

    def append_r(self):
        self.r_child = Node(self.depth+1)
        return self.r_child


class Leaf():
    def __init__(self, Y, n_classes):
        self.is_leaf = True
        self.distribution = np.bincount(Y[:, 0], minlength=n_classes)/float(Y.shape[0])
        self.class_max = np.argmax(self.distribution)


class ClassificationTree():
    """ """

    def __init__(self, n_classes, max_depth=2, min_samples=1, randomness=0):
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.r = randomness
        self.n_feature_dims = None
        self.root = Node(0)

    def _fit(self, X, Y, node):
        indices = np.random.permutation(X.shape[1])[:self.n_feature_dims]
        (fi, th), (Xl, Yl), (Xr, Yr) = utils.best_split(X, Y, indices=indices)
        node.fi = fi
        node.th = th

        if Yl.shape[0] > self.min_samples and np.unique(Yl).shape[0] > 1 and node.depth < self.max_depth:
            self._fit(Xl, Yl, node.append_l())
        else:
            node.l_child = Leaf(Yl, self.n_classes)

        if Yr.shape[0] > self.min_samples and np.unique(Yr).shape[0] > 1 and node.depth < self.max_depth:
            self._fit(Xr, Yr, node.append_r())
        else:
            node.r_child = Leaf(Yr, self.n_classes)

    def fit(self, X, Y):
        self.n_feature_dims = max(np.rint(X.shape[1]*(1-self.r)), 1)
        self._fit(X, Y, self.root)

    def _decide(self, node, X, indices, result):
        if node.is_leaf:
            result[indices] = node.distribution
        else:
            indices_l = X[:, node.fi] < node.th
            indices_r = X[:, node.fi] >= node.th
            self._decide(node.l_child, X, indices_l, result)
            self._decide(node.r_child, X, indices_r, result)

    def predict(self, X):
        result = np.zeros((X.shape[0], self.n_classes))
        self._decide(self.root, X, indices=np.ones(X.shape[0]), result=result)
        return result
