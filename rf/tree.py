import numpy as np

import utils


class BinaryTree():
    """ Array based binary tree. """

    def __init__(self, node_dim, leaf_dim, depth):
        self.depth = depth
        self.nodes = np.array((node_dim, 2**self.depth-1))
        self.leaves = np.array((leaf_dim, 2**self.depth))

    def l_child_idx(idx):
        pass

    def r_child_idx(idx):
        pass

    def _depth(self, idx):
        pass

    def get_node(ind):
        pass

    def is_leaf(ind):
        pass


class DecisionTree(BinaryTree):
    """ """

    def __init__(self, max_depth=2, min_samples=1):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.split_fi = np.empty(2**self.max_depth-1, dtype=np.uint8)
        self.split_th = np.empty(2**self.max_depth-1, dtype=np.float32)
        self.leaves = None

    def fit(self, X, Y, idx=0):
        if len(Y) > self.min_samples and self._depth(idx) <= self.max_depth:
            fi, th, Xl, Yl, Xr, Yr = utils.best_split(X, Y)
            self.split_fi[idx] = fi
            self.split_th[idx] = th
            self.fit(Xl, Yl, self._l_child(idx))
            self.fit(Xr, Yr, self._r_child(idx))
        else:
            # create leaf
            pass

    def _decide(self, node, X):
        l_child = self._l_child_ind(node)
        r_child = self._r_child_ind(node)
        indices = (self.result == node)

        l, r = self.node_func(self.tree[node], X[indices])
        self.result[indices][l] = l_child
        self.result[indices][r] = r_child

        self._decide(l_child, X)
        self._decide(r_child, X)

    def predict(self, X):
        self._decide(0, X)
        return self.result

