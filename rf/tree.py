import numpy as np
"""

"""


def linear(w, X, b=0):
    return w.dot(X) + b


class BinaryTree():
    """ Array based binary tree. """

    def __init__(self, node_dim, depth):
        self.depth = depth
        self.tree = np.array((node_dim, 2**self.depth-1))

    def _l_child_ind(ind):
        pass

    def _r_child_ind(ind):
        pass

    def is_leaf(ind):
        pass


class DecisionTree(BinaryTree):
    """ """

    def __init__(self, dim_in, node_func=None, max_depth=2):
        self.node_func = node_func
        self.max_depth = max_depth
        self.tree = np.array((dim_in, 2**self.max_depth-1))
        self.result = np.zeros((2*max_depth))

    def _split(self, X, Y):
        
        self._split()
        self._split()
        
    def fit(self, X, Y):
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
