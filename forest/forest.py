import multiprocessing as mp

from tree import ClassificationTree as CT
from tree import fit_worker
from tree import predict_worker


class ClassificationForest():
    """ """

    def __init__(self, n_trees, n_classes, max_depth=10, min_samples=1, randomness=1):
        self.n_classes = n_classes
        self.n_trees = n_trees
        self.r = randomness
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.forest = [CT(n_classes, max_depth, min_samples, randomness) for i in range(n_trees)]

    def fit(self, X, Y):
        # for tree in self.forest:
        #     fit_worker((tree, X, Y))
        args = [(tree, X, Y) for tree in self.forest]
        pool = mp.Pool(processes=4)
        self.forest = pool.map(fit_worker, args)
        pool.close()
        pool.join()

    def predict(self, X):
        # results = []
        # for tree in self.forest:
        #     results.append(predict_worker((tree, X)))
        args = [(tree, X) for tree in self.forest]
        pool = mp.Pool(processes=4)
        results = pool.map(predict_worker, args)
        pool.close()
        pool.join()

        return sum(results)
