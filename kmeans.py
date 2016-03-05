import numpy as np

import matplotlib.pyplot as plt


class kMeans(object):

    def __init__(self):
        self.means = None

    def _new_means(self, data, labels, k):
        means = np.empty((k, data.shape[1]))
        for i in range(k):
            indices = labels == i
            if np.sum(indices) > 0:
                means[i] = np.average(data[indices], axis=0)
            else:
                means[i] = self.means[i]
        return means

    def _min_dist_means(self, data):
        labels = np.empty((data.shape[0]))
        for i, d in enumerate(data):
            labels[i] = np.argmin(np.sum(np.square(self.means-d), axis=1))
        return labels

    def fit(self, data, k, epsilon=10**-3, visualize=False):
        """Clusters data into k clusters.
        Parameters:
        -----------
        data: numpy array (n, d)
        k: int
        """
        n = data.shape[0]
        self.means = data[np.random.choice(n, k, replace=False)]

        if visualize:
            plt.ion()
            plt.figure()

        delta = epsilon
        while(delta >= epsilon):

            labels = self._min_dist_means(data)
            new_means = self._new_means(data, labels, k)
            delta = np.sum(np.abs(new_means - self.means))
            self.means = new_means

            if visualize:
                plt.clf()
                plt.scatter(data[:, 0], data[:, 1], c=labels, s=50)
                plt.scatter(self.means[:, 0], self.means[:, 1], c='b', s=100)
                plt.draw()
                plt.waitforbuttonpress()

        plt.close()

        return labels

    def predict(self, data):
        return self._min_dist_means(data)
