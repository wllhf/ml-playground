import numpy as np
import matplotlib.pyplot as plt

from data.generator import Generator

import kmeans

N_CLASSES = 10
N_SAMPLES = 100
N_DIMS = 2

g = Generator(N_DIMS, N_CLASSES, 1.0)
g.generate(N_SAMPLES)

classifier = kmeans.kMeans()
labels = classifier.fit(g.X, 10, visualize=True)

fig = plt.figure()
plt.scatter(g.X[:, 0], g.X[:, 1], c=labels, s=50)
plt.scatter(classifier.means[:, 0], classifier.means[:, 1], c='b', s=100)
plt.show()
