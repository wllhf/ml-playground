from util.generator import Generator
from clustering.kmeans import kMeans

import matplotlib.pyplot as plt

N_CLASSES = 10
N_SAMPLES = 1000
N_DIMS = 2

g = Generator(N_DIMS, N_CLASSES, 50.0)
g.generate_gaussian(N_SAMPLES)

classifier = kMeans()

plt.ion()
f = plt.figure()
labels = classifier.fit(g.X, N_CLASSES, visualize=True)
plt.close()
