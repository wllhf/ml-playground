from util.generator import Generator
from dim_reduction.pca import PCA

import matplotlib
matplotlib.use('GTK3Agg')
# matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt


N_CLASSES = 1
N_SAMPLES = 1000
N_DIMS = 2

g = Generator(N_DIMS, N_CLASSES)
g.generate(N_SAMPLES)
data = g.X

transformation = PCA()

f = plt.figure()
transformation.fit(data)
data_t = transformation.transform(data)
plt.scatter(data[:, 0], data[:, 1], c='b', s=50)
plt.scatter(data_t[:, 0], data_t[:, 1], c='g', s=50)
plt.show()
