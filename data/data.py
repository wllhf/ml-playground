import numpy as np
import matplotlib.pyplot as plt

from generator import Generator

N_CLASSES = 2
N_SAMPLES = 1000

g = Generator(2, N_CLASSES)
g.generate_gaussian(N_SAMPLES, True)
# g.save('data_c_3_n_1000')

print g.X.shape
print np.mean(g.X, axis=0)

fig = plt.figure()
plt.scatter(g.X[:, 0], g.X[:, 1], c=g.Y, s=50)
plt.show()
