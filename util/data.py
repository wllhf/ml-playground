import numpy as np
import matplotlib.pyplot as plt

from data.generator import Generator

N_CLASSES = 3
N_SAMPLES = 1000

g = Generator(2, N_CLASSES)
g.generate_gaussian(N_SAMPLES)
g.save('data_c_3_n_1000')

fig = plt.figure()
plt.scatter(g.X[:, 0], g.X[:, 1], c=g.Y, s=50)
plt.show()
