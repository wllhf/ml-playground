import pickle
import numpy as np

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from tree import ClassificationTree
from data.generator import Generator

N_DIMS = 2
N_CLASSES = 3
N_SAMPLES = 1000

g = Generator(N_DIMS, N_CLASSES)
g.generate_gaussian(N_SAMPLES)
# with open(b'data_c_3_n_1000', 'rb') as inp:
#    g = pickle.load(inp)
g.shuffle()

X_train = g.X[:N_SAMPLES*N_CLASSES-100, :]
Y_train = g.Y[:N_SAMPLES*N_CLASSES-100]
X_test = g.X[N_SAMPLES*N_CLASSES-10:, :]
Y_test = g.Y[N_SAMPLES*N_CLASSES-10:]

t = ClassificationTree(N_CLASSES, 5, 100)
t.fit(X_train, Y_train)
result = t.predict(X_test)
result = np.expand_dims(np.argmax(result, axis=1), axis=1)
diff = Y_test - result
print "correct ", float(np.sum(diff == 0))/diff.shape[0]
