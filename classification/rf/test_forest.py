import pickle
import numpy as np

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from tree import ClassificationTree
from forest import ClassificationForest
from data.generator import Generator

N_DIMS = 3
N_CLASSES = 3
N_SAMPLES = 1000
N_TREES = 10
MAX_DEPTH = 5
MIN_SAMPLES = 100
RANDOMNESS = 0.5

N_TEST_SAMPLES = 1000


g = Generator(N_DIMS, N_CLASSES)
g.generate(N_SAMPLES)
# with open(b'data_c_3_n_1000', 'rb') as inp:
#    g = pickle.load(inp)
g.shuffle()

X_train = g.X[:N_SAMPLES*N_CLASSES-N_TEST_SAMPLES, :]
Y_train = g.Y[:N_SAMPLES*N_CLASSES-N_TEST_SAMPLES]
X_test = g.X[N_SAMPLES*N_CLASSES-N_TEST_SAMPLES:, :]
Y_test = g.Y[N_SAMPLES*N_CLASSES-N_TEST_SAMPLES:]

t = ClassificationTree(N_CLASSES, MAX_DEPTH, MIN_SAMPLES, 0)
f = ClassificationForest(N_TREES, N_CLASSES, MAX_DEPTH, MIN_SAMPLES, RANDOMNESS)

#t.fit(X_train, Y_train)
f.fit(X_train, Y_train)

#rt = t.predict(X_test)
rf = f.predict(X_test)
#rt = np.expand_dims(np.argmax(rt, axis=1), axis=1)
rf = np.expand_dims(np.argmax(rf, axis=1), axis=1)
#diff_t = Y_test - rt
diff_f = Y_test - rf
#print "correct ", float(np.sum(diff_t == 0))/diff_t.shape[0]
print "correct ", float(np.sum(diff_f == 0))/diff_f.shape[0]
