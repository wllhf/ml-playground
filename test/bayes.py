import numpy as np
from scipy import stats
from classification.bayes import NaiveBayes
from util.generator import Generator

import matplotlib.pyplot as plt

gen = Generator(1, 2)
dists = gen.generate_gaussian(20)
gen.shuffle()
d_train = gen.X[:30]
l_train = gen.Y[:30]
d_test = gen.X[30:]
l_test = gen.Y[30:]

classifier = NaiveBayes(2)
classifier.fit(d_train, l_train)

result = classifier.predict(d_test)
print np.squeeze(l_test.T)
print result
print 1 - np.sum(np.abs(result - np.squeeze(l_test)))/float(result.shape[0])

f = plt.figure()
x = np.linspace(-3, 3, 100)
plt.plot(x, stats.multivariate_normal.pdf(x, dists[0][0], dists[0][1]))
plt.plot(x, stats.multivariate_normal.pdf(x, dists[1][0], dists[1][1]))
plt.plot(x, stats.multivariate_normal.pdf(x, classifier.thetas[0][0], classifier.thetas[0][1]))
plt.plot(x, stats.multivariate_normal.pdf(x, classifier.thetas[1][0], classifier.thetas[1][1]))

plt.show()
