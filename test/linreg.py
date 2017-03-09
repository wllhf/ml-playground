import numpy as np
from scipy import stats
from regression.linear_regression import LinRegression
from data.generator import RegressionData

import matplotlib.pyplot as plt


def lin(x):
    return 2*x + 10


gen = RegressionData(lin, sigma=10)
X, Y = gen.generate()

reg = LinRegression()
reg.fit_ols(Y, X)

plt.scatter(X, Y)
plt.plot(X, reg.predict(X))

plt.show()
