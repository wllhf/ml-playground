import numpy as np
from regression.linear_regression import LinRegression
from data.generator import RegressionData

import matplotlib.pyplot as plt


def lin(x):
    return 2*x + 10


def square(x):
    return x**2 + 2


def lin_test():
    gen = RegressionData(lin, sigma=10)
    X, Y = gen.generate()

    reg = LinRegression()
    reg.fit(Y, X)

    plt.scatter(X, Y)
    plt.plot(X, reg.predict(X))

    plt.show()


def square_test():
    gen = RegressionData(square, sigma=10)
    X, Y = gen.generate()
    X_ = np.stack([X, np.square(X)], axis=1)

    reg = LinRegression()
    reg.fit(Y, X_)

    plt.scatter(X, Y)
    plt.plot(X, reg.predict(X_))

    plt.show()


lin_test()
square_test()
