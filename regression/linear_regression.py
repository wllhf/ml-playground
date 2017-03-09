import numpy as np


class LinRegression(object):

    def __init__(self):
        self.alpha = None
        self.beta = None

    def fit_ols(self, Y, X):
        X = X if len(X.shape) > 1 else np.expand_dims(X, axis=1)
        Y = Y if len(Y.shape) > 1 else np.expand_dims(Y, axis=1)
        Yc = Y - Y.mean()
        Xc = X - X.mean()
        self.beta = np.squeeze(np.linalg.inv((Xc.T).dot(Xc)).dot(Xc.T).dot(Yc))
        self.alpha = (Y - self.beta.dot(X)).sum()/float(X.shape[0])

    def fit_ml(self):
        pass

    def predict(self, X):
        return self.beta.dot(X) + self.alpha
