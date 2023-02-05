import numpy as np
import utils
import findMin



class LassoRegression():
    def __init__(self, lammy = 1, maxEvals = 3000):
        self.lammy = lammy
        self.maxEvals = maxEvals

    def fit(self,X,y):
        n, d = X.shape
        self.w = np.zeros((d,1))
        self.w, f = findMin.findMinL1(self.funObj, self.w, self.lammy, self.maxEvals, X, y)

    def funObj(self,w,X,y):
        if w.ndim == 1:
            w = w[:, np.newaxis]

        l = X@w-y
        f = np.sum(l**2)/2/X.shape[0]
        g = X.T@l / X.shape[0]

        return f,g

    def predict(self, X):
        return X@self.w
