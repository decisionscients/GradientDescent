# =========================================================================== #
#                              ANALYTICAL                                     #
# =========================================================================== #
''' This module contains analytical solutions to optimization problems'''
# --------------------------------------------------------------------------- #
import numpy as np
from numpy.linalg import inv
# --------------------------------------------------------------------------- #
class Normal():

    def _init_(self):
        self.alg = "Normal Equation"
        self.theta = None
        return self

    def fit(self, X, y):   
        self.alg = "Normal Equation"      
        self.theta = inv(X.T.dot(X)).dot(X.T).dot(y)
        return self

