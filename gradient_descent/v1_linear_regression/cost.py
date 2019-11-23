# =========================================================================== #
#                                    COST                                     #
# =========================================================================== #
"""Cost functions and gradient computations."""
from abc import ABC, abstractmethod
import numpy as np

class Cost(ABC):

    @abstractmethod
    def __call__(self, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def gradient(self, X, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class RegressionCostFunction(Cost):
    """Base class for regression cost functions."""
    @abstractmethod
    def __call__(self, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def gradient(self, X, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class BinaryClassificationCostFunction(Cost):
    """Base class for binary classification cost functions."""
    @abstractmethod
    def __call__(self, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def gradient(self, X, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class MultinomialClassificationCostFunction(Cost):
    """Base class for multinomial classification cost functions."""
    @abstractmethod
    def __call__(self, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def gradient(self, X, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

# --------------------------------------------------------------------------- #
#                      REGRESSION COST FUNCTIONS                              #
# --------------------------------------------------------------------------- #
class Quadratic(RegressionCostFunction):
    """Computes cost."""

    def __init__(self):        
        self.name = "Quadratic Loss Function"

    def __call__(self, y, y_pred):
        """Computes quadratic costs e.g. squared error cost"""
        e = y_pred - y 
        J = 1/2 * np.mean(e**2)
        return(J)

    def gradient(self, X, y, y_pred):
        """Computes quadratic costs gradient with respect to weights"""
        n_samples = y.shape[0]
        y = np.atleast_2d(y).reshape(-1,1)
        y_pred = np.atleast_2d(y_pred).reshape(-1,1)
        dW = 1/n_samples * X.T.dot(y_pred-y)
        return(dW)   

class RegressionCostFactory():
    """Returns the requested cost class."""

    def __call__(self,cost='quadratic'):

        dispatcher = {'quadratic': Quadratic()}
        return(dispatcher.get(cost, False))

