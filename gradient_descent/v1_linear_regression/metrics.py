# =========================================================================== #
#                                  METRICS MODULE                             #
# =========================================================================== #
"""Classification and regression metrics classes."""
from abc import ABC, abstractmethod
import math
import numpy as np

class Metric(ABC):
    """Abstract base class for all metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class RegressionMetric(Metric):
    """Base class for regression metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

# --------------------------------------------------------------------------- #
#                           REGRESSION METRICS                                #
# --------------------------------------------------------------------------- #
class MAE(RegressionMetric):
    """Computes mean absolute error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'mean_absolute_error'
        self.label = "Mean Absolute Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = abs(y-y_pred)
        return np.mean(e)


class MSE(RegressionMetric):
    """Computes mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'mean_squared_error'
        self.label = "Mean Squared Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return np.mean(e**2)

class RMSE(RegressionMetric):
    """Computes root mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'root_mean_squared_error'
        self.label = "Root Mean Squared Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return np.sqrt(np.mean(e**2)) 

class RegressionMetricFactory:
    """Returns the requested score class."""

    def __call__(self, metric='mean_squared_error'):

        dispatcher = {'mean_absolute_error': MAE(),
                      'mean_squared_error': MSE(),                                            
                      'root_mean_squared_error': RMSE()}
        return(dispatcher.get(metric,False))
