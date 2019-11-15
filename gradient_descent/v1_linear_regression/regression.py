# =========================================================================== #
#                          REGRESSION CLASSES                                 #
# =========================================================================== #
"""Regression classes."""
from abc import abstractmethod
import numpy as np

from .gradient_descent import GradientDescent
from ..operations.metrics import RegressionMetric
from ..operations.metrics import RegressionMetrics
from ..operations.cost import RegressionCostFunction
from ..operations.cost import RegressionCostFunctions

import warnings

# --------------------------------------------------------------------------- #
#                          REGRESSION CLASS                                   #
# --------------------------------------------------------------------------- #

class Regression(GradientDescent):
    """Base class for all regression classes."""

    DEFAULT_METRIC = 'mean_squared_error'
    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='quadratic', metric='mean_squared_error', 
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(Regression, self).__init__(learning_rate=learning_rate,
                                              batch_size=batch_size,
                                              theta_init=theta_init, epochs=epochs,
                                              cost=cost, metric=metric,                                               
                                              verbose=verbose,
                                              checkpoint=checkpoint, 
                                              name=name, seed=seed)    
 

    def _get_cost_function(self):
        """Obtains the cost function associated with the cost parameter."""
        cost_function = RegressionCostFunctions()(cost=self.cost)
        if not isinstance(cost_function, RegressionCostFunction):
            msg = str(self.cost) + ' is not a supported regression cost function.'
            raise ValueError(msg)
        else:
            return cost_function

    def _get_scorer(self):
        """Obtains the scoring function associated with the metric parameter."""
        scorer = RegressionMetrics()(metric=self.metric)
        if not isinstance(scorer, RegressionMetric):
            msg = str(self.metric) + ' is not a supported regression metric.'
            raise ValueError(msg)
        else:
            return scorer     
        
    def _predict(self, X):
        """Computes predictions during training with current weights."""
        self._validate_data(X)
        y_pred = self._linear_prediction(X)
        return y_pred.ravel()

    def predict(self, X):
        """Predicts output as a linear function of inputs and final parameters.

        The method computes predictions based upon final parameters; therefore,
        the model must have been trained.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for which predictions will be rendered.

        Returns
        -------
        array, shape(n_samples,)
            Returns the linear regression prediction.        
        """
        return self._predict(X)

    def score(self, X, y):
        """Computes a score for the current model, given inputs X and output y.

        The score uses the class associated the metric parameter from class
        instantiation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for which predictions will be rendered.

        y : numpy array, shape (n_samples,)
            Target values             

        Returns
        -------
        float
            Returns the score for the designated metric.
        """
        self._validate_data(X, y)
        y_pred = self.predict(X)
        if self.metric:
            score = self.scorer(y=y, y_pred=y_pred)    
        else:
            score = RegressionMetrics()(metric=self.DEFAULT_METRIC)(y=y, y_pred=y_pred)        
        return score

# --------------------------------------------------------------------------- #
#                         LINEAR REGRESSION CLASS                             #
# --------------------------------------------------------------------------- #


class LinearRegression(Regression):
    """Performs linear regression with gradient descent."""

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='quadratic', 
                 metric='mean_squared_error', early_stop=None, 
                 verbose=False, checkpoint=100, name=None, 
                 seed=None):
        super(LinearRegression, self).__init__(learning_rate=learning_rate,
                                              batch_size=batch_size,
                                              theta_init=theta_init, epochs=epochs,
                                              cost=cost,
                                              metric=metric, early_stop=early_stop,
                                              verbose=verbose,
                                              checkpoint=checkpoint, name=name, seed=seed)

