# =========================================================================== #
#                          GRADIENT DESCENT CLASS                             #
# =========================================================================== #
"""Regression classes."""
from abc import ABC, abstractmethod, ABCMeta
import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils import shuffle
import warnings

from ..utils.data_manager import batch_iterator, data_split
from ..operations.monitor import History, Progress

# --------------------------------------------------------------------------- #

class GradientDescent(ABC, BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """Base class gradient descent estimator."""

    DEFAULT_METRIC = 'mean_squared_error'

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='quadratic', metric='mean_squared_error', 
                 verbose=False,  checkpoint=100, name=None, seed=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.theta_init = theta_init
        self.epochs = epochs
        self.cost = cost
        self.metric = metric        
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.name = name
        self.seed = seed
        # Instance variables
        self.epoch = 0
        self.batch = 0
        self.converged = False
        self.theta = None
        self.cost_function = None              
        self.algorithm = None
        # Attributes
        self.coef_ = None
        self.intercept_ = None       
        self.epochs_ = 0 

    def set_params(self, **kwargs):
        """Sets parameters to **kwargs and validates."""
        super().set_params(**kwargs)
        self._validate_params()
        return self

    def _validate_params(self):
        """Validate parameters."""
        if self.batch_size is not None:
            if not isinstance(self.batch_size, int):
                raise TypeError("batch_size must provide an integer.")            
        if self.theta_init is not None:
            if not isinstance(self.theta_init, (list, pd.core.series.Series, np.ndarray)):
                raise TypeError("theta must be an array like object.")            
        if not isinstance(self.epochs, int):
            raise TypeError("epochs must be an integer.")        
        if self.metric is not None:
            if not isinstance(self.metric, str):
                raise TypeError("metric must be string containing name of metric for scoring")                
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be either True or False")
        if self.checkpoint is not None:
            if not isinstance(self.checkpoint, int):
                raise TypeError(
                    "checkpoint must be a positive integer or None.")
            elif self.checkpoint < 0:
                raise ValueError(
                    "checkpoint must be a positive integer or None.")
            elif self.checkpoint > self.epochs:
                warnings.warn(UserWarning(
                    "checkpoint must not be greater than the number of epochs."))
        if self.seed is not None:
            if not isinstance(self.seed, int):
                raise TypeError("seed must be a positive integer.")

    def _validate_data(self, X, y=None):
        """Validates and reports n_features."""
        if not isinstance(X, (np.ndarray)):
            raise TypeError("X must be of type np.ndarray")
        if y is not None:
            if not isinstance(y, (np.ndarray)):
                raise TypeError("y must be of type np.ndarray")            
            if len(y.shape) > 1:
                raise ValueError("y should be of shape (m,) not %s" % str(y.shape))
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y have incompatible lengths")        

    def _prepare_data(self, X, y):
        """Prepares training (and validation) data."""
        self.X = self.y = None
        # Add a column of ones to train the intercept term
        self.X = np.insert(X, 0, 1, axis=1)  
        self.y = y
        self.n_features_ = self.X.shape[1]

    def _evaluate_epoch(self, log=None):
        """Computes training (and validation) costs and scores."""
        log = log or {}
        # Update log with current learning rate and parameters theta
        log['epoch'] = self.epoch
        log['theta'] = self.theta.copy()        
        # Compute costs 
        y_pred = self._predict(self.X)
        log['train_cost'] = self.cost_function(y=self.y, y_pred=y_pred)
        # Compute scores if a metric was provided
        if self.metric is not None:            
            log['train_score'] = self.score(self.X, self.y)

        return log

    @abstractmethod
    def _get_cost_function(self):
        """Obtains the cost function for the cost parameter."""
        pass

    @abstractmethod        
    def _get_scorer(self):
        """Obtains the scoring function for the metric parameter."""
        pass

    def _compile(self):
        """Obtains external objects and add key functions to the log."""
        self.cost_function = self._get_cost_function()
        self.scorer = self._get_scorer()        

    def _init_weights(self):
        """Initializes weights"""        
        if self.theta_init is not None:
            if self.theta_init.shape[0] != self.X.shape[1]:
                raise ValueError("theta_init shape mispatch. Expected shape %s,"
                                 " but theta_init.shape = %s." % ((self.X.shape[1],1),
                                 self.theta_init.shape))
            else:
                self.theta = np.atleast_2d(self.theta_init).reshape(-1,1)
        else:
            n_features = self.X.shape[1]
            np.random.seed(seed=self.seed)
            self.theta = np.random.normal(size=n_features).reshape(-1,1)

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""        
        self._validate_params()
        self._validate_data(log.get('X'), log.get('y'))        
        self._prepare_data(log.get('X'), log.get('y'))
        self._init_weights()   
        self._compile()

    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]        
        self.epochs_ = self.epoch

    def _begin_epoch(self):
        """Increment the epoch count and shuffle the data."""
        self.epoch += 1
        self.X, self.y = shuffle(self.X, self.y, random_state=self.seed)
        if self.seed:
            self.seed += 1

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        # Compute performance statistics for epoch and post to history
        log = self._evaluate_epoch()

    def _begin_batch(self, log=None):
        self.batch += 1

    def _end_batch(self, log=None):
        pass

    def fit(self, X, y):
        """Trains model for a predesignated number of epochs.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        y : numpy array, shape (n_samples,)
            Target values 

        Returns
        -------
        self : returns instance of self.
        """
        train_log = {'X': X, 'y': y}
        self._begin_training(train_log)
        
        while (self.epoch < self.epochs):

            self._begin_epoch()

            for X_batch, y_batch in batch_iterator(self.X, self.y, batch_size=self.batch_size):

                self._begin_batch()
                # Compute prediction
                y_pred = self._predict(X_batch)
                # Compute costs
                J = self.cost_function(y=y_batch, y_pred=y_pred)
                # Update batch log with weights and cost
                batch_log = {'batch': self.batch, 'batch_size': X_batch.shape[0],
                             'theta': self.theta.copy(), 'train_cost': J}
                # Compute gradient and update weights
                gradient = self.cost_function.gradient(
                    X_batch, y_batch, y_pred)
                # Update parameters              
                self.theta -= self.learning_rate * gradient
                # Update batch log
                self._end_batch(batch_log)

            # Wrap up epoch
            self._end_epoch()

        self._end_training()
        return self
    
    def _linear_prediction(self, X):
        """Computes prediction as linear combination of inputs and thetas."""
        if X.shape[1] == self.theta.shape[0]:
            y_pred = X.dot(self.theta)
        else:
            if not hasattr(self, 'coef_') or self.coef_ is None:
                raise Exception("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})              
            y_pred = self.intercept_ + X.dot(self.coef_)  
        return y_pred            

    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y):
        pass

