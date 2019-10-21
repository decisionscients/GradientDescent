# =========================================================================== #
#                            TRANSFORMERS                                     #
# =========================================================================== #
''' This module contains custom transformers for the sklearn Pipeline Class'''
# --------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# --------------------------------------------------------------------------- #
class AddBiasTerm(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):         
        return self

    def transform(self, X):
        bias = np.ones(shape=(X.shape[0], 1))
        if isinstance(X, pd.DataFrame):
            bias = pd.DataFrame({'X0':bias[:,0]})
            X_new = pd.concat([bias, X], axis=1)
        elif len(X.shape) == 1:
            X = np.reshape(X,(-1,1))        
            X_new = np.append(bias, X, axis=1)
        else:
            X_new = np.append(bias, X, axis=1)
        return(X_new)
        
class RangeScaler(BaseEstimator, TransformerMixin):

    def __init__(self, df=False):
        self._scaler = None
        self._df = df

    def fit(self, X, y=None): 
        self._scaler = MinMaxScaler()
        return self

    def transform(self, X):
        if len(X.shape) == 1:
            X_new = np.reshape(X, (-1,1))
            X_new = self._scaler.fit_transform(X_new)
            X_new = X_new.flatten()
        else:
            X_new = self._scaler.fit_transform(X)
        return(X_new)

