# %%
# ============================================================================ #
#                                AMES_DEMO                                     #
# ============================================================================ #
# Renders gradient descent search and fitfor ames housing dataset sample.
# Import System modules
import inspect
import os
import sys
# Add MLStudio Path
home = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
mls = os.path.join(os.path.dirname(home), "MLStudio")
sys.path.append(mls)

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)

from ml_studio.legos.scorer import Scorer
from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.visual.animations import SingleModelSearch3D, SingleModelFit2D


# ---------------------------------------------------------------------------- #
#                                 DATA                                         #
# ---------------------------------------------------------------------------- #
X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=5)
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)
# Reshape features
X_train = X_train.values.reshape(-1,1)
# Standardize Data
X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
#%%
# Set Parameters
learning_rate = 0.01
maxiter = 10000  
batch_size=None
precision = 0.01
early_stop=False
patience = 10
metric = 'neg_root_mean_squared_error'

 
# Instantiate BGD object and fit data
lr = LinearRegression(learning_rate=learning_rate, maxiter=maxiter, early_stop=early_stop, 
                      batch_size=batch_size, metric=metric, print_costs=None)
lr.fit(X,y)
#%%
sms = SingleModelSearch3D()      
directory = "./figures"
filename = "ames_search.gif"
ames_search = sms.search(lr, directory=directory, filename=filename)
#%%
smf = SingleModelFit2D()      
directory = "./figures"
filename = "ames_fit.gif"
ames_fit = smf.fit(lr, directory=directory, filename=filename)
