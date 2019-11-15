# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
#%%
from IPython.display import HTML
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.visual.animations import MultiModelSearch3D, MultiModelFit2D
from ml_studio.utils.data_manager import StandardScaler
# --------------------------------------------------------------------------- #
# Set figure parameters
directory = "./content/figures"
params = {'axes.titlesize':'x-large'}
plt.rcParams.update(params)
# --------------------------------------------------------------------------- #
# Obtain data
X, y = datasets.load_boston(return_X_y=True)
X = X[:,5]
X = X.reshape(-1,1)
# --------------------------------------------------------------------------- #
# Standardize Feature
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# --------------------------------------------------------------------------- #
# Run models and create animations                               
#%%
models = {}
learning_rates = [0.001, 0.01, 0.1]
names = ['Learning Rate: 0.001','Learning Rate: 0.01', 'Learning Rate: 0.1']

for i in range(len(learning_rates)):
    bgd = LinearRegression(epochs=500, learning_rate=learning_rates[i],
                                   name=names[i])
    models[names[i]] = bgd.fit(X,y)
ani = MultiModelSearch3D()
ani.search(models, directory=directory, filename='search_by_learning_rate.gif')
ani = MultiModelFit2D()
ani.fit(models, directory=directory, filename='fit_by_learning_rate.gif')


#%%
