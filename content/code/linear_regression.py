# ============================================================================ #
#                                 CONVEXITY                                    #
# ============================================================================ #
# Renders stacked line charts showing convex and non-convex obj function
#%%
import pandas as pd
import plotly.offline as po
import plotly.graph_objs as go
import numpy as np
from sklearn import datasets
from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.utils.data_manager import StandardScaler
from ml_studio.visual.animations import SingleModelFit2D, SingleModelSearch3D
directory = "./content/figures/"
# ---------------------------------------------------------------------------- #
#                                   DATA                                       #
# ---------------------------------------------------------------------------- #
#%%
# Data
boston = datasets.load_boston()
bdf = pd.DataFrame(boston['data'], columns=boston['feature_names'])
bdf['Price']=boston['target']
bdf = bdf[['RM', 'Price']]
bdf = bdf.rename(columns={"RM": 'Rooms'})
X = bdf['Rooms']
X = X.to_numpy().reshape(-1,1)
y = bdf['Price']
y = y.to_numpy()
# Data transformation
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
# ---------------------------------------------------------------------------- #
#                                    PLOT                                      #
# ---------------------------------------------------------------------------- #
#%%
# Linear Regression Scatterplot
data = go.Scatter(
    x=X[:,0],
    y=y,
    mode='markers',
    marker=dict(color='steelblue')
)
layout = go.Layout(title='Boston Housing Prices by Rooms', 
                   height=400,
                   width=800,
                   showlegend=False,
                   margin=dict(l=10,r=10,t=40,b=10),
                   template='plotly_white'
                   )
fig = go.Figure(data=data, layout=layout)      
fig.show()
po.plot(fig, filename = "./content/figures/boston.html", auto_open=False)             
# ---------------------------------------------------------------------------- #
#                             LINEAR REGRESSION                                #
# ---------------------------------------------------------------------------- #
#%%
# Linear Regression
lr = LinearRegression(epochs=50, learning_rate=0.05)
lr.fit(X_scaled,y)
plot = SingleModelSearch3D()
plot.search(lr, directory=directory, filename="linear_regression_search.gif")
plot = SingleModelFit2D()
plot.fit(lr, directory=directory, filename="linear_regression_fit.gif")
#%%


# %%
