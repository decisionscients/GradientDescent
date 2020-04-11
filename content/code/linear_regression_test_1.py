# ============================================================================ #
#                             LINEAR REGRESSION                                #
# ============================================================================ #
# Renders stacked line charts showing convex and non-convex obj function
#%%
import pandas as pd
import plotly.offline as po
import plotly.graph_objs as go
import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDRegressor
from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.supervised_learning.training.metrics import R2
from ml_studio.utils.data_manager import StandardScaler, data_split
from ml_studio.visual.animations import SingleModelFit2D, SingleModelSearch3D
directory = "./content/figures/"
# ---------------------------------------------------------------------------- #
#                                   DATA                                       #
# ---------------------------------------------------------------------------- #
#%%
# Data
X, y, coef = datasets.make_regression(n_samples=1000, n_features=1, bias=10, 
                                noise=40, coef=True, random_state=50)
X_train, X_test, y_train, y_test = data_split(X,y, test_size=0.3, seed=50)     
# ---------------------------------------------------------------------------- #
#                                SCATTER PLOT                                  #
# ---------------------------------------------------------------------------- #
#%%
# Linear Regression Scatterplot
data = go.Scatter(
    x=X_train.flatten(),
    y=y_train,
    mode='markers',
    marker=dict(color='steelblue')
)
layout = go.Layout(title='Simulated Data', 
                   height=400,
                   width=800,
                   showlegend=False,
                   xaxis_title="X",
                   yaxis_title="Y",
                   margin=dict(l=10,r=10,t=40,b=10),
                   template='plotly_white'
                   )
fig = go.Figure(data=data, layout=layout)      
fig.show()
po.plot(fig, filename = "./content/figures/simulated_training_data.html", auto_open=False)             
# ---------------------------------------------------------------------------- #
#                             LINEAR REGRESSION                                #
# ---------------------------------------------------------------------------- #
#%%
# Linear Regression
scaler = StandardScaler()
scaler.fit(X_train)                                
X_train = scaler.transform(X_train)
lr = LinearRegression(epochs=1000, learning_rate=0.01, val_size=0.2, patience=40,
                      early_stop=True, metric='r2', verbose=True, checkpoint=100)
lr.fit(X_train,y_train)
print(lr.intercept_)
print(lr.coef_.shape)
# ---------------------------------------------------------------------------- #
#                                ANIMATIONS                                    #
# ---------------------------------------------------------------------------- #
#%%
# Animations
plot = SingleModelSearch3D()
plot.search(lr, directory=directory, filename="linear_regression_search_test.gif")
plot = SingleModelFit2D()
plot.fit(lr, directory=directory, filename="linear_regression_fit_test.gif")
#%%
# ---------------------------------------------------------------------------- #
#                                  TEST                                        #
# ---------------------------------------------------------------------------- #
scaler.fit(X_test)                                
X_test = scaler.transform(X_test)

# %%
# ---------------------------------------------------------------------------- #
#                   PREDICTED VS ACTUAL ON TEST DATA                           #
# ---------------------------------------------------------------------------- #
def f_actual(x):
    y = (10 + np.dot(coef, x)).flatten()
    return y
def f_pred(x):
    y = (lr.intercept_ + X.dot(lr.coef_)).flatten()
    return y
x = np.linspace(start=np.min(X_test), stop=np.max(X_test), num=100)
y_actual = f_actual(x)
y_pred = f_pred(x)
# ---------------------------------------------------------------------------- #
#%%
# Print actual vs predicted fit
data = [go.Scatter(x=X_test.flatten(),y=y_test,
                mode='markers',
                name='Test Data',
                marker=dict(color='steelblue')),
        go.Scatter(x=x, y=y_actual,
                mode='lines',
                name='Actual Fit'),
        go.Scatter(x=x, y=y_pred,
                mode='lines',
                name='Predicted Fit')]

layout = go.Layout(title='Actual vs Predicted Simulated Test Data', 
                   height=400,
                   width=800,
                   showlegend=True,
                   xaxis_title="X",
                   yaxis_title="Y",
                   margin=dict(l=10,r=10,t=40,b=10),
                   template='plotly_white'
                   )
fig = go.Figure(data=data, layout=layout)      
fig.show()
po.plot(fig, filename = "./content/figures/linear_regression_test_actual_v_predicted_.html", auto_open=False)  

# ---------------------------------------------------------------------------- #
#                   PREDICTED VS ACTUAL R2 ON TEST DATA                        #
# ---------------------------------------------------------------------------- #
# %%
r2_pred = lr.score(X_test, y_test)
y_test_pred = f_actual(X_test)
r2_actual = R2()(y_test, y_test_pred)
# Print actual vs predicted R2
x = ['Actual Model', 'Learned Model']
y = [r2_actual, r2_pred]
data = [go.Bar(x=x,y=y,
                text=np.round(y,4),
                textposition='auto',
                showlegend=False,
                marker_color='rgb(55, 83, 109)')]

layout = go.Layout(title='R2 for Actual vs Predicted Simulated Test Data', 
                   height=400,
                   width=800,
                   showlegend=True,
                   xaxis_title="X",
                   yaxis_title="Y",
                   margin=dict(l=10,r=10,t=40,b=10),
                   template='plotly_white'
                   )
fig = go.Figure(data=data, layout=layout)      
fig.show()
po.plot(fig, filename = "./content/figures/linear_regression_test_actual_v_predicted_r2.html", auto_open=False)  

# %%
