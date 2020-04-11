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
from ml_studio.supervised_learning.training.metrics import MSE
from ml_studio.utils.data_manager import StandardScaler, data_split
from ml_studio.visual.animations import SingleModelFit2D, SingleModelSearch3D
from ml_studio.visual.plots import plot_loss, plot_score
directory = "./content/figures/"
# ---------------------------------------------------------------------------- #
#                                   DATA                                       #
# ---------------------------------------------------------------------------- #
#%%
# Data
X, y, coef = datasets.make_regression(n_samples=1000000, n_features=1000, bias=10, 
                                noise=40, coef=True, random_state=50)
scaler = StandardScaler()
scaler.fit(X)                                
X = scaler.transform(X)                                
X_train, X_test, y_train, y_test = data_split(X,y, test_size=0.3, seed=50)     

# ---------------------------------------------------------------------------- #
#                             LINEAR REGRESSION                                #
# ---------------------------------------------------------------------------- #
#%%
# Linear Regression
lr = LinearRegression(epochs=1000, learning_rate=0.01, val_size=0.2, patience=40,
                      early_stop=True, metric='mean_squared_error', 
                      verbose=True, checkpoint=100)
lr.fit(X_train,y_train)
# ---------------------------------------------------------------------------- #
#                      PLOT LOSS AND SCORES BY EPOCH                           #
# ---------------------------------------------------------------------------- #
#%%
# Plot loss and scores by epoch
plot_loss(lr, directory=directory, filename="Loss History on Simulated Data 2.png")
plot_score(lr, directory=directory, filename="Score History on Simulated Data 2.png")


# %%
# ---------------------------------------------------------------------------- #
#                   SCORES FOR LEARNED VERSUS ACTUAL MODEL                     #
# ---------------------------------------------------------------------------- #
def f_actual(x):
    y = (10 + np.dot(coef, x)).flatten()
    return y
y_test_actual = f_actual(X_test)
score_actual = MSE()(y_test, y_test_actual)
score_pred = lr.score(X_test, y_test)
print(score_actual)
print(score_pred)

# ---------------------------------------------------------------------------- #
#                   PLOT SCORES FOR ACTUAL VS LEARNED MODEL                    #
# ---------------------------------------------------------------------------- #
# %%
# Print actual vs predicted R2
x = ['Actual Model', 'Learned Model']
y = [score_actual, score_pred]
data = [go.Bar(x=x,y=y,
                text=np.round(y,4),
                textposition='auto',
                showlegend=False,
                marker_color='rgb(55, 83, 109)')]

layout = go.Layout(title='Mean Squared Error for Actual vs Learned Model on Test Data', 
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
po.plot(fig, filename = "./content/figures/linear_regression_test_actual_v_predicted_mse.html", auto_open=False)  

# %%
