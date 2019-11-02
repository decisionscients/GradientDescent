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
from ml_studio.supervised_learning.classification import LogisticRegression
from ml_studio.utils.data_manager import StandardScaler, train_test_split

directory = "./content/figures/"
# ---------------------------------------------------------------------------- #
#                                   DATA                                       #
# ---------------------------------------------------------------------------- #
#%%
# Data
X, y = datasets.load_breast_cancer(return_X_y=True)
# Data transformation
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test =  train_test_split(X_scaled,y)
# ---------------------------------------------------------------------------- #
#                            LOGISTIC REGRESSION                               #
# ---------------------------------------------------------------------------- #
#%%
# Linear Regression
clf = LogisticRegression(epochs=500, learning_rate=0.05, metric='accuracy')
clf.fit(X_train,y_train)
history = clf.history
costs = history.epoch_log['train_cost']
# ---------------------------------------------------------------------------- #
#                            LEARNING CURVE                                    #
# ---------------------------------------------------------------------------- #
#%%
# Learning Curve
data = go.Scatter(
    x=np.linspace(0,len(costs), len(costs)),
    y=costs,
    mode='lines',
    line=dict(color='steelblue')
)
layout = go.Layout(title='Wisconsin Breast Cancer Dataset Learning Curve', 
                   xaxis_title="Epochs",
                   yaxis_title='Cross-Entropy Cost',
                   height=400,
                   width=800,
                   showlegend=False,
                   margin=dict(l=10,r=10,t=40,b=10),
                   template='plotly_white'
                   )
fig = go.Figure(data=data, layout=layout)      
#fig.show()
po.plot(fig, filename = "./content/figures/breast_cancer.html", auto_open=False)  
# ---------------------------------------------------------------------------- #
#                            EVALUATE                                          #
# ---------------------------------------------------------------------------- #
#%%
# Evaluate
intercept = clf.intercept
coef = clf.coef
score = clf.score(X_test, y_test)
print(score)
#%%