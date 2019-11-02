# ============================================================================ #
#                        MULTINOMIAL LOGISTIC REGRESSION                       #
# ============================================================================ #
# Renders stacked line charts showing convex and non-convex obj function
#%%
# Load libraries
import plotly.offline as po
import plotly.graph_objs as go
import numpy as np
from sklearn import datasets
from ml_studio.supervised_learning.classification import MultinomialLogisticRegression
from ml_studio.utils.data_manager import StandardScaler, train_test_split

directory = "./content/figures/"
# ---------------------------------------------------------------------------- #
#                                   DATA                                       #
# ---------------------------------------------------------------------------- #
#%%
# Load wine dataset 
X, y = datasets.load_wine(return_X_y=True)
# Standardize Features
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
# Split data into training and test set
X_train, X_test, y_train, y_test =  train_test_split(X_scaled,y, seed=5)
# ---------------------------------------------------------------------------- #
#                            LOGISTIC REGRESSION                               #
# ---------------------------------------------------------------------------- #
#%%
# Create Multinomial Linear Regression Classifier
clf = MultinomialLogisticRegression(epochs=500, learning_rate=0.001, 
                                    metric='accuracy', seed=5)
# Train the model                                    
clf.fit(X_train,y_train)
# ---------------------------------------------------------------------------- #
#                            LEARNING CURVE                                    #
# ---------------------------------------------------------------------------- #
#%%
# Plot Learning Curve
history = clf.history
costs = history.epoch_log['train_cost']
data = go.Scatter(
    x=np.linspace(0,len(costs), len(costs)),
    y=costs,
    mode='lines',
    line=dict(color='steelblue')
)
layout = go.Layout(title='Wine Dataset Learning Curve', 
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
po.plot(fig, filename = "./content/figures/wine.html", auto_open=False)  
# ---------------------------------------------------------------------------- #
#                            EVALUATE                                          #
# ---------------------------------------------------------------------------- #
#%%
# Evaluate model on test set
score = clf.score(X_test, y_test)
# Print score to sysout
print(score)
#%%
