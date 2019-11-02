# ============================================================================ #
#                                 SIGMOID                                      #
# ============================================================================ #
# Renders stacked line charts showing convex and non-convex obj function
#%%
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
# ---------------------------------------------------------------------------- #
#                                 CONVEX                                       #
# ---------------------------------------------------------------------------- #
# Convex Formula
def sigmoid(z):
    return 1/(1+np.exp(-z))
# ---------------------------------------------------------------------------- #
# Data
x = np.linspace(-10,10,100)
y = sigmoid(x)
# ---------------------------------------------------------------------------- #
# Trace
data = go.Scatter(x=x, y=y,
                      mode='lines',
                      name='Sigmoid Function',
                      marker=dict(
                          size=2,
                          color='#1560bd'
                          )
                    )

# ---------------------------------------------------------------------------- #
# Layout
layout = go.Layout(title='Sigmoid Function', 
                   height=400,
                   width=800,
                   showlegend=False,
                   margin=dict(l=10,r=10,t=40,b=10),
                   template='plotly_white'
                   )
# ---------------------------------------------------------------------------- #
# Figure and plot
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename = "./content/figures/sigmoid.html", auto_open=False)
