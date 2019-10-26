# ============================================================================ #
#                                 CONVEXITY                                    #
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
def convex(x):
    return np.square(x)
# ---------------------------------------------------------------------------- #
# Convex Data
x_convex = np.linspace(-5, 5, 100)
y_convex = convex(x_convex)
# ---------------------------------------------------------------------------- #
# Convex points definitions
convex_points_list = [
    {'label': 'x0', 'value':-3, 'color':'black'},
    {'label': 'Global Minimum', 'value': 0, 'color': 'red'},
    {'label': 'x1', 'value': 3, 'color':'black'}]
# ---------------------------------------------------------------------------- #
# Convex Traces
convex_points = [go.Scatter(
    x=[p['value']],
    y=[convex(p['value'])],
    mode='markers+text',
    marker=dict(
        size=8,
        opacity=1,
        color=p['color']
    ),
    text=p['label'],
    textposition='bottom center'
    )
    for p in convex_points_list]

t_convex = go.Scatter(x=x_convex, y=y_convex,
                      mode='lines',
                      name='Convex Function',
                      marker=dict(
                          size=2,
                          color='#2F5168'
                          )
                    )

# ---------------------------------------------------------------------------- #
# Convex Layout
layout = go.Layout(title='Convex Objective Function', 
                   height=400,
                   width=800,
                   showlegend=False,
                   margin=dict(l=10,r=10,t=40,b=10),
                   template='plotly_white'
                   )
convex_points.append(t_convex)
data = convex_points
# ---------------------------------------------------------------------------- #
# Figure and plot
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename = "./content/figures/convex.html", auto_open=False)
#%%
# ---------------------------------------------------------------------------- #
#                                 CONVEX                                       #
# ---------------------------------------------------------------------------- #
# Nonconvex formula
def nonconvex(x):
    return np.sin(x)+np.sin((10/3)*x)    
# ---------------------------------------------------------------------------- #
# Nonconvex data
x_nonconvex = np.linspace(-2, 2, 100)
y_nonconvex = nonconvex(x_nonconvex)
# ---------------------------------------------------------------------------- #
# Find minimums
x_range = np.linspace(-1,0,100)
x_global_min = x_range[np.argmin(nonconvex(x_range))]
x_range = np.linspace(1,2,100)
x_local_min = x_range[np.argmin(nonconvex(x_range))]

# ---------------------------------------------------------------------------- #
# Nonconvex points 
nonconvex_points_list = [
    {'label': 'Global Minimum', 'value':x_global_min, 'color': 'red'},
    {'label': 'x0', 'value':0.1, 'color':'black'},
    {'label': 'x1', 'value': 1, 'color':'black'},
    {'label': 'Local Minimum', 'value':x_local_min, 'color': 'green'}]
# ---------------------------------------------------------------------------- #
# Nonconvex Traces
nonconvex_points = [go.Scatter(
    x=[p['value']],
    y=[nonconvex(p['value'])],
    mode='markers+text',
    marker=dict(
        opacity=1,
        size=8,
        color=p['color']
    ),
    text=p['label'],
    textposition='bottom center'
    )
    for p in nonconvex_points_list]

t_nonconvex = go.Scatter(x=x_nonconvex, y=y_nonconvex,
                      mode='lines',
                      name='Non-Convex Function',
                      marker=dict(
                          size=2,
                          color='#2F5168'
                          )
                    )

# ---------------------------------------------------------------------------- #
# Nonconvex Layout
layout = go.Layout(title='Non-Convex Objective Function', 
                   height=400,
                   width=800,
                   showlegend=False,
                   margin=dict(l=10,r=10,t=40,b=10),
                   template='plotly_white'
                   )
nonconvex_points.append(t_nonconvex)
data = nonconvex_points
# ---------------------------------------------------------------------------- #
# Figure and plot
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename = "./content/figures/nonconvex.html", auto_open=False)

