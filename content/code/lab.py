#%%
import plotly.graph_objects as go
import plotly.offline as po
import numpy as np
# ---------------------------------------------------------------------------- #
#%%
# Formulas
def f(x):
    return np.square(x)
# Compute derivative    
def df(x):
    return 2 * x
# Compute coordinates of tangent line    
def tangent(x,y,d=10):
    p = np.array([x,y])
    m = df(x)
    v = np.array([1,m]) 
    l = (1**2+m**2)**(1/2)
    u = v/l
    line = np.array([p-d*u,p+d*u])
    xx = np.linspace(line[0][0], line[1][0],10)
    yy = np.linspace(line[0][1], line[1][1],10)
    return xx, yy
# ---------------------------------------------------------------------------- #
#%%
# Compute data
x = np.linspace(-2, 10, 100)
y = f(x)
xm = np.min(x) - 3
xM = np.max(x) + 3
ym = np.min(y) - 3
yM = np.max(y) + 3
delta_x = 6
x0 = 3
y0 = f(x0)
N = 100
x_Q = np.linspace(x0, x0+delta_x, N)
y_Q = f(x_Q)
# ---------------------------------------------------------------------------- #
#%%
# Create curve
curve = go.Scatter(x=x, y=y,
            mode="lines",
            visible=True,
            line=dict(width=2, color="blue"))
# ---------------------------------------------------------------------------- #
#%%
# Point x0            
point_x0 = go.Scatter(x=[x0], y=[0], 
            mode='markers+text',
            name='x0',        
            marker=dict(color='black'),
            text=r'x0',
            textposition='bottom center')            
# ---------------------------------------------------------------------------- #
#%%
# Point P            
point_P = go.Scatter(x=[x0], y=[y0], 
            mode='markers+text',
            name='P',        
            marker=dict(color='black'),
            text=r'P',
            textposition='top center')    
# ---------------------------------------------------------------------------- #
#%%
# Create figure
fig = go.Figure(
    data=[curve, point_x0,point_P],
    layout=go.Layout(
        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
        title_text="Kinematic Generation of a Planar Curve", hovermode="closest",
        showlegend=False,
        template='plotly_white',
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])]),
    frames=[go.Frame(
        data=[curve,
            point_x0,
            point_P,
            go.Scatter(
                x=[x_Q[k]],
                y=[y_Q[k]],
                mode="markers+text",            
                marker=dict(color="red", size=10),
                text=r'Q',
                textposition='top center'),
            go.Scatter(
                x = tangent(x_Q[k], y_Q[k])[0],
                y = tangent(x_Q[k], y_Q[k])[1],
                mode="lines",
                line=dict(width=2, color="red")),
            go.Scatter(
                x = tangent(x_Q[k], y_Q[k])[0],
                y = tangent(x_Q[k], y_Q[k])[1],
                mode="lines",
                line=dict(width=2, color="red"))
                ])
        for k in reversed(range(N))]
)

po.plot(fig, filename='./content/figures/kinematic.html',include_mathjax='cdn', auto_open=True)
#%%