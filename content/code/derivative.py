# ============================================================================ #
#                               DERIVATIVE                                     #
# ============================================================================ #
# Plot showing derivative as slope of tangent line
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
# Compute coordinates of secant line
def secant(P,Q):
    # Compute slope
    m = (P[1]-Q[1])/(P[0]-Q[0])
    # Compute interept
    b = P[1]- m * P[0]
    # Designate x-lim
    x = np.linspace(P[0]-1,Q[0]+1,50)
    y = m*x+b
    return x, y

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
P = np.array([x0,y0])
N = 50
x_Q = np.linspace(x0, x0+delta_x, N)
y_Q = f(x_Q)
# ---------------------------------------------------------------------------- #
#%%
# Create curve
curve = go.Scatter(x=x, y=y,
            mode="lines",
            visible=True,
            line=dict(width=2, color="#1560bd"))
# ---------------------------------------------------------------------------- #
#%%
# Point x0            
point_x0 = go.Scatter(x=[x0], y=[0], 
            mode='markers+text',
            name='x',        
            marker=dict(color='black'),
            text=r'x',
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
# Point P projection            
projection_P = go.Scatter(x=np.full(10,x0), y=np.linspace(0,y0,10), 
            mode='lines',                  
            line=dict(dash='dash',color='grey')
)
# ---------------------------------------------------------------------------- #
#%%
# Create frames
def texts(t,n):
    if n != 0:
        return t
    else:
        return ""
frames = []
for k in reversed(range(N)):
    frame=go.Frame(
        data=[
            go.Scatter(
                x=[x_Q[k]],
                y=[y_Q[k]],
                mode="markers+text",            
                marker=dict(color="black"),
                text=r'Q',
                textposition='top center'),
            go.Scatter(
                x = secant(P=P,Q=np.array([x_Q[k],y_Q[k]]))[0],
                y = secant(P=P,Q=np.array([x_Q[k],y_Q[k]]))[1],
                mode="lines",
                line=dict(width=2, color="#f28500")),
            go.Scatter(
                x = secant(P=P,Q=np.array([x_Q[k],y_Q[k]]))[0],
                y = secant(P=P,Q=np.array([x_Q[k],y_Q[k]]))[1],
                mode="lines",
                line=dict(width=2, color="#f28500")),                
            go.Scatter(
                x=[x_Q[k]],
                y=[0],
                mode='markers+text',
                name=r'$x+h$',        
                marker=dict(color='black'),
                text=texts(t=r'$x+h$',n=k),
                textposition='bottom center'),
            go.Scatter(
                x=np.full(10,[x_Q[k]]),
                y=np.linspace(0,y_Q[k],10),
                mode='lines',                  
                line=dict(dash='dash',color='grey')
            ),
            go.Scatter(
                x=np.linspace(x0,x_Q[k]+1,10),
                y=np.full(10,y0),
                mode='lines',                  
                line=dict(dash='dash',color='grey')
            ),
            go.Scatter(
                x=[x_Q[k]+1],
                y=[y0+(y_Q[k]-y0)/2],
                mode='text',
                name = r"$f(x+h)-f(x)$",
                text=r"$f(x+h)-f(x)$",
                textposition="middle right"
            ),
            go.Scatter(
                x=[x0+(x_Q[k]-x0)/2],
                y=[y0-1],
                mode='text',
                name=r'$h$',        
                text=texts(t=r'$h$',n=k),
                textposition='bottom center')
        ]
        )
    frames.append(frame)

final_frame=go.Frame(
    data=[
        go.Scatter(
            x=tangent(x0,y0)[0],
            y=tangent(x0,y0)[1],
            mode="lines",
            line=dict(width=2, color="#f28500")
        )
    ]
)
frames.append(final_frame)
    
# ---------------------------------------------------------------------------- #
#%%
# Create figure
ax = dict(
  zeroline = False,
  showline = False,
  showticklabels = False,
  showgrid = False    
) 
fig = go.Figure(
    data=[curve, curve, curve, curve, curve, curve, curve, curve,
          point_x0, point_x0, point_P, 
          point_P, projection_P],
    layout=go.Layout(
        xaxis=dict(range=[xm, xM], showticklabels=False, autorange=False, zeroline=False),
        yaxis=dict(range=[ym, yM], showticklabels=False, autorange=False, zeroline=False),
        title_text="Derivative as a Limit", hovermode="closest",
        width=800,
        height=600,
        margin=dict(l=40,r=40,t=40,b=0),
        showlegend=False,
        template='plotly_white',
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])]),
    frames=frames
)


po.plot(fig, filename='./content/figures/derivative.html',include_mathjax='cdn', auto_open=False)

