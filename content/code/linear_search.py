# ============================================================================ #
#                           LINEAR SEARCH                                      #
# ============================================================================ #
# Interactive plot showing linear function optimization with one step
#%%
import plotly.offline as po
import plotly.graph_objs as go
import numpy as np
po.init_notebook_mode()

#%%
# ---------------------------------------------------------------------------- #
# Formula
theta = np.array([-5,-3])
b = 4
def f(X, theta=theta, b=b):
    return np.dot(X.T, theta) + b
# ---------------------------------------------------------------------------- #
#%%
# Compute vector given point, length and angle
def vector(x, angle, length=1):
    angle = angle * np.pi / 180
    return np.array([length*np.cos(angle)+x[0], length*np.sin(angle)+x[1]])
x=np.array([0,0])
z_start = f(x)

# ---------------------------------------------------------------------------- #
#%%
# Get data
x0 = np.linspace(0, 2, 10)
x1 = np.linspace(0, 2, 10)
X0, X1 = np.meshgrid(x0, x1)
X = np.array((X0,X1))
Z = f(X)
# ---------------------------------------------------------------------------- #
#%%
# Create unit vector in opposite direction of theta 
u = -theta / (theta**2).sum()**0.5 
# Compute the value of the function at (x+u)
z = f(x+u)
# Compute empirical solution
angles = np.linspace(0,90,90)
theta_hats = [vector(x, angle) for angle in angles]
costs = [f(x0,x1) for x0, x1 in theta_hats]
angle_prime = np.argmin(costs)
theta_hat = theta_hats[np.argmin(costs)]

# ---------------------------------------------------------------------------- #
#%%
# Create figure
fig = go.Figure()
# ---------------------------------------------------------------------------- #
#%%
# Add surface and starting point traces
fig.add_trace(
    go.Surface(x=X0, y=X1, z=Z, colorbar=None, 
        name='Linear Surface',
        colorscale='Viridis',        
        showscale=False,
        opacity=0.5)
)
# Starting point at z=0
fig.add_trace(
    go.Scatter3d(x=[x[0]], y=[x[1]], z=[np.min(Z)],
        mode='markers+text',
        name='Start Point',        
        marker=dict(
            size=4,
            color='blue'
            ),
        text=r'x',
        textposition='top center',
        )        
)
# Starting point at z != 0
fig.add_trace(
    go.Scatter3d(x=[x[0]], y=[x[1]], z=[f(x)],
        mode='markers+text',
        name='Start Point',
        marker=dict(
            size=4,
            color='blue'
        ),
        text='A',
        textposition='top center')
)      
# Add background scatterplot to be used as frame for text
fig.add_trace(
    go.Scatter(
        x=[0,10], 
        y=[0,10], 
        mode='text',        
        name='Text Frame',
        text=[]
        )
)
# ---------------------------------------------------------------------------- #
#%%
# Add iterative traces
for angle in np.arange(0,90,1):
    # Get new solution
    u = vector(x, angle)
    # Compute new vector components 
    u_x = np.linspace(x[0],u[0],10)
    u_y = np.linspace(x[1],u[1],10)
    u_z0 = np.full(len(u_y), np.min(Z)) 
    u_z1 = np.linspace(z_start,f(x+u),10)
    # Compute projection line
    p_x = np.full(50,u[0])
    p_y = np.full(50,u[1])
    p_z = np.linspace(np.min(Z),f(x+u),50)
    # Print current solution
    text = r'$\vec{{u}} = \text{{{}}}$'.format(str(np.round(u,4)))
    fig.add_trace(
        go.Scatter(
            x=[5],
            y=[9],
            mode='text',        
            name='Theta Hat',
            text=[text],
            visible=False,
            textposition="top center",
            textfont=dict(
                size=18
            )
        )
    )
    # Print current costs
    text = r'$f(\vec{{x}}+\vec{{u}}) = \text{{{}}}$'.format(str(np.round(f(x+u), 4)))    
    fig.add_trace(
        go.Scatter(
            x=[5],
            y=[8.3],
            mode='text',        
            name='Cost',
            text=[text],
            visible=False,
            textposition="top center",
            textfont=dict(
                size=18
            )            
        )
    )    
    # Add new point (z=0) trace
    fig.add_trace(
        go.Scatter3d(x=[u[0]], y=[u[1]], z=[u_z0[0]],
            mode='markers+text',
            name = 'New Point',
            visible=False,
            marker=dict(
                size=4,
                color='red'
            )
        )
    )
    # Add new point trace
    fig.add_trace(
        go.Scatter3d(x=[u[0]], y=[u[1]], z=[u_z1[-1]],
            mode='markers+text',
            name = 'New Point',
            visible=False,
            marker=dict(
                size=4,
                color='red'
            ),
            text='B',
            textposition='top center')                                                          

    )
    # Add new path (z=0)
    fig.add_trace(
        go.Scatter3d(x=u_x, y=u_y, z=u_z0,
            mode='lines',
            name = 'New Path (z=0)',
            visible=False,
            line=dict(
                width=2,
                color='blue'
            )
        )
    )
    # Add new path 
    fig.add_trace(
        go.Scatter3d(x=u_x, y=u_y, z=u_z1,
            mode='lines',
            name = 'New Path (z=0)',
            visible=False,
            line=dict(
                width=2,
                color='blue'
            )
        )
    )          
    # Annotate path with unit vector symbol  
    fig.add_trace(
        go.Scatter3d(x=[u_x[5]], y=[u_y[5]], z=[u_z0[5]],
            mode='text',
            name = 'Unit Vector',
            visible=False,
            text=r'u',
            textposition='top center'
        )
    )                                            
    # Add projection line 
    fig.add_trace(
        go.Scatter3d(x=p_x, y=p_y, z=p_z,
            mode='lines',
            name = 'Projection Line',
            visible=False,
            line=dict(
                width=0.5,
                color='blue'
            )
        )
    )                        
# ---------------------------------------------------------------------------- #
#%%
# Set visibility for all steps
steps = []
num_angles = 90
num_traces_update = 7
for i in range(num_angles):
    # Hide all traces
    step = dict(
        method = 'restyle',
        args = ['visible', [False] * len(fig.data)],
        label=str(i)
    )
    # Show static traces (surface, start points)
    step['args'][1][0] = True
    step['args'][1][1] = True
    step['args'][1][2] = True
    step['args'][1][3] = True
    step['args'][1][4+(i*num_traces_update)] = True
    step['args'][1][4+(i*num_traces_update)+1] = True
    step['args'][1][4+(i*num_traces_update)+2] = True
    step['args'][1][4+(i*num_traces_update)+3] = True
    step['args'][1][4+(i*num_traces_update)+4] = True
    step['args'][1][4+(i*num_traces_update)+5] = True
    step['args'][1][4+(i*num_traces_update)+6] = True
    step['args'][1][4+(i*num_traces_update)+7] = True
    # Add step to steps
    steps.append(step)

# ---------------------------------------------------------------------------- #
#%%
# Create slider
sliders = [dict(
  steps=steps,
  currentvalue = dict(
      prefix='Direction in Degrees: '
  )
  )]
# ---------------------------------------------------------------------------- #
#%%
# Add layout and scene

scene = dict(
    camera=dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=2, z=0.1)
        ),
    xaxis=dict(title='x0'),
    yaxis=dict(title='x1'),
    zaxis=dict(title='z')
)
ax = dict(
  zeroline = False,
  showline = False,
  showticklabels = False,
  showgrid = False    
)                     
fig.update_layout(title='Surface Plot for Linear Function in 2D', 
                  scene=scene,
                  height=600,
                  width=900,
                  xaxis=ax,
                  yaxis=ax,
                  sliders=sliders,
                  showlegend=False,
                  margin=dict(l=0,r=0,t=40,b=0),
                  template='plotly_white'
                )                     
po.plot(fig, filename='./content/figures/linear_search.html',include_mathjax='cdn', auto_open=True)


#%%
