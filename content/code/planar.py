# ============================================================================ #
#                                 PLANAR                                       #
# ============================================================================ #
# Renders t
#%%
import plotly.offline as po
import plotly.graph_objs as go
import numpy as np
from ipywidgets import interactive, HBox, VBox, widgets
po.init_notebook_mode()
# ---------------------------------------------------------------------------- #
# Formula
def f(x0,x1):
    x = np.array((x0,x1))    
    theta = np.array([-5,-3])
    b = 4
    return np.dot(x.T, theta) + b
# ---------------------------------------------------------------------------- #
#%%
# Compute vector given point, length and angle
def vector(x, y, angle, length=1):
    angle = angle * np.pi / 180
    return [length*np.cos(angle)+x, length*np.sin(angle)+y]
x_start=0
y_start=0    
z_start = f(x_start, y_start)

#%%
# ---------------------------------------------------------------------------- #
#%%
# Get data
x0 = np.linspace(0, 2, 10)
x1 = np.linspace(0, 2, 10)
X0, X1 = np.meshgrid(x0, x1)
Z = f(X0,X1)
# ---------------------------------------------------------------------------- #
#%%
# Compute antipated solution
theta = np.array([-5,-3])
theta_prime = -theta / (theta**2).sum()**0.5 # Compute unit vector
cost_prime = f(theta_prime[0], theta_prime[1])
# Compute empirical solution
angles = np.linspace(0,90,90)
theta_hats = [vector(x_start, y_start, angle) for angle in angles]
costs = [f(x0,x1) for x0, x1 in theta_hats]
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
    go.Scatter3d(x=[x_start], y=[y_start], z=[np.min(Z)],
        mode='markers+text',
        name='Start Point',        
        marker=dict(
            size=4,
            color='blue'
            )
        )
)
# Starting point at z != 0
fig.add_trace(
    go.Scatter3d(x=[x_start], y=[y_start], z=[f(x_start, y_start)],
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

# Text theoretical solution
text = r'$\\theta^\\prime = ' + str(theta_prime)
fig.add_trace(
    go.Scatter(
        x=[2], 
        y=[8], 
        mode='text',        
        name='Theta Prime',
        text=[text]
        )
)

# Text theoretical costs
text = 'cost: ' + str(cost_prime)
fig.add_trace(
    go.Scatter(
        x=[5], 
        y=[8], 
        mode='text',        
        name='Costs Prime',
        text=[text]
        )
)
# ---------------------------------------------------------------------------- #
#%%
# Add iterative traces
for angle in np.arange(0,90,1):
    # Get new solution
    v = vector(x_start,y_start,angle)
    # Compute new vector components 
    v_x = np.linspace(x_start,v[0],10)
    v_y = np.linspace(y_start,v[1],10)
    v_z0 = np.full(len(v_y), np.min(Z)) 
    v_z1 = np.linspace(z_start,f(v[0], v[1]),10)
    # Compute projection line
    u_x = np.full(50,v[0])
    u_y = np.full(50,v[1])
    u_z = np.linspace(np.min(Z),f(v[0], v[1]),50)
    # Print current solution
    text = r'$\\hat{\\theta} = ' + str(v) 
    fig.add_trace(
        go.Scatter(
            x=[2],
            y=[6],
            mode='text',        
            name='Theta Hat',
            text=[text]
        )
    )
    # Print current costs
    text = 'cost: ' + str(f(v[0], v[1]))
    fig.add_trace(
        go.Scatter(
            x=[5],
            y=[6],
            mode='text',        
            name='Theta Hat',
            text=[text]
        )
    )    
    # Add new point (z=0) trace
    fig.add_trace(
        go.Scatter3d(x=[v[0]], y=[v[1]], z=[v_z0[0]],
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
    # Add new point trace
    fig.add_trace(
        go.Scatter3d(x=[v[0]], y=[v[1]], z=[v_z1[0]],
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
        go.Scatter3d(x=v_x, y=v_y, z=v_z0,
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
        go.Scatter3d(x=v_x, y=v_y, z=v_z1,
            mode='lines',
            name = 'New Path (z=0)',
            visible=False,
            line=dict(
                width=2,
                color='blue'
            )
        )
    )                                                      
    # Add projection line 
    fig.add_trace(
        go.Scatter3d(x=u_x, y=u_y, z=u_z,
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
num_traces_update = 5
for i in range(num_angles):
    # Hide all traces
    step = dict(
        method = 'restyle',
        args = ['visible', [False] * len(fig.data)]
    )
    # Show static traces (surface, start points)
    step['args'][1][0] = True
    step['args'][1][1] = True
    step['args'][1][2] = True
    step['args'][1][3] = True
    step['args'][1][4] = True
    step['args'][1][5] = True
    step['args'][1][6+(i*num_traces_update)] = True
    step['args'][1][6+(i*num_traces_update)+1] = True
    step['args'][1][6+(i*num_traces_update)+2] = True
    step['args'][1][6+(i*num_traces_update)+3] = True
    step['args'][1][6+(i*num_traces_update)+4] = True
    step['args'][1][6+(i*num_traces_update)+5] = True
    step['args'][1][6+(i*num_traces_update)+6] = True
    # Add step to steps
    steps.append(step)

# ---------------------------------------------------------------------------- #
#%%
# Create slider
sliders = [dict(
  steps=steps,
  currentvalue = dict(
      prefix='',
      suffix='-deg'
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
fig.update_layout(title='Linear Function in 2D', 
                  scene=scene,
                  height=600,
                  width=900,
                  sliders=sliders,
                  showlegend=False,
                  margin=dict(l=0,r=0,t=40,b=0),
                  template='plotly_white'
                )                     
po.plot(fig, filename='planar.html', auto_open=True)


#%%
