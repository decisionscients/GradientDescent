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
x_start=-1
y_start=-1    

#%%
# ---------------------------------------------------------------------------- #
#%%
# Get data
x0 = np.linspace(-1, 1, 10)
x1 = np.linspace(-1, 1, 10)
X0, X1 = np.meshgrid(x0, x1)
Z = f(X0,X1)
# ---------------------------------------------------------------------------- #
#%%
# Initialize new point and vector
angle=66
v = vector(x_start,y_start,angle)
v_x = np.linspace(x_start,v[0],10)
v_y = np.linspace(y_start,v[1],10)
v_z = np.full(len(v_y), np.min(Z))
# Vector projecting from point to x-y plane
u_x = np.full(50,v[0])
u_y = np.full(50,v[1])
u_z = np.linspace(np.min(Z),f(v[0], v[1]),50)
# ---------------------------------------------------------------------------- #
#%%
# Create traces, scene, layout
surface = go.Surface(x=X0, y=X1, z=Z, colorbar=None, 
                     colorscale='Viridis',
                     showscale=False,
                     opacity=0.5)
start_point_z = go.Scatter3d(x=[x_start], y=[y_start], z=[f(x_start, y_start)],
                       mode='markers',
                       name='Start Point',
                       marker=dict(
                           size=4,
                           color='blue'
                           ),
                       text='A',
                       textposition='top center')      

start_point = go.Scatter3d(x=[x_start], y=[y_start], z=[v_z[0]],
                       mode='markers',
                       name='Start Point',
                       marker=dict(
                           size=4,
                           color='blue'
                           ),
                       text='A',
                       textposition='top center')      

new_point = go.Scatter3d(x=[v[0]], y=[v[1]], z=[v_z[0]],
                       mode='markers',
                       name = 'New Point',
                       marker=dict(
                           size=4,
                           color='red'
                           ),
                       text='B',
                       textposition='top center')                                                          


new_point_z = go.Scatter3d(x=[v[0]], y=[v[1]], z=[f(v[0], v[1])],
                       mode='markers',
                       name = 'New Point',
                       marker=dict(
                           size=4,
                           color='red'
                           ),
                       text='B',
                       textposition='top center')                                                          

projection = go.Scatter3d(x=u_x, y=u_y, z=u_z,
                       mode='lines',
                       line=dict(
                           width=0.5,
                           color='blue',
                           colorscale='Viridis'
                       ))  

step = go.Scatter3d(x=v_x, y=v_y, z=v_z,
                       mode='lines',
                       name='Step',
                       line=dict(
                           width=2,
                           color='blue',
                           colorscale='Viridis'
                       ))  

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
layout = go.Layout(title='Linear Function in 2D', 
                   scene=scene,
                   height=600,
                   width=900,
                   showlegend=False,
                   margin=dict(l=0,r=0,t=40,b=0),
                   template='plotly_white'
                   )                     
data = [surface, start_point, start_point_z, new_point, new_point_z, step, projection]                   
fig = go.Figure(data=data, layout=layout)
# ---------------------------------------------------------------------------- #
#%%
# Create figure widget

fw = go.FigureWidget(data=fig.data, layout=fig.layout)
# ---------------------------------------------------------------------------- #
#%%
# Create slider
angle_slider = widgets.IntSlider(
    value=90,
    min=0,
    max=90,
    step=1,
    description='Angle'
)
# ---------------------------------------------------------------------------- #
#%%
# Create updater
def updater(angle):
    v = vector(x_start, y_start, angle=angle)
    v_x = np.linspace(x_start,v[0],20)
    v_y = np.linspace(y_start,v[1],20)
    v_z = np.full(len(v_y), np.min(Z))
    # Update point
    fw.data[3].x=[v[0]]
    fw.data[3].y=[v[1]]
    fw.data[3].z=[np.min(Z)]
    # Update point (z)
    fw.data[4].x=[v[0]]
    fw.data[4].y=[v[1]]
    fw.data[4].z=[f(v[0], v[1])]
    # # Update vector
    fw.data[5].x = v_x
    fw.data[5].y = v_y
    fw.data[5].z = v_z
    # # Update projection
    fw.data[6].x = u_x
    fw.data[6].y = u_y
    fw.data[6].z = u_z

vb = VBox((fw, interactive(updater, angle=angle_slider)))
vb.layout.align_items = 'center'
vb
    

#po.plot(fig, filename='planar.html', auto_open=True)


#%%
