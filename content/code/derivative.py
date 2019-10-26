# ============================================================================ #
#                               DERIVATIVE                                     #
# ============================================================================ #
# Plot showing derivative as slope of tangent line
#%%
import plotly.offline as po
import plotly.graph_objs as go
import numpy as np
po.init_notebook_mode()

#%%
# ---------------------------------------------------------------------------- #
# Formula
def f(x):
    return np.square(x)
# ---------------------------------------------------------------------------- #
#%%
# Get data
x = np.linspace(-2, 10, 100)
y = f(x)
x0 = 3
y0 = f(x0)
p_x = np.full(10,x0)
p_y = np.linspace(0,y0,10)
# ---------------------------------------------------------------------------- #
#%%
# Create figure
fig = go.Figure()
# ---------------------------------------------------------------------------- #
#%%
# Plot of function
fig.add_trace(
    go.Scatter(x=x, y=y,         
        name=r'y=x^2',
        mode='lines',
        line=dict(
            color='red'
        )
    )
)
# Point x0
fig.add_trace(
    go.Scatter(x=[x0], y=[0], 
        mode='markers+text',
        name='x0',        
        marker=dict(
            color='black'
            ),
        text=r'x0',
        textposition='bottom center'
        )        
)
# Point P
fig.add_trace(
    go.Scatter(x=[x0], y=[y0], 
        mode='markers+text',
        name=r'P=(x_0, y_0)',        
        marker=dict(
            color='black'
            ),
        text=r'P',
        textposition='top center'
        )        
)
# Point P projection line
fig.add_trace(
    go.Scatter(x=p_x, y=p_y, 
        mode='lines',
        name='Projection of x0',                
        line=dict(
            width=1,
            color='black'
        ),
        showlegend=False
        )        
)
# Horizontal Line from P
fig.add_trace(
    go.Scatter(
        x=np.linspace(x0,10,10), 
        y=np.full(10, f(x0)), 
        mode='lines',
        name='Horizontal Line from P',                
        line=dict(
            width=1,
            color='black'
        ),
        showlegend=False
        )        
)
# ---------------------------------------------------------------------------- #
#%%
# Variable traces
for delta_x in np.arange(0,10):
    # x0+delta x on x axis
    fig.add_trace(
        go.Scatter(x=[x0+delta_x], y=[0], 
            mode='markers+text',
            name=r'x_0+\Delta x',   
            visible=False,     
            marker=dict(
                color='black'
                ),
            text=r'x_0+\Delta x',
            textposition='top center'
            )        
    )
    # Point Q
    fig.add_trace(
        go.Scatter(x=[x0+delta_x], y=[f(x0+delta_x)], 
            mode='markers+text',
            name=r'Q',    
            visible=False,    
            marker=dict(
                color='black'
                ),
            text=r'Q',
            textposition='top center'
            )        
    )    
    # Point Q projection line
    fig.add_trace(
        go.Scatter(
            x=np.full(10,delta_x), 
            y=np.linspace(x0+delta_x,f(x0+delta_x),10), 
            mode='lines',
            name='Projection to point Q',                
            visible=False,
            line=dict(
                width=1,
                color='black'
            ),
            showlegend=False
            )        
    )
    # Point of intersection projection lines
    fig.add_trace(
    go.Scatter(x=[x0+delta_x], y=[f(x0)], 
        mode='markers',
        name='Intersection of Projection Lines',        
        visible=False,
        marker=dict(
            color='black'
            )
        )        
    )
    # Horizontal Line from Q
    fig.add_trace(
        go.Scatter(
            x=np.full(10,f(x0+delta_x)), 
            y=np.linspace(x0+delta_x, 10, 10), 
            mode='lines',
            name='Horizontal Line from Q',                
            visible=False,
            line=dict(
                width=1,
                color='black'
            ),
            showlegend=False
            )        
    )                        
# ---------------------------------------------------------------------------- #
#%%
# Set visibility for all steps
steps = []
num_traces_static = 5
num_traces_update = 5
for i in np.arange(0,10):
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
    step['args'][1][4] = True
    step['args'][1][num_traces_static+(i*num_traces_update)] = True
    step['args'][1][num_traces_static+(i*num_traces_update)+1] = True
    step['args'][1][num_traces_static+(i*num_traces_update)+2] = True
    step['args'][1][num_traces_static+(i*num_traces_update)+3] = True
    step['args'][1][num_traces_static+(i*num_traces_update)+4] = True
    # Add step to steps
    steps.append(step)

# ---------------------------------------------------------------------------- #
#%%
# Create slider
sliders = [dict(
  steps=steps,
  currentvalue = dict(
      prefix=r'\Delta x'
  )
  )]
# ---------------------------------------------------------------------------- #
#%%
# Add layout and scene

ax = dict(
  zeroline = False,
  showline = False,
  showticklabels = False,
  showgrid = False    
)                     
fig.update_layout(title='Derivative as Limit', 
                  height=600,
                  width=900,
                  xaxis=ax,
                  yaxis=ax,
                  sliders=sliders,
                  showlegend=False,
                  margin=dict(l=0,r=0,t=40,b=0),
                  template='plotly_white'
                )                     
po.plot(fig, filename='./content/figures/derivative.html',include_mathjax='cdn', auto_open=True)


#%%
