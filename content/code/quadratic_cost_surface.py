# ============================================================================ #
#                             LINEAR REGRESSION                                #
# ============================================================================ #
# Renders stacked line charts showing convex and non-convex obj function
#%%
import plotly.offline as po
import plotly.graph_objs as go
import numpy as np
directory = "./content/figures/"
# ---------------------------------------------------------------------------- #
#                                    DATA                                      #
# ---------------------------------------------------------------------------- #
#%%
# Generate Data
x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
x_grid, y_grid = np.meshgrid(x,y)
z = x_grid**2 + y_grid**2

# ---------------------------------------------------------------------------- #
#                                    PLOT                                      #
# ---------------------------------------------------------------------------- #
#%%
# 3D Surface Plot
data = go.Surface(
    x=x,
    y=y,
    z=z,
    colorscale='RdBu',
    showscale=False
)
layout = go.Layout(title='Quadratic Cost Surface', 
                   height=400,
                   width=800,
                   showlegend=False,
                   margin=dict(l=10,r=10,t=40,b=10),
                   scene=dict(
                    xaxis_title="θ_0",
                    yaxis_title="θ_1",
                    zaxis_title="J(θ)"
                   ),
                   template='plotly_white'
                   )
fig = go.Figure(data=data, layout=layout)      
fig.show()
po.plot(fig, filename = "./content/figures/quadratic_cost_surface.html",
        include_mathjax='cdn', auto_open=False)             


# %%
