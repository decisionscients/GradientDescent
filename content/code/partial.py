# %%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib import animation, rc, rcParams
from IPython.display import HTML

from content.code.filemanager import save_gif, save_fig

# ---------------------------------------------------------------------------- #
directory = "./content/figures"
def f(x,y):
    return(x**2 + x*y - y**2)

def f_prime_x(x,y):
    return(2*x + y)

def f_prime_y(x,y):
    return(x-2*y)

def tangent_wrt_x(x_0, y_0, x,y):    
    tan = f(x_0, y_0) + f_prime_x(x_0, y_0) * (x-x_0)
    return tan

def tangent_wrt_y(x_0, y_0, x,y):    
    tan = f(x_0, y_0) + f_prime_y(x_0, y_0) * (y-y_0)
    return tan

# ---------------------------------------------------------------------------- #
#                               Plot defaults                                  #
# ---------------------------------------------------------------------------- #
def plot_init():
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    # Remove grid
    ax.grid(False)
    # Set face, tick,and label colors 
    ax.set_facecolor('w')
    ax.tick_params(colors='k')
    ax.xaxis.label.set_color('k')
    ax.yaxis.label.set_color('k')
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # Add the labels
    ax.set_xlabel('X' )
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax
# ---------------------------------------------------------------------------- #
#                             Function Surface Plot                            #
# ---------------------------------------------------------------------------- #
# Initialize plot
fig, ax = plot_init()
# Setup data
x = np.arange(-5, 5, .05)
y = np.arange(-5, 5, .05)
X, Y = np.meshgrid(x, y)
zs = np.array([f(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
z_range = np.arange(min(zs), max(zs), .01)
Z = zs.reshape(X.shape)
# Plot surface and point
ax.plot_surface(X, Y, Z, alpha=.6)
ax.plot([-2.], [-2.], [f(-2,-2)], markerfacecolor='r', markeredgecolor='r',
        marker='o', markersize=10, alpha=1)
# Set Title
ax.set_title(r"$f(x,y) = x^2+xy-y^2$", fontsize=14, color='black')
# Save Image
filename = "multivariable_function.png"
save_fig(fig, directory=directory, filename=filename)
plt.tight_layout()
plt.show()
#%%
# ---------------------------------------------------------------------------- #
#                        Partial Derivative w.r.t X                            #
# ---------------------------------------------------------------------------- #
# Initialize Plot
fig, ax = plot_init()
# Setup data
x = np.arange(-5, 5, .05)
y = np.arange(-2, 5, .05)
X, Y = np.meshgrid(x, y)
zs = np.array([f(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
z_range = np.arange(min(zs), max(zs), .01)
Z = zs.reshape(X.shape)
# Plot surface and point
ax.plot_surface(X, Y, Z, alpha=.6)
ax.plot([-2.], [-2.], [f(-2,-2)], markerfacecolor='r', markeredgecolor='r',
        marker='o', markersize=10, alpha=1)
# Plot Equation with plane 
xx_wrt_x, zz_wrt_x = np.meshgrid(x, z_range)
yy_wrt_x = -2
ax.plot_surface(xx_wrt_x, yy_wrt_x, zz_wrt_x, alpha=.5)
# Prepare 2D Partial line with respect to x
x_line_wrt_x = x
y_line_wrt_x = [-2] * len(x_line_wrt_x)
z_line_wrt_x = np.array([f(x, y) for x, y in zip(np.ravel(x_line_wrt_x), np.ravel(y_line_wrt_x))])
ax.plot3D(x_line_wrt_x, y_line_wrt_x, z_line_wrt_x, 'blue', alpha=1)
# Prepare 2D Partial with respect to x tangent line
z_line_wrt_x = np.array([tangent_wrt_x(-2, -2, x, y) for x, y in zip(np.ravel(x_line_wrt_x), np.ravel(y_line_wrt_x))])
ax.plot3D(x_line_wrt_x, y_line_wrt_x, z_line_wrt_x, 'red', alpha=1)
# Set Title
ax.set_title(r"$f(x,y) = x^2+xy-y^2$" + "\n" + "Partial Derivative w.r.t X", fontsize=14, color='black')
# Save Image
filename = "partial_wrt_x.png"
save_fig(fig, directory=directory, filename=filename)
plt.tight_layout()
plt.show()
#%%
# ---------------------------------------------------------------------------- #
#                        Partial Derivative w.r.t y                            #
# ---------------------------------------------------------------------------- #
# Initialize Plot
fig, ax = plot_init()
# Setup data
x = np.arange(-5, -2, .05)
y = np.arange(-5, 5, .05)
X, Y = np.meshgrid(x, y)
zs = np.array([f(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
z_range = np.arange(min(zs), max(zs), .01)
Z = zs.reshape(X.shape)
# Plot surface and point
ax.plot_surface(X, Y, Z, alpha=.6)
ax.plot([-2.], [-2.], [f(-2,-2)], markerfacecolor='r', markeredgecolor='r',
        marker='o', markersize=10, alpha=1)
# Plot Equation with plane 
yy_wrt_y, zz_wrt_y = np.meshgrid(y, z_range)
xx_wrt_y = -2
ax.plot_surface(xx_wrt_y, yy_wrt_y, zz_wrt_y, alpha=.5)
# Prepare 2D Partial line with respect to y
y_line_wrt_y = y
x_line_wrt_y = [-2] * len(y_line_wrt_y)
z_line_wrt_y = np.array([f(x, y) for x, y in zip(np.ravel(x_line_wrt_y), np.ravel(y_line_wrt_y))])
ax.plot3D(x_line_wrt_y, y_line_wrt_y, z_line_wrt_y, 'blue', alpha=1)
# Prepare 2D Partial with respect to x tangent line
z_line_wrt_y = np.array([tangent_wrt_y(-2, -2, x, y) for x, y in zip(np.ravel(x_line_wrt_y), np.ravel(y_line_wrt_y))])
ax.plot3D(x_line_wrt_y, y_line_wrt_y, z_line_wrt_y, 'red', alpha=1)
# Set Title
ax.set_title(r"$f(x,y) = x^2+xy-y^2$" + "\n" + "Partial Derivative w.r.t Y", fontsize=14, color='black')
# Save Image
filename = "partial_wrt_y.png"
save_fig(fig, directory=directory, filename=filename)
plt.tight_layout()
plt.show()
#%%
