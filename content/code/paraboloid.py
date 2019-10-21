# =========================================================================== #
#                                 COST MESH                                   #
# =========================================================================== #
#%%
# --------------------------------------------------------------------------- #
import inspect
import os
import sys

from IPython.display import HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import seaborn as sns

from .filemanager import save_fig, save_csv, save_gif


def z(a, b, THETA):
    return(((THETA[0]**2)/a**2) + ((THETA[1]**2)/b**2))

def paraboloid(x, y, a, b, directory=None, filename=None):
    '''Plots surface plot on two dimensional problems only 
    '''        
    # Designate plot area
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sns.set(style="whitegrid", font_scale=1)

    # Establish boundaries of plot
    theta0_mesh = np.linspace(-x, x, 50)
    theta1_mesh = np.linspace(-y, y, 50)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh, theta1_mesh)

    # Create cost grid based upon x,y the grid of thetas
    Zs = np.array([z(a,b, THETA)
                for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
    Zs = Zs.reshape(theta0_mesh.shape)

    # Set Title
    title = "Convex Cost Surface"
    ax.set_title(title, color='k', pad=30)               
    ax.text2D(0.3,0.92, '', transform=ax.transAxes, color='k')             

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
    # Make surface plot
    cs = ax.plot_surface(theta0_mesh, theta1_mesh, Zs, rstride=1,
            cstride=1, cmap='jet', alpha=.8, linewidth=0)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_0$')
    ax.set_zlabel('Cost')        
    ax.view_init(elev=30., azim=30)
    # Create colorbar
    fig.colorbar(cs)    

    if directory is not None:
        if filename is None:
            filename = "convex_cost_surface.png"
        save_fig(fig, directory, filename)
    return(fig)
directory = "./figures/"
paraboloid(x=5, y=5, a=1, b=1, directory=directory)