#%%
import inspect
import os
import sys
codedir = "c:\\Users\\John\\Documents\\Data Science\\Projects\\GradientDescent\\code"
sys.path.append(codedir) 

from IPython.display import HTML, Image
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import colors as mcolors
from matplotlib import rc, rcParams
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from numpy import array, newaxis
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from filemanager import save_fig, save_csv, save_gif

class GradientAnimation3D():

    def __init__(self):
        pass

    def _cost_mesh(self,X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y)))    

    def animate(self, X, y, models, interval=200, repeat_delay=5, 
                 secs=10, maxframes=500, fontsize=None, blit=True, show=False,
                 directory=None, filename=None):

        # Designate plot area
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        sns.set(style="whitegrid", font_scale=1)

        # Extract model names                 
        methods = [v.results()['summary']['alg'].item()  for (k,v) in models.items()]

        # Extract paths
        self._paths=[]
        self._zpaths=[]
        for k, v in models.items():    
            self._paths.append(np.array(v.results()['detail'][['theta_0', 'theta_1']]).T)
            self._zpaths.append(np.array(v.results()['detail']['cost']))

        # Get number of frames and frames per second (fps)
        frames = max(path.shape[1] for path in self._paths)    
        min_frames = min(path.shape[1] for path in self._paths)    
        fps = math.floor(min_frames/secs) if min_frames >=secs else math.floor(secs/min_frames)      

        # Get minimum and maximum thetas and establish boundaries of plot
        theta0_min = min([min(v.results()['detail']['theta_0']) for (k,v) in models.items()])
        theta1_min = min([min(v.results()['detail']['theta_1']) for (k,v) in models.items()])
        theta0_max = max([max(v.results()['detail']['theta_0']) for (k,v) in models.items()])
        theta1_max = max([max(v.results()['detail']['theta_1']) for (k,v) in models.items()])
        theta0_mesh = np.linspace(theta0_min, theta0_max, 100)
        theta1_mesh = np.linspace(theta1_min, theta1_max, 100)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh, theta1_mesh)

        # Create cost grid based upon x,y the grid of thetas
        Js = np.array([self._cost_mesh(X, y, THETA) 
                    for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
        Js = Js.reshape(theta0_mesh.shape)

        # Set Title
        title = 'Gradient Descent Trajectories'
        self.ax.set_title(title, color='k', pad=30)                       

        # Set face, tick,and label colors 
        self.ax.set_facecolor('w')
        self.ax.tick_params(colors='k')
        self.ax.xaxis.label.set_color('k')
        self.ax.yaxis.label.set_color('k')
        # make the panes transparent
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        self.ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        self.ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        self.ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # Make surface plot
        self.ax.plot_surface(theta0_mesh, theta1_mesh, Js, rstride=1,
                cstride=1, cmap='jet', alpha=0.5, linewidth=0)
        self.ax.set_xlabel(r'Intercept($\theta_0$)')
        self.ax.set_ylabel(r'Slope($\theta_1$)')
        self.ax.set_zlabel('Error')        
        self.ax.view_init(elev=30., azim=30)

        # Build the empty line plot at the initiation of the animation
        colors = ['b-', 'g-', 'r-', 'c-', 'm-', 'w-']
        self._lines = [self.ax.plot([], [], [], c, label=method, lw=2)[0] 
                      for c, _, method in itertools.zip_longest(colors, self._paths, methods)]       

        def init_anim():
            for line in self._lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return (self._lines)      

        def animate(i):
            for line, path, zpath in zip(self._lines, self._paths, self._zpaths):
                line.set_data(*path[::,:i])
                line.set_3d_properties(zpath[:i])
            return (self._lines)        
                 
   
        # Initialize funcAnimation
        ani = animation.FuncAnimation(self.fig, animate, init_func=init_anim,
                                                  frames=frames, interval=interval, blit=blit,
                                                  repeat_delay=repeat_delay)  


        if directory is not None:
            if filename is None:
                filename = 'Gradient Descent Trajectory.gif'
            save_gif(ani, directory, filename, fps=5)
       
        return(ani)    

  