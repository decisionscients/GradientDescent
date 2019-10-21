# %%
# =========================================================================== #
#                               GRADIENT VISUAL                               #
# =========================================================================== #

# --------------------------------------------------------------------------- #
import inspect
import os
import sys

from IPython.display import HTML
import datetime
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from itertools import zip_longest
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib import animation, rc, rcParams
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from numpy import array, newaxis
import pandas as pd
import seaborn as sns
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from textwrap import wrap
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .analytical import Normal
from .filemanager import save_fig, save_csv, save_gif


# --------------------------------------------------------------------------- #
#                             GRADIENTVISUAL CLASS                            #  
# --------------------------------------------------------------------------- #
class GradientVisual:
    '''
    Base class for gradient descent plots
    '''

    def __init__(self):
        pass        

    def _get_label(self, x):
        labels = {'learning_rate': 'Learning Rate',
                  'learning_rates': 'Learning Rates',
                  'learning_rate_sched': 'Learning Rate Schedule',
                  'stop_metric': 'Stop Metric',
                  'epochs_stable': 'Stable Iterations',
                  'batch_size': 'Batch Size',
                  'time_decay': 'Time Decay',
                  'step_decay': 'Step Decay',
                  'decay_rate': 'Decay Rate',
                  'step_epochs': 'Epochs per Step',
                  'exp_decay': 'Exponential Decay',
                  'precision': 'Precision',
                  'theta': "Theta",
                  'duration': 'Computation Time (ms)',
                  'iterations': 'Iterations',
                  'iteration': 'Iteration',
                  'epoch': 'Epoch',
                  'cost': 'Training Set Costs',
                  'initial_train_error': 'Initial Training Error',                  
                  'final_train_error': 'Final Training Error',
                  'initial_dev_set_error': 'Initial Development Set Error',                  
                  'final_dev_set_error': 'Final Development Set Error',
                  'c': 'Constant Learning Rate',
                  't': 'Time Decay Learning Rate',
                  's': 'Step Decay Learning Rate',
                  'e': 'Exponential Decay Learning Rate',
                  'j': 'Training Set Costs',
                  'v': 'Validation Set Error',
                  'g': 'Gradient Norm'}
        return(labels.get(x,x))

    def distplot(self, ax,  data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)        

        # Plot time by learning rate 
        ax = sns.distplot(a=data[x])
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel(self._get_label(x))
        ax.set_ylabel(self._get_label(y))
        ax.set_title(title, color='k')
        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax)             
    
    def scatterplot(self, ax,  data, x, y, z=None, title=None,
                 log=False, xlim=None,  ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)        

        # Plot time by learning rate 
        ax = sns.scatterplot(x=x, y=y, hue=z, data=data, ax=ax, legend='full')
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel(self._get_label(x))
        ax.set_ylabel(self._get_label(y))
        ax.set_title(title, color='k')
        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax)         

    def barplot(self, ax,  data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)

        # Plot time by learning rate 
        ax = sns.barplot(x=x, y=y, hue=z, data=data, ax=ax)
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel(self._get_label(x))
        ax.set_ylabel(self._get_label(y))
        ax.set_title(title, color='k')
        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax)     

    def boxplot(self, ax,  data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)  

        # Plot time by learning rate 
        ax = sns.boxplot(x=x, y=y, hue=z, data=data, ax=ax)
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel(self._get_label(x))
        ax.set_ylabel(self._get_label(y))
        ax.set_title(title, color='k')
        # Change to log scale and limit y-axis if requested
        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax) 

    def lineplot(self, ax, data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)

        # Plot time by learning rate 
        ax = sns.lineplot(x=x, y=y, hue=z, data=data, legend='full', ax=ax)
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel(self._get_label(x))
        ax.set_ylabel(self._get_label(y))
        ax.set_title(title, color='k')
        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax) 

    def figure(self, data, x, y, z=None, title=None, func=None, 
               cols=2, directory=None, filename=None, show=False, height=1, 
               width=1, log=False, xlim=None, ylim=None):

        sns.set(style="whitegrid", font_scale=1)                
        # Designate plot title
        if title is not None:
            if y is not None:
                title = title + '\n' + self._get_label(y) + ' By ' + self._get_label(x)
        else:
            if y is not None:
                title = self._get_label(y) + ' By ' + self._get_label(x)
        if z:
            title = title + ' and ' + self._get_label(z)             
        # Establish plot dimensions and initiate matplotlib objects
        fig_width = math.floor(12*width)
        fig_height = math.floor(4*height)                
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))    
        # Render plot    
        ax = func(ax=ax, data=data, x=x, y=y, z=z, 
                    title=title, log=log, xlim=xlim, ylim=ylim)            

        # Finalize and save
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = title.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig)      
         
        return(fig)     

  

    def learning_curve(self, gd, directory=None, filename=None, show=False):
        # Obtain data
        alg = gd.alg
        train = gd.get_train_costs()
        validation = gd.get_validation_costs()
        df_train = pd.DataFrame({'Dataset': 'Train', 
                                 'Epoch': train['epoch'],
                                 'Error': train['error']})
        df_val = pd.DataFrame({'Dataset': 'Validation', 
                                 'Epoch': validation['epoch'],
                                 'Error': validation['error']})                                 
        df = pd.concat([df_train, df_val], axis=0)                                 

        # Initialize and render plot
        fig, ax = plt.subplots(figsize=(12,4)) 
        ax = self.lineplot(ax, df, x='Epoch', y='Error', z='Dataset')        

        # Titles 
        suptitle = alg + '\n' + 'Training and Validation Error Curves'
        fig.suptitle(suptitle, y=1.05)

        # Finalize and save
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = suptitle.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig)      
        return(fig)         


    def plot_cv_scores(self, alg, gs, x, z=None, cols=2, top=None,
                       xleft=None, xright=None,
                       height=1, width=1,  directory=None, 
                       scale=None, filename=None, show=False):

        # Get results
        results = pd.DataFrame.from_dict(gs.cv_results_)

        # Filter top results if required
        if top is not None:
            if top > 1:
                # Return top n by rank test score
                results = results[results.rank_test_score <= top]
            else:
                # Return results within top % of best score
                low = gs.best_score_ + gs.best_score_ * top
                high = gs.best_score_ - gs.best_score_ * top
                results = results[(results.mean_test_score >= low) & (results.mean_test_score <= high)]

        # Filter x-range if required
        if xleft is not None:
            results = results[results[x] >= xleft]
        if xright is not None:
            results = results[results[x] <= xright]

        # Validate parameters and obtain labels 
        params = list(results.filter(like='param_').columns)        
        if x not in params:
            raise Exception("Invalid parameter name")
        x_label = self._get_label(x.split('__')[1])
        y_label = 'Mean Test Scores'
        if z is not None:
            z_label = self._get_label(z.split('__')[1])  

        # Obtain and initialize matplotlib figure
        fig_width = math.floor(12*width)
        fig_height = math.floor(4*height)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height)) 
        
        # Set Title
        title = alg + '\n' + 'GridSearchCV Scores' + '\n' + y_label + ' By ' + x_label
        
        # Print line plot
        ax = self.lineplot(x=x, y='mean_test_score', z=z, data=results, ax=ax, title=title)

        # Change to log scale if requested
        if scale == 'log':
             ax.set_xscale('log')
        
        # Finalize, show and save
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = alg + ' GridSearchCV Score Analysis.png'
            save_fig(fig, directory, filename)
        plt.close(fig)             

    def plot_cv_times(self, alg, gs, x, z=None, cols=2, top=None,
                      xleft=None, xright=None,
                      height=1, width=1,  scale=None,
                      directory=None, filename=None, show=False):

        # Get results
        results = pd.DataFrame.from_dict(gs.cv_results_)

        # Filter top results if required
        if top is not None:
            if top > 1:
                # Return top n by rank test score
                results = results[results.rank_test_score <= top]
            else:
                # Return results within top % of best score
                low = gs.best_score_ + gs.best_score_ * top
                high = gs.best_score_ - gs.best_score_ * top
                results = results[(results.mean_test_score >= low) & (results.mean_test_score <= high)]

        # Filter x-range if required
        if xleft is not None:
            results = results[x >= xleft]
        if xright is not None:
            results = results[x <= xright]


        # Validate parameters and obtain labels 
        params = list(results.filter(like='param_').columns)        
        if x not in params:
            raise Exception("Invalid parameter name")
        x_label = self._get_label(x.split('__')[1])
        y_label = 'Mean Fit Time'
        if z is not None:
            z_label = self._get_label(z.split('__')[1])  

        # Obtain and initialize matplotlib figure
        fig_width = math.floor(12*width)
        fig_height = math.floor(4*height)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height)) 
        
        # Set Title
        title = alg + '\n' + 'GridSearchCV Computation Times' + '\n' + y_label + ' By ' + x_label
        
        # Print line plot
        ax = self.lineplot(x=x, y='mean_fit_time', z=z, data=results, ax=ax, title=title)

        # Change to log scale if requested
        if scale == 'log':
             ax.set_xscale('log')        
        
        # Finalize, show and save
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = alg + ' GridSearchCV Score Analysis.png'
            save_fig(fig, directory, filename)
        plt.close(fig)                                


    def plotfit(self, X,y, models, directory=None, filename=None, 
                show=False, height=1, width=1, subtitle=None):

        # line color vector
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

        # Obtain and initialize matplotlib figure
        fig_width = math.floor(12*width)
        fig_height = math.floor(4*height)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))            
        sns.set(style="whitegrid", font_scale=1, palette="Set1")                

        # Render scatterplot                
        ax = sns.scatterplot(x=X[:,1], y=y, ax=ax)

        # Format Data
        df = pd.DataFrame()
        for m in models:
            theta = m.theta
            intercept = theta[0]
            coef = theta[1]
            y = intercept + X[:,1].dot(coef)    
            alg = m.alg
            df2 = pd.DataFrame({'Algorithm': alg, 'x':X[:,1], 'y':y})        
            df = pd.concat([df, df2], axis=0)

        # Render line plots
        ax = sns.lineplot(x='x', y='y', data=df, ax=ax, hue='Algorithm')    

        # Title and aesthetics
        title = 'Validation Set Fit'
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title, color='k')
        
        # Finalize and save
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = title.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig) 


            

    def _cost_mesh(self,X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y)))        

    def show_search(self, alg, X, y, detail, summary, directory=None, filename=None, fontsize=None,
                    interval=200, secs=10, maxframes=500):
        '''Plots surface plot on two dimensional problems only 
        '''        
        # Designate plot area
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        sns.set(style="whitegrid", font_scale=1)

        # Create index for n <= maxframes number of points
        idx = np.arange(0,detail.shape[0])
        nth = math.floor(detail.shape[0]/maxframes)
        nth = max(nth,1) 
        idx = idx[::nth]

        # Create the x=theta0, y=theta1 grid for plotting
        iterations = detail['iteration']
        costs = detail['cost']     
        theta0 = detail['theta_0']
        theta1 = detail['theta_1']

        # Format frames per second
        fps = math.floor(len(iterations)/secs) if len(iterations) >=secs else math.floor(secs/len(iterations))      

        # Establish boundaries of plot
        theta0_min = min(-1, min(theta0))
        theta1_min = min(-1, min(theta1))
        theta0_max = max(1, max(theta0))
        theta1_max = max(1, max(theta1))
        theta0_mesh = np.linspace(theta0_min, theta0_max, 100)
        theta1_mesh = np.linspace(theta1_min, theta1_max, 100)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh, theta1_mesh)

        # Create cost grid based upon x,y the grid of thetas
        Js = np.array([self._cost_mesh(X, y, THETA)
                    for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
        Js = Js.reshape(theta0_mesh.shape)

        # Set Title
        title = alg + '\n' + r' $\alpha$' + " = " + str(round(summary['learning_rate'].item(),3))
        if fontsize:
            ax.set_title(title, color='k', pad=30, fontsize=fontsize)                            
            display = ax.text2D(0.1,0.92, '', transform=ax.transAxes, color='k', fontsize=fontsize)
        else:
            ax.set_title(title, color='k', pad=30)               
            display = ax.text2D(0.3,0.92, '', transform=ax.transAxes, color='k')             
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
        ax.plot_surface(theta0_mesh, theta1_mesh, Js, rstride=1,
                cstride=1, cmap='jet', alpha=0.5, linewidth=0)
        ax.set_xlabel(r'Intercept($\theta_0$)')
        ax.set_ylabel(r'Slope($\theta_1$)')
        ax.set_zlabel('Error')        
        ax.view_init(elev=30., azim=30)

        # Build the empty line plot at the initiation of the animation
        line3d, = ax.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
        line2d, = ax.plot([], [], [], 'b-', label = 'Gradient descent', lw = 1.5)
        point3d, = ax.plot([], [], [], 'bo')
        point2d, = ax.plot([], [], [], 'bo')

        def init():

            # Initialize 3d line and point
            line3d.set_data([],[])
            line3d.set_3d_properties([])
            point3d.set_data([], [])
            point3d.set_3d_properties([])

            # Initialize 2d line and point
            line2d.set_data([],[])
            line2d.set_3d_properties([])
            point2d.set_data([], [])
            point2d.set_3d_properties([])

            # Initialize display
            display.set_text('')
            return (line2d, point2d, line3d, point3d, display,)

        # Animate the regression line as it converges
        def animate(i):
            # Animate 3d Line
            line3d.set_data(theta0[:idx[i]], theta1[:idx[i]])
            line3d.set_3d_properties(costs[:idx[i]])

            # Animate 3d points
            point3d.set_data(theta0[idx[i]], theta1[idx[i]])
            point3d.set_3d_properties(costs[idx[i]])

            # Animate 2d Line
            line2d.set_data(theta0[:idx[i]], theta1[:idx[i]])
            line2d.set_3d_properties(0)

            # Animate 2d points
            point2d.set_data(theta0[idx[i]], theta1[idx[i]])
            point2d.set_3d_properties(0)

            # Update display
            display.set_text('Iteration: '+ str(iterations[idx[i]]) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[idx[i]],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[idx[i]],3)) +
                            ' Cost: ' + str(np.round(costs[idx[i]], 5)))


            return(line3d, point3d, line2d, point2d, display)

        # create animation using the animate() function
        surface_ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(idx),
                                            interval=interval, blit=True, repeat_delay=3000)
        if directory is not None:
            if filename is None:
                filename = alg + ' Search Path Learning Rate ' + str(summary['learning_rate'].item()) +  '.gif'
            save_gif(surface_ani, directory, filename, fps)
        plt.close(fig)
        return(surface_ani)

    def show_fit(self, alg, X, y, detail, summary, directory=None, filename=None, fontsize=None,
                 interval=50, secs=10, maxframes=500):
        '''Shows animation of regression line fit for 2D X Vector 
        '''

        # Create index for n <= maxframes number of points
        idx = np.arange(0,detail.shape[0])
        nth = math.floor(detail.shape[0]/maxframes)
        nth = max(nth,1) 
        idx = idx[::nth]

        # Extract data for plotting
        x = X[:,1]
        iterations = detail['iteration']
        costs = detail['cost']        
        theta0 = detail['theta_0']
        theta1 = detail['theta_1']
        theta = np.array([theta0, theta1])

        # Format frames per second
        fps = math.floor(len(iterations)/secs) if len(iterations) >=secs else math.floor(secs/len(iterations))
        
        # Render scatterplot
        fig, ax = plt.subplots(figsize=(12,8))
        sns.set(style="whitegrid", font_scale=1)
        sns.scatterplot(x=x, y=y, ax=ax)
        # Set face, tick,and label colors 
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_ylim(7.5,20)
        # Initialize line
        line, = ax.plot([],[],'r-', lw=2)
        # Set Title, Annotations and label
        title = alg + '\n' + r' $\alpha$' + " = " + str(round(summary['learning_rate'].item(),3)) 
        if fontsize:
            ax.set_title(title, color='k', fontsize=fontsize)                                    
            display = ax.text(0.1, 0.9, '', transform=ax.transAxes, color='k', fontsize=fontsize)
        else:
            ax.set_title(title, color='k')                                    
            display = ax.text(0.35, 0.9, '', transform=ax.transAxes, color='k')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        fig.tight_layout()

        # Build the empty line plot at the initiation of the animation
        def init():
            line.set_data([],[])
            display.set_text('')
            return (line, display,)

        # Animate the regression line as it converges
        def animate(i):

            # Animate Line
            y=X.dot(theta[:,idx[i]])
            line.set_data(x,y)

            # Animate text
            display.set_text('Iteration: '+ str(iterations[idx[i]]) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[idx[i]],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[idx[i]],3)) +
                            ' Cost: ' + str(round(costs[idx[i]], 3)))
            return (line, display)

        # create animation using the animate() function
        line_gd = animation.FuncAnimation(fig, animate, init_func=init, frames=len(idx),
                                            interval=interval, blit=True, repeat_delay=3000)
        if directory is not None:
            if filename is None:
                filename = title = alg + ' Fit Plot Learning Rate ' + str(round(summary['learning_rate'].item(),3)) + '.gif'  
            save_gif(line_gd, directory, filename, fps)
        plt.close(fig)  
        return(line_gd)

# --------------------------------------------------------------------------- #
#                         GRADIENT ANIMATION 3D CLASS                         #  
# --------------------------------------------------------------------------- #
class GradientAnimation3D(animation.FuncAnimation):

    def __init__(self):
        pass

    def _anim8(self, *paths, zpaths, methods=[], frames=None, 
                 interval=60, repeat_delay=5, blit=False, **kwargs):

        self.paths = paths
        self.zpaths = zpaths

        if frames is None:
            frames = max(path.shape[1] for path in paths)
            self.nth = [1 for p in paths]
        else:
            self.nth = [max(1,math.floor(path.shape[1] / frames)) for path in paths]
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0,1,len(paths)))
        self.lines = [self.ax.plot([], [], [], alpha=0.7, label=method, c=c, lw=2)[0] 
                      for _, method, c in zip_longest(paths, methods, colors)]

        ani = animation.FuncAnimation(self.fig, self._update, init_func=self._init,
                                      frames=frames, interval=interval, blit=blit,
                                      repeat_delay=repeat_delay, **kwargs)
        self.ax.legend(loc='upper left')                                      
        return(ani)                                                  

    def _init(self):
        for line in self.lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return self.lines

    def _update(self, i):
        for line, nth, path, zpath in zip(self.lines, self.nth, self.paths, self.zpaths):
            self.ax.view_init(elev=10., azim=i*.30)
            line.set_data(*path[::,:i*nth])
            line.set_3d_properties(zpath[:i*nth])
        return self.lines

    def _cost_mesh(self,X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y))) 

    def _surface(self, X, y, models):

        # Designate plot area
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        sns.set(style="whitegrid", font_scale=1)

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

    def _get_data(self, models):
        
        paths=[]
        zpaths=[]
        methods = []
        for k, v in models.items():    
            paths.append(np.array(v.results()['detail'][['theta_0', 'theta_1']]).T)
            zpaths.append(np.array(v.results()['detail']['cost'])) 
            methods.append(k)  

        return(paths, zpaths, methods)

    def animate(self, X, y, models, frames=None, directory=None, filename=None, fps=5):

        self._surface(X, y, models)
        paths, zpaths, methods = self._get_data(models)
        ani = self._anim8(*paths, zpaths=zpaths, methods=methods, frames=frames)
        if directory is not None:
            save_gif(ani, directory, filename, fps)
        return(ani)


# --------------------------------------------------------------------------- #
#                           GRADIENT FIT 3D CLASS                             #   
# --------------------------------------------------------------------------- #
class GradientFit3D(animation.FuncAnimation):

    def __init__(self):
        pass

    def _anim8(self, *paths, methods=[], frames=None, 
                 interval=60, repeat_delay=5, blit=False, **kwargs):

        self.paths = paths

        if frames is None:
            frames = max(path.shape[1] for path in paths)
            self.nth = [1 for p in paths]
        else:
            self.nth = [max(1,math.floor(path.shape[1] / frames)) for path in paths]
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0,1,len(paths)))
        self.lines = [self.ax.plot([], [], label=method, c=c, lw=2)[0] 
                      for _, method, c in zip_longest(paths, methods, colors)]

        ani = animation.FuncAnimation(self.fig, self._update, init_func=self._init,
                                      frames=frames, interval=interval, blit=blit,
                                      repeat_delay=repeat_delay, **kwargs)
        self.ax.legend(loc='upper left')                                      
        return(ani)                                                  

    def _init(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def _update(self, i):
        for line, nth, path, in zip(self.lines, self.nth, self.paths):
            print(path)
            line.set_data(*path[::,:i*nth])            
        return self.lines

    def _scatterplot(self, X, y):

        # Designate plot area
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        sns.set(style="whitegrid", font_scale=1)
        sns.scatterplot(x=X[:,1], y=y, ax=self.ax)

        # Set face, tick,and label colors 
        self.ax.set_facecolor('w')
        self.ax.tick_params(colors='k')
        self.ax.xaxis.label.set_color('k')
        self.ax.yaxis.label.set_color('k')
        self.ax.set_ylim(7.5,20)        

        # Set Title, Annotations and label
        title = 'Model Fit'
        self.ax.set_title(title, color='k')
        self.ax.set_xlabel('Living Area')
        self.ax.set_ylabel('Sale Price')
        self.fig.tight_layout()      

    def _get_data(self, models):
        
        paths=[]
        methods = []
        for k, v in models.items():    
            paths.append(np.array(v.results()['detail'][['theta_0', 'theta_1']]).T)
            methods.append(k)  

        return(paths, methods)

    def animate(self, X, y, models, frames=None, directory=None, filename=None, fps=5):

        self._scatterplot(X, y)
        paths, methods = self._get_data(models)
        ani = self._anim8(*paths, methods=methods, frames=frames)
        if directory is not None:
            save_gif(ani, directory, filename, fps)
        return(ani)        