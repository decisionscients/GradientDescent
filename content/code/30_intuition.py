# ============================================================================ #
#                                 30_intuition                                 #
# ============================================================================ #
#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import UnivariateSpline

from .filemanager import save_fig
directory = "./figures"
params = {'axes.titlesize':'x-large'}
pylab.rcParams.update(params)
# ---------------------------------------------------------------------------- #
#                                 UNIVARIATE                                   #
# ---------------------------------------------------------------------------- #
# Functions
def convex(x):
    return x**2
def nonconvex(x):
    return np.sin(x)+0.1*x
# ---------------------------------------------------------------------------- #
# Lines 
x = np.linspace(-5, 5, 100)
y_convex = convex(x)
y_non_convex = nonconvex(x)
# ---------------------------------------------------------------------------- #
# Data points
x_points = [-3,3]
y_points_convex = [convex(x_points[0]),convex(x_points[1])]
y_points_non_convex = [nonconvex(x_points[0]),nonconvex(x_points[1])]
# ---------------------------------------------------------------------------- #
# Designate plot objects
sns.set(style="white", font_scale=2)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(12,6))
# ---------------------------------------------------------------------------- #
# Plot convex function
ax1.set_title(r"Convex Function""\n"r"$f(x)=x^2$", pad=30)
ax1.plot(x_points, y_points_convex, 'ro')
sns.lineplot(x=x, y=y_convex, ci=None, ax=ax1)
ax1.text(x=x_points[0], y=convex(x_points[0])*1.01, s='A')
ax1.text(x=x_points[1], y=convex(x_points[1])*1.01, s='B')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
# ---------------------------------------------------------------------------- #
# Plot non-convex function
ax2.set_title(r"Non-Convex Function""\n"r"$f(x)=nonconvex(x)$", pad=30)
ax2.plot(x_points, y_points_non_convex, 'ro')
sns.lineplot(x=x, y=y_non_convex, ci=None, ax=ax2)
ax2.text(x=x_points[0], y=nonconvex(x_points[0])*1.01, s='A')
ax2.text(x=x_points[1], y=nonconvex(x_points[1])*1.01, s='B')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')
# ---------------------------------------------------------------------------- #
# Combine plots into single figure and save
fig.tight_layout()
filename = "univariate.png"
save_fig(fig, directory = directory, filename=filename)
#%%