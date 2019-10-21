# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
#%%
import inspect
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import UnivariateSpline


# ============================================================================ #
#                                 DATA                                         #
# ============================================================================ #
x = np.linspace(-1, 5, 100)
y = f(x)
x_points = [0, 2]
y_points = [0, 2**2]

# ============================================================================ #
#                               Functions                                      #
# ============================================================================ #
def f(x):
    return x**2

def deriv(f,x):

    h = 0.000000001                 #step-size 
    return (f(x+h) - f(x))/h        #definition of derivative

def tangent_line(f,x_0,a,b):
    x = np.linspace(a,b,200)
    y = f(x) 
    y_0 = f(x_0)
    y_tan = deriv(f,x_0) * (x - x_0) + y_0 
   
  #plotting
    sns.set(style="white", font_scale=2)
    fig, ax = plt.subplots(figsize=(12,8))
    sns.scatterplot(x=x_points, y=y_points)
    ax.text(x=2.1, y=2**2, s='A')
    ax.text(x=-0.1, y=0.2, s='B')
    
    plt.plot(x,y,'r-')
    plt.plot(x,y_tan)
    plt.axis([a,b,a,b])
    plt.xlabel('x')     
    plt.ylabel('y')    
    plt.title(r'$f(x)=x^2$, with Tangent Line')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()  

tangent_line(f, 2,-1,5)

