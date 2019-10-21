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
def f(x):
    return(x**2)


x = np.linspace(0, 3.5, 100)
y = f(x)
x_points = [0, 2, 2.5]
y_points = [0, 2**2, 2.5**2]
aq_x_points = [2, 2.5]
aq_y_points = [2**2, 2.5**2]

# ============================================================================ #
#                        Secant and Tangent Lines                              #
# ============================================================================ #
secant_x = np.linspace(0, 5, 10)
s = UnivariateSpline(aq_x_points, aq_y_points, k=1)
secant_y = s(secant_x)

a = 1
h = 0.1
fprime = (f(a+h)-f(a))/h
tan_x = np.linspace(1.5, 2.5, 10)

def tan_y(x):
    return(f(a)+fprime*(x-a))

# ============================================================================ #
#                        Vertical and Horizontal Lines                         #
# ============================================================================ #
x0_x = [1, 1]
x0_y = [-5, 1]
x1_x = [3.5, 3.5]
x1_y = [-5, 3.5**2]
x2_x = [1, 4.5]
x2_y = [1, 1]
x3_x = [3.5, 4.5]
x3_y = [3.5**2, 3.5**2]
pq_x = [3.5]
pq_y = [1]    
# ============================================================================ #
#                                Plot                                          #
# ============================================================================ #
sns.set(style="white", font_scale=2)
fig, ax = plt.subplots(figsize=(12,8))

ax = plt.plot(x_points, y_points, 'ro')
ax = sns.lineplot(x=x, y=y, ci=None)
ax = sns.lineplot(x=secant_x, y=secant_y, ci=None)
ax = sns.lineplot(x=tan_x, y=tan_y(tan_x), ci=None)
ax.lines[1].set_linestyle("--")

ax.text(x=2.1, y=2**2, s='A')
ax.text(x=-0.1, y=0.2, s='B')
ax.text(x=2.5, y=2.5**2, s='Q')

plt.title(r'$f(x)=x^2$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()
