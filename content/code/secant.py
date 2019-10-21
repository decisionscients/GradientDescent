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

x = np.linspace(-5, 5, 100)
y = f(x)
x_points = [0, 2, 3.5]
y_points = [0, 2**2, 3.5**2]
secant_x_points = [2, 3.5]
secant_y_points = [2**2, 3.5**2]

# ============================================================================ #
#                               Secant Line                                    #
# ============================================================================ #
secant_x = np.linspace(1.5, 4, 10)
s = UnivariateSpline(secant_x_points, secant_y_points, k=1)
secant_y = s(secant_x)

# ============================================================================ #
#                        Vertical and Horizontal Lines                         #
# ============================================================================ #
x0_x = [2, 2]
x0_y = [-4, 2**2]
x1_x = [3.5, 3.5]
x1_y = [-4, 3.5**2]
x2_x = [2, 4.5]
x2_y = [4, 2**2]
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

ax.annotate('Secant', xy=(3, s(3)), xytext=(1, 10),
        arrowprops=dict(facecolor='black', shrink=0.05))

ax.text(x=1.5, y=2**2, s='A')
ax.text(x=-0.1, y=0.2, s='B')
ax.text(x=3.5*.9, y=f(3.5)*1.05, s='Q')
ax.text(x=2.7, y=-2.5, s=r'$h$', color='dimgrey', fontsize=14)
ax.text(x=3.75, y=8, s=r'$f(a+h)-f(a)$', color='dimgrey',
        fontsize=14)


ax.annotate('', xy=(2, -2), xytext=(2.5, -2),
        arrowprops=dict(facecolor='grey', shrink=0.05,
                        width=2, headwidth=5))
ax.annotate('', xy=(3.5, -2), xytext=(3, -2),
        arrowprops=dict(facecolor='grey', shrink=0.05,
                        width=2, headwidth=5))
ax.annotate('', xy=(4, 3.5**2), xytext=(4, 9),
        arrowprops=dict(facecolor='grey', shrink=0.05,
                        width=2, headwidth=5))

ax.annotate('', xy=(4, 5), xytext=(4, 8),
        arrowprops=dict(facecolor='grey', shrink=0.05,
                        width=2, headwidth=5))

plt.plot(x0_x, x0_y, color='grey', lw=1, linestyle='--')
plt.plot(x1_x, x1_y, color='grey', lw=1, linestyle='--')
plt.plot(x2_x, x2_y, color='grey', lw=1, linestyle='--')
plt.plot(x3_x, x3_y, color='grey', lw=1, linestyle='--')

x0_tick = r'$a$'
x1_tick = r'$a+h$'
plt.title(r'$f(x)=x^2$ with Secant Line')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()
#%%
