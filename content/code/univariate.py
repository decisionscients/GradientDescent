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
x_points = [0, 2]
y_points = [0, 2**2]

# ============================================================================ #
#                                Plot                                          #
# ============================================================================ #
sns.set(style="white", font_scale=2)
fig, ax = plt.subplots(figsize=(12,8))

ax = plt.plot(x_points, y_points, 'ro')
ax = sns.lineplot(x=x, y=y, ci=None)

ax.text(x=2.1, y=2**2, s='A')
ax.text(x=-0.1, y=0.2, s='B')

plt.title(r'$f(x)=x^2$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()
