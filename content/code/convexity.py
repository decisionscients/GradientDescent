# %%
# ============================================================================ #
#                                 CONVEXITY                                    #
# ============================================================================ #
# Renders a convex curve.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Convex Data
def convex(x): 
    return(x**2)
x1 = np.linspace(-2,2)
y1 = convex(x1)
df_c = pd.DataFrame({
    'x': [-1.5,1],
    'y': [convex(-1.5), convex(1)],
    'group': ['A', 'B']
})

# NonConvex Data
def non_convex(x): 
    return(np.sin(x)+np.sin((10/3)*x))
x2 = np.linspace(2.7,7.5)
y2 = non_convex(x2)
df_nc = pd.DataFrame({
    'x': [3,6],
    'y': [non_convex(3), non_convex(6)],
    'group': ['A', 'B']
})

#%%

sns.set(style="white", font_scale=2)
sns.set_palette("GnBu_d")
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
ax1 = sns.lineplot(x=x1, y=y1, ax=ax1)
ax1 = sns.regplot(data=df_c, x='x', y='y', fit_reg=False, marker='o', color='steelblue', ax=ax1)
for line in range(0,df_c.shape[0]):
     ax1.text(df_c.x[line]+0.2, df_c.y[line], df_c.group[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
ax1.set_title("Convex Function")
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.tick_params(    
    which='both',      # both major and minor ticks are affected
    left=False,
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelleft=False,
    labelbottom=False)

ax2 = sns.lineplot(x=x2, y=y2, ax=ax2)
ax2 = sns.regplot(data=df_nc, x='x', y='y', fit_reg=False, marker='o', color='steelblue', ax=ax2)
for line in range(0,df_nc.shape[0]):
     ax2.text(df_nc.x[line]+0.2, df_nc.y[line], df_nc.group[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
ax2.set_title("Non-Convex Function")     
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.tick_params(    
    which='both',      # both major and minor ticks are affected
    left=False,
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelleft=False,
    labelbottom=False)
plt.show()
