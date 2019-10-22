# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
#%%
import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np

from content.code.filemanager import save_fig
# ============================================================================ #
#                                 DATA                                         #
# ============================================================================ #
def f(x):
    theta = 1.5
    b = 2
    return(x*theta+b)


x = np.linspace(0, 3, 10)
x0=1.5
y = f(x)
x_points = [x0]
y_points = [f(x0)]

# ============================================================================ #
#                                Plot                                          #
# ============================================================================ #
sns.set(style="white", font_scale=2)
fig, ax = plt.subplots(figsize=(8,8))

ax = plt.plot(x_points, y_points, 'ro')
ax = sns.lineplot(x=x, y=y, ci=None)

ax.text(x=x0-x0*0.05, y=f(x0)+f(x0)*.03, s='A')

plt.title(r'$f(x)=\theta^Tx+b$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

directory = "./figures/"
filename = "linear.png"
save_fig(fig, directory = directory, filename=filename)


#%%
