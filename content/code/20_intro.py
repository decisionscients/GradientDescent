# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
#%%
from IPython.display import HTML
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.visual.animations import MultiModelSearch3D, MultiModelFit2D
from .filemanager import save_fig
directory = "./figures"
params = {'axes.titlesize':'x-large'}
pylab.rcParams.update(params)
# --------------------------------------------------------------------------- #
# Obtain data
# --------------------------------------------------------------------------- #
X, y = datasets.load_boston(return_X_y=True)
X = X[:,5]
data={'Rooms':X, 'Price': y}
data = pd.DataFrame(data)
X = X.reshape(-1,1)

# --------------------------------------------------------------------------- #
# Plot home prices by rooms
# --------------------------------------------------------------------------- #
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(12,6))
ax = sns.scatterplot(x='Rooms', y='Price', data=data)
ax.set_title(r'Median Home Price by Rooms')
plt.xlabel(r'Mean # of Rooms')
plt.ylabel(r'Price ($000)')
filename = "price_by_rooms.png"
save_fig(fig, directory = directory, filename=filename)
# =========================================================================== #
# Plot home prices by rooms regression line
# =========================================================================== #
#%%
lr = LinearRegression()
lr.fit(X,y)
y_pred = lr.predict(X)
fig, ax = plt.subplots(figsize=(12,6))
ax = sns.scatterplot(x='Rooms', y='Price', data=data)
plt.plot(X, y_pred, color='red')
ax.set_title('Median Home Price by Rooms\nOrdinary Least Squares Regression Line')
plt.xlabel(r'Mean # of Rooms')
plt.ylabel(r'Price ($000)') 
filename = "price_by_rooms_regression.png"
save_fig(fig, directory = directory, filename=filename)
# =========================================================================== #
#                                 COST MESH                                   #
# =========================================================================== #
#%%
# Standardize X
scaler = StandardScaler()    
X = scaler.fit_transform(X)
# --------------------------------------------------------------------------- #
def z(a, b, THETA):
    return(((THETA[0]**2)/a**2) + ((THETA[1]**2)/b**2))

def paraboloid(x, y, a, b, directory=None, filename=None):
    '''Plots surface plot on two dimensional problems only 
    '''        
    # Designate plot area
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    sns.set(style="whitegrid", font_scale=1)

    # Establish boundaries of plot
    #theta0_mesh = np.linspace(-x, x, 50)
    #theta1_mesh = np.linspace(-y, y, 50)
    theta0_mesh = np.linspace(-100, 100, 50)
    theta1_mesh = np.linspace(-100, 100, 50)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh, theta1_mesh)

    # Create cost grid based upon x,y the grid of thetas
    Zs = np.array([z(a,b, THETA)
                for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
    Zs = Zs.reshape(theta0_mesh.shape)

    # Set Title
    title = "Quadratic Cost Surface"
    ax.set_title(title, fontsize=20, color='k', pad=30)               
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
    ax.plot_surface(theta0_mesh, theta1_mesh, Zs, rstride=1,
             cstride=1, cmap='jet', alpha=.8, linewidth=0)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_0$')
    ax.set_zlabel('Cost')        
    ax.view_init(elev=30., azim=220)
    
    if directory is not None:
        if filename is None:
            filename = "quadratic_cost_surface.png"
        save_fig(fig, directory, filename)
    return(fig)
fig = paraboloid(x=max(X), y=max(y), a=1, b=1, directory=directory)
# =========================================================================== #
#                          LEARNING RATE IMPACT                               #
# =========================================================================== #
#%%
models = {}
learning_rates = [0.001, 0.01, 0.1]
names = ['Learning Rate: 0.001','Learning Rate: 0.01', 'Learning Rate: 0.1']

for i in range(len(learning_rates)):
    bgd = LinearRegression(epochs=500, learning_rate=learning_rates[i],
                                   name=names[i])
    models[names[i]] = bgd.fit(X,y)
ani = MultiModelSearch3D()
ani.search(models, directory=directory, filename='search_by_learning_rate.gif')
ani = MultiModelFit2D()
ani.fit(models, directory=directory, filename='fit_by_learning_rate.gif')


#%%
