#=============================================================================#
#                             MOUNT ELIAS                                     #
#=============================================================================#
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
import geoplot
import pyproj
import folium
import fiona
from shapely import geometry
from cartopy.io import shapereader
import shapefile as shp
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import mplleaflet
#%%
from osgeo import gdal
#-----------------------------------------------------------------------------#
wrstvlf = gpd.read_file('./content/figures/wrangell/wrstvlf.shp')
wrstmin = gpd.read_file('./content/figures/wrangell/wrstmin.shp')
wrstgsl = gpd.read_file('./content/figures/wrangell/wrstgsl.shp')
wrstgol = gpd.read_file('./content/figures/wrangell/wrstgol.shp')
wrstglga = gpd.read_file('./content/figures/wrangell/wrstglga.shp')
wrstglg = gpd.read_file('./content/figures/wrangell/wrstglg.shp')
wrstgfl = gpd.read_file('./content/figures/wrangell/wrstgfl.shp')
wrstflt = gpd.read_file('./content/figures/wrangell/wrstflt.shp')
wrstdke = gpd.read_file('./content/figures/wrangell/wrstdke.shp')
wrstasl = gpd.read_file('./content/figures/wrangell/wrstasl.shp')
#-----------------------------------------------------------------------------#
wrangell = wrstvlf.append(wrstmin, ignore_index=True)

#-----------------------------------------------------------------------------#
#%%
# Plot population estimates with an accurate legend
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(1, 1, figsize=(18,12))

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)

wrstvlf.plot(ax=ax, legend=True, cax=cax)
geoplot.polyplot(wrstmin, ax=ax)
print(wrstmin)
#-----------------------------------------------------------------------------#
#%%
# Plot shape records
wrangell = shp.Reader('./content/figures/wrangell/wrstgfl.shp')
plt.figure(figsize=(16,12))
for shape in wrangell.shapeRecords():
    x=[i[0] for i in shape.shape.points[:]]
    y=[i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)
plt.show()



# %%
