#=============================================================================#
#                         MOUNT FAIRWEATHER                                   #
#=============================================================================#
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
import geoplot
import pyproj
import folium
from shapely import geometry
from cartopy.io import shapereader
import shapefile as shp
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import mplleaflet

#-----------------------------------------------------------------------------#
tmn_derived_names = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/TNMDerivedNames.shp')
nhyd_water_body = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/NHDWaterbody.shp')
nhyd_line = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/NHDLine.shp')
nhyd_flowline = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/NHDFlowline.shp')
nhyd_area = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/NHDArea.shp')
landcover_woodland = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/LANDCOVER_WOODLAND.shp')
gu_stateorterritory = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/GU_StateOrTerritory.shp')
gu_reserve = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/GU_Reserve.shp')
gu_plsstownship = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/GU_PLSSTownship.shp')
gu_plss_first_division = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/GU_PLSSFirstDivision.shp')
gu_native_american_area = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/GU_NativeAmericanArea.shp')
gu_county_or_equivalent = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/GU_CountyOrEquivalent.shp')
elev_contour = gpd.read_file('./content/figures/VECTOR_Healy_B-4_NW_AK_7_5_Min_Shape/Shape/Elev_Contour.shp')
cellgrid_7_5_minute = gpd.read_file('./content/figures/VECTOR_Mount_Fairweather_D-6_NW_AK_7_5_Min_Shape/Shape/CellGrid_7_5Minute.shp')
#-----------------------------------------------------------------------------#
fairweather = tmn_derived_names.append(nhyd_water_body, ignore_index=True)
fairweather = fairweather.append(nhyd_line, ignore_index=True)
fairweather = fairweather.append(nhyd_flowline, ignore_index=True)
fairweather = fairweather.append(nhyd_area, ignore_index=True)
fairweather = fairweather.append(landcover_woodland, ignore_index=True)
fairweather = fairweather.append(gu_stateorterritory, ignore_index=True)
fairweather = fairweather.append(gu_reserve, ignore_index=True)
fairweather = fairweather.append(gu_plsstownship, ignore_index=True)
fairweather = fairweather.append(gu_plss_first_division, ignore_index=True)
fairweather = fairweather.append(gu_native_american_area, ignore_index=True)
fairweather = fairweather.append(gu_county_or_equivalent, ignore_index=True)
fairweather = fairweather.append(elev_contour, ignore_index=True)
fairweather = fairweather.append(cellgrid_7_5_minute, ignore_index=True)
#-----------------------------------------------------------------------------#
#%%
# Plot population estimates with an accurate legend
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(1, 1, figsize=(18,12))

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)

elev_contour.plot(ax=ax, legend=True, cax=cax)
geoplot.polyplot(gu_plss_first_division, ax=ax)
print(elev_contour)
#-----------------------------------------------------------------------------#
#%%
# Plot shape records
ec = shp.Reader('./content/figures/VECTOR_Healy_B-4_NW_AK_7_5_Min_Shape/Shape/Elev_Contour.shp')
plt.figure(figsize=(16,12))
for shape in ec.shapeRecords():
    x=[i[0] for i in shape.shape.points[:]]
    y=[i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)
plt.show()
# %%
#-----------------------------------------------------------------------------#
#%%
# Plot interactive maps
