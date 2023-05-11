# ---- Processing NHD Stream Shapefile ----
# written by Danielle Tadych
# 4/26/23

# The purpose of this code is to make the NHD stream shapefile less of a monster for graphs
# File needed to create these streams: 'NHD_Important_Rivers.shp' on cyverse
# Dictionary of NHD Terms is here: https://specx.nationalmap.gov/reports/DD/?report_type=2&product=144&theme=2&feature_class=all&scale=142&table=All 

#%%
from typing import Mapping
#import affine
import matplotlib
from matplotlib.cbook import report_memory
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import datetime as dt
import pandas as pd
import geopandas as gp
import cartopy
import netCDF4
import rasterio
#import rasterstats as rstats
#from xrspatial import zonal_stats
import easymore
import glob

# %% Data paths
datapath = '../MergedData'
outputpath = '../MergedData/Output_files/'
shapepath = '../MergedData/Shapefiles/'

filename_georeg = 'NHD_Important_Rivers.shp'
filepath = os.path.join(shapepath, filename_georeg)
nhd_rivers = gp.read_file(filepath)
# %%
nhd_rivers

# %% Individual River Lookup
water_name = 'Walnut Creek'
nhd_rivers[nhd_rivers['GNIS_Name'] == water_name]


# %%
nhd_rivers['GNIS_Name'].unique()
# %%
nhd_rivers[(nhd_rivers['FType'] == 460) & (nhd_rivers['GNIS_Name'].notna())].plot()

# %%
to_dissolve = nhd_rivers[['GNIS_Name','geometry']]
dissolved = to_dissolve.dissolve(by='GNIS_Name')
dissolved

#%% Check to see it worked
# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))
dissolved.reset_index().plot(column='GNIS_Name',
                            ax=ax)
ax.set_axis_off()
plt.axis('equal')
plt.show() 
# %% Calculate stream lengths

dissolved['length'] = dissolved['geometry'].length
dissolved.head()
# %%
dissolved.describe()
# %%
narrowed = dissolved[(dissolved.length > 0.23)]
narrowed
# %%
narrowed.plot()
# %%
narrowed.describe()
# %%
more_narrowing = narrowed[(narrowed.length > 1.9)]
more_narrowing
# %%
more_narrowing.plot()
# %%
more_narrowing = more_narrowing.reset_index()
more_narrowing
#%%
more_narrowing.loc[[8],'geometry'].plot()
# %%
ImportantStreams = more_narrowing
ImportantStreams = ImportantStreams.drop([0,3,8,9,11])
ImportantStreams
# %%
ImportantStreams.plot()
# %%
more_narrowing.to_file('../MergedData/Output_files/Narrowed_Important_SWFlowlines.shp')
# %%
