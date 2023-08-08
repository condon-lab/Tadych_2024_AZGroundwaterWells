# The purpose of this script is to create a code to spatially analyze all the wells in 
# the combined database based on management. 
# Written by Danielle Tadych

# WORKFLOW
# 1. Read in the master ADWR database static database, water level database, and 
#       georegions shapefile created in QGIS
# 2. Overlayed region shapefile on static well database shapefile
# 3. Exported a dataframe (registry list) of combined ID's with the columns we want 
#       (regulation, etc.)
# 4. Joined the registry list with the timeseries database so every well has water 
#       levels and is tagged with a category we want
# 5. Create pivot tables averaging water levels based on categories (e.g. regulation, 
#       access to SW, or georegion (finer scale))
# 6. Export pivot tables into .csv's for easy analyzing later
#       * Note: after reading in packages, skip to line 197 to avoid redoing steps 1-5
# 7. Graphs for days (starting around line 214)
# 8. Statistical analyses
#       - Linear Regression (~line 929)
#       - Pearson/Spearman Correlation (~line 212)
#       - lagged Correlation analyses


# %%
from optparse import Values
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import datetime
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from shapely.geometry import box
import geopandas as gp
#import earthpy as et
import scipy.stats as sp
from scipy.stats import kendalltau, pearsonr, spearmanr
import pymannkendall as mk
import Custom_functions as cf

# Data paths
datapath_web = 'https://data.cyverse.org/dav-anon/iplant/home/dtadych/AZ_Spatial_Analysis/Data/'
outputpath_web = 'https://data.cyverse.org/dav-anon/iplant/home/dtadych/AZ_Spatial_Analysis/Data/Output_files/'
shapepath_web = 'https://data.cyverse.org/dav-anon/iplant/home/dtadych/AZ_Spatial_Analysis/Data/Shapefiles/'

datapath_local = '../Data'
outputpath_local = '../Data/Output_files/'
shapepath_local = '../Data/Shapefiles/'

# %%  Load in the master databases
filename_mdb_nd = 'Master_ADWR_database_noduplicates.shp'
filepath = os.path.join(outputpath_web, filename_mdb_nd)
print(filepath)

masterdb = gp.read_file(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(masterdb.info())

# %%
filename_mdb_w = 'Master_ADWR_database_water.shp'
filepath = os.path.join(outputpath_web, filename_mdb_w)
print(filepath)

masterdb_water = gp.read_file(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(masterdb_water.info())
# %%
# Reading in the shapefile
filename_georeg = 'georeg_reproject_fixed.shp'
filepath = os.path.join(shapepath_web, filename_georeg)
georeg = gp.read_file(filepath)

# %%
#georeg.boundary.plot()
georeg.plot(cmap='viridis')

#%%
georeg['GEOREGI_NU'] = georeg['GEOREGI_NU'].astype('int64')
georeg.info()
#%%
# Read in the annual time series database
filename_ts = 'Wells55_GWSI_WLTS_DB_annual.csv'
filepath = os.path.join(outputpath_web, filename_ts)
print(filepath)
annual_db = pd.read_csv(filepath, header=1, index_col=0)
# annual_db = annual_db.iloc[1:158,:] # fixing the first row
annual_db

# %%
# annual_db.index = annual_db.index.astype('float')
annual_db.index = annual_db.index.astype('int64')
#%%
annual_db.head()

#%%
only_special = masterdb[masterdb['WELL_TYPE_']=='OTHER']
only_special.info()
#%%
monitoring = masterdb[masterdb['WELL_TYPE_']=='MONITOR']
monitoring.info()

# %%
exempt = masterdb[masterdb['WELL_TYPE_']=='EXEMPT']
exempt.info()
#%%
nonexempt = masterdb[masterdb['WELL_TYPE_']=='NON-EXEMPT']
nonexempt.info()
# %% Overlay georegions onto the static database
# Going to use sjoin based off this website: https://geopandas.org/docs/user_guide/mergingdata.html
print("Non-cancelled: ", masterdb.crs, "Water Wells: ", masterdb_water.crs, "Georegions: ", georeg.crs)

# %%
georeg = georeg.to_crs(epsg=26912)
masterdb = masterdb.set_crs(epsg=26912)
masterdb_water = masterdb_water.set_crs(epsg=26912)
# %%
static_geo = gp.sjoin(masterdb, georeg, how="inner", op='intersects')
static_geo.head()
print(str(filename_mdb_nd) + " and " + str(filename_georeg) + " join complete.")

# %% Exporting or reading in the static geodatabase instead of rerunning
static_geo.to_csv(outputpath_local+'/Final_Static_geodatabase_allwells.csv')

# %% Rerunning this but for the water wells
static_geo2 = gp.sjoin(masterdb_water, georeg, how="inner", op='intersects')
static_geo2.head()
print(str(filename_mdb_nd) + " and " + str(filename_georeg) + " join complete.")


#%%
static_geo2.to_csv(outputpath_local+'/Final_Static_geodatabase_waterwells.csv')

# %%
filename = "Final_Static_geodatabase_allwells.csv"
filepath = os.path.join(outputpath_local, filename)
static_geo = pd.read_csv(filepath)
static_geo

# %% Create a dataframe of Final_Region and Well ID's
reg_list = static_geo[['Combo_ID', 'GEO_Region', 'GEOREGI_NU','Water_CAT', 'Loc','Regulation','WELL_DEPTH','WELL_TYPE_']]
reg_list

# %% Converting Combo_ID to int
reg_list['Combo_ID'] = reg_list['Combo_ID'].astype(int, errors = 'raise')
# %%
annual_db2
# %%
annual_db2 = annual_db.reset_index(inplace=True)
annual_db2 = annual_db.rename(columns = {'year':'Combo_ID'})
annual_db2.head()

# %% Add list to the annual database
combo = annual_db2.merge(reg_list, how="outer")
combo.info()

# This worked!!
# %% set index to Combo_ID
combo.set_index('Combo_ID', inplace=True)

# %% Sort the values
combo = combo.sort_values(by=['GEOREGI_NU'])
combo

# %% Exporting the combo table
combo.to_csv(outputpath_local+'Final_WaterLevels_adjusted.csv')

# %% Reading in so we don't have to redo the combining, comment as appropriate
filepath = outputpath_web+'Final_WaterLevels_adjusted.csv'
# filepath = outputpath_local+'Final_WaterLevels_adjusted.csv'
combo = pd.read_csv(filepath, index_col=0)
combo.head()

# %% in order to filter deep/mid/shallow wells
shallow = 200
deep = 500

wd1 = combo[(combo["WELL_DEPTH"] > deep)]
wd2 = combo[(combo["WELL_DEPTH"] <= deep) & (combo["WELL_DEPTH"] >= shallow)]
wd3 = combo[(combo["WELL_DEPTH"] < shallow)]

# %% in order to make it where we can actually group these bitches
whatever = [combo,wd1,wd2,wd3]
for i in whatever:
        del i['WELL_DEPTH']

# %% Now for aggregating by category for the timeseries
# to narrow by depth database
# combo = wd1

cat_wl_georeg = combo.groupby(['GEOREGI_NU']).mean()
cat_wl_reg = combo.groupby(['Regulation']).mean()
cat_wl_SW = combo.groupby(['Water_CAT']).mean()

cat_wl_georeg.info()

# %%
test = cat_wl_reg.copy()
del test['Combo_ID']
test = test.transpose()
test
#%%
test.plot()

# %%
wdc1_reg = wd1.groupby(['Regulation']).mean() # deep
wdc2_reg = wd2.groupby(['Regulation']).mean() # midrange
wdc3_reg = wd3.groupby(['Regulation']).mean() # shallow

wdc1_SW = wd1.groupby(['Water_CAT']).mean()
wdc2_SW = wd2.groupby(['Water_CAT']).mean()
wdc3_SW = wd3.groupby(['Water_CAT']).mean()

# %%
i = wdc1_reg
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc1_reg = f

i = wdc2_reg
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc2_reg = f

i = wdc3_reg
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc3_reg = f

i = wdc1_SW
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc1_SW = f

i = wdc2_SW
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc2_SW = f

i = wdc3_SW
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc3_SW = f
# %% 
cat_wl2_georeg = cat_wl_georeg.copy()
cat_wl2_reg = cat_wl_reg.copy()
cat_wl2_SW = cat_wl_SW.copy()

cat_wl2_georeg = cat_wl2_georeg.sort_values(by=['GEOREGI_NU'])
cat_wl2_SW = cat_wl2_SW.sort_values(by=['GEOREGI_NU'])

# Clean up the dataframe for graphing

i = cat_wl2_georeg
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
cat_wl2_georeg = f
        
i = cat_wl2_reg
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
cat_wl2_reg = f

i = cat_wl2_SW
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
cat_wl2_SW = f
# %% Going to export all these as CSV's
cat_wl2_georeg.to_csv(outputpath_local+'Waterlevels_georegions.csv')
cat_wl2_reg.to_csv(outputpath_local+'Waterlevels_Regulation.csv')
cat_wl2_SW.to_csv(outputpath_local+'Waterlevels_AccesstoSW.csv')

# %%  ==== Reading in the data we created above ====
#          Comment as appropriate
# For regulation
filepath = outputpath_web+'/Waterlevels_Regulation.csv'
# filepath = outputpath_local+'Waterlevels_Regulation.csv'
cat_wl2_reg = pd.read_csv(filepath, index_col=0)
cat_wl2_reg.head()

# For Access to SW
filepath = outputpath_web+'/Waterlevels_AccesstoSW.csv'
# filepath = outputpath_local+'Waterlevels_AccesstoSW.csv'
cat_wl2_SW = pd.read_csv(filepath, index_col=0)
cat_wl2_SW.head()

# For georegion number
filepath = outputpath_web+'Waterlevels_georegions.csv'
# filepath = outputpath_local+'Waterlevels_georegions.csv'
cat_wl2_georeg = pd.read_csv(filepath, index_col=0)
# cat_wl2_georeg.head()
# %%
cat_wl2_georeg = cat_wl2_georeg.transpose()
cat_wl2_georeg
# %%
cat_wl2_georeg.reset_index(inplace=True)
cat_wl2_georeg['index'] = pd.to_numeric(cat_wl2_georeg['index'])
cat_wl2_georeg.set_index('index', inplace=True)
cat_wl2_georeg.info()

# %%
cat_wl2_georeg = cat_wl2_georeg.transpose()
cat_wl2_georeg

# %% -- Linear regression --
# For Depth to Water by SW Access
ds = cat_wl2_SW
dt = "Depth to Water 08072023"
min = 1975
mx = 2020
betterlabels = ['Recieves CAP (Regulated)'
                ,'GW Dominated (Regulated)'
                ,'Surface Water Dominated'
                ,'GW Dominated'
                ,'Mixed Source']
cf.linearregress(cat_wl2_SW,dt,min,mx,betterlabels)
# %% 
ds = cat_wl2_SW
data_type = "Depth to Water 08072023"
min_yr = 1975
mx_yr = 2020
betterlabels = ['Recieves CAP (Regulated)'
                ,'GW Dominated (Regulated)'
                ,'Surface Water Dominated'
                ,'GW Dominated'
                ,'Mixed Source']
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
for i in column_list:
        df = f[i]
        # df = f[i].pct_change()
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
        stats = stats.append({'slope': slope, 
                              'int':intercept, 
                              'rsq':r_value*r_value, 
                              'p_val':p_value, 
                              'std_err':std_err, 
                              'mean': np.mean(y),
                              'var': np.var(y),
                              'sum': np.sum(y)
                              },
                              ignore_index=True)


stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)
# %% -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
m1 = round(stats1.loc['slope',betterlabels[0]], 2)
m2 = round(stats1.loc['slope',betterlabels[3]], 2)
m3 = round(stats1.loc['slope',betterlabels[4]], 2)
m4 = round(stats1.loc['slope',betterlabels[1]], 2)
m5 = round(stats1.loc['slope',betterlabels[2]], 2)
yint1 = round(stats1.loc['int',betterlabels[0]], 2)
yint2 = round(stats1.loc['int',betterlabels[3]], 2)
yint3 = round(stats1.loc['int',betterlabels[4]], 2)
yint4 = round(stats1.loc['int',betterlabels[1]], 2)
yint5 = round(stats1.loc['int',betterlabels[2]], 2)
rsq1 = round(stats1.loc['rsq', betterlabels[0]], 4)
rsq2 = round(stats1.loc['rsq', betterlabels[3]], 4)
rsq3 = round(stats1.loc['rsq', betterlabels[4]], 4)
rsq4 = round(stats1.loc['rsq', betterlabels[1]], 4)
rsq5 = round(stats1.loc['rsq', betterlabels[2]], 4)
pval1 = round(stats1.loc['p_val', betterlabels[0]], 4)
pval2 = round(stats1.loc['p_val', betterlabels[3]], 4)
pval3 = round(stats1.loc['p_val', betterlabels[4]], 4)
pval4 = round(stats1.loc['p_val', betterlabels[1]], 4)
pval5 = round(stats1.loc['p_val', betterlabels[2]], 4)
yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2
yf3 = (m3*xf)+yint3
yf4 = (m4*xf)+yint4
yf5 = (m5*xf)+yint5

fig, ax = plt.subplots(1, 1, figsize = (7,4.5))
# fig, ax = plt.subplots(figsize = (16,9))

ax.plot(xf1, yf1,"-.",color=cap,label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color=GWdom, lw=1)
ax.plot(xf1, yf3,"-.",color=mixed, lw=1)
ax.plot(xf1, yf4,"-.",color='#CCC339', lw=1)
ax.plot(xf1, yf5,"-.",color=swdom, lw=1)

# f.plot(ax=ax,marker='o', ls='', label=betterlabels)
# Trying to draw lines with better shit 

minyear=1975
maxyear=2020
min_y = 75
max_y = 300
fsize = 12

ax.plot(ds['CAP'], label=betterlabels[0], color=cap)
ax.plot(ds['No_CAP'], label=betterlabels[1], color='#CCC339') 
ax.plot(ds['SW'], label=betterlabels[2], color=swdom) 
ax.plot(ds['Mix'], label=betterlabels[4], color=mixed)
ax.plot(ds['GW'], label=betterlabels[3], color=GWdom)  

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = 10)
# # Drought Year Shading
# a = 1988.5
# b = 1990.5
# c = 1995.5
# d = 1996.5
# e = 2001.5
# f = 2003.5
# g = 2005.5
# h = 2007.5
# i = 2011.5
# j = 2014.5
# k = 2017.5
# l= 2018.5
# plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)

ax.minorticks_on()

fig.set_dpi(600.0)

# ax.set_xlim(min_yr, mx_yr)
ax.set_ylim(75,300)
# ax.set_title(Name)
vertshift = 0
plt.figtext(0.95, 0.5 - vertshift, 'CAP equation: y = '+str(m1)+'x + '+str(yint1))
plt.figtext(0.98, 0.45 - vertshift, 'rsq = '+ str(rsq1) + '; p-value = ' + str(pval1))
plt.figtext(0.95, 0.4 - vertshift, 'Unregulated GW equation: y = '+str(m2)+'x + '+str(yint2))
plt.figtext(0.98, 0.35 - vertshift, 'rsq = '+ str(rsq2) +'; p-value = ' + str(pval2))
plt.figtext(0.95, 0.3 - vertshift, 'Mixed SW/GW equation: y = '+str(m3)+'x + '+str(yint3))
plt.figtext(0.98, 0.25 - vertshift, 'rsq = '+ str(rsq3) +'; p-value = ' + str(pval3))
plt.figtext(0.95, 0.2 - vertshift, 'Regulated GW equation: y = '+str(m4)+'x + '+str(yint4))
plt.figtext(0.98, 0.15 - vertshift, 'rsq = '+ str(rsq4) +'; p-value = ' + str(pval4))
plt.figtext(0.95, 0.1 - vertshift, 'SW equation: y = '+str(m5)+'x + '+str(yint5))
plt.figtext(0.98, 0.05 - vertshift, 'rsq = '+ str(rsq5) +'; p-value = ' + str(pval5))

ax.legend(
        loc = [1.065, 0.65]
        )
plt.savefig(outputpath+'Stats/Water_CAT/'+Name, bbox_inches = 'tight')
plt.savefig(outputpath+'Stats/Water_CAT/'+Name, bbox_inches = 'tight')
stats1.to_csv(outputpath+'Stats/Water_CAT/'+Name+'.csv')

# %% Piecewise Linear Regression
# For Depth to Water by SW Access
ds = cat_wl2_SW
data_type = "Depth to Water"
# -- Piece 1 --
min_yr = 1975
mx_yr = 1985
betterlabels = ['Recieves CAP (Regulated)'
                ,'GW Dominated (Regulated)'
                ,'Surface Water Dominated'
                ,'GW Dominated'
                ,'Mixed Source'] 
Name1 = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name1)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
for i in column_list:
        df = f[i]
        # df = f[i].pct_change()
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
        stats = stats.append({'slope': slope, 
                              'int':intercept, 
                              'rsq':r_value*r_value, 
                              'p_val':p_value, 
                              'std_err':std_err, 
                              'mean': np.mean(y),
                              'var': np.var(y),
                              'sum': np.sum(y)
                              },
                              ignore_index=True)


stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)
# -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
m1 = round(stats1.loc['slope',betterlabels[0]], 2)
m2 = round(stats1.loc['slope',betterlabels[3]], 2)
m3 = round(stats1.loc['slope',betterlabels[4]], 2)
m4 = round(stats1.loc['slope',betterlabels[1]], 2)
m5 = round(stats1.loc['slope',betterlabels[2]], 2)
yint1 = round(stats1.loc['int',betterlabels[0]], 2)
yint2 = round(stats1.loc['int',betterlabels[3]], 2)
yint3 = round(stats1.loc['int',betterlabels[4]], 2)
yint4 = round(stats1.loc['int',betterlabels[1]], 2)
yint5 = round(stats1.loc['int',betterlabels[2]], 2)
rsq1 = round(stats1.loc['rsq', betterlabels[0]], 4)
rsq2 = round(stats1.loc['rsq', betterlabels[3]], 4)
rsq3 = round(stats1.loc['rsq', betterlabels[4]], 4)
rsq4 = round(stats1.loc['rsq', betterlabels[1]], 4)
rsq5 = round(stats1.loc['rsq', betterlabels[2]], 4)
pval1 = round(stats1.loc['p_val', betterlabels[0]], 4)
pval2 = round(stats1.loc['p_val', betterlabels[3]], 4)
pval3 = round(stats1.loc['p_val', betterlabels[4]], 4)
pval4 = round(stats1.loc['p_val', betterlabels[1]], 4)
pval5 = round(stats1.loc['p_val', betterlabels[2]], 4)
yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2
yf3 = (m3*xf)+yint3
yf4 = (m4*xf)+yint4
yf5 = (m5*xf)+yint5

fig, ax = plt.subplots(1, 1, figsize = (12,7))
# fig, ax = plt.subplots(figsize = (16,9))

# ax.plot(xf1, yf1,"-.",color=c_2,label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color=c_7, lw=1)
ax.plot(xf1, yf3,"-.",color=c_5, lw=1)
ax.plot(xf1, yf4,"-.",color=c_3, lw=1)
# ax.plot(xf1, yf5,"-.",color=c_4, lw=1)

vertshift = -0.3
horshift = 0
plt.figtext(0.94+horshift, 0.55 - vertshift, 'Regression for ' +str(min_yr)+' to '+str(mx_yr)+':')
plt.figtext(0.95+horshift, 0.5 - vertshift, 'CAP equation: y = '+str(m1)+'x + '+str(yint1))
plt.figtext(0.98+horshift, 0.45 - vertshift, 'rsq = '+ str(rsq1) + '; p-value = ' + str(pval1))
plt.figtext(0.95+horshift, 0.4 - vertshift, 'Unregulated GW equation: y = '+str(m2)+'x + '+str(yint2))
plt.figtext(0.98+horshift, 0.35 - vertshift, 'rsq = '+ str(rsq2) +'; p-value = ' + str(pval2))
plt.figtext(0.95+horshift, 0.3 - vertshift, 'Mixed SW/GW equation: y = '+str(m3)+'x + '+str(yint3))
plt.figtext(0.98+horshift, 0.25 - vertshift, 'rsq = '+ str(rsq3) +'; p-value = ' + str(pval3))
plt.figtext(0.95+horshift, 0.2 - vertshift, 'Regulated GW equation: y = '+str(m4)+'x + '+str(yint4))
plt.figtext(0.98+horshift, 0.15 - vertshift, 'rsq = '+ str(rsq4) +'; p-value = ' + str(pval4))
plt.figtext(0.95+horshift, 0.1 - vertshift, 'SW equation: y = '+str(m5)+'x + '+str(yint5))
plt.figtext(0.98+horshift, 0.05 - vertshift, 'rsq = '+ str(rsq5) +'; p-value = ' + str(pval5))

# -- Piece 2 --
min_yr = 1985
mx_yr = 1995
Name2 = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name2)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
for i in column_list:
        df = f[i]
        # df = f[i].pct_change()
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
        stats = stats.append({'slope': slope, 
                              'int':intercept, 
                              'rsq':r_value*r_value, 
                              'p_val':p_value, 
                              'std_err':std_err, 
                              'mean': np.mean(y),
                              'var': np.var(y),
                              'sum': np.sum(y)
                              },
                              ignore_index=True)


stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)
# -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
m1 = round(stats1.loc['slope',betterlabels[0]], 2)
m2 = round(stats1.loc['slope',betterlabels[3]], 2)
m3 = round(stats1.loc['slope',betterlabels[4]], 2)
m4 = round(stats1.loc['slope',betterlabels[1]], 2)
m5 = round(stats1.loc['slope',betterlabels[2]], 2)
yint1 = round(stats1.loc['int',betterlabels[0]], 2)
yint2 = round(stats1.loc['int',betterlabels[3]], 2)
yint3 = round(stats1.loc['int',betterlabels[4]], 2)
yint4 = round(stats1.loc['int',betterlabels[1]], 2)
yint5 = round(stats1.loc['int',betterlabels[2]], 2)
rsq1 = round(stats1.loc['rsq', betterlabels[0]], 4)
rsq2 = round(stats1.loc['rsq', betterlabels[3]], 4)
rsq3 = round(stats1.loc['rsq', betterlabels[4]], 4)
rsq4 = round(stats1.loc['rsq', betterlabels[1]], 4)
rsq5 = round(stats1.loc['rsq', betterlabels[2]], 4)
pval1 = round(stats1.loc['p_val', betterlabels[0]], 4)
pval2 = round(stats1.loc['p_val', betterlabels[3]], 4)
pval3 = round(stats1.loc['p_val', betterlabels[4]], 4)
pval4 = round(stats1.loc['p_val', betterlabels[1]], 4)
pval5 = round(stats1.loc['p_val', betterlabels[2]], 4)
yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2
yf3 = (m3*xf)+yint3
yf4 = (m4*xf)+yint4
yf5 = (m5*xf)+yint5

# ax.plot(xf1, yf1,"-.",color=c_2, lw=1)
ax.plot(xf1, yf2,"-.",color=c_7, lw=1)
ax.plot(xf1, yf3,"-.",color=c_5, lw=1)
ax.plot(xf1, yf4,"-.",color=c_3, lw=1)
# ax.plot(xf1, yf5,"-.",color=c_4, lw=1)

vertshift = -0.3
horshift = 0.3
plt.figtext(0.94+horshift, 0.55 - vertshift, 'Regression for ' +str(min_yr)+' to '+str(mx_yr)+':')
plt.figtext(0.95+horshift, 0.5 - vertshift, 'CAP equation: y = '+str(m1)+'x + '+str(yint1))
plt.figtext(0.98+horshift, 0.45 - vertshift, 'rsq = '+ str(rsq1) + '; p-value = ' + str(pval1))
plt.figtext(0.95+horshift, 0.4 - vertshift, 'Unregulated GW equation: y = '+str(m2)+'x + '+str(yint2))
plt.figtext(0.98+horshift, 0.35 - vertshift, 'rsq = '+ str(rsq2) +'; p-value = ' + str(pval2))
plt.figtext(0.95+horshift, 0.3 - vertshift, 'Mixed SW/GW equation: y = '+str(m3)+'x + '+str(yint3))
plt.figtext(0.98+horshift, 0.25 - vertshift, 'rsq = '+ str(rsq3) +'; p-value = ' + str(pval3))
plt.figtext(0.95+horshift, 0.2 - vertshift, 'Regulated GW equation: y = '+str(m4)+'x + '+str(yint4))
plt.figtext(0.98+horshift, 0.15 - vertshift, 'rsq = '+ str(rsq4) +'; p-value = ' + str(pval4))
plt.figtext(0.95+horshift, 0.1 - vertshift, 'SW equation: y = '+str(m5)+'x + '+str(yint5))
plt.figtext(0.98+horshift, 0.05 - vertshift, 'rsq = '+ str(rsq5) +'; p-value = ' + str(pval5))

# -- Piece 3 --
min_yr = 1995
mx_yr = 2020
Name3 = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name3)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
for i in column_list:
        df = f[i]
        # df = f[i].pct_change()
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
        stats = stats.append({'slope': slope, 
                              'int':intercept, 
                              'rsq':r_value*r_value, 
                              'p_val':p_value, 
                              'std_err':std_err, 
                              'mean': np.mean(y),
                              'var': np.var(y),
                              'sum': np.sum(y)
                              },
                              ignore_index=True)


stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)
# -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
m1 = round(stats1.loc['slope',betterlabels[0]], 2)
m2 = round(stats1.loc['slope',betterlabels[3]], 2)
m3 = round(stats1.loc['slope',betterlabels[4]], 2)
m4 = round(stats1.loc['slope',betterlabels[1]], 2)
m5 = round(stats1.loc['slope',betterlabels[2]], 2)
yint1 = round(stats1.loc['int',betterlabels[0]], 2)
yint2 = round(stats1.loc['int',betterlabels[3]], 2)
yint3 = round(stats1.loc['int',betterlabels[4]], 2)
yint4 = round(stats1.loc['int',betterlabels[1]], 2)
yint5 = round(stats1.loc['int',betterlabels[2]], 2)
rsq1 = round(stats1.loc['rsq', betterlabels[0]], 4)
rsq2 = round(stats1.loc['rsq', betterlabels[3]], 4)
rsq3 = round(stats1.loc['rsq', betterlabels[4]], 4)
rsq4 = round(stats1.loc['rsq', betterlabels[1]], 4)
rsq5 = round(stats1.loc['rsq', betterlabels[2]], 4)
pval1 = round(stats1.loc['p_val', betterlabels[0]], 4)
pval2 = round(stats1.loc['p_val', betterlabels[3]], 4)
pval3 = round(stats1.loc['p_val', betterlabels[4]], 4)
pval4 = round(stats1.loc['p_val', betterlabels[1]], 4)
pval5 = round(stats1.loc['p_val', betterlabels[2]], 4)
yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2
yf3 = (m3*xf)+yint3
yf4 = (m4*xf)+yint4
yf5 = (m5*xf)+yint5

# ax.plot(xf1, yf1,"-.",color=c_2, lw=1)
ax.plot(xf1, yf2,"-.",color=c_7, lw=1)
ax.plot(xf1, yf3,"-.",color=c_5, lw=1)
ax.plot(xf1, yf4,"-.",color=c_3, lw=1)
# ax.plot(xf1, yf5,"-.",color=c_4, lw=1)

vertshift = -0.3
horshift = 0.6
plt.figtext(0.94+horshift, 0.55 - vertshift, 'Regression for ' +str(min_yr)+' to '+str(mx_yr)+':')
plt.figtext(0.95+horshift, 0.5 - vertshift, 'CAP equation: y = '+str(m1)+'x + '+str(yint1))
plt.figtext(0.98+horshift, 0.45 - vertshift, 'rsq = '+ str(rsq1) + '; p-value = ' + str(pval1))
plt.figtext(0.95+horshift, 0.4 - vertshift, 'Unregulated GW equation: y = '+str(m2)+'x + '+str(yint2))
plt.figtext(0.98+horshift, 0.35 - vertshift, 'rsq = '+ str(rsq2) +'; p-value = ' + str(pval2))
plt.figtext(0.95+horshift, 0.3 - vertshift, 'Mixed SW/GW equation: y = '+str(m3)+'x + '+str(yint3))
plt.figtext(0.98+horshift, 0.25 - vertshift, 'rsq = '+ str(rsq3) +'; p-value = ' + str(pval3))
plt.figtext(0.95+horshift, 0.2 - vertshift, 'Regulated GW equation: y = '+str(m4)+'x + '+str(yint4))
plt.figtext(0.98+horshift, 0.15 - vertshift, 'rsq = '+ str(rsq4) +'; p-value = ' + str(pval4))
plt.figtext(0.95+horshift, 0.1 - vertshift, 'SW equation: y = '+str(m5)+'x + '+str(yint5))
plt.figtext(0.98+horshift, 0.05 - vertshift, 'rsq = '+ str(rsq5) +'; p-value = ' + str(pval5))


# --- Code for Main Plot ---
ds = cat_wl2_SW
minyear=1975
maxyear=2020
min_y = 75
max_y = 300
fsize = 14

# ax.plot(ds['CAP'], label='CAP', color=c_2)
ax.plot(ds['No_CAP'], label='Regulated GW', color=c_3) 
# ax.plot(ds['SW'], label='Surface Water', color=c_4) 
ax.plot(ds['Mix'], label='Mixed SW/GW', color=c_5)
ax.plot(ds['GW'], label='Unregulated GW', color=c_7)  

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
# ax.set_title(name, fontsize=20)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = fsize)

ax.minorticks_on()

fig.set_dpi(600.0)

# ax.set_xlim(min_yr, mx_yr)
ax.set_ylim(75,300)
# ax.set_title(Name)
ax.set_title('Linear Regression Depth to Water and Access to Surface Water Categories')
ax.legend(
        # loc = [1.065, 0.75]
        )
# plt.savefig(outputpath+'Stats/Water_CAT/'+Name+'_all', bbox_inches = 'tight')
plt.savefig(outputpath+'Stats/Water_CAT/'+Name+'_GW_3pieces', bbox_inches = 'tight')
# stats1.to_csv(outputpath+'Stats/Water_CAT/'+Name+'_GW.csv')

# %% For Depth to Water by regulation
ds = cat_wl2_reg
data_type = "Depth to Water"
min_yr = 1975
mx_yr = 2020
betterlabels = ['Regulated','Unregulated'] 
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
# for i in range(1, 12, 1):
for i in column_list:
        df = f[i]
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
        # print('Georegion Number: ', i, '\n', 
        #        'slope = ', slope, '\n', 
        #        'intercept = ', intercept, '\n', 
        #        'r^2 = ', r_value, '\n', 
        #        'p-value = ', p_value, '\n', 
        #        'std error = ', std_err)      
        stats = stats.append({'slope': slope, 
                              'int':intercept, 
                              'rsq':r_value*r_value, 
                              'p_val':p_value, 
                              'std_err':std_err, 
                              'mean': np.mean(y),
                              'var': np.var(y),
                              'sum': np.sum(y)
                              },
                              ignore_index=True)
        # xf = np.linspace(min(x),max(x),100)
        # xf1 = xf.copy()
        # xf1 = pd.to_datetime(xf1)
        # yf = (slope*xf)+intercept
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(xf1, yf,label='Linear fit', lw=3)
        # df.plot(ax=ax,marker='o', ls='')
        # ax.set_ylim(0,max(y))
        # ax.legend()


stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)

# -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
#xf1 = pd.to_datetime(xf1)
m1 = round(stats1.loc['slope','Regulated'], 2)
m2 = round(stats1.loc['slope','Unregulated'], 2)
yint1 = round(stats1.loc['int','Regulated'], 2)
yint2 = round(stats1.loc['int','Unregulated'], 2)
pval1 = round(stats1.loc['p_val', 'Regulated'], 4)
pval2 = round(stats1.loc['p_val', 'Unregulated'], 4)

yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2

fig, ax = plt.subplots(1, 1, figsize = (7,4.5))
ax.plot(xf1, yf1,"-.",color=cap,label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color=GWdom, lw=1)

ds = cat_wl2_reg
minyear=1975
maxyear=2020
min_y = 75
max_y = 300
fsize = 12

ax.plot(ds['R'], label='Regulated', color=cap) 
ax.plot(ds['U'], label='Unregulated', color=GWdom) 

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = fsize)
# # Drought Year Shading
# a = 1988.5
# b = 1990.5
# c = 1995.5
# d = 1996.5
# e = 2001.5
# f = 2003.5
# g = 2005.5
# h = 2007.5
# i = 2011.5
# j = 2014.5
# k = 2017.5
# l= 2018.5
# plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)

ax.minorticks_on()

fig.set_dpi(600.0)

# ax.set_xlim(min_yr, mx_yr)
ax.set_ylim(75,300)
# ax.set_title(Name)

plt.figtext(0.95, 0.4, 'Regulated equation: y= '+str(m1)+'x + '+str(yint1))
plt.figtext(0.96, 0.35, 'p-value = ' + str(pval1))
plt.figtext(0.95, 0.6, 'Unregulated equation: y= '+str(m2)+'x + '+str(yint2))
plt.figtext(0.96, 0.55, 'p-value = ' + str(pval2))
ax.legend()
plt.savefig(outputpath+'Stats/'+Name, bbox_inches = 'tight')
# stats1.to_csv(outputpath+'Stats/'+Name+'.csv')
