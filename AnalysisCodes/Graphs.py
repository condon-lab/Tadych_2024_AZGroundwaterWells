# ----- All Paper Graphs -----
# %% Load the packages
from cProfile import label
from operator import ge
from optparse import Values
import os
from geopandas.tools.sjoin import sjoin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import datetime
from matplotlib.transforms import Bbox
import seaborn as sns
import numpy as np
import pandas as pd
from shapely.geometry import box
import geopandas as gp
#import earthpy as et
import scipy.stats as sp

# === Assign Data paths ===

# This is for accessing our data on Cyverse
datapath_web = 'https://data.cyverse.org/dav-anon/iplant/home/dtadych/AZ_Spatial_Analysis/Data/'
outputpath_web = 'https://data.cyverse.org/dav-anon/iplant/home/dtadych/AZ_Spatial_Analysis/Data/Output_files/'
shapepath_web = 'https://data.cyverse.org/dav-anon/iplant/home/dtadych/AZ_Spatial_Analysis/Data/Shapefiles/'

# This is if you created your own database
datapath_local = '../Data'
outputpath_local = '../Data/Output_files/'
shapepath_local = '../Data/Shapefiles/'

# Change this based on whether you're running off local or web data
# Cyverse:
datapath = datapath_web
outputpath = outputpath_web
shapepath = shapepath_web

# Local: 
# datapath = datapath_local
# outputpath = outputpath_local
# shapepath = shapepath_local
# %% Read in the data
# Shallow and Deep and drilling depth cutoffs
shallow = 200
deep = 500

#%% Importing the Depth categories for Well Counts
wdc1_reg = pd.read_csv(outputpath+'Final_Welldepth_regulation' + str(deep) + 'plus.csv',
                        header=1, index_col=0)
wdc1_reg = wdc1_reg.iloc[1:,:]
wdc2_reg = pd.read_csv(outputpath+'Final_Welldepth_regulation' + str(shallow) + 'to' + str(deep) + '.csv',
                        header=1, index_col=0)
wdc2_reg = wdc2_reg.iloc[1:,:]
wdc3_reg = pd.read_csv(outputpath+'Final_Welldepth_regulation' + str(shallow) + 'minus.csv',
                        header=1, index_col=0)
wdc3_reg = wdc3_reg.iloc[1:,:]

wdc1_wc = pd.read_csv(outputpath+'Final_Welldepth_sw' + str(deep) + 'plus.csv',
                        header=1, index_col=0)
wdc1_wc = wdc1_wc.iloc[1:,:]
wdc2_wc = pd.read_csv(outputpath+'Final_Welldepth_sw' + str(shallow) + 'to' + str(deep) + '.csv',
                        header=1, index_col=0)
wdc2_wc = wdc2_wc.iloc[1:,:]
wdc3_wc = pd.read_csv(outputpath+'Final_Welldepth_sw' + str(shallow) + 'minus.csv',
                        header=1, index_col=0)
wdc3_wc = wdc3_wc.iloc[1:,:]

wdc1_reg_ex = pd.read_csv(outputpath+'Final_Welldepth_regulation_exemptstatus' + str(deep) + 'plus.csv',
                        header=[1,2], index_col=0)
wdc1_reg_ex = wdc1_reg_ex.iloc[:,:]
# wdc1_reg_ex = wdc1_reg_ex.drop('MONITOR',axis=1,level=1)
wdc2_reg_ex = pd.read_csv(outputpath+'Final_Welldepth_regulation_exemptstatus' + str(shallow) + 'to' + str(deep) + '.csv',
                        header=[1,2], index_col=0)
wdc2_reg_ex = wdc2_reg_ex.iloc[:,:]
# wdc2_reg_ex = wdc2_reg_ex.drop('MONITOR',axis=1,level=1)
wdc3_reg_ex = pd.read_csv(outputpath+'Final_Welldepth_regulation_exemptstatus' + str(shallow) + 'minus.csv',
                        header=[1,2], index_col=0)
wdc3_reg_ex = wdc3_reg_ex.iloc[:,:]
# wdc3_reg_ex = wdc3_reg_ex.drop('MONITOR',axis=1,level=1)
wdc1_wc_ex = pd.read_csv(outputpath+'Final_Welldepth_sw_exemptstatus' + str(deep) + 'plus.csv',
                        header=[1,2], index_col=0)
wdc1_wc_ex = wdc1_wc_ex.iloc[:,:]
# wdc1_wc_ex = wdc1_wc_ex.drop('MONITOR',axis=1,level=1)
wdc2_wc_ex = pd.read_csv(outputpath+'Final_Welldepth_sw_exemptstatus' + str(shallow) + 'to' + str(deep) + '.csv',
                        header=[1,2], index_col=0)
wdc2_wc_ex = wdc2_wc_ex.iloc[:,:]
# wdc2_wc_ex = wdc2_wc_ex.drop('MONITOR',axis=1,level=1)
wdc3_wc_ex = pd.read_csv(outputpath+'Final_Welldepth_sw_exemptstatus' + str(shallow) + 'minus.csv',
                        header=[1,2], index_col=0)
wdc3_wc_ex = wdc3_wc_ex.iloc[:,:]
# wdc3_wc_ex = wdc3_wc_ex.drop('MONITOR',axis=1,level=1)
wdc3_wc_ex

# %% Now importing Well Densities
dens_wdc1_reg = pd.read_csv(outputpath+'FinalDensities_Welldepth_regulation' + str(deep) + 'plus.csv',
                        header=1, index_col=0)
dens_wdc1_reg = dens_wdc1_reg.iloc[1:,:]
dens_wdc2_reg = pd.read_csv(outputpath+'FinalDensities_Welldepth_regulation' + str(shallow) + 'to' + str(deep) + '.csv',
                        header=1, index_col=0)
dens_wdc2_reg = dens_wdc2_reg.iloc[1:,:]
dens_wdc3_reg = pd.read_csv(outputpath+'FinalDensities_Welldepth_regulation' + str(shallow) + 'minus.csv',
                        header=1, index_col=0)
dens_wdc3_reg = dens_wdc3_reg.iloc[1:,:]

dens_wdc1_wc = pd.read_csv(outputpath+'FinalDensities_Welldepth_sw' + str(deep) + 'plus.csv',
                        header=1, index_col=0)
dens_wdc1_wc = dens_wdc1_wc.iloc[1:,:]
dens_wdc2_wc = pd.read_csv(outputpath+'FinalDensities_Welldepth_sw' + str(shallow) + 'to' + str(deep) + '.csv',
                        header=1, index_col=0)
dens_wdc2_wc = dens_wdc2_wc.iloc[1:,:]
dens_wdc3_wc = pd.read_csv(outputpath+'FinalDensities_Welldepth_sw' + str(shallow) + 'minus.csv',
                        header=1, index_col=0)
dens_wdc3_wc = dens_wdc3_wc.iloc[1:,:]

dens_wdc1_reg_ex = pd.read_csv(outputpath+'FinalDensities_Welldepth_regulation_exemptstatus' + str(deep) + 'plus.csv',
                        header=[1,2], index_col=0)
dens_wdc1_reg_ex = dens_wdc1_reg_ex.iloc[:,:]
# dens_wdc1_reg_ex = dens_wdc1_reg_ex.drop('MONITOR',axis=1,level=1)
dens_wdc2_reg_ex = pd.read_csv(outputpath+'FinalDensities_Welldepth_regulation_exemptstatus' + str(shallow) + 'to' + str(deep) + '.csv',
                        header=[1,2], index_col=0)
dens_wdc2_reg_ex = dens_wdc2_reg_ex.iloc[:,:]
# dens_wdc2_reg_ex = dens_wdc2_reg_ex.drop('MONITOR',axis=1,level=1)
dens_wdc3_reg_ex = pd.read_csv(outputpath+'FinalDensities_Welldepth_regulation_exemptstatus' + str(shallow) + 'minus.csv',
                        header=[1,2], index_col=0)
dens_wdc3_reg_ex = dens_wdc3_reg_ex.iloc[:,:]
# dens_wdc3_reg_ex = dens_wdc3_reg_ex.drop('MONITOR',axis=1,level=1)
dens_wdc1_wc_ex = pd.read_csv(outputpath+'FinalDensities_Welldepth_sw_exemptstatus' + str(deep) + 'plus.csv',
                        header=[1,2], index_col=0)
dens_wdc1_wc_ex = dens_wdc1_wc_ex.iloc[:,:]
# dens_wdc1_wc_ex = dens_wdc1_wc_ex.drop('MONITOR',axis=1,level=1)
dens_wdc2_wc_ex = pd.read_csv(outputpath+'FinalDensities_Welldepth_sw_exemptstatus' + str(shallow) + 'to' + str(deep) + '.csv',
                        header=[1,2], index_col=0)
dens_wdc2_wc_ex = dens_wdc2_wc_ex.iloc[:,:]
# dens_wdc2_wc_ex = dens_wdc2_wc_ex.drop('MONITOR',axis=1,level=1)
dens_wdc3_wc_ex = pd.read_csv(outputpath+'FinalDensities_Welldepth_sw_exemptstatus' + str(shallow) + 'minus.csv',
                        header=[1,2], index_col=0)
dens_wdc3_wc_ex = dens_wdc3_wc_ex.iloc[:,:]
# dens_wdc3_wc_ex = dens_wdc3_wc_ex.drop('MONITOR',axis=1,level=1)
dens_wdc3_wc_ex

# %% Importing Water Level Values
# For regulation
filepath = outputpath_web+'/Waterlevels_Regulation.csv'
# filepath = '../Data/Output_files/Waterlevels_Regulation.csv'
cat_wl2_reg = pd.read_csv(filepath, index_col=0)
cat_wl2_reg.head()

# For Access to SW
filepath = outputpath_web+'/Waterlevels_AccesstoSW.csv'
# filepath = '../Data/Output_files/Waterlevels_AccesstoSW.csv'
cat_wl2_SW = pd.read_csv(filepath, index_col=0)
cat_wl2_SW.head()

# For georegion number
filepath = outputpath_web+'Waterlevels_georegions.csv'
# filepath = '../Data/Output_files/Waterlevels_georegions.csv'
cat_wl2_georeg = pd.read_csv(filepath, index_col=0)
# cat_wl2_georeg.head()
# %% Creating colors
c_1 = '#8d5a99' # Reservation
c_2 = "#d7191c" # Regulated with CAP (Water Category Color)
c_3 = '#e77a47' # Regulated without CAP (Water Category Color)
c_4 = '#2cbe21' # Lower CO River - SW (Water Category Color)
c_5 = '#2f8c73' # Upper CO River - Mixed (Water Category Color)
c_6 = '#6db7e8' # SE - GW
c_7 = '#165782' # NW - GW (Water Category color)
c_8 = '#229ce8' # SC - GW
c_9 = '#1f78b4' # NE - GW
c_10 = '#41bf9e' # N - Mixed
c_11 = '#7adec4' # C - Mixed
drought_color = '#ffa6b8'
wet_color = '#b8d3f2'

reg_colors = [c_2,c_7]
georeg_colors = [c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11]
SW_colors = [c_2,c_3,c_4,c_5,c_7]

bar_watercatc = [c_2,c_3,c_4,c_5,c_7]

# Color blind palette
# https://jacksonlab.agronomy.wisc.edu/2016/05/23/15-level-colorblind-friendly-palette/
blind =["#000000","#004949","#009292","#ff6db6","#ffb6db",
 "#490092","#006ddb","#b66dff","#6db6ff","#b6dbff",
 "#920000","#924900","#db6d00","#24ff24","#ffff6d"]

# Matching new map

cap = '#C6652B'
# noCAP = '#EDE461' # This is one from the map
noCAP = '#CCC339' # This color but darker for lines
GWdom = '#3B76AF'
mixed = '#6EB2E4'
swdom = '#469B76'

# %% # Plot all of the different depths 3 in a line
# Formatting correctly
test = wdc1_reg.copy()
test = test.reset_index()
test['Regulation'] = test['Regulation'].astype(float)
test['Regulation'] = test['Regulation'].astype(int)
test.set_index('Regulation', inplace=True)
test.info()
wdc1_reg = test

test = wdc2_reg.copy()
test = test.reset_index()
test['Regulation'] = test['Regulation'].astype(float)
test['Regulation'] = test['Regulation'].astype(int)
test.set_index('Regulation', inplace=True)
test.info()
wdc2_reg = test

test = wdc3_reg.copy()
test = test.reset_index()
test['Regulation'] = test['Regulation'].astype(float)
test['Regulation'] = test['Regulation'].astype(int)
test.set_index('Regulation', inplace=True)
test.info()
wdc3_reg = test

ds1 = wdc1_reg
ds2 = wdc2_reg
ds3 = wdc3_reg

name = 'New Wells by Drilling Depths over Time'
ylabel = "Well Count (#)"
minyear=1975.0
maxyear=2020.0
#min_y = -15
#max_y = 7
fsize = 14

columns = ds1.columns
labels = ds1.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(1,3,figsize=(15,5))
#fig.tight_layout()
# fig.suptitle(name, fontsize=18, y=1.00)
fig.supylabel(ylabel, fontsize = 14, x=0.07)
ax[0].plot(ds3[labels[0]],'--', label='Regulated', color='black')
ax[0].plot(ds3[labels[2]], label='Unregulated', color='black')
ax[1].plot(ds2[labels[0]],'--', label='Regulated', color='black')
ax[1].plot(ds2[labels[2]], label='Unregulated', color='black')
ax[2].plot(ds1[labels[0]],'--', label='Regulated', color='black')
ax[2].plot(ds1[labels[2]], label='Unregulated', color='black')

ax[0].set_xlim(minyear,maxyear)
ax[1].set_xlim(minyear,maxyear)
ax[2].set_xlim(minyear,maxyear)

#ax[0].set_ylim(min_y,max_y)
#ax[1].set_ylim(min_y,max_y)
#ax[2].set_ylim(min_y,max_y)

# ax[0].set_title('Shallow (<200 ft)', loc='center')
# ax[1].set_title('Midrange (200-500ft)', loc='center')
# ax[2].set_title('Deep (> 500ft)', loc='center')


ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

fig.set_dpi(600.0)

# plt.savefig(outputpath_local+name+'_3horizontalpanel_regulated', bbox_inches='tight')

# %%
# Plot all of the different depths 3 in a line
# Formatting correctly
test = wdc1_wc.copy()
test = test.reset_index()
test['Water_CAT'] = test['Water_CAT'].astype(float)
test['Water_CAT'] = test['Water_CAT'].astype(int)
test.set_index('Water_CAT', inplace=True)
test.info()
wdc1_wc = test

test = wdc2_wc.copy()
test = test.reset_index()
test['Water_CAT'] = test['Water_CAT'].astype(float)
test['Water_CAT'] = test['Water_CAT'].astype(int)
test.set_index('Water_CAT', inplace=True)
test.info()
wdc2_wc = test

test = wdc3_wc.copy()
test = test.reset_index()
test['Water_CAT'] = test['Water_CAT'].astype(float)
test['Water_CAT'] = test['Water_CAT'].astype(int)
test.set_index('Water_CAT', inplace=True)
test.info()
wdc3_wc = test

ds1 = wdc1_wc
ds2 = wdc2_wc
ds3 = wdc3_wc

name = 'New Wells by Drilling Depths over Time'
ylabel = "Well Count (#)"
minyear=1975
maxyear=2022
#min_y = -15
#max_y = 7
fsize = 14

columns = ds1.columns
labels = ds1.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(1,3,figsize=(15,5))
#fig.tight_layout()
# fig.suptitle(name, fontsize=18, y=1.00)
fig.supylabel(ylabel, fontsize = 14, x=0.06)

ax[0].plot(ds3[labels[0]], label='Receives CAP (Regulated)', color=cap)
ax[0].plot(ds3[labels[3]], label='GW Dominated (Regulated)', color=noCAP)
ax[0].plot(ds3[labels[4]], label='Surface Water Dominated', color=swdom)
ax[0].plot(ds3[labels[2]], label='Mixed Source', color=mixed)
ax[0].plot(ds3[labels[1]], label='GW Dominated', color=GWdom)

ax[1].plot(ds2[labels[0]], label='Receives CAP (Regulated)', color=cap)
ax[1].plot(ds2[labels[3]], label='GW Dominated (Regulated)', color=noCAP)
ax[1].plot(ds2[labels[4]], label='Surface Water Dominated', color=swdom)
ax[1].plot(ds2[labels[2]], label='Mixed Source', color=mixed)
ax[1].plot(ds2[labels[1]], label='GW Dominated', color=GWdom)

ax[2].plot(ds1[labels[0]], label='Receives CAP (Regulated)', color=cap)
ax[2].plot(ds1[labels[3]], label='GW Dominated (Regulated)', color=noCAP)
ax[2].plot(ds1[labels[4]], label='Surface Water Dominated', color=swdom)
ax[2].plot(ds1[labels[2]], label='Mixed Source', color=mixed)
ax[2].plot(ds1[labels[1]], label='GW Dominated', color=GWdom)


ax[0].set_xlim(minyear,maxyear)
ax[1].set_xlim(minyear,maxyear)
ax[2].set_xlim(minyear,maxyear)

#ax[0].set_ylim(min_y,max_y)
#ax[1].set_ylim(min_y,max_y)
#ax[2].set_ylim(min_y,max_y)

# ax[0].set_title('Shallow (<200 ft)', loc='center')
# ax[1].set_title('Midrange (200-500ft)', loc='center')
# ax[2].set_title('Deep (> 500ft)', loc='center')


ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

fig.set_dpi(600.0)

plt.savefig(outputpath_local+name+'_watercat_3horizontalpanel', bbox_inches='tight')

# %% -- Grouped bar chart - Had to create some summarazing dataframes --
# Check the commented code to turn on whichever graph you want to make
#   - dens = well densities
#   - wdc1 = water depth category 1 (deep)
#     wdc2 = midrange
#     wdc3 = shallow

# Below is for Groundwater Regulation
ds = wdc1_reg.copy()
# ds = dens_wdc1_reg.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['Regulated'] = ds[labels[0]]
ds1['Unregulated'] = ds[labels[2]]

dft1 = ds1.copy()
dft1


ds = wdc2_reg.copy()
# ds = dens_wdc2_reg.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['Regulated'] = ds[labels[0]]
ds1['Unregulated'] = ds[labels[2]]

dft2 = ds1.copy()
dft2


ds = wdc3_reg.copy()
# ds = dens_wdc3_reg.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['Regulated'] = ds[labels[0]]
ds1['Unregulated'] = ds[labels[2]]

dft3 = ds1.copy()
dft3

df1 = pd.DataFrame(dft1.sum())
df1 = df1.transpose()
df1 = df1.reset_index()
df1['index'] = 'Deep'
df1.set_index('index', inplace=True)
df1

df2 = pd.DataFrame(dft2.sum())
df2 = df2.transpose()
df2 = df2.reset_index()
df2['index'] = 'Midrange'
df2.set_index('index', inplace=True)
df2

df3 = pd.DataFrame(dft3.sum())
df3 = df3.transpose()
df3 = df3.reset_index()
df3['index'] = 'Shallow'
df3.set_index('index', inplace=True)
df3

df_test = df3.append([df2,df1])
df_test = df_test.transpose()
df_test = df_test.rename_axis(None,axis=1)
df_test

# group_colors = ['cornflowerblue','slategrey','darkblue']
group_colors = ['lightsteelblue','cornflowerblue','darkblue']

# name = 'Well Densities by Groundwater Regulation'
# horlabel = 'Well Densities (well/km^2)'
name = 'Number of wells by Groundwater Regulation'
horlabel = 'Number of Wells (#)'
fsize = 14

df_test.plot(figsize = (9,6),
        kind='bar',
        stacked=False,
        # title=name,
        color = group_colors,
        zorder = 2,
        width = 0.85,
        fontsize = fsize
        )
# plt.title(name, fontsize = (fsize+2))
plt.ylabel(horlabel, fontsize = fsize)
plt.xticks(rotation=0)
plt.grid(axis='y', linewidth=0.5, zorder=0)
plt.legend(fontsize = fsize)


# plt.savefig(outputpath_local+name+'groupedchart', dpi=600)


# %% Below is for Water Access Category
ds = wdc1_wc.copy()
# ds = dens_wdc1_wc.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['Receives \nCAP \n(Regulated)'] = ds[labels[0]]
ds1['GW \nDominated \n(Regulated)'] = ds[labels[3]]
ds1['Surface \nWater \nDominated'] = ds[labels[5]]
ds1['Mixed Source'] = ds[labels[2]]
ds1['GW \nDominated'] = ds[labels[1]]

dft1 = ds1.copy()
dft1


ds = wdc2_wc.copy()
# ds = dens_wdc2_wc.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['Receives \nCAP \n(Regulated)'] = ds[labels[0]]
ds1['GW \nDominated \n(Regulated)'] = ds[labels[3]]
ds1['Surface \nWater \nDominated'] = ds[labels[5]]
ds1['Mixed Source'] = ds[labels[2]]
ds1['GW \nDominated'] = ds[labels[1]]

dft2 = ds1.copy()
dft2


ds = wdc3_wc.copy()
# ds = dens_wdc3_wc.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['Receives \nCAP \n(Regulated)'] = ds[labels[0]]
ds1['GW \nDominated \n(Regulated)'] = ds[labels[3]]
ds1['Surface \nWater \nDominated'] = ds[labels[5]]
ds1['Mixed Source'] = ds[labels[2]]
ds1['GW \nDominated'] = ds[labels[1]]

dft3 = ds1.copy()
dft3

df1 = pd.DataFrame(dft1.sum())
df1 = df1.transpose()
df1 = df1.reset_index()
df1['index'] = 'Deep'
df1.set_index('index', inplace=True)
df1

df2 = pd.DataFrame(dft2.sum())
df2 = df2.transpose()
df2 = df2.reset_index()
df2['index'] = 'Midrange'
df2.set_index('index', inplace=True)
df2

df3 = pd.DataFrame(dft3.sum())
df3 = df3.transpose()
df3 = df3.reset_index()
df3['index'] = 'Shallow'
df3.set_index('index', inplace=True)
df3

df_test = df3.append([df2,df1])
df_test = df_test.transpose()
df_test = df_test.rename_axis(None,axis=1)
df_test

# group_colors = ['cornflowerblue','slategrey','darkblue']
group_colors = ['lightsteelblue','cornflowerblue','darkblue']

# name = 'Well Densities by Access to Surface Water'
# horlabel = 'Well Densities (well/km^2)'
name = 'Number of wells by Access to Surface Water'
horlabel = 'Number of Wells (#)'
fsize = 14

df_test.plot(figsize = (9,6),
        kind='bar',
        stacked=False,
        # title=name,
        color = group_colors,
        zorder = 2,
        width = 0.85,
        fontsize = fsize
        )
# plt.title(name, fontsize = (fsize+2))
plt.ylabel(horlabel, fontsize = fsize)
plt.xticks(rotation=0)
plt.grid(axis='y', linewidth=0.5, zorder=0)
plt.legend(fontsize = fsize)

# plt.savefig(outputpath_local+name+'groupedchart_version2',dpi=600)
# %% Now for plotting with exempt/non-exempt on the bar graph for Number wells
# Summing the data

# Shallow
ds = wdc3_reg_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds1 = ds.transpose()

# Midrange
ds = wdc2_reg_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds2 = ds.transpose()

# Deep
ds = wdc1_reg_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds3 = ds.transpose()

shallow_exempt = np.array([ds1.iloc[0,0],ds1.iloc[0,3]])
shallow_nonexempt = np.array([ds1.iloc[0,2],ds1.iloc[0,5]])
mid_exempt = np.array([ds2.iloc[0,0],ds2.iloc[0,3]])
mid_nonexempt = np.array([ds2.iloc[0,2],ds2.iloc[0,5]])
deep_exempt = np.array([ds3.iloc[0,0],ds3.iloc[0,3]])
deep_nonexempt = np.array([ds3.iloc[0,2],ds3.iloc[0,5]])
big_categories = ['Regulated', 'Unregulated']
depth_colors = ['lightsteelblue','cornflowerblue','darkblue']

with sns.axes_style("white"):
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # plot details
    bar_width = 0.25
    epsilon = .0
    line_width = 1
    opacity = 0.7
    left_bar_positions = np.arange(len(shallow_exempt))
    middle_bar_positions = left_bar_positions + bar_width
    right_bar_positions = middle_bar_positions + bar_width

    # make bar plots
    plt.figure(figsize=(10, 8))

    shallow_Exempt_Bar = plt.bar(left_bar_positions, shallow_exempt, bar_width-epsilon,
                              color=depth_colors[0],
                              edgecolor='000000',
                              linewidth=line_width,
                              hatch='//',
                              label='Shallow: Small Wells')
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width,
                              bottom=shallow_exempt,
                              linewidth=line_width,
                              edgecolor='000000',
                            #   alpha=opacity,
                              color=depth_colors[0],
                              label='     "      : Large Wells')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width-epsilon,
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              linewidth=line_width,
                              label='Midrange: Small Wells')
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width,
                              bottom=mid_exempt, # On top of first category
                              edgecolor='000000',
                              linewidth=line_width,
                              color=depth_colors[1],
                              label='      "       : Large Wells')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width-epsilon,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//',
                              label='Deep: Small Wells')
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width,
                              bottom=deep_exempt,
                              linewidth=line_width,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              label='   "    : Large Wells')

    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Number of Wells')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    # plt.legend(bbox_to_anchor=(1.1, 1.05))  
    sns.despine()  
    # plt.savefig(outputpath_local+'NumberWells_Regulation_BarGraph', bbox_inches='tight')

# %% Now for plotting with exempt/non-exempt on the bar graph for densities
# Summing the data

# Shallow
ds = dens_wdc3_reg_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds1 = ds.transpose()

# Midrange
ds = dens_wdc2_reg_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds2 = ds.transpose()

# Deep
ds = dens_wdc1_reg_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds3 = ds.transpose()

shallow_exempt = np.array([ds1.iloc[0,0],ds1.iloc[0,3]])
shallow_nonexempt = np.array([ds1.iloc[0,2],ds1.iloc[0,5]])
mid_exempt = np.array([ds2.iloc[0,0],ds2.iloc[0,2]])
mid_nonexempt = np.array([ds2.iloc[0,1],ds2.iloc[0,3]])
deep_exempt = np.array([ds3.iloc[0,0],ds3.iloc[0,2]])
deep_nonexempt = np.array([ds3.iloc[0,1],ds3.iloc[0,3]])
big_categories = ['Regulated', 'Unregulated']
depth_colors = ['lightsteelblue','cornflowerblue','darkblue']

with sns.axes_style("white"):
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # plot details
    bar_width = 0.3
    epsilon = .0
    line_width = 1
    opacity = 0.7
    left_bar_positions = np.arange(len(shallow_exempt))
    middle_bar_positions = left_bar_positions + bar_width
    right_bar_positions = middle_bar_positions + bar_width

    # make bar plots
    plt.figure(figsize=(10, 8))
    shallow_Exempt_Bar = plt.bar(left_bar_positions, shallow_exempt, bar_width-epsilon,
                              color=depth_colors[0],
                              edgecolor='000000',
                              linewidth=line_width,
                              hatch='//',
                              label='Shallow: Small Wells')
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width,
                              bottom=shallow_exempt,
                              linewidth=line_width,
                              edgecolor='000000',
                            #   alpha=opacity,
                              color=depth_colors[0],
                              label='     "      : Large Wells')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width-epsilon,
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              linewidth=line_width,
                              label='Midrange: Small Wells')
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width,
                              bottom=mid_exempt, # On top of first category
                              edgecolor='000000',
                              linewidth=line_width,
                              color=depth_colors[1],
                              label='      "       : Large Wells')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width-epsilon,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//',
                              label='Deep: Small Wells')
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width,
                              bottom=deep_exempt,
                              linewidth=line_width,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              label='   "    : Large Wells')

    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Well Densities (well/km^2)')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    # plt.legend(bbox_to_anchor=(1.75, 1.05))  
    sns.despine()  
    plt.savefig(outputpath_local+'WellDensities_Regulation_BarGraph', bbox_inches='tight')
 

# %% Now for plotting with exempt/non-exempt on the bar graph by water category
# Summing the data

# Shallow
ds = wdc3_wc_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds1 = ds.transpose()

# Midrange
ds = wdc2_wc_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds2 = ds.transpose()

# Deep
ds = wdc1_wc_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds3 = ds.transpose()

shallow_exempt = np.array([ds1.iloc[0,0],ds1.iloc[0,9],ds1.iloc[0,12],ds1.iloc[0,6],ds1.iloc[0,3]])
shallow_nonexempt = np.array([ds1.iloc[0,2],ds1.iloc[0,11],ds1.iloc[0,14],ds1.iloc[0,8],ds1.iloc[0,5]])
mid_exempt = np.array([ds2.iloc[0,0],ds2.iloc[0,9],ds2.iloc[0,12],ds2.iloc[0,6],ds2.iloc[0,3]])
mid_nonexempt = np.array([ds2.iloc[0,2],ds2.iloc[0,11],ds2.iloc[0,14],ds2.iloc[0,8],ds2.iloc[0,5]])
deep_exempt = np.array([ds3.iloc[0,0],ds3.iloc[0,9],ds3.iloc[0,12],ds3.iloc[0,6],ds3.iloc[0,3]])
deep_nonexempt = np.array([ds3.iloc[0,2],ds3.iloc[0,11],ds3.iloc[0,14],ds3.iloc[0,8],ds3.iloc[0,5]])
big_categories = ['Receives\nCAP\n(Regulated)'
                  , 'GW\nDominated\n(Regulated)'
                  , ' Surface\nWater\nDominated'
                  , 'Mixed\nSource','GW\nDominated']
depth_colors = ['lightsteelblue','cornflowerblue','darkblue']

with sns.axes_style("white"):
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # plot details
    bar_width = 0.3
    epsilon = .0
    line_width = 1
    opacity = 0.7
    left_bar_positions = np.arange(len(shallow_exempt))
    middle_bar_positions = left_bar_positions + bar_width
    right_bar_positions = middle_bar_positions + bar_width
    plt.figure(figsize=(10, 8))
    # make bar plots
    shallow_Exempt_Bar = plt.bar(left_bar_positions, shallow_exempt, bar_width-epsilon,
                              color=depth_colors[0],
                              edgecolor='000000',
                              linewidth=line_width,
                              hatch='//',
                              label='Shallow: Small Wells')
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width,
                              bottom=shallow_exempt,
                              linewidth=line_width,
                              edgecolor='000000',
                            #   alpha=opacity,
                              color=depth_colors[0],
                              label='     "      : Large Wells')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width-epsilon,
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              linewidth=line_width,
                              label='Midrange: Small Wells')
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width,
                              bottom=mid_exempt, # On top of first category
                              edgecolor='000000',
                              linewidth=line_width,
                              color=depth_colors[1],
                              label='      "       : Large Wells')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width-epsilon,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//',
                              label='Deep: Small Wells')
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width,
                              bottom=deep_exempt,
                              linewidth=line_width,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              label='   "    : Large Wells')

    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Number of Wells')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    plt.legend(bbox_to_anchor=(1.0, 1.05))  
    sns.despine()
    plt.savefig(outputpath_local+'NumberWells_WC_BarGraph', bbox_inches='tight')

# %% Totals
ds1 = ds1.drop('MONITOR',axis=1,level=1)
ds2 = ds2.drop('MONITOR',axis=1,level=1)
ds3 = ds3.drop('MONITOR',axis=1,level=1)

# %%
allRegUreg = ds1 + ds2 + ds3
allRegUreg

# %%
print('Regulated total = ', (allRegUreg.iloc[0,0]+allRegUreg.iloc[0,1]))
print('Unregulated total = ', (allRegUreg.iloc[0,2]+allRegUreg.iloc[0,3]))
# %% Now for plotting with exempt/non-exempt on the bar graph by water category
# Summing the data

# Shallow
ds = dens_wdc3_wc_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds1 = ds.transpose()

# Midrange
ds = dens_wdc2_wc_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds2 = ds.transpose()

# Deep
ds = dens_wdc1_wc_ex.copy()
ds = ds.drop('Res',axis=1)
ds = pd.DataFrame(ds.sum())
ds3 = ds.transpose()

shallow_exempt = np.array([ds1.iloc[0,0],ds1.iloc[0,9],ds1.iloc[0,12],ds1.iloc[0,6],ds1.iloc[0,3]])
shallow_nonexempt = np.array([ds1.iloc[0,2],ds1.iloc[0,11],ds1.iloc[0,14],ds1.iloc[0,8],ds1.iloc[0,5]])
mid_exempt = np.array([ds2.iloc[0,0],ds2.iloc[0,9],ds2.iloc[0,12],ds2.iloc[0,6],ds2.iloc[0,3]])
mid_nonexempt = np.array([ds2.iloc[0,2],ds2.iloc[0,11],ds2.iloc[0,14],ds2.iloc[0,8],ds2.iloc[0,5]])
deep_exempt = np.array([ds3.iloc[0,0],ds3.iloc[0,9],ds3.iloc[0,12],ds3.iloc[0,6],ds3.iloc[0,3]])
deep_nonexempt = np.array([ds3.iloc[0,2],ds3.iloc[0,11],ds3.iloc[0,14],ds3.iloc[0,8],ds3.iloc[0,5]])
big_categories = ['Receives\nCAP\n(Regulated)'
                  , 'GW\nDominated\n(Regulated)'
                  , ' Surface\nWater\nDominated'
                  , 'Mixed\nSource','GW\nDominated']
depth_colors = ['lightsteelblue','cornflowerblue','darkblue']

with sns.axes_style("white"):
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # plot details
    bar_width = 0.3
    epsilon = .0
    line_width = 1
    opacity = 0.7
    left_bar_positions = np.arange(len(shallow_exempt))
    middle_bar_positions = left_bar_positions + bar_width
    right_bar_positions = middle_bar_positions + bar_width
    plt.figure(figsize=(10, 8))
    # make bar plots
    shallow_Exempt_Bar = plt.bar(left_bar_positions, shallow_exempt, bar_width-epsilon,
                              color=depth_colors[0],
                              edgecolor='000000',
                              linewidth=line_width,
                              hatch='//',
                              label='Shallow: Small Wells')
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width,
                              bottom=shallow_exempt,
                              linewidth=line_width,
                              edgecolor='000000',
                            #   alpha=opacity,
                              color=depth_colors[0],
                              label='     "      : Large Wells')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width-epsilon,
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              linewidth=line_width,
                              label='Midrange: Small Wells')
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width,
                              bottom=mid_exempt, # On top of first category
                              edgecolor='000000',
                              linewidth=line_width,
                              color=depth_colors[1],
                              label='      "       : Large Wells')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width-epsilon,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//',
                              label='Deep: Small Wells')
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width,
                              bottom=deep_exempt,
                              linewidth=line_width,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              label='   "    : Large Wells')

    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Well Densities (well/km^2)')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    plt.legend(bbox_to_anchor=(1.0, 1.05))  
    sns.despine()
    plt.savefig(outputpath_local+'WellDensities_WC_BarGraph', bbox_inches='tight')

# %%  === Plotting Water Level Madness ===
 
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
# This is testing whether or not the slope is positive or negative (2-way)
#       For our purposes, time is the x variable and y is
#       1. Depth to Water
#       2. Number of Wells
#       3. Well Depths

# Actual documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
# Tutorial from https://mohammadimranhasan.com/linear-regression-of-time-series-data-with-pandas-library-in-python/

# For Depth to Water of georegions
ds = cat_wl2_georeg
min_yr = 2002
mx_yr = 2020
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression:"
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]

# -- For Multiple years --
# Name = "Linear Regression for Non-drought years: "
# wetyrs = [2005, 2008, 2009, 2010, 2016, 2017, 2019]
# dryyrs = [2002, 2003, 2004, 2006, 2007, 2011, 2012, 2013, 2014, 2015, 2018]
# #f = ds[(ds.index == wetyrs)]

# f = pd.DataFrame()
# for i in dryyrs:
#         wut = ds[(ds.index == i)]
#         f = f.append(wut)
# print(f)
# -----------------------

stats = pd.DataFrame()
for i in range(1, 12, 1):
        df = f[i]
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
#        print('Georegion Number: ', i, '\n', 
#                'slope = ', slope, '\n', 
#                'intercept = ', intercept, '\n', 
#                'r^2 = ', r_value, '\n', 
#                'p-value = ', p_value, '\n', 
#                'std error = ', std_err)
        
        # row1 = pd.DataFrame([slope], index=[i], columns=['slope'])
        # row2 = pd.DataFrame([intercept], index=[i], columns=['intercept'])
        # stats = stats.append(row1)
        # stats = stats.append(row2)
        # stats['intercept'] = intercept
        stats = stats.append({'slope': slope, 
                        #       'int':intercept, 
                              'rsq':r_value, 
                              'p_val':p_value, 
                              'std_err':std_err,
                              'mean': np.mean(y),
                              'var': np.var(y)}, 
                              ignore_index=True)
        xf = np.linspace(min(x),max(x),100)
        xf1 = xf.copy()
        #xf1 = pd.to_datetime(xf1)
        yf = (slope*xf)+intercept
        fig, ax = plt.subplots(1, 1)
        ax.plot(xf1, yf,label='Linear fit', lw=3)
        df.plot(ax=ax,marker='o', ls='')
        ax.set_ylim(0,max(y))
        ax.legend()


# stats = stats.append(slope)
#        stats[i] = stats[i].append(slope)

#   df = df.append({'A': i}, ignore_index=True)
stats1 = stats.transpose()
stats1

# %% Linear Regression
# For Depth to Water by SW Access
ds = cat_wl2_SW
data_type = "Depth to Water"
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
# -- For Multiple years --
# Name = "Linear Regression during Wet and Normal years for " + data_type
# wetyrs = [2005, 2008, 2009, 2010, 2016, 2017, 2019]
# dryyrs = [2002, 2003, 2004, 2006, 2007, 2011, 2012, 2013, 2014, 2015, 2018]
# dryyrs = [1975,1976,1977
#           ,1981,1989,1990
#           ,1996,1997,
#           1999,2000,2001,2002,2003,2004
#           ,2006,2007,2008,2009
#           ,2011, 2012, 2013, 2014, 2015,2017,2018]
# wetyrs = [1978,1979,1980,1982,1983,1984,1984,1986,1987,1988
#           , 1991,1992,1993,1994,1995,
#           1998,2005,2010,2019]

#f = ds[(ds.index == wetyrs)]

# f = pd.DataFrame()
# for i in wetyrs:
#         wut = ds[(ds.index == i)]
#         f = f.append(wut)
# print(f)
columns = ds.columns
column_list = ds.columns.tolist()
# ------------------------

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

fig, ax = plt.subplots(1, 1, figsize = (7,4.5))
# fig, ax = plt.subplots(figsize = (16,9))

ax.plot(xf1, yf1,"-.",color=cap,label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color=GWdom, lw=1)
ax.plot(xf1, yf3,"-.",color=mixed, lw=1)
ax.plot(xf1, yf4,"-.",color='#CCC339', lw=1)
ax.plot(xf1, yf5,"-.",color=swdom, lw=1)

# f.plot(ax=ax,marker='o', ls='', label=betterlabels)
# Trying to draw lines with better shit 

ds = cat_wl2_SW
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
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
# ax.set_title(name, fontsize=20)
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
plt.savefig(outputpath_local+Name, bbox_inches = 'tight')
# plt.savefig(outputpath+'Stats/Water_CAT/'+Name, bbox_inches = 'tight')
stats1.to_csv(outputpath_local+Name+'.csv')

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
plt.savefig(outputpath_local+Name+'_GW_3pieces', bbox_inches = 'tight')
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
plt.savefig(outputpath_local+Name, bbox_inches = 'tight')
# stats1.to_csv(outputpath+'Stats/'+Name+'.csv')

# %% Figure out which water level database you want
cat_wl2 = cat_wl2_reg.copy() 
# cat_wl2 = cat_wl2_SW.copy()
# cat_wl2 = cat_wl2_georeg.copy()

# cat_wl2 = wdc1_reg.copy()
# cat_wl2 = wdc2_reg.copy()
# cat_wl2 = wdc3_reg.copy()
# cat_wl2 = wdc1_SW.copy()
# cat_wl2 = wdc2_SW.copy()
# cat_wl2 = wdc3_SW.copy()

# Water Analysis period
wlanalysis_period = cat_wl2[cat_wl2.index>=1975]
# %%
state_wellcount = wdc1_reg + wdc2_reg + wdc3_reg
state_wellcount = state_wellcount[['R','U']]
state_wellcount = state_wellcount[state_wellcount.index>=1975]
# %%
test = state_wellcount.copy()
test = test.reset_index()
test['Regulation'] = test['Regulation'].astype(float)
test.set_index('Regulation', inplace=True)
test = test[(test.index>=1975)&(test.index<=2020)]
test

# %% Scatterplot of correlation values
ds = wlanalysis_period
# name = 'Comparing PDSI with Depth to Water Anomalies by Access to SW'
name = 'Comparing Number of wells with Depth to Water Levels by Regulation'
# del ds['Res']
columns = ds.columns
column_list = ds.columns.tolist()
# betterlabels = ['Receives CAP (Regulated)','GW Dominated (Regulated)','Surface Water Dominated','GW Dominated','Mixed Source'] 
betterlabels = ['Regulated','Unregulated'] 
colors=[cap, GWdom]
# colors=[cap,noCAP, swdom, mixed, GWdom]

fig, ax = plt.subplots(figsize = (7,5))
x = test['R']
y = ds['R']
ax.scatter(x,y,label=betterlabels[0],color=colors[0])
z = np.polyfit(x,y,1)
p = np.poly1d(z)
plt.plot(x,p(x),'-',color=colors[0])
x = test['U']
y = ds['U']
ax.scatter(x,y,label=betterlabels[1],color=colors[1])
z = np.polyfit(x,y,1)
p = np.poly1d(z)
plt.plot(x,p(x),'-',color=colors[1])

ax.set_xlabel('Number of Wells')
ax.set_ylabel('Depth to Water Levels (ft)')
ax.set_title(name,loc='center')
# ax.set_ylim(0,400)
fig.set_dpi(600)
plt.legend(loc = [1.05, 0.40])

# plt.savefig(outputpath+name, bbox_inches='tight') 

# %% If running a shifted correlation analysis,
#    change this to however many # years; 0 is no lag
lag = 0

print('Kendall Correlation coefficient')
for i in column_list:
        # print(' '+i+':')
        print(' '+str(i)+':')
# To normalize the data 
        # df1 = ds[i].pct_change()
        # df2 = drought_indices.PDSI.pct_change()
        df1 = ds[i]
        df2 = drought_indices.PDSI.shift(lag)
        print('  tau = ',round(df1.corr(df2, method='kendall'),3))
        print('  pval = ',round(df1.corr(df2, method=kendall_pval),4))

# %%
print('Spearman Correlation coefficient')
for i in column_list:
        print(' '+str(i)+':')
        # df1 = ds[i].pct_change()
        # df2 = drought_indices.PDSI.pct_change()
        df1 = ds[i]
        df2 = drought_indices.PDSI.shift(lag)
        print('  rho = ',round(df1.corr(df2, method='spearman'),3))
        print('  pval = ',round(df1.corr(df2, method=spearmanr_pval),4))

# %%
print('Pearson Correlation coefficient')
for i in column_list:
        print(' '+str(i)+':')
        # df1 = ds[i].pct_change()
        # df2 = drought_indices.PDSI.pct_change()
        df1 = ds[i]
        df2 = drought_indices.PDSI.shift(lag)
        r = df1.corr(df2, method='pearson')
        print('  rsq = ',round(r*r,3))
        print('  pval = ',round(df1.corr(df2, method=pearsonr_pval),4))