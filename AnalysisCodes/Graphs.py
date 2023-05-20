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

# Assign Data paths
datapath = '../Data'
outputpath = '../Data/Output_files/'
shapepath = '../Data/Shapefiles/'

# %% Read in the data
# Set shallow and drilling depths
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

# %%
# Plot all of the different depths 3 in a line
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
fig, ax = plt.subplots(1,3,figsize=(20,5))
#fig.tight_layout()
fig.suptitle(name, fontsize=18, y=1.00)
fig.supylabel(ylabel, fontsize = 14, x=0.08)
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

ax[0].set_title('Shallow (<200 ft)', loc='center')
ax[1].set_title('Midrange (200-500ft)', loc='center')
ax[2].set_title('Deep (> 500ft)', loc='center')


ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

fig.set_dpi(600.0)

# plt.savefig(outputpath+name+'_3horizontalpanel_regulated', bbox_inches='tight')

# %%
# Plot all of the different depths 3 in a line
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
fig.suptitle(name, fontsize=18, y=1.00)
fig.supylabel(ylabel, fontsize = 14, x=0.08)

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

ax[0].set_title('Shallow (<200 ft)', loc='center')
ax[1].set_title('Midrange (200-500ft)', loc='center')
ax[2].set_title('Deep (> 500ft)', loc='center')


ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

fig.set_dpi(600.0)

# plt.savefig(outputpath+name+'_watercat_3horizontalpanel', bbox_inches='tight')

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


# plt.savefig(outputpath+name+'groupedchart', dpi=600)


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

# plt.savefig(outputpath+name+'groupedchart_version2',dpi=600)
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

shallow_exempt = np.array([ds1.iloc[0,0],ds1.iloc[0,2]])
shallow_nonexempt = np.array([ds1.iloc[0,1],ds1.iloc[0,3]])
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
    bar_width = 0.25
    epsilon = .015
    line_width = 1
    opacity = 0.7
    left_bar_positions = np.arange(len(shallow_exempt))
    middle_bar_positions = left_bar_positions + bar_width
    right_bar_positions = middle_bar_positions + bar_width

    # make bar plots
    shallow_Exempt_Bar = plt.bar(left_bar_positions, shallow_exempt, bar_width,
                              color=depth_colors[0],
                              label='Shallow Exempt')
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width-epsilon,
                              bottom=shallow_exempt,
                            #   alpha=opacity,
                              color=depth_colors[0],
                              edgecolor='000000',
                              linewidth=line_width,
                              hatch='//',
                              label='Shallow Non-exempt')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width,
                              color=depth_colors[1],
                              label='Mid exempt')
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width-epsilon,
                              bottom=mid_exempt, # On top of first category
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              linewidth=line_width,
                              label='Mid Non-exempt')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width,
                              color=depth_colors[2],
                              label='Deep Exempt')
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width-epsilon,
                              bottom=deep_exempt,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//',
                              label='Deep Non-exempt')

    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Number of Wells')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    plt.legend(bbox_to_anchor=(1.1, 1.05))  
    sns.despine()  
    plt.show()  

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
    bar_width = 0.25
    epsilon = .015
    line_width = 1
    opacity = 0.7
    left_bar_positions = np.arange(len(shallow_exempt))
    middle_bar_positions = left_bar_positions + bar_width
    right_bar_positions = middle_bar_positions + bar_width

    # make bar plots
    shallow_Exempt_Bar = plt.bar(left_bar_positions, shallow_exempt, bar_width,
                              color=depth_colors[0],
                              label='Shallow Exempt')
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width-epsilon,
                              bottom=shallow_exempt,
                            #   alpha=opacity,
                              color=depth_colors[0],
                              edgecolor='000000',
                              linewidth=line_width,
                              hatch='//',
                              label='Shallow Non-exempt')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width-epsilon,
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              linewidth=line_width,
                              label='Mid exempt')
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width,
                              bottom=mid_exempt, # On top of first category
                              color=depth_colors[1],
                              label='Mid Non-exempt')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width,
                              color=depth_colors[2],
                              label='Deep Exempt')
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width-epsilon,
                              bottom=deep_exempt,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//',
                              label='Deep Non-exempt')

    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Well Densities (well/km^2)')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    plt.legend(bbox_to_anchor=(1.1, 1.05))  
    sns.despine()  
    plt.show() 

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
    bar_width = 0.2
    epsilon = .05
    line_width = 1
    opacity = 0.7
    left_bar_positions = np.arange(len(shallow_exempt))
    middle_bar_positions = left_bar_positions + bar_width
    right_bar_positions = middle_bar_positions + bar_width

    # make bar plots
    shallow_Exempt_Bar = plt.bar(left_bar_positions, shallow_exempt, bar_width,
                              color=depth_colors[0],
                              edgecolor='000000',
                              linewidth=line_width,
                              hatch='//',
                              label='Shallow Exempt')
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width,
                              bottom=shallow_exempt,
                            #   alpha=opacity,
                              color=depth_colors[0],
                              label='Shallow Non-exempt')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width-epsilon,
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              label='Mid exempt')
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width,
                              bottom=mid_exempt, # On top of first category
                              color=depth_colors[1],
                              edgecolor='#000000',
                              linewidth=line_width,
                              label='Mid Non-exempt')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width-epsilon,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//',
                              label='Deep Exempt')
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width,
                              bottom=deep_exempt,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              label='Deep Non-exempt')

    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Number of Wells')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    plt.legend(bbox_to_anchor=(1.1, 1.05))  
    sns.despine()  
    plt.show()  
# %% From the internet
# https://gist.github.com/ctokheim/6435202a1a880cfecd71
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# make up some fake data
pos_mut_pcts = np.array([20, 10, 5, 7.5, 30, 50])
pos_cna_pcts = np.array([10, 0, 0, 7.5, 10, 0])
pos_both_pcts = np.array([10, 0, 0, 0, 0, 0])
neg_mut_pcts = np.array([10, 30, 5, 0, 10, 25])
neg_cna_pcts = np.array([5, 0, 7.5, 0, 0, 10])
neg_both_pcts = np.array([0, 0, 0, 0, 0, 10])
genes = ['PIK3CA', 'PTEN', 'CDKN2A', 'FBXW7', 'KRAS', 'TP53']

with sns.axes_style("white"):
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # plot details
    bar_width = 0.35
    epsilon = .015
    line_width = 1
    opacity = 0.7
    pos_bar_positions = np.arange(len(pos_mut_pcts))
    neg_bar_positions = pos_bar_positions + bar_width

    # make bar plots
    hpv_pos_mut_bar = plt.bar(pos_bar_positions, pos_mut_pcts, bar_width,
                              color='#ED0020',
                              label='HPV+ Mutations')
    hpv_pos_cna_bar = plt.bar(pos_bar_positions, pos_cna_pcts, bar_width-epsilon,
                              bottom=pos_mut_pcts,
                              alpha=opacity,
                              color='white',
                              edgecolor='#ED0020',
                              linewidth=line_width,
                              hatch='//',
                              label='HPV+ CNA')
    hpv_pos_both_bar = plt.bar(pos_bar_positions, pos_both_pcts, bar_width-epsilon,
                               bottom=pos_cna_pcts+pos_mut_pcts,
                               alpha=opacity,
                               color='white',
                               edgecolor='#ED0020',
                               linewidth=line_width,
                               hatch='0',
                               label='HPV+ Both')
    hpv_neg_mut_bar = plt.bar(neg_bar_positions, neg_mut_pcts, bar_width,
                              color='#0000DD',
                              label='HPV- Mutations')
    hpv_neg_cna_bar = plt.bar(neg_bar_positions, neg_cna_pcts, bar_width-epsilon,
                              bottom=neg_mut_pcts,
                              color="white",
                              hatch='//',
                              edgecolor='#0000DD',
                              ecolor="#0000DD",
                              linewidth=line_width,
                              label='HPV- CNA')
    hpv_neg_both_bar = plt.bar(neg_bar_positions, neg_both_pcts, bar_width-epsilon,
                               bottom=neg_cna_pcts+neg_mut_pcts,
                               color="white",
                               hatch='0',
                               edgecolor='#0000DD',
                               ecolor="#0000DD",
                               linewidth=line_width,
                               label='HPV- Both')
    plt.xticks(neg_bar_positions, genes, rotation=0)
    plt.ylabel('Percentage of Samples')
    plt.legend(bbox_to_anchor=(1.1, 1.05))  
    sns.despine()  
    plt.show()  
# %%
