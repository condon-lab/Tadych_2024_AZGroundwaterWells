# ----- All Paper Graphs except graphics from QGIS -----
# All maps (Figure 1, 2, and 7a) were created in QGIS using the data from this repository.
# For help recreating them, feel free to contact the author.
# Otherwise, all other graphs were created using the code below.

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
import scipy.stats as sp
from scipy.stats import kendalltau, pearsonr, spearmanr
import pymannkendall as mk
import Custom_functions

# === Assign Data paths ===

# This is for accessing our data on Cyverse
# 
datapath_web = 'https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/Tadych_AzGroundwaterSpatialAnalysis_Aug2023/Data/'
outputpath_web = 'https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/Tadych_AzGroundwaterSpatialAnalysis_Aug2023/Data/Output_files/'
shapepath_web = 'https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/Tadych_AzGroundwaterSpatialAnalysis_Aug2023/Data/Shapefiles/'

# This is if you created your own database
datapath_local = '../Data'
outputpath_local = '../Data/Output_files/'
shapepath_local = '../Data/Shapefiles/'
figurepath = '../Data/Figures/'

# Change this based on whether you're running off local or web
# Cyverse:
# outputpath = outputpath_web
# shapepath = shapepath_web

# Local: 
outputpath = outputpath_local
shapepath = shapepath_local
# %% Read in the data
# Shallow and Deep and drilling depth cutoffs
shallow = 200
deep = 500

# Importing the Depth categories for Well Counts
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
# filepath = outputpath+'/Waterlevels_Regulation.csv'
filepath = outputpath+'/Waterlevels_Regulation_updated.csv'
# filepath = '../Data/Output_files/Waterlevels_Regulation_comboID.csv'
cat_wl2_reg = pd.read_csv(filepath, index_col=0)
cat_wl2_reg.head()

# For Access to SW
# filepath = outputpath+'/Waterlevels_AccesstoSW.csv'
filepath = outputpath+'/Waterlevels_AccesstoSW_updated.csv'
# filepath = '../Data/Output_files/Waterlevels_AccesstoSW_comboID.csv'
cat_wl2_SW = pd.read_csv(filepath, index_col=0)
cat_wl2_SW.head()

# For georegion number
# filepath = outputpath+'Waterlevels_georegions.csv'
filepath = outputpath+'Waterlevels_georegions_updated.csv'
# filepath = '../Data/Output_files/Waterlevels_georegions.csv'
cat_wl2_georeg = pd.read_csv(filepath, index_col=0)
# cat_wl2_georeg.head()

# %% Importing GRACE analyses
filepath = outputpath+'grace_stateavg_yearly.csv'
# filepath = outputpath_local+'gracse_remapped_yearly.csv'
grace_yearly = pd.read_csv(filepath, index_col=0)
grace_yearly = grace_yearly[:-1]

# Reading in the shapefile - note, figure 2 is created through QGIS
filename_georeg = 'georeg_reproject_fixed.shp'
filepath = os.path.join(shapepath_web, filename_georeg)

#%% Importing Population Data
filepath = datapath_local+'/Input_files/Arizona_Population.csv'
# filepath = outputpath_local+'gracse_remapped_yearly.csv'
population = pd.read_csv(filepath, index_col=0, header=0)
population = population[:-7]
population
# %% Creating colors
# Matching map
cap = '#C6652B'
# noCAP = '#EDE461' # This is one from the map but it's too bright and hard to see
noCAP = '#CCC339' # This color but darker for lines
GWdom = '#3B76AF'
mixed = '#6EB2E4'
swdom = '#469B76'
specialyears = 'darkgray'

# %% Number of New wells for Figure 3
allnewwells = wdc1_reg+wdc2_reg+wdc3_reg
exempt_newwells = wdc1_reg_ex+wdc2_reg_ex+wdc3_reg_ex
del allnewwells['Res']
del exempt_newwells['Res']
allnewwells = allnewwells.sum(axis=1)

ds = exempt_newwells['R',     'EXEMPT'] + exempt_newwells['U',     'EXEMPT']

test = allnewwells.copy()
test = test.reset_index()
test['Regulation'] = test['Regulation'].astype(float)
test['Regulation'] = test['Regulation'].astype(int)
test.set_index('Regulation', inplace=True)
allnewwells = test

test = ds.copy()
test = test.reset_index()
test['In_year'] = test['In_year'].astype(float)
test['In_year'] = test['In_year'].astype(int)
test.set_index('In_year', inplace=True)
exempt_newwells = test

test = population.copy()
test = test.reset_index()
test['Year'] = test['Year'].astype(float)
test['Year'] = test['Year'].astype(int)
test.set_index('Year', inplace=True)
test.info()
population = test

ds = allnewwells
ds2 = exempt_newwells
min_yr = 1975
mx_yr = 2022

fig, ax = plt.subplots(1, 1, figsize = (9,5))

min_y = 0
max_y = 4000
fsize = 12

#Putting Population on a secondary axis
ax2 = ax.twinx()
ax2.bar(population.index, population['Percent_Change'], 
        alpha=0.5, 
        label='Arizona Population', color='#989FA6')
ax2.set_ylim([0, 8])
ax2.set_ylabel('Percent Change (%)',fontsize=fsize)
ax2.legend(loc='upper right')

# New Well Lines
ax.plot(ds, label='All New Wells', color='black',lw=2) 
ax.plot(ds2, '--',label='Exempt Wells', color='black') 

ax.set_xlim(min_yr,mx_yr)
ax.set_ylim(min_y,max_y)
ax.grid(visible=True,which='major',color='#D5D5D5')
ax.set_ylabel('Number of New Wells',fontsize=fsize)
fig.set_dpi(600.0)
ax.legend(loc = "upper left")

# Remove frame and x-grid lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.xaxis.grid(False)

# Remove major tick marks
ax.tick_params(axis='both', which='major', length=0)
ax2.tick_params(axis='both', which='major', length=0)

plt.savefig(figurepath+'Figure3', bbox_inches = 'tight')

# %% Grouped Bar Chart for Figure 4a
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
                              hatch='//'
                              # label='Shallow: Small Wells'
                              )
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width,
                              bottom=shallow_exempt,
                              linewidth=line_width,
                              edgecolor='000000',
                            #   alpha=opacity,
                              color=depth_colors[0],
                              label='Shallow')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width-epsilon,
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              linewidth=line_width
                              # label='Midrange: Small Wells'
                              )
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width,
                              bottom=mid_exempt, # On top of first category
                              edgecolor='000000',
                              linewidth=line_width,
                              color=depth_colors[1],
                              label='Midrange')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width-epsilon,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//'
                              # label='Deep: Small Wells'
                              )
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width,
                              bottom=deep_exempt,
                              linewidth=line_width,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              label='Deep')
    
    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Number of Wells')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    plt.title('a)', pad = 20, loc='left',fontsize=22)
    # plt.legend(bbox_to_anchor=(1.1, 1.05))  
    sns.despine()  
    plt.savefig(figurepath+'Figure4a', bbox_inches='tight')

# %% Figure 4b
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
                              hatch='//')
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width,
                              bottom=shallow_exempt,
                              linewidth=line_width,
                              edgecolor='000000',
                            #   alpha=opacity,
                              color=depth_colors[0],
                              label='Shallow')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width-epsilon,
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              linewidth=line_width)
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width,
                              bottom=mid_exempt, # On top of first category
                              edgecolor='000000',
                              linewidth=line_width,
                              color=depth_colors[1],
                              label='Midrange')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width-epsilon,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//')
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width,
                              bottom=deep_exempt,
                              linewidth=line_width,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              label='Deep')

    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Well Densities (well/km^2)')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    plt.title('b)', pad = 20, loc='left',fontsize=22)
    # plt.legend(bbox_to_anchor=(1.75, 1.05))  
    sns.despine()  
    plt.savefig(figurepath+'Figure4b', bbox_inches='tight')
 
# %% Figure 4c
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
                              hatch='//')
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width,
                              bottom=shallow_exempt,
                              linewidth=line_width,
                              edgecolor='000000',
                            #   alpha=opacity,
                              color=depth_colors[0],
                              label='Shallow')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width-epsilon,
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              linewidth=line_width)
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width,
                              bottom=mid_exempt, # On top of first category
                              edgecolor='000000',
                              linewidth=line_width,
                              color=depth_colors[1],
                              label='Midrange')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width-epsilon,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//')
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width,
                              bottom=deep_exempt,
                              linewidth=line_width,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              label='Deep')

    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Number of Wells')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    plt.title('c)', pad = 20, loc='left',fontsize=22)
    # plt.legend(bbox_to_anchor=(1.0, 1.05))  
    sns.despine()
    plt.savefig(figurepath+'Figure4c', bbox_inches='tight')

# %% Figure 4d
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
                              hatch='//')
    shallow_Nonexempt_Bar = plt.bar(left_bar_positions, shallow_nonexempt, bar_width,
                              bottom=shallow_exempt,
                              linewidth=line_width,
                              edgecolor='000000',
                            #   alpha=opacity,
                              color=depth_colors[0],
                              label='Shallow')

    Mid_Exempt_bar = plt.bar(middle_bar_positions, mid_exempt, bar_width-epsilon,
                              color=depth_colors[1],
                              hatch='//',
                              edgecolor='#000000',
                              ecolor="#000000",
                              linewidth=line_width)
    Mid_Nonexempt_bar = plt.bar(middle_bar_positions, mid_nonexempt, bar_width,
                              bottom=mid_exempt, # On top of first category
                              edgecolor='000000',
                              linewidth=line_width,
                              color=depth_colors[1],
                              label='Midrange')
    
    Deep_Exempt_Bar = plt.bar(right_bar_positions, deep_exempt, bar_width-epsilon,
                              color=depth_colors[2],
                              edgecolor='lightsteelblue',
                              linewidth=line_width,
                              hatch='//')
    Deep_Nonexempt_Bar = plt.bar(right_bar_positions, deep_nonexempt, bar_width,
                              bottom=deep_exempt,
                              linewidth=line_width,
                            #   alpha=opacity,
                              color=depth_colors[2],
                              label='Deep')
    dummy_bar = plt.bar(0, 0, bar_width-epsilon,
                              color='white',
                              edgecolor='black',
                              linewidth=line_width,
                              hatch='//',
                              label='Small Wells')

    plt.xticks(middle_bar_positions, big_categories
               , rotation=0
               )
    plt.ylabel('Well Densities (well/km^2)')
    plt.grid(axis='y', linewidth=0.5, zorder=0)
    plt.title('d)', pad = 20, loc='left',fontsize=22)
    plt.legend(bbox_to_anchor=(1.0, 1.05))  
    sns.despine()
    plt.savefig(figurepath+'Figure4d', bbox_inches='tight')

# %% # Plot for Figure 5 a) Shallow, b) Midrange, and c) Deep wells by Regulation
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
fsize = 14

columns = ds1.columns
labels = ds1.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(1,3,figsize=(15,5))
fig.supylabel(ylabel, fontsize = fsize, x=0.07)

ax[0].plot(ds3[labels[0]],'--', label='Regulated', color='black')
ax[0].plot(ds3[labels[2]], label='Unregulated', color='black')
ax[1].plot(ds2[labels[0]],'--', label='Regulated', color='black')
ax[1].plot(ds2[labels[2]], label='Unregulated', color='black')
ax[2].plot(ds1[labels[0]],'--', label='Regulated', color='black')
ax[2].plot(ds1[labels[2]], label='Unregulated', color='black')


# Lines for 1993 and 2007
style = ':'
ax[0].axvline(1993,ls = style, color=specialyears, zorder=1)
ax[1].axvline(1993,ls = style,color=specialyears, zorder=1)
ax[2].axvline(1993,ls = style,color=specialyears, zorder=1)
ax[0].axvline(2007,ls = style,color=specialyears, zorder=1)
ax[1].axvline(2007,ls = style,color=specialyears, zorder=1)
ax[2].axvline(0,ls = style,color='white', zorder=1,label=" ")
ax[2].axvline(2007,ls = style,color=specialyears, zorder=1,label='1993 and 2007')

ax[0].set_xlim(minyear,maxyear)
ax[1].set_xlim(minyear,maxyear)
ax[2].set_xlim(minyear,maxyear)

ax[0].set_title('a)', fontsize= fsize+2, loc='left')
ax[1].set_title('b)', fontsize = fsize+2, loc='left')
ax[2].set_title('c)', fontsize = fsize+2, loc='left')

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

ax[0].tick_params(labelsize=fsize)
ax[1].tick_params(labelsize=fsize)
ax[2].tick_params(labelsize=fsize)

ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

fig.set_dpi(600.0)

plt.savefig(figurepath+'Figure5_abc', bbox_inches='tight')

# %%
# Plot for Figure 5 d) Shallow, e) Midrange, and f) Deep wells by Access to SW
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

style = ':'
# Lines for 1993 and 2007
ax[0].axvline(1993,ls = style, color=specialyears)
ax[1].axvline(1993,ls = style,color=specialyears)
ax[2].axvline(1993,ls = style,color=specialyears)
ax[0].axvline(2007,ls = style,color=specialyears)
ax[1].axvline(2007,ls = style,color=specialyears)
ax[2].axvline(2007,ls = style,color=specialyears)

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

ax[0].set_title('d)', fontsize= fsize+2, loc='left')
ax[1].set_title('e)', fontsize = fsize+2, loc='left')
ax[2].set_title('f)', fontsize = fsize+2, loc='left')

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

ax[0].tick_params(labelsize=fsize)
ax[1].tick_params(labelsize=fsize)
ax[2].tick_params(labelsize=fsize)

ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

fig.set_dpi(600.0)

plt.savefig(figurepath+'Figure5_def', bbox_inches='tight')

# %% Figure 6a
# For Depth to Water by regulation
ds = cat_wl2_reg
min_yr = 1975
mx_yr = 2022
betterlabels = ['Regulated','Unregulated'] 

# del ds['Res']

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
#xf1 = pd.to_datetime(xf1)
m1 = round(stats1.loc['slope','Regulated'], 2)
m2 = round(stats1.loc['slope','Unregulated'], 2)
yint1 = round(stats1.loc['int','Regulated'], 2)
yint2 = round(stats1.loc['int','Unregulated'], 2)
pval1 = round(stats1.loc['p_val', 'Regulated'], 4)
pval2 = round(stats1.loc['p_val', 'Unregulated'], 4)

yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2

fig, ax = plt.subplots(1, 1, figsize = (11,8))

min_y = 0
max_y = 300
fsize = 18

ax.plot(ds['R'], label='Regulated', color=cap) 
ax.plot(ds['U'], label='Unregulated', color=GWdom) 

ax.plot(xf1, yf1,"-.",color='k',label='Linear Trendline', lw=1)
ax.plot(xf1, yf1,"-.",color=cap, lw=1)
ax.plot(xf1, yf2,"-.",color=GWdom, lw=1)

ax.set_xlim(min_yr,mx_yr)
ax.set_ylim(min_y,max_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.minorticks_on()
fig.set_dpi(600.0)
ax.set_title('a)',loc='left',pad=15)
# ax.legend(loc = [1.2,0.5])
ax.legend(loc = "lower left")

#Putting Grace on a secondary axis
ax2 = ax.twinx()
ax2.plot(grace_yearly['0'], label='State Average LWE', color='k',zorder=1)
ax2.set_ylim([15, -15])
ax2.set_ylabel(u'Δ LWE (cm)',fontsize=fsize)
ax2.legend(loc='lower right')

# Combine legends for both axes
# lines, labels = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax.legend(lines + lines2, labels + labels2, loc=[1.2,0.5])

plt.savefig(figurepath+'Figure6a', bbox_inches = 'tight')

# %% Figure 6c
# For Depth to Water by SW Access
ds = cat_wl2_SW
min_yr = 1975
mx_yr = 2022
betterlabels = ['Recieves CAP (Regulated)'
                ,'GW Dominated (Regulated)'
                ,'Surface Water Dominated'
                ,'GW Dominated'
                ,'Mixed Source'] 

# del ds['Res']

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
        stats = stats.append({'slope': slope, 'int':intercept, 
                              'rsq':r_value*r_value, 'p_val':p_value, 
                              'std_err':std_err, 'mean': np.mean(y),
                              'var': np.var(y),'sum': np.sum(y)},
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

fig, ax = plt.subplots(1, 1, figsize = (11,8))

ax.plot(xf1, yf1,"-.",color=cap, lw=1)
ax.plot(xf1, yf2,"-.",color=GWdom, lw=1)
ax.plot(xf1, yf3,"-.",color=mixed, lw=1)
ax.plot(xf1, yf4,"-.",color='#CCC339', lw=1)
ax.plot(xf1, yf5,"-.",color=swdom, lw=1)

min_y = 0
max_y = 300
fsize = 18

ax.plot(ds['CAP'], label=betterlabels[0], color=cap,zorder=2)
ax.plot(ds['No_CAP'], label=betterlabels[1], color='#CCC339',zorder=2) 
ax.plot(ds['SW'], label=betterlabels[2], color=swdom,zorder=2) 
ax.plot(ds['Mix'], label=betterlabels[4], color=mixed,zorder=2)
ax.plot(ds['GW'], label=betterlabels[3], color=GWdom,zorder=2)  

ax.set_xlim([min_yr,mx_yr])
ax.set_ylim(min_y, max_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.minorticks_on()
fig.set_dpi(600.0)
# ax.set_title('c)',fontsize = fsize,loc='left',pad=15)
# ax.legend()

#Putting Grace on a secondary axis
ax2 = ax.twinx()
ax2.plot(grace_yearly['0'], label='State Average LWE', color='k',zorder=1)
ax2.set_ylim([15, -15])
ax2.set_ylabel(u'Δ LWE (cm)',fontsize=fsize)
# ax2.legend(loc='lower right')
ax.legend(loc = [1.1,0.7])

plt.savefig(figurepath+'Figure6c', bbox_inches = 'tight')

# %%
# For Figure 7b, see "IndividualSlopes.ipynb"

# For Figure 8, see "NarrowedAreas.ipynb"