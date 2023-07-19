# Evaluation of Spatial and Temporal Patterns of Well Distributions and Groundwater: A Historical and Statistical Analysis of Arizona

*by Danielle E. Tadych, Matthew Ford, Laura E. Condon, Bonnie G. Colby*

<br>
The purpose of this repository is to be able to recreate databases and timeseries from our paper (under review).  All data for the paper can be found on cyverse here <link later>; however, the goals of this study are for others to recreate these or run new analyses with the raw data.  Below are instructions on where to download and place the data on this repository so the code runs properly.
</br>
<br></br>

## Choose your adventure!

### ** To only reproduce the graphs from our paper (reccomended): **
### 1. First Clone This Repository
### 2. Create Graphs
- In the Analysis Codes folder, run "Graphs.py".
  - This code automatically links to our data on Cyverse

### To create your own graphs of the databases using our methodology: 
### 1. First Clone This Repository
### 2. Download Data
- Well Registry (also known as Wells55) can be downloaded <a href="https://gisdata2016-11-18t150447874z-azwater.opendata.arcgis.com/datasets/azwater::well-registry/explore?location=34.114115%2C-111.970052%2C8.10">here</a>: 
    - Move .shp and .dbf from both into Data/Shapefiles folder
    - Move Well_Registry.csv into the Data/Input_files/Wells55 folder
- Groundwater Site Inventory (GWSI) can be downloaded <a href="https://gisdata2016-11-18t150447874z-azwater.opendata.arcgis.com/maps/gwsi-app/about">here</a>:
    - Move .shp and .dbf from both into Data/Shapefiles folder
    - Move the Data_Tables from GWSI downloaded data into the Data/Input_files/GWSI folder
- CSR GRACE Mascons (v6) used in this study can be downloaded <a href="https://www2.csr.utexas.edu/grace/RL06_mascons.html">here</a>:
    - Move into the Data/Input_files/GRACE folder
- Georegions or custom shapefile into the Data/Shapefiles folder
    - If you would like to run a custom analysis, you can use your own shapefile
    - If you are intending to recreate the files from this paper, download our Georegions.shp and Georegions.dbf files on Cyverse

### 3. Create Combined Databases
- In the AnalysisCodes/ folder, run the following codes:
    1. Well_Timeseries_Merge.py
        - Creates a database of timeseries data from GWSI
    2. Wells55_GWSI_Static_Merge.py
        - Creates combined databases of static information from both GWSI and Wells55 
### 4. Analyze Data
- In the AnalysisCodes/ folder, run the following codes:
    1. Well_Count_Analysis.py
        - Creates .csv files for new well installations per year and for well densities (# Well/km^2) in each georegion
    2. Spatial_Analysis_Wells_FinalGeoreg.py
        - Creates depth to water databases based on the categories in the shapefiles
    3. Spatial_Analysis_GRACE.py
        - Re-maps GRACE Satellite .nc files to shapefiles and exports both csv and shapefiles
### 5. Visualize Data
- In the AnalysisCodes/ folder, run Graphs.py

You did it!

