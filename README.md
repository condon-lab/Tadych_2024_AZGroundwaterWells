# Historical Patterns of Well Drilling and Groundwater Depth in Arizona Considering Groundwater Regulation and Surface Water Access

*by Danielle E. Tadych, Matthew Ford, Bonnie G. Colby, Laura E. Condon*

<br>
The purpose of this repository is to be able to recreate databases and timeseries from our paper (under review).  All data for the paper can be found on cyverse at <a href='http://doi.org/10.25739/1ckh-tx39'>http://doi.org/10.25739/1ckh-tx39</a>; however, the goals of this study are for others to recreate these datasets or run new analyses with the raw data.  Below are instructions on where to download and place the data on this repository so the code runs properly.
</br>

<br></br>
By using this repository or data in your publications, you agree to cite:
> *Tadych, D.E., Ford, M., Colby, B.G, Condon, L.E..: Historical Patterns of Well Drilling and Groundwater Depth in Arizona Considering Groundwater Regulation and Surface Water Access [under review], 2023.*

> Danielle Tadych (2023). Az Groundwater Spatiotemporal Analysis. CyVerse Data Commons. DOI 10.25739/1ckh-tx39
<br></br>

## Prepare for your journey!
To run the code you can use the provided conda environment. To clone the repo and create the environment use:
```
git clone git@github.com:dtadych/Az-Well-GRACE-Spatiotemp-Analysis.git
cd Az-Well-GRACE-Spatiotemp-Analysis
conda env create -f environment.yml
```
Once you have the environment set up you can activate it by running:

```
conda activate azgwspatialanalysis
```
If you have trouble installing the .yml file, you can always install packages through conda-forge or pip install manually.  It is reccomended to use conda-forge when possible.
```
conda create --name azgwspatialanalysis python=3.8
conda activate azgwspatialanalysis
conda install -c conda-forge <package>
```

The packages for basic graphing include:
- numpy
- matplotlib
- pandas
- geopandas
- scipy
- datetime

For running statistical analyses, also install: 
- seaborn
- pymannkendall

To re-create the study with brand new data, install all of the above plus the following:
- xarray
- rasterio
- netCDF4
- rioxarray
- easymore
- shapely

## Choose your adventure!

### **To only reproduce the graphs from our paper (reccomended):**
### 1. Create Graphs
- In the Analysis Codes folder, run "Graphs.py".
  - This code automatically links to our data on Cyverse and outputs images into the Data/Figures folder
<br></br>

### **To re-run well statistics using our regions:**

### 1. Downlaod data
- go to our <a href='http://doi.org/10.25739/1ckh-tx39'>Cyverse data hub</a> and download the following files from "Data/Output_files" to your local "Data/Output_files":
    - Waterlevels_Regulation.csv
    - Waterlevels_AccesstoSW.csv
### 2. Run Statistics
- In the Analysis Codes folder, run "Spatial_Analysis_Wells_FinalGeoreg.py"
  - *Follow workflow 1*
  - Runs a linear regression on our formatted data

### 3. Create Graphs
- In the Analysis Codes folder, run "Graphs.py".
  - This code automatically links to our data on Cyverse and outputs images into the Data/Figures folder

### **To create your own graphs of the databases using our methodology (most time consumptive):**

### 1. Download Data
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
    - If you are intending to recreate the files from this paper, download georeg_reproject_fixed files on Cyverse in the Data/Shapefiles folder <a href='http://doi.org/10.25739/1ckh-tx39'>here</a>

### 2. Create Combined Databases
- In the AnalysisCodes/ folder, run the following codes:
    1. Well_Timeseries_Merge.py
        - Creates a database of timeseries data from GWSI
    2. Wells55_GWSI_Static_Merge.py
        - Creates combined databases of static information from both GWSI and Wells55 
### 3. Analyze Data
- In the AnalysisCodes/ folder, run the following codes:
    1. Well_Count_Analysis.py
        - Creates .csv files for new well installations per year and for well densities (# Well/km^2) in each georegion
    2. Spatial_Analysis_Wells_FinalGeoreg.py
        - *Follow Workflow 2*
        - Creates depth to water databases based on the categories in the shapefiles
        - Runs statistical analyses on the desired categories
    3. Spatial_Analysis_GRACE.py
        - Re-maps GRACE Satellite .nc files to shapefiles and exports both csv and shapefiles
### 4. Visualize Data
- In the AnalysisCodes folder, run Graphs.py
   - make sure to change the "output" paths to local

You did it!