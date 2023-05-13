# Code for Analyzing Well and CSR GRACE Data in Arizona through Surface Water Access, Groundater Management, and Drought
The purpose of this repository is to be able to recreate databases and timeseries from my paper


## 1. First Clone This Repository
## 2. Download Data
All data for the paper can be found on cyverse here <link later>; however, the goals of this study are for others to recreate these or run new analyses with the raw data.  Below are instructions on where to download and place the data on this repository so the code runs properly
- Well Registry (also known as Wells55) and Groundwater Site Inventory can be downloaded here:
    - Move .shp and .dbf from both into Data > Shapefiles folder
    - Move Well_Registry.csv into the Data > Input_files > Wells55 folder
    - Move the Data_Tables from GWSI downloaded data into the 
- CSR GRACE Mascons (v6) used in this study can be downloaded here:
    - Move into the Data > Input_files > GRACE folder
- Georegions or custom shapefile into the Shapefiles folder
    - If you would like to run a custom analysis, you can use your own shapefile
    - If you are intending to recreate the files from this paper

## 3. Create Databases
- In the AnalysisCodes folder, run the following codes:



