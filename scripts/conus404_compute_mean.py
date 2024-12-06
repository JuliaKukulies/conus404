"""
This script computes the mean of a given variable over all historical CONUS404 files and all PGW CONUS404 files.

Contact: kukulies@ucar.edu

"""
import sys
from tqdm import tqdm 
import numpy as np
import xarray as xr
from pathlib import Path

#### file paths ####
present = Path('/glade/campaign/collections/rda/data/ds559.0/' )
future = Path('/glade/campaign/ncar/USGS_Water/CONUS404_PGW/'  )
output_path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/mean_values/')

def compute_total_mean(file_paths, var, chunk_size= None):
    """
    
    This function computes the mean over all given C404 files for a given variable. 

    """
    total_count = 0
    total_sum = None 
    print('loop through all files...in total ', len(file_paths), 'files found.',  flush = True)
    for file_path in tqdm(file_paths, desc="Processing monthly files..."): 
        ds = xr.open_dataset(file_path, chunks=chunk_size)
        variable_data = ds[var].squeeze()
        # update sum per file
        if total_sum is None:
            total_sum = variable_data
        else:
            total_sum += variable_data
            total_count += 1

        ds.close()

    # compute mean over all months 
    total_mean = variable_data / total_count 
    return total_mean

######################################### main program ###################################


var = str(sys.argv[1])
# do the calculations per year
years = np.arange(1980,2000)

for year in years:
    year = str(year)
    fname = output_path / (str(var)+'_' + year + '_conus404_hist.nc')
    if not fname.is_file():
        print('processing files for year ', year, flush = True)
        # get all files
        present_files = list(present.glob(  str('wy'+year+'/??????/wrf2d*nc'    )))
        future_files = list(future.glob(  str('WY'+year +'/wrf2d_*00'         )))
        print(len(present_files), len(future_files), flush = True)

        total_mean_hist = compute_total_mean( present_files, var=var, chunk_size = {'south_north': 200, 'west_east': 200})
        total_mean_pgw = compute_total_mean( future_files, var=var, chunk_size = {'south_north': 200, 'west_east': 200})
        total_mean_hist.to_netcdf(output_path / (str(var)+'_' + year + '_conus404_hist.nc'))
        total_mean_pgw.to_netcdf(output_path / (str(var)+ '_' + year + '_conus404_pgw.nc'))
    else:
        print(str(fname), ' already exists.', flush = True)
