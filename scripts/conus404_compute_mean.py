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
# get all files
present_files = list(present.glob('wy????/??????/wrf2d*nc'    ))
future_files = list(future.glob('WY????/wrf2d_*00'            ))

def compute_total_mean(file_paths, var, chunk_size= None):
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
        del variable_data

    # compute mean over all months 
    total_mean = variable_data / total_count 
    return total_mean

#### main program ####
var = str(sys.argv[1])

total_mean_hist = compute_total_mean( present_files, var=var, chunk_size = {'south_north': 200, 'west_east': 200})
total_mean_pgw = compute_total_mean( future_files, var=var, chunk_size = {'south_north': 200, 'west_east': 200})
total_mean_hist.to_netcdf(output_path / (str(var)+ '_conus404_hist.nc'))
total_mean_pgw.to_netcdf(output_path / (str(var)+ '_conus404_pgw.nc'))