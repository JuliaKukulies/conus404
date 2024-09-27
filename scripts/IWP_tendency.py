"""
Script to compute the IWP tendency for every feature in the CONUS404 dataset. 

Contact: kukulies@ucar.edu

"""
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import sys
# Specify the path to the directory containing your package
package_path = '/glade/work/kukulies/pe_conus404/conus404/scripts'
# Add the directory to the Python path
sys.path.append(package_path)
# Now you can import your package
import utils
import warnings
warnings.filterwarnings('ignore')

#####################################
# directories 
path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/tracked_storms/')
conus_path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/processed/')

######################################
# all files for input year yyyy
year = str(sys.argv[1])

mask_files = list(path.glob( str('tobac_storm_mask_'+ year +'*nc')))
track_files = list(path.glob( str('tobac_storm_tracks_'+ year +'*nc')))
track_files.sort()
mask_files.sort()
print(len(track_files), year, flush = True)


months = np.arange(8,13)
for idx, month in enumerate(months):
    mon = str(month).zfill(2)
    fname_mask = mask_files[idx]
    fname_track = track_files[idx]
    print(fname_mask, fname_track)
    example_mask = xr.open_dataset(fname_mask)
    example_track = xr.open_dataset(fname_track).to_dataframe()
    conus_ds = xr.open_dataset(conus_path / str('conus404_' + year+ mon+ '.nc' ))
    tiwp = conus_ds.tiwp
    segments = example_mask.segmentation_mask[1:,:,:]
    features = utils.get_stats_iwp_tendency(example_track, segments, tiwp)
    print(year, mon, np.nanmean(features.total_iwpten.values), flush = True)
    # save new features with IWP tendency to file
    new_features = features.to_xarray()
    outfile = path / str('features_iwp_' + year + mon + '.nc') 
    new_features.to_netcdf(outfile)



    

