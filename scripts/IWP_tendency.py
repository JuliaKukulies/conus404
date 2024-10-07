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
package_path = '/glade/work/kukulies/pe_conus404/conus404/scripts/'
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

months = np.arange(1,13) 
for month in months:
    mon = str(month).zfill(2)
    fname_mask = path / str('tobac_storm_mask_'  + year + '_' + mon + '.nc')
    fname_track = path / str('tobac_storm_tracks_'+ year + '_' + mon + '.nc')
    print(fname_mask, fname_track)
    mask = xr.open_dataset(fname_mask)
    tracks = xr.open_dataset(fname_track).to_dataframe()
    conus_ds = xr.open_dataset(conus_path / str('conus404_' + year+ mon+ '.nc' ))
    tiwp = conus_ds.tiwp
    segments = mask.segmentation_mask[1:,:,:]
    tracks = utils.get_stats_iwp_tendency(tracks, segments, tiwp)
    print(year, mon, np.nanmean(tracks.max_iwpten.values), flush = True)
    # save new features with IWP tendency to file
    new_features = tracks.to_xarray()
    outfile = path / str('features_iwp_' + year + mon + '.nc') 
    new_features.to_netcdf(outfile)



    

