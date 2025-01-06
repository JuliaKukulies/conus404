"""
Script to compute the IWP tendency for every feature in the CONUS404 dataset.
"""
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime 
import sys
import utils
feature = 'CY_z500'

#####################################
# directories 
path = Path( str( '/glade/work/kukulies/pe_conus404/moaap_features/conus404/') + str(feature) )
conus_path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/processed/')
mask_path = Path('/glade/campaign/mmm/c3we/prein/Papers/2022_CONUS404-Features/data/V1/CONUS404/'\
)

######################################

months = np.arange(1,13)
for year in np.arange(2001,2020):
    year = str(year)
    for month in months:
        print(year, str(month))
        mon = str(month).zfill(2)
        #features moaap
        fname_track = path / str('features_'+ feature+ '_' +  year + '_' + mon + '.nc')
        ds = xr.open_dataset( mask_path / str(year + '01_CONUS404_ObjectMasks__dt-1h_MOAAP-masks.nc'))
        monthly_ds= ds.sel(time=ds.time.dt.month.isin([int(month)]))
        atmospheric_feature = monthly_ds[ str( feature + '_Objects') ]
        lats = ds.lat
        lons = ds.lon
        conus_file = conus_path / str('conus404_' + year+ mon+ '.nc' )
        
        if conus_file.is_file() and fname_track.is_file():
            tracks = xr.open_dataset(fname_track).to_dataframe()
            if not tracks.empty:
                conus_ds = xr.open_dataset( conus_file)
                tiwp = conus_ds.tiwp
                timedim = conus_ds.surface_precip.shape[-1]

                # initiate empty matrix
                atmospheric_feature_4km = np.zeros((conus_ds.surface_precip.shape[0], conus_ds.surface_precip.shape[1],timedim))

                # regridding the mask file at ERA5 grid to CONUS404 grid
                for idx, tt in enumerate(conus_ds.time.values):
                    era_var = atmospheric_feature.values[idx].flatten()
                    atmospheric_feature_4km_t = utils.regrid_data(era_var, lons, lats, conus_ds)
                    atmospheric_feature_4km[:,:,idx] =  atmospheric_feature_4km_t

                # get mask in right format
                segments = conus_ds.surface_precip.copy()
                segments.values = atmospheric_feature_4km.astype(np.int64)
                print(segments.shape, segments.dims)
                # cut off first timestep because we are working with the tendencies 
                segments = segments[:,:,1:]
                tracks = utils.get_stats_iwp_tendency(tracks, segments,tiwp)
                
                print(year, mon, np.nanmean(tracks.max_iwpten.values), flush = True)
                # save new features with IWP tendency to file
                new_features = tracks.to_xarray()
                outfile = path / str('features_iwp_' +feature+ '_'+ year + mon + '.nc') 
                new_features.to_netcdf(outfile)

            else:
                print(np.unique(tracks.feature.values), ' --> no features detected', flush = True)
                continue
                
        else:
                      print(str(conus_file), conus_file.is_file(), str(fname_track), fname_track.is_file())
                      continue


