"""
This python script calculates bulk statistics for each detected hurricane feature based on
the IBTrACS database of recorded hurricanes that make landfall in the US between 2000 and 2021.

Email: kukulies@ucar.edu

"""

### import libraries ###
import utils
import s3fs
import ccic 
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from datetime import datetime
from pathlib import Path 
import numpy as np 
import xarray as xr
import h5py 
from tqdm import tqdm 
import pandas as pd
from scipy.interpolate import interp1d, griddata
import warnings
warnings.filterwarnings('ignore')

### select special functions ###

def extract_hurricanes_conus(ds_filtered, us_lon_min = -120, us_lon_max = -70, us_lat_min= 25, us_lat_max = 45, threshold = 64):
    """
    Select only storms that are at least Cat1 hurricanes that originate over the Northern Atlantic and make landfall over the US. 

    The extent of the target region and threshold for maximum sustained wind speeds (kts) can be modified.
    
    """    
    landfall_storms = []
    # iterate over storm dimensions (all hurricanes after 1999)
    for storm in ds_filtered.storm.values:
        # data per storm 
        storm_data = ds_filtered.sel(storm=storm)
        
        # landfall? 
        landfall = np.any((storm_data['lon'].values >= us_lon_min) &
                          (storm_data['lon'].values <= us_lon_max) &
                          (storm_data['lat'].values >= us_lat_min) &
                          (storm_data['lat'].values <= us_lat_max))
    
        # check if the storm reached hurricane strength over selected area 
        if landfall and np.any(storm_data['wmo_wind'].values >= threshold) and storm_data['basin'].values[0] == b'NA' :  
            landfall_storms.append(storm)
            
    # filter out the storms that made landfall in the US from original subset 
    landfall_ds = ds_filtered.sel(storm=np.isin(ds_filtered['storm'].values, landfall_storms))
    return landfall_ds


# Function to extract date from the filename (the YYYYMMDD part of the path)
def extract_datetime_from_gpm_filename(file_path):
    """
    extract date (YYYYMMDDHH) from a GPM IMERG filename 

    """
    date_str = file_path.stem.split('.')[4]
    date_hour_str = date_str.split('S')[0] + date_str.split('S')[1][:2]
    return np.datetime64(datetime.strptime(date_hour_str, '%Y%m%d-%H'))


def extract_datetime_from_ccic_filename(file_path):
    """
    extract the datetime object from a CCIC filename (from AWS listing)
    """
    file_path = Path(file_path)
    date_str = file_path.stem.split('_')[2]  # get the '201710010100' part of the fname
    return np.datetime64(datetime.strptime(date_str, '%Y%m%d%H%M'))

### directories ###
output_dir = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/hurricane_stats/')
conus404_hist  = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/processed/')
conus404_pgw   = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/PGW/')
ibtracs_data = Path('/glade/work/kukulies/pe_conus404/IBTrACS.ALL.v04r01.nc')

# read in IBTrACS data and subset 
ibtracs = xr.open_dataset(ibtracs_data)
first_time_per_storm = pd.to_datetime(ibtracs['time'].isel(date_time=0)) 
years = first_time_per_storm.year
valid_storms = (years >= 2000) & (years <= 2021)
ibtracs_hurricanes = ibtracs.isel(storm=valid_storms)
landfall_ds = extract_hurricanes_conus(ibtracs_hurricanes)

print(landfall_ds.storm.size, ' hurricanes make landfall in the US', flush = True)

############################################################

### Processing per hurricane ###

############################################################

for storm in landfall_ds.storm.values:
    hurricane_data = landfall_ds.sel(storm = storm)
    observed_lons = hurricane_data['lon'].dropna(dim = 'date_time')
    observed_lats = hurricane_data['lat'].dropna(dim = 'date_time')

    # derive the time info 
    year = hurricane_data.time.dt.year.values.astype(int)[0]
    month = hurricane_data.time.dt.month.values.astype(int)[0]
    #month_end = hurricane_data.time.dropna(dim ='date_time').dt.year.values.astype(int)[-1]
    start_date = str(hurricane_data.time.values[0])
    hurricane_name = str(hurricane_data.name.values)
    
    # check if hurricane has already been processed 
    output_hurricane_file= str(output_dir) + '/hurricane_' + hurricane_name + '_'+  start_date + '_ibtracs.csv'

    if Path(output_hurricane_file).is_file():
        print(output_hurricane_file, ' has already been processed.', flush = True)
        continue
    else:
        print('processing hurricane ', hurricane_name, flush = True)
        # interpolate 3-hourly TC centers to hourly values
        start = hurricane_data.time.values[0]
        end = hurricane_data.time.dropna(dim ='date_time').values[-1]
        new_time_points =  pd.date_range(start, end,  freq = '1h').to_numpy()
        print('hurricane persisted for '+ str(new_time_points.size)+ ' hours.', flush = True)
        time_points = hurricane_data.time.dropna(dim = 'date_time' ).values
        time_indices = (time_points - time_points[0]).astype('timedelta64[s]').astype(float)
        new_time_indices = (new_time_points - new_time_points[0]).astype('timedelta64[s]').astype(float)
        # interpolate hurricane track center coordinates 
        interp_lat = interp1d(time_indices, observed_lats, kind='linear', fill_value="extrapolate")
        interp_lon = interp1d(time_indices, observed_lons, kind='linear', fill_value="extrapolate")
        hourly_latitudes = interp_lat(new_time_indices)
        hourly_longitudes = interp_lon(new_time_indices)
        
        # get CCIC and GPM data cropped over CONUS, regridded to 4km CCIC grid
        print('getting the CCIC data for CONUS extent...', flush = True)
              
        ### CCIC ###
        s3 = s3fs.S3FileSystem(anon=True)
        aws_path = Path('chalmerscloudiceclimatology/record/cpcir/')
        file_list = s3.glob(str(aws_path) + '/'+ str(year) +  '/*'+ str(year) +str(month).zfill(2)+'*zarr')
        file_list.sort()
        filtered_files_ccic = [fpath for fpath in file_list if start <= extract_datetime_from_ccic_filename(fpath) <= end]

        for fname in filtered_files_ccic:
            ds_ccic = xr.open_zarr(s3.get_mapper(fname))
            tiwp_global = ds_ccic.tiwp
            tiwp_cropped = utils.subset_data_to_conus(tiwp_global, 'latitude', 'longitude')
            tiwp_data = tiwp_cropped.load().mean('time')
            if fname == filtered_files_ccic[0]:
                tiwp = tiwp_data
            else:
                tiwp = xr.concat([tiwp, tiwp_data], dim = 'time')

        tiwp = tiwp.transpose('latitude', 'longitude', 'time')
        print('getting GPM IMERG data for CONUS extent and CCIC grid...', flush = True)

        ### GPM IMERG ###
        file_path = Path( str('/glade/campaign/mmm/c3we/prein/observations/GPM_IMERG_V07/' + str(year) + '/'))
        file_list = list(file_path.glob(str('3B*IMERG*'+ str(year) + str(month).zfill(2)+'*')))
        file_list.sort()
        # filter file list to get only the files for hurricane lifetime
        filtered_files = [fpath for fpath in file_list if start <= extract_datetime_from_gpm_filename(fpath) <= end ]

        datasets = []
        for i, fname in enumerate(filtered_files):
            with h5py.File(fname, 'r') as f:
                if i == 0:
                    lat_coords = f['Grid/lat'][:]
                    lon_coords = f['Grid/lon'][:]

                data = f['Grid/precipitation'][:]
                coords = {'lat': lat_coords,  'lon': lon_coords}  
                xarray_data = xr.DataArray(np.array(data).squeeze(), coords=coords, dims=['lon', 'lat'])
                # crop region over CONUS
                cropped_gpm = utils.crop_gpm_to_conus(xarray_data, lon_coords, lat_coords)
              
                # regrid to CCIC grid
                lat_grid, lon_grid = np.meshgrid( cropped_gpm.lat.values, cropped_gpm.lon.values,) 
                target_lons = tiwp.longitude.values
                target_lats = tiwp.latitude.values
                target_lat_grid, target_lon_grid = np.meshgrid(target_lats, target_lons)

                points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
                target_points = np.vstack((target_lon_grid.flatten(), target_lat_grid.flatten())).T

                flattened_data = cropped_gpm.values.flatten()
                interpolated = griddata(points, flattened_data, (target_lon_grid, target_lat_grid), method='nearest')
                datasets.append(interpolated)
                del data

        times = pd.date_range(start, end, freq = '30min')
        regridded_data = np.stack(datasets, axis=0)
        # create a new xarray DataArray with the regridded data
        regridded_xarray = xr.DataArray(regridded_data,coords=[times, target_lons, target_lats],dims=['time', 'lon', 'lat'])
        regridded_gpm = regridded_xarray.transpose('lat', 'lon', 'time')
        # resample to hourly data 
        regridded_gpm = stacked.resample(time = '1h').mean()

        print('create the TC mask...', flush = True)

        # create TC mask around hourly centers (based on CCIC grid as well)
        tiwp_lats = tiwp_cropped.latitude
        tiwp_lons = tiwp_cropped.longitude
        lon, lat = np.meshgrid( tiwp_lons, tiwp_lats)
        time = tiwp.time
        data = tiwp
        # quick check to see that interpolated hourly values are same shape as the time dimension of CCIC
        
        assert time.size == regridded_gpm.time.size ==  hourly_latitudes.size == hourly_longitudes.size
        # define the radius around TC in degrees 
        radius_deg = 5
        R = 6371.0
        radius_km = radius_deg * (np.pi / 180) * R

        # create an empty mask with the same shape as the xarray DataArray
        mask = xr.DataArray(np.zeros(data.shape, dtype=int), 
                    coords = data.coords,
                    dims=data.dims)
        
        # loop through each time step
        for idx, t in tqdm(enumerate(range(len(time)))):
            lat_ref  = hourly_latitudes[idx]
            lon_ref  = hourly_longitudes[idx]
            lon_grid =  lon
            lat_grid =  lat 
            lon_ref_rad = np.radians(lon_ref)
            lat_ref_rad = np.radians(lat_ref)
            lon_grid_rad = np.radians(lon_grid)
            lat_grid_rad = np.radians(lat_grid)

            dlon = lon_grid_rad - lon_ref_rad
            dlat = lat_grid_rad - lat_ref_rad
            a = np.sin(dlat / 2)**2 + np.cos(lat_ref_rad) * np.cos(lat_grid_rad) * np.sin(dlon / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = R * c 
            condition = distance <= radius_km
            mask.loc[{"time": mask.time[t]}] = condition.astype(int)


        mask.to_netcdf('hurricane_mask_'+ hurricane_name+'_'+ start_date+ '.nc')
        print('mask saved.', flush = True)
            
        # corresponding feature dataframe in which the statistics will be saved            
        feature_mask = mask.copy()
        # start with this nr for unique d=feature label assignment
        feature_counter = 1
        features = []

        for t_idx, time in tqdm(enumerate(mask.time.values)):
            mask_t = mask.isel(time=t_idx)
            feature_mask[:,:,t_idx] = mask_t.where(mask_t != 1 , feature_counter )
            if np.unique(mask_t.values).size == 2:
                features.append({"feature": feature_counter, "time": time})
                feature_counter += 1  

        df_features = pd.DataFrame(features)

        # CALCULATE STATISTICS based on df_features, feature_mask, regridded_gpm and tiwp
        # quick check that TC mask, CCIC, and GPM data all have the same shape
        assert feature_mask.shape == subset_tiwp.shape == regridded_gpm.shape

        print('calculating the statistics...', flush = True)
        observed_track = utils.get_statistics_obs(df_features, feature_mask, regridded_gpm, tiwp, timedim = 2 , inplace = True) 
        # save observed statistics to netcdf 
        observed_track.to_xarray().to_netcdf(output_hurricane_file) 


    
    
  
    


    
    











