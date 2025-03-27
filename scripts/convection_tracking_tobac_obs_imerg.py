"""
This script is used to track convective storms based on IR brightness temperatures from geostationary satellites and surface precipitation from the StageIV dataset using the tracking library tobac. Mesoscale convective systems are a subset of the resulting storm database. These are identified using extra criteria for a large cloud shield <= 241 K and the occurrence of a cold core and heavy surface precipitation. 

The script also calculates bulk statistics (incl. total precipitation, condensation and ice water path) for each detected storm feature. These statistics can be used to estimate storm-scale precipitation efficiencies. 

Contact: kukulies@ucar.edu

"""
import io
import contextlib 
import calendar
import sys
from scipy.interpolate import griddata 
from datetime import datetime
import numpy as np 
from pathlib import Path 
import xarray as xr 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import utils 
import tobac
import h5py
print('using tobac version ', tobac.__version__, flush = True) 

#### input parameter ####
year = str(sys.argv[1])

#### DIRECTORIES ####

gpm_path = Path( str('/glade/campaign/mmm/c3we/prein/observations/GPM_IMERG_V07/' + str(year) + '/'))

# MERGIR data
mergir = Path('/glade/campaign/mmm/c3we/prein/observations/GPM_MERGIR/data')

# output data path 
savedir = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/tracked_storms_obs/')

### TRACKING PARAMETERS ###
# 4km horizontal grid spacing, hourly data (in meter and seconds)
dxy,dt= 4000, 3600 

parameters_features={}
parameters_features['position_threshold']='weighted_diff'
parameters_features['sigma_threshold']= 0.5
parameters_features['n_min_threshold']= 10 
parameters_features['target']='minimum'
parameters_features['threshold']=[241, 230, 225, 220, 210, 200]
parameters_features['statistic'] = {"feature_min_tb": np.nanmin, 'feature_max_iwp': np.nanmax, 'feature_mean_tb': np.nanmean}
parameters_features['position_threshold'] = "center"

parameters_linking={}
parameters_linking['d_max']=10*dxy
parameters_linking['stubs']= 2  
parameters_linking['adaptive_stop']=0.2
parameters_linking['adaptive_step']=0.95
parameters_linking['method_linking']= 'predict'

parameters_segmentation = {}
parameters_segmentation['threshold']=242
parameters_segmentation['target'] = "minimum"
parameters_segmentation['statistic'] = {"object_min_tb": np.nanmin, 'object_max_tb': np.nanmax, 'object_mean_tb': np.nanmean}

parameters_merge = dict(
    distance=dxy*20,)

################################ processing monthly files ######################################
month =  str(sys.argv[2]).zfill(2)

# get monthly file for Tb (crop, regrid, and interpolate nan)
monthly_file_ir = list((mergir/ year ).glob(str('merg_'+year + month +'*_4km-pixel.nc4') )) 
monthly_file_ir.sort()
print(len(monthly_file_ir), ' files for MERGIR.', flush = True) 
ds = xr.open_mfdataset(monthly_file_ir)
print(ds.dims, flush = True )
# latitudes are flipped in IR data, so fix that 
tbb  = np.flip(ds.Tb, axis =1 )
tbb['lat'] = np.flip(tbb.lat, axis = 0)
start= datetime(int(year), int(month),1,0,0 )
# replace with right number of days for respective month!
last_day =  int(calendar.monthrange(int(year), int(month))[1]) 
end = datetime(int(year), int(month), last_day,23,30)
times_30min = pd.date_range(start,end, freq = '30min')
tbb = tbb.resample(time = 'H').mean()
times = tbb.time.values

# crop Tb data 
tbb_cropped = utils.subset_data_to_conus(tbb, 'lat', 'lon')
#tbb_cropped['lat'] = -np.flip(tbb_cropped.lat, axis = 0)
# fix meta data so data array can be converted to iris cube
attributes= {'units':'K', 'long_name':'brightness_temperature'}
tbb_cropped = tbb_cropped.assign_attrs(attributes, inplace = True)
tbb_cropped = tbb_cropped.transpose('lat', 'lon', 'time')
# interpolate nan values (crucial for tracking on native grid,
# because otherwise way less features are identified in the obs data compared to model) 
tbb_filled= np.flip(tbb_cropped.interpolate_na(dim = 'lat'), axis = 0 )
tbb_filled['lat'] = -tbb_filled.lat

# convert tracking fields to iris 
tb_iris = tbb_filled.to_iris()

# get monthly file for CCIC (crop, regrid, and set nan)
import ccic
import s3fs

### CCIC ###
s3 = s3fs.S3FileSystem(anon=True)
aws_path = Path('chalmerscloudiceclimatology/record/cpcir/')
file_list_ccic = s3.glob(str(aws_path) + '/'+ str(year) +  '/*'+ str(year)+ str(month)+'*zarr')
file_list_ccic.sort()
print('files for CCIC:', len(file_list_ccic), flush = True)

for fname in file_list_ccic:
    ds_ccic = xr.open_zarr(s3.get_mapper(fname))
    tiwp_global = ds_ccic.tiwp
    # crop data (no regridding) 
    tiwp_cropped = utils.subset_data_to_conus(tiwp_global, 'latitude', 'longitude')
    tiwp_data = tiwp_cropped.load().mean('time')
    if fname == file_list_ccic[0]:
        tiwp_xr = tiwp_data
    else:
        tiwp_xr = xr.concat([tiwp_xr, tiwp_data], dim = 'time')
        
tiwp_lats = tiwp_cropped.latitude
tiwp_lons = tiwp_cropped.longitude
tiwp = xr.DataArray(tiwp_xr.data,coords=[times, tiwp_lats.values,tiwp_lons.values],dims=['time', 'latitude', 'longitude'])
tiwp = tiwp.transpose('latitude', 'longitude', 'time')
print('processing done for CCIC data', flush = True)
print('dimensions for tracking input ', tiwp.dims, flush = True)

# get monthly data for GPM IMERG 
gpm_file_list = list(gpm_path.glob(str('3B*IMERG*'+ str(year) + str(month)+ '*')))
gpm_file_list.sort()
print(len(gpm_file_list), ' files for GPM for month', str(month), str(year),  flush = True)

# get all 30-min files together and process (from TC script) 
datasets = []
gpm_times = np.array(())

for i, fname in enumerate(gpm_file_list):
    with h5py.File(fname, 'r') as f:
        if i == 0:
            lat_coords = f['Grid/lat'][:]
            lon_coords = f['Grid/lon'][:]
        gpm_times = np.append(gpm_times, np.array(f['Grid/time'][:]))
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

regridded_data = np.stack(datasets, axis=0)
# create a new xarray DataArray with the regridded data 
regridded_xarray = xr.DataArray(regridded_data,coords=[times_30min, target_lons, target_lats],dims=['time', 'lon', 'lat'])
regridded_gpm = regridded_xarray.transpose('lat', 'lon', 'time')
# resample to hourly data, this data contains the hourly average rain rate of GPM over CONUS
# CCIC grid (same that is used for TC tracking)
precip = regridded_gpm.resample(time = '1h').mean()
precip = np.flip(precip, axis = 0 )
precip['lat'] = np.flip(precip.lat, axis = 0 ) 
print('GPM IMERG precip pre-processing done,data ready to be used in tracking', flush = True)
print('input dims of precip: ', precip.dims, flush = True)

# read in StageIV 
stageIV= Path('/glade/campaign/mmm/c3we/prein/observations/STAGE_II_and_IV/data/')
stageIV_conus = Path('/glade/campaign/mmm/c3we/prein/observations/STAGE_II_and_IV/DEM_STAGE-IV/STAGE4_A.nc')
stageIV_coords = xr.open_dataset(stage_iv_conus, decode_times = False)
monthly_file_prec = stageIV / str('LEVEL_2-4_hourly_precipitation_' + year + month +'.nc')
ds_prec = xr.open_dataset(monthly_file_prec)
stage_precip = ds_prec.Precipitation
stage_precip = precip.transpose("rlat", "rlon", "time")
stage_precip['lat'] = stage_coords.lat
stage_precip['lon'] = stage_coords.lon

# regrid to Tb/CCIC grid 

# use this data to set Tb and Precip NaN, where StageIV is NaN


####################################### Tracking ##############################################

### monthly input:  tiwp, tb_iris, precip 
print(precip.shape, tb_iris.shape, tiwp.shape, flush = True)
monthly_file = savedir / str('tobac_storm_tracks_' + year + '_' + month + '_IMERG.nc')
print('start tracking for ', year, month, str(datetime.now()), flush = True)

if monthly_file.is_file() is False:
    # feature detection based on Tb
    print(f"Commencing feature detection using Tb for ", year, month, flush=True)
    features=tobac.feature_detection_multithreshold(tb_iris ,dxy, **parameters_features)

    # linking  
    print(f"Commencing tracking", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        tracks = tobac.linking_trackpy(features, tb_iris, dt, dxy, **parameters_linking)
    # reduce tracks to valid cells and those cells that contain a cold core
    tracks = tracks[tracks.cell != -1]
    tracks_cold_core = tracks.groupby("cell").feature_min_tb.min()
    valid_cells = tracks_cold_core.index[tracks_cold_core < 225]
    tracks = tracks[np.isin(tracks.cell, valid_cells)]

    # use merging and splitting module and perform segmentation
    print(f"Calculating merges and splits", flush = True )
    merges = tobac.merge_split.merge_split_MEST(tracks, dxy, **parameters_merge)
    # add track identifiers to feature dataframe 
    tracks["track"] = merges.feature_parent_track_id.data.astype(np.int64)
    track_start_time = tracks.groupby("track").time.min()
    tracks["time_track"] = tracks.time - track_start_time[tracks.track].to_numpy()

    # Tb segmentation 
    print(f"Commencing segmentation", flush=True)
    mask, tracks = tobac.segmentation.segmentation(tracks, tb_iris, dxy, **parameters_segmentation)

    # Bulk statistics for identified cloud objects
    print('Calculating the statistics...', flush = True)
    # make sure track and mask are both in datetime format
    #times = np.array([x.to_datetime64() for x in tracks.time])
    times = np.array([np.datetime64(f'{cftime_obj.year}-{cftime_obj.month:02d}-{cftime_obj.day:02d} '
                                    f'{cftime_obj.hour:02d}:{cftime_obj.minute:02d}')
                      for cftime_obj in tracks.time])
    tracks['time'] = times.astype('datetime64[ns]')
    mask_xr = xr.DataArray.from_iris(mask)
    mask_xr["time"] = precip.time.values
    
    # check if time coordinates match
    assert (mask_xr.time.values == tiwp.time.values).all()
    assert (precip.time.values == tiwp.time.values).all()
    #print(tracks.time.values[0:10] , mask_xr.time.values[0:10], tracks.time.dtype, mask_xr.time.dtype, flush = True)
    print(tbb_filled.lat[0:10], mask_xr.lat[0:10], precip.lat[0:10], flush = True)
    tracks = utils.get_statistics_obs(tracks, mask_xr, precip, np.flip(tiwp, axis = 0), inplace = True)
    lonname = "longitude"
    latname= "latitude"

    # MCS classification
    tracks, clusters = utils.process_clusters(tracks, lonname, latname)
    mcs_flag = utils.is_track_mcs_cluster(clusters)

    # A little check for the MCS flag result 
    print(mcs_flag[mcs_flag == True].shape[0], 'identified storms are MCSs', flush = True)
    assert np.unique(tracks.track.values).size == mcs_flag.shape[0]

    # OUTPUT DATA FRAME
    # remove redundant columns 
    redundant = ['idx', 'num', 'timestr', 'time_cell']
    tracks.drop(redundant, axis = 1, inplace= True)
    # checks 
    assert merges.track_child_cell_count.shape == mcs_flag.shape
    assert merges.cell_parent_track_id.shape == merges.cell_child_feature_count.shape

    # Add MCS flag (per track) 
    df = mcs_flag.rename('mcs_flag').to_frame()
    tracks = tracks.merge(df, on='track', how='left') 
    # Add how many cells belong to each track (per track) 
    tracks = tracks.merge(merges.track_child_cell_count.to_dataframe(), on='track', how='left')   
    # Add ID of parent track (per cell) 
    tracks = tracks.merge( merges.cell_parent_track_id.to_dataframe(), on='cell', how='left') 
    # Add how many features belong to each cell (per cell) 
    tracks = tracks.merge( merges.cell_child_feature_count.to_dataframe(), on='cell', how='left')
    del mask_xr.attrs['inplace']
    print(mask_xr.dtype, mask_xr, flush = True)

    # Save output data (mask and track files)
    tracks.to_xarray().to_netcdf(savedir / str('tobac_storm_tracks_' + year + '_' + month + '_IMERG.nc'))    
    mask_xr.to_netcdf(savedir / str('tobac_storm_mask_' + year + '_' + month + '_IMERG.nc'))
    print('files saved', str(datetime.now()), flush = True)

else:
    print(str(monthly_file), ' does already exist.', flush = True)
   



    
