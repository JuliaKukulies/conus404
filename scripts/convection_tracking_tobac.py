"""
This script is used to track convective storms based on IR brightness temperatures and surface precipitation in the CONUS404 dataset using the tracking library tobac. Mesoscale convective systems are a subset of the resulting storm database. These are identified using extra criteria for a large cloud shield <= 241 K and the occurrence of a cold core and heavy surface precipitation. 

The script also calculates bulk statistics (incl. total precipitation, condensation and ice water path) for each detected storm feature. These statistics can be used to estimate storm-scale precipitation efficiencies. 

Contact: kukulies@ucar.edu

"""

import sys
import numpy as np 
from pathlib import Path 
import xarray as xr 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import utils 
import tobac
print('using tobac version ', tobac.__version__, flush = True) 

#### input parameter ####
year = str(sys.argv[1])

#### DIRECTORIES ####

# processed CONUS404 data with monthly files with hourly 2D variables needed for the calculation 
data2d = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/processed/')
monthly_files = list(data2d.glob( str('conus404_' + year + '*.nc')))  
print(len(monthly_files), ' files detected for year ', year, flush = True)
monthly_files.sort()

# data path
savedir = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/tracked_storms/')

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
################################ processing monthly files ##################################

for monthly_file in monthly_files:
    month =  str(monthly_file)[-5:-3]
    output_file = savedir / str('tobac_storm_tracks_' + year + '_' + month + '_present.nc')
    print(monthly_file, month, flush = True)
    if output_file.is_file() is False:
        ds = xr.open_dataset(monthly_file)
        # make longitudes and latitudes coordinates (instead of variables) 
        coords = {'lon': (["south_north", "west_east"], ds.lons.values), 'lat': (["south_north", "west_east"], ds.lats.values)}
        ds = ds.assign_coords(coords)
        ds = ds.drop_vars(["lats", "lons"])
        lonname = 'lon'
        latname =  'lat'
        conus_lats, conus_lons = ds.lat, ds.lon

        # read in StageIV 
        stageIV= Path('/glade/campaign/mmm/c3we/prein/observations/STAGE_II_and_IV/data/')
        stageIV_conus = Path('/glade/campaign/mmm/c3we/prein/observations/STAGE_II_and_IV/DEM_STAGE-IV/STAGE4_A.nc')
        stage_coords = xr.open_dataset(stageIV_conus, decode_times = False)
        monthly_file_prec = stageIV / str('LEVEL_2-4_hourly_precipitation_' + year + month +'.nc')
        ds_prec = xr.open_dataset(monthly_file_prec)
        stage_precip = ds_prec.Precipitation
        stage_precip = stage_precip.transpose("rlat", "rlon", "time")
        stage_precip['lat'] = stage_coords.lat
        stage_precip['lon'] = stage_coords.lon

        # regrid to CONUS grid 
        datasets = []
        stage_times = []

        print('reading in StageIV data and regridded it to IMERGIR grid', flush = True)
        for time_idx in np.arange(stage.time.values.size): 
            precip_t = stage_precip[:,:,time_idx]
            tt = stage.time.values[time_idx]
            stage_times.append(tt )

            # regrid to CONUS                                                                      
            lat_grid, lon_grid = stage_precip.lat.values, stage_precip.lon.values
            target_lat_grid, target_lon_grid = conus_lats.values, conus_lons.values

            points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
            target_points = np.vstack((target_lon_grid.flatten(), target_lat_grid.flatten())).T

            flattened_data = precip_t.values.flatten()
            interpolated = griddata(points, flattened_data, (target_lon_grid, target_lat_grid), method='nearest')
            datasets.append(interpolated)

        regridded_data = np.stack(datasets, axis=0)                                         
        regridded_xarray = xr.DataArray(
            regridded_data,
            coords={
                'time': stage_times,  # 1D time coordinate
                'lon': (('lat', 'lon'), conus_lons.values),  # 2D lon coordinate with dims
                'lat': (('lat', 'lon'), conus_lats.values),  # 2D lat coordinate with dims
            },
            dims=['time', 'lat', 'lon']  # Order matches regridded_data's shape
        )
        regridded_stage = regridded_xarray.transpose('lat', 'lon', 'time')  
            
        print('setting datasets to NaN where there is no StageIV coverage', flush = True)
        # use this data to set Tb and Precip NaN, where StageIV is NaN
        precip = ds.surface_precip.where(~np.isnan(regridded_stage.data), np.nan)
        tbb = ds.tb.where(~np.isnan(regridded_stage.data), np.nan) 
        
        # convert tracking fields to iris cubes 
        tb_iris = precip.to_iris()
        precip_iris = tbb.to_iris()
        
        ############################# Tracking #################################

        # feature detection based on Tb
        print(f"Commencing feature detection using Tb for ", year, month, flush=True)
        features=tobac.feature_detection_multithreshold(tb_iris ,dxy, **parameters_features)

        # linking  
        print(f"Commencing tracking", flush=True)
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
        mask, tracks = tobac.segmentation_2D(tracks, tb_iris, dxy, **parameters_segmentation)

        # Bulk statistics for identified cloud objects
        print('Calculating the statistics for ', tracks.feature.unique().size, ' features.', flush = True)
        # get_statistics_conus(tracks, mask, ds, inplace = True) 
        tracks = utils.get_stats_conus(tracks, mask, ds, timedim = 0)
        #print(mask.dims, flush = True)

        # MCS classification
        tracks, clusters = utils.process_clusters(tracks, lonname, latname )
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
        # Add whether the cell starts with a split (per cell) 
        #tracks =tracks.merge( merges.cell_starts_with_split.to_dataframe(), on='cell', how='left')
        # Addinfo whether cell ends with a merge (per cell)
        #tracks =tracks.merge( merges.cell_ends_with_merge.to_dataframe(), on='cell', how='left')

        # Save output data (mask and track files)
        xr.DataArray.from_iris(mask).to_netcdf(savedir / str('tobac_storm_mask_' + year + '_' + month + '_present.nc'))
        tracks.to_xarray().to_netcdf(savedir / str('tobac_storm_tracks_' + year + '_' + month + '_present.nc'))    
        print('files saved', flush = True)


    else:
        print(str(monthly_file), ' does already exist.', flush = True)
        continue

    


    
