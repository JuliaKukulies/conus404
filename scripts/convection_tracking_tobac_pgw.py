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
import logging
#logging.basicConfig(filename='PGW_tracking.log', level=logging.DEBUG)

print('using tobac version ', tobac.__version__, flush = True) 

#### input parameter ####
year = str(sys.argv[1])

#### DIRECTORIES ####

# processed CONUS404 data with monthly files with hourly 2D variables needed for the calculation 
data2d = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/PGW/')
monthly_files = list(data2d.glob( str('conus404_' + year + '*.nc')))  
print(len(monthly_files), ' files detected for year ', year, flush = True)
monthly_files.sort()

# data path
savedir = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/tracked_storms_pgw/')

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

for monthly_file in monthly_files:
    month =  str(monthly_file)[-5:-3]
    output_file = savedir / str('tobac_storm_tracks_' + year + '_' + month + '.nc')
    print(monthly_file, month, flush = True)
    if output_file.is_file() is False:
        ds = xr.open_dataset(monthly_file)
        
        # get coordinates from present climate pendant
        ds_present = xr.open_dataset('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/processed/conus404_' + year + month+ '.nc')
        coords = {'lon': (["south_north", "west_east"], ds_present.lons.values), 'lat': (["south_north", "west_east"], ds_present.lats.values)}
        ds = ds.assign_coords(coords)
        ds["time"] = ds_present.time.values

        # convert tracking fields to iris cubes 
        tb_iris = ds.tb.to_iris()
        precip_iris = ds.surface_precip.to_iris()

        try: 

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
            print('Calculating the statistics...', flush = True)
            tracks = utils.get_statistics_conus(tracks, mask, ds, inplace = True)

            lonname = 'lon'
            latname= 'lat'
            # MCS classification
            tracks, clusters = utils.process_clusters(tracks, lonname, latname)
            mcs_flag = utils.is_track_mcs_cluster(clusters)

            # A little check for the MCS flag result 
            print(mcs_flag[mcs_flag == True].shape[0], 'identified storms are MCSs', flush = True)
            assert np.unique(tracks.track.values).size == mcs_flag.shape[0]

        except Exception as e:
            logging.error("An error occurred during the tracking procedure: %s", e)


        # OUTPUT DATA FRAME
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
        # Save output data (mask and track files)
        xr.DataArray.from_iris(mask).to_netcdf(savedir / str('tobac_storm_mask_' + year + '_' + month + '.nc'))
        tracks.to_xarray().to_netcdf(savedir / str('tobac_storm_tracks_' + year + '_' + month + '.nc'))    
        print('files saved', flush = True)    
    else:
        print(str(monthly_file), ' does already exist.', flush = True)
        continue

    


    
