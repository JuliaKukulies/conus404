"""

This python script calculates bulk statistics for each detected hurricane feature based on a TempestExtreme tracking in the CONUS404 dataset.

Email: kukulies@ucar.edu

"""

### import libraries ### 
import utils 
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from datetime import datetime
from pathlib import Path 
import numpy as np 
import xarray as xr
from tqdm import tqdm 
import pandas as pd

def get_hurricane_centers(hurricane_mask, lats, lons): 
    """
    
    Get TC centers from TempestExtreme mask. 
    
    """
    # Ensure that the input is a binary mask (1 for hurricane, 0 for no hurricane)
    assert np.all(np.isin(hurricane_mask.values, [0, 1])), "Mask values should be binary (0 or 1)"
    
    # Create an empty list to store the center coordinates (lon, lat) for each timestep
    hurricane_centers = []

    # Iterate through each timestep
    for t in hurricane_mask.time.values:
        mask_t = hurricane_mask.sel(time = t ).data

        lat_values = lats.where(mask_t== 1)
        lon_values = lons.where(mask_t == 1)
        
        center_lat = np.nanmean(lat_values.values)
        center_lon = np.nanmean(lon_values.values)
        
        hurricane_centers.append((center_lat, center_lon))
    return hurricane_centers


### directories ###
output_dir = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/hurricane_stats/')
conus404_hist  = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/processed/')
conus404_pgw   = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/PGW/')

### get files, sorted by individual hurricanes ###
path = Path('/glade/derecho/scratch/alyssas/CONUS404/for_julia/Hist/')
file_paths = list(path.glob('*deg_filtered*nc'))
files = [str(path) for path in file_paths]
files.sort()
file_pattern = r'wrf2d_d01_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})\.nc_precip\.nc_5deg_filtered\.nc'

hurricane_tracks = []
sorted_files = sorted(files, key=lambda x: datetime.strptime(re.search(file_pattern, x).group(1), '%Y-%m-%d_%H:%M:%S')) 

current_track = []
previous_time = None
time_gap_threshold = timedelta(hours=24)

for file in sorted_files:
    match = re.search(file_pattern, file) 
    if match:
        timestamp_str = match.group(1)
        current_time = datetime.strptime(timestamp_str, '%Y-%m-%d_%H:%M:%S')
        if previous_time is None or (current_time - previous_time) <= time_gap_threshold:
            current_track.append(file)
        else:
            hurricane_tracks.append(current_track)
            current_track = [file]  
        previous_time = current_time

if current_track:
    hurricane_tracks.append(current_track)

############################################################

### Processing per hurricane ###

############################################################

print(len(files), 'hourly timesteps of all hurricane')
print(len(hurricane_tracks), ' individual hurricane in dataset') 
hurricane_tracks.sort()

for hurricane_files in tqdm(hurricane_tracks):
    hurricane_files.sort()
    hurricane_ds = xr.open_mfdataset(hurricane_files, decode_times= False)
    year = Path(hurricane_files[0]).name[10:14]
    month = Path(hurricane_files[0]).name[15:17]
    start_date = Path(hurricane_tracks[44][100]).name[10:29]
    fname = conus404_hist / str('conus404_' +year + month + '.nc')
    if fname.is_file():
        conus_data    = xr.open_dataset(fname)  
        #conus_data_pgw    = xr.open_dataset(conus404_pgw / str('conus404_' + year + month + '.nc'))
        #conus_data_pgw['time'] = conus_data.time
        lons = conus_data.lons
        lats = conus_data.lats
        base_time = pd.Timestamp('1979-10-01 00:00:00')
        simulated_times = base_time + pd.to_timedelta(hurricane_ds.time.values, unit='m')
        hurricane_ds["time"] = simulated_times
        print('processing hurricane that occurred: ', simulated_times[0], simulated_times[-1])

        # assign unique labels to hurricane feature detected in each timestep 
        tc_mask = hurricane_ds.TCmask.copy().load().astype('int64')
        unique_labels = []

        for t_idx in range(tc_mask.shape[0]): 
            if np.any(tc_mask[t_idx].values == 1): 
                label = int(t_idx) 
                tc_mask[t_idx] = label * tc_mask[t_idx].data
                unique_labels.append(label)
            else:
                unique_labels.append(0) 

        time_indices = np.arange(tc_mask.shape[0])  
        actual_times = tc_mask.time.values 

        df = pd.DataFrame({
            'time': actual_times,
            'time_index': time_indices,
            'feature': unique_labels})

        # remove timesteps where there is no TC in the conus domain yet 
        df = df[df.feature > 0 ]
        # assure that df rows match time dimension
        if df.shape[0] < hurricane_ds.time.size:
            tc_mask_binary = hurricane_ds.TCmask[1:]
        else:
            tc_mask_binary = hurricane_ds.TCmask

        # subset conus data to same extent as TC mask
        start = tc_mask.time.values[0]
        end  = tc_mask.time.values[-1] 
        start_time = conus_data.time.sel(time=start, method='nearest')
        end_time =  conus_data.time.sel(time=end, method='nearest')
        print(start_time, start, end_time, end )
        subset_conus = conus_data.sel(time=slice(start_time, end_time), drop = True)
        print(tc_mask.time.values.size, subset_conus.time.values.size)
        
        # CALCULATE STATISTICS
        stats_df = utils.get_stats_conus(df, tc_mask, subset_conus)

        # get center lons and lats add add to dataframe 
        hurricane_centers_simulated = np.array(get_hurricane_centers(tc_mask_binary, lats, lons))
        center_lon= hurricane_centers_simulated[:, 1 ]
        center_lat = hurricane_centers_simulated[:, 0 ]
        stats_df['center_lat'] = center_lat
        stats_df['center_lon'] = center_lon 
        stats_df.to_csv('hurricane_' + start_date + '_C404_hist.csv', index = False)
    else:
        print('no CONUS data for this hurricane', hurricane_files[0]) 
        continue

    


    
    
  
    


    
    











