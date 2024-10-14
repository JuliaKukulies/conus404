"""

Calculating the MCS frequency and contribution to total precip for future tracks.


"""
from pathlib import Path 
import xarray as xr
import numpy as np
conus_path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/PGW/')
future_path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/tracked_storms_pgw/')
mask_pgw_files = list(future_path.glob('tobac_storm_mask_????_??_pgw.nc'))
track_pgw_files = list(future_path.glob('tobac_storm_tracks_????_??_pgw.nc'))
mask_pgw_files.sort()
track_pgw_files.sort()
assert len(mask_pgw_files) == len(track_pgw_files)
print(len(track_pgw_files), ' files detected.')

# initialize
data_array =  xr.open_dataset(mask_pgw_files[0]).segmentation_mask[0]
data_array2 =  xr.open_dataset(mask_pgw_files[0]).segmentation_mask[0]
total_precip = np.zeros(data_array.data.shape )
mcs_precip = np.zeros(data_array.data.shape )
mcs_frequency   = np.zeros( data_array.data.shape )
total_timesteps = 0

for idx, fname in enumerate(mask_pgw_files):
    year = str(fname.name)[-14:-10]
    month = str(fname.name)[-9:-7]
    conus_file = conus_path / str('conus404_' + year + '_' + month + '.nc')
    print(str(conus_file), flush = True)
    
    if conus_file.is_file():
        print('processing...', fname, flush = True)
        print('file', idx + 1, ' of', len(mask_pgw_files), ' files.', flush = True)
        segmentation_mask = xr.open_dataset(fname).segmentation_mask

        ##### select only features that also belong to an MCS track #####
        df = xr.open_dataset(track_pgw_files[idx]).to_dataframe()
        feature_labels = df[df.mcs_flag == True].feature.unique()
        mask = segmentation_mask.isin(feature_labels)
        segmentation_mask = segmentation_mask.where(mask, np.nan)

        # convert feature labels into 1 
        mask = segmentation_mask > 0 
        # add over time dimension 
        count_array = mask.sum(dim='time')
        mcs_frequency += count_array
        total_timesteps += segmentation_mask.time.size

        # add corresponding precip file
        precip = xr.open_dataset(conus_file).surface_precip
        precip_sum = precip.sum('time').data
        # get total sum of precip and the precip associated with MCSs 
        total_precip += precip_sum 
        mcs_precip += precip.where(segmentation_mask > 0 ).sum('time').data
        # close datasets and vars 
        segmentation_mask.close() 
        precip.close() 
        del precip_sum
        
    else:
        continue
    
# save to netcdf 
data_array.data = mcs_frequency / total_timesteps 
data_array.to_netcdf(future_path / 'mcs_frequency_pgw.nc')
data_array2.data = mcs_precip / total_precip 
data_array2.to_netcdf(future_path / 'mcs_contribution_precip_pgw.nc')
