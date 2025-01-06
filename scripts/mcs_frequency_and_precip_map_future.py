"""

Calculating the MCS frequency and contribution to total precip for present MCS tracks.


"""
from pathlib import Path 
import xarray as xr
import numpy as np

conus_path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/PGW/')
future_path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/tracked_storms_pgw/')
mask_files = list(future_path.glob('tobac_storm_mask_????_??_pgw.nc'))
track_files = list(future_path.glob('tobac_storm_tracks_????_??_pgw.nc'))
mask_files.sort()
track_files.sort()

# initialize
data_array = xr.open_dataset(mask_files[0]).segmentation_mask[0]
total_precip = np.zeros(data_array.data.shape )
mcs_precip = np.zeros(data_array.data.shape )
mcs_frequency   = np.zeros( data_array.data.shape )
total_timesteps = 0

for idx, fname in enumerate(mask_files):
    year = str(fname.name)[-14:-10]
    month = str(fname.name)[-9:-7]
    conus_file = conus_path / str('conus404_' + year + month + '.nc')
    
    if conus_file.is_file():
        print('processing...', fname, flush = True)
        print('file', idx + 1, ' of', len(mask_files), ' files.', flush = True)
        segmentation_mask = xr.open_dataset(fname).segmentation_mask

        ##### select only features that also belong to an MCS track #####
        df = xr.open_dataset(track_files[idx]).to_dataframe()
        feature_labels = df[df.mcs_flag == True].feature.unique()
        mask = segmentation_mask.isin(feature_labels)
        count_array = mask.sum(dim='time').data
        mcs_frequency += count_array
        total_timesteps += segmentation_mask.time.size

        # add corresponding precip file
        precip = xr.open_dataset(conus_file).surface_precip
        precip = precip.transpose(*segmentation_mask.dims) 
        precip_sum = precip.sum('time').data
        # get total sum of precip and the precip associated with MCSs 
        total_precip += precip_sum 
        mcs_precip += precip.where(mask.data).sum('time').data
        # close datasets and vars 
        segmentation_mask.close() 
        precip.close() 
        del precip_sum
        
    else:
        continue

# save to netcdf 
print('timesteps: ', total_timesteps) 
xr.DataArray(mcs_frequency).to_netcdf(future_path / 'MCS_frequency_pgw.nc') 
xr.DataArray(mcs_precip).to_netcdf(future_path / 'MCS_precip_pgw.nc')
xr.DataArray(total_precip).to_netcdf(future_path / 'total_precip_pgw.nc')
