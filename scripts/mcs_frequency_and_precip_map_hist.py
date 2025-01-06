"""

Calculating the MCS frequency and contribution to total precip for present MCS tracks.


"""
from pathlib import Path 
import xarray as xr
import numpy as np
conus_path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/processed/')
future_path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/tracked_storms_pgw/')
mask_pgw_files = list(future_path.glob('tobac_storm_mask_????_??_pgw.nc'))
track_pgw_files = list(future_path.glob('tobac_storm_tracks_????_??_pgw.nc'))
mask_pgw_files.sort()
track_pgw_files.sort()
assert len(mask_pgw_files) == len(track_pgw_files)
print(len(track_pgw_files), ' files detected.')

# get corresponding present-climate tracks
present_path = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/tracked_storms/')
# all
#mask_files = list(present_path.glob('tobac_storm_mask_*nc'))
#track_files = list(present_path.glob('tobac_storm_tracks_*nc'))

track_files = []
mask_files = []

# just find files that correspond to future tracks 
for i, trackfile in enumerate(track_pgw_files): 
    fname = str(trackfile.name)[:-7] + '.nc'
    present_track = present_path / fname
    track_files.append(present_track) 
    fname = str(mask_pgw_files[i].name)[:-7] + '.nc'
    present_mask = present_path / fname
    mask_files.append(present_mask) 

assert len(track_files) == len(track_pgw_files)

# initialize
data_array = xr.open_dataset(mask_files[0]).segmentation_mask[0]
print(data_array.data.shape)
total_precip = np.zeros(data_array.data.shape )
mcs_precip = np.zeros(data_array.data.shape )
mcs_frequency   = np.zeros( data_array.data.shape )
total_timesteps = 0

for idx, fname in enumerate(mask_files):
    year = str(fname.name)[-10:-6]
    month = str(fname.name)[-5:-3]
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
        precip_sum = precip.sum('time').data
        # get total sum of precip and the precip associated with MCSs 
        total_precip += precip_sum 
        mcs_precip += precip.where(mask ).sum('time').data
        # close datasets and vars 
        segmentation_mask.close() 
        precip.close() 
        del precip_sum
        
    else:
        continue

# save to netcdf 
print('timesteps: ', total_timesteps) 
xr.DataArray(mcs_frequency).to_netcdf(present_path / 'MCS_frequency_present.nc') 
xr.DataArray(mcs_precip).to_netcdf(present_path / 'MCS_precip_present.nc')
xr.DataArray(total_precip).to_netcdf(present_path / 'total_precip_present.nc')
