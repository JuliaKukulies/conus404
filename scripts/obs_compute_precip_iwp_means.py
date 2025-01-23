"""
This script derives histograms and monthly mean maps for CCIC, Stage IV and GPM IMERG, regridded to the 4km CONUS404 grid. 

"""

import numpy as np
import xarray as xr
import h5py
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import ccic
import s3fs
from scipy.stats import pearsonr
from tqdm import tqdm
import sys 
module_path = '/glade/work/kukulies/pe_conus404/conus404/scripts/'
sys.path.append(module_path) 
import utils
from utils import regrid_to_conus
from utils import crop_gpm_to_conus

##### directories #####
stageIV_path = Path('/glade/campaign/mmm/c3we/prein/observations/STAGE_II_and_IV/data/')
gpm_path = Path('/glade/campaign/mmm/c3we/prein/observations/GPM_IMERG_V07/')
aws_path = Path('chalmerscloudiceclimatology/record/cpcir/')
out_dir = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/mean_values/OBS/')
conus_dir = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/processed/')

year = str(sys.argv[1])
#month = str(sys.argv[2]).zfill(2)

##### bins for histograms #####
bin_edges_precip = np.linspace(1, 200, 200)  
bin_edges_iwp = np.linspace(0.1,100,200)


#### steps for CCIC ####
nx, ny = 1015, 1367 # dimensions of conus domain

# looping through months  
for month in np.arange(1,13):
    month = str(month).zfill(2)
    if Path(out_dir / str('ccic_monthly_mean_' + year + month + '.nc' )).is_file():
        continue
    else:
        s3 = s3fs.S3FileSystem(anon=True)
        fnames = s3.glob(str(aws_path) + '/'+ year +  '/*'+ year +month +'*zarr')
        fnames.sort()

        # read in corresponding C404 file (to get evaluation metrics on instantaneous values)
        conus_ds = xr.open_dataset(conus_dir / str('conus404_' + year + month + '.nc') )  

        # regrid data for each timestep
        nt = len(fnames)
        regridded_data = np.zeros((nt, nx, ny ))

        print('derive and regrid CCIC data..', flush = True)
        for t, fname in enumerate(fnames):
            # read in one file 
            ds_ccic = xr.open_zarr(s3.get_mapper(fname))
            ccic = ds_ccic.tiwp[0]
            # crop to CONUS 
            ccic_cropped = utils.subset_data_to_conus(ccic, 'latitude', 'longitude') 
            ccic_lats = ccic_cropped.latitude
            ccic_lons = ccic_cropped.longitude
            ccic_on_conus = regrid_to_conus(ccic_cropped, 'latitude', 'longitude', ds = 'CCIC')
            regridded_data[t] = ccic_on_conus

        monthly_mean = np.nanmean(regridded_data, axis = 0)

        # make xarray with conus dimensions
        dataset = xr.Dataset(
            {
                'tiwp': (['west_east', 'south_north'], monthly_mean)
            },
            coords={
                'south_north': np.arange(ny),
                'west_east': np.arange(nx)
            }
        )
        # save to netcdf 
        dataset.to_netcdf(out_dir / str('ccic_monthly_mean_' + year + month + '.nc'))

        # histogram of instantanous values
        iwp_hist, _ = np.histogram(regridded_data, bins = bin_edges_iwp)
        np.save(out_dir / str('ccic_histogram_' + year+ month  ), iwp_hist)

        # RMSE, bias, correlation with CONUS404 instantanous values
        model = conus_ds.tiwp.transpose('time', 'south_north', 'west_east').values

        valid_mask = ~np.isnan(model) & ~np.isnan(regridded_data)  
        rmse = np.sqrt(np.nanmean((model[valid_mask] - regridded_data[valid_mask]) ** 2))
        bias = np.nanmean(model[valid_mask]- regridded_data[valid_mask])

        # remove nan values 
        valid_indices = ~np.isnan(model) & ~np.isnan(regridded_data)
        cleaned_model = model[valid_indices]
        cleaned_regridded_data = regridded_data[valid_indices]
        corr = pearsonr(cleaned_model.flatten(), cleaned_regridded_data.flatten())[0]
        np.save(out_dir / str('ccic_instantaneous_stats_' + year + month ), [rmse, bias, corr])

        del regridded_data    
        print('CCIC processing done.', flush = True)

        #### steps for Stage IV ####

        # read file 
        monthly_file_prec = stageIV_path / str('LEVEL_2-4_hourly_precipitation_' +  year  +  month  + '.nc')
        ds_prec = xr.open_dataset(monthly_file_prec)
        precip = ds_prec.Precipitation
        stage_iv_conus = Path('/glade/campaign/mmm/c3we/prein/observations/STAGE_II_and_IV/DEM_STAGE-IV/STAGE4_A.nc') 
        ds_coords = xr.open_dataset(stage_iv_conus, decode_times = False) 
        precip = precip.transpose("rlat", "rlon", "time")
        precip['lat'] = ds_coords.lat
        precip['lon'] = ds_coords.lon

        # regrid data for each timestep
        nt = precip.time.values.size
        regridded_data = np.zeros((nt, nx, ny))

        print('deriving and processing Stage IV data', flush = True)

        for t, time in enumerate(precip.time.values): 
            precip_t= precip.sel(time=time ) 
            stageiv_on_conus = regrid_to_conus(precip_t, 'lat', 'lon', ds = 'StageIV')
            regridded_data[t] = stageiv_on_conus

        # monthly mean
        monthly_mean = np.nanmean(regridded_data, axis = 0)

        # make xarray with conus dimensions
        dataset = xr.Dataset(
            {
                'surface_precip': (['west_east', 'south_north'], monthly_mean)
            },
            coords={
                'south_north': np.arange(ny),
                'west_east': np.arange(nx)
            }
        )
        # save to netcdf 
        dataset.to_netcdf(out_dir / str('stage_iv_monthly_mean_' + year + month + '.nc'))

        # histogram of instantanous values
        precip_hist, _ = np.histogram(regridded_data, bins = bin_edges_precip)
        np.save(out_dir / str('stageiv_histogram_' + year+ month  ), precip_hist)

        # RMSE, bias, correlation with CONUS404 instantanous values
        model = conus_ds.surface_precip.transpose('time', 'south_north', 'west_east').values
        valid_mask = ~np.isnan(model) & ~np.isnan(regridded_data)  
        rmse = np.sqrt(np.nanmean((model[valid_mask] - regridded_data[valid_mask]) ** 2))
        bias = np.nanmean(model[valid_mask]- regridded_data[valid_mask])

        valid_indices = ~np.isnan(model) & ~np.isnan(regridded_data)
        cleaned_model = model[valid_indices]
        cleaned_regridded_data = regridded_data[valid_indices]
        corr = pearsonr(cleaned_model.flatten(), cleaned_regridded_data.flatten())[0]
        np.save(out_dir / str('stageiv_instantaneous_stats_' + year + month ), [rmse, bias, corr])

        del regridded_data
        print('Stage IV processing done.', flush = True )

        #### steps for GPM IMERG ####

        fnames = list( (gpm_path / str(year)).glob( str('3B*IMERG*'+str(year) + str(month).zfill(2) +'*')))
        fnames.sort()
        print(len(fnames), 'files found for GPM', flush = True)

        #regrid data for each timestep
        nt = len(fnames)
        regridded_data = np.zeros((nt, nx, ny))

        print('processing GPM data...',flush = True)

        for i, fname in enumerate(fnames): 
            with h5py.File(fname, 'r') as f:
                if i == 0:
                    lat_coords = f['Grid/lat'][:]
                    lon_coords = f['Grid/lon'][:]
                data = np.array(f['Grid/precipitation'][:])
                coords = {'lat': lat_coords,  'lon': lon_coords}
                xarray_data = xr.DataArray(data.squeeze(), coords=coords, dims=['lon', 'lat'])
                # crop region over CONUS 
                cropped_gpm = crop_gpm_to_conus(xarray_data, lon_coords, lat_coords)
                # regrid data
                gpm_on_conus = regrid_to_conus(cropped_gpm.T, 'lat', 'lon', ds = 'GPM')
                regridded_data[i] = gpm_on_conus

        # hourly mean
        hours = model.shape[0]
        reshaped_data = regridded_data.reshape(hours, 2, 1015, 1367)
        regridded_data_hourly = reshaped_data.mean(axis=1)
        monthly_mean = np.nanmean(regridded_data, axis = 0)

        # make xarray with conus dimensions
        dataset = xr.Dataset(
            {
                'surface_precip': (['west_east', 'south_north'], monthly_mean)
            },
            coords={
                'south_north': np.arange(ny),
                'west_east': np.arange(nx)
            }
        )
        # save to netcdf 
        dataset.to_netcdf(out_dir / str('gpm_imerg_monthly_mean_' + year + month + '.nc'))
        # histogram of instantanous values
        precip_hist, _ = np.histogram(regridded_data, bins = bin_edges_precip)
        np.save(out_dir / str('gpm_imerg_histogram_' + year+ month  ), precip_hist)

        # RMSE, bias, correlation with CONUS404 instantanous values 
        valid_mask = ~np.isnan(model) & ~np.isnan(regridded_data_hourly)  
        rmse = np.sqrt(np.nanmean((model[valid_mask] - regridded_data_hourly[valid_mask]) ** 2))
        bias = np.nanmean(model[valid_mask]- regridded_data_hourly[valid_mask])
        valid_indices = ~np.isnan(model) & ~np.isnan(regridded_data_hourly)
        cleaned_model = model[valid_indices]
        cleaned_regridded_data = regridded_data_hourly[valid_indices]
        corr = pearsonr(cleaned_model.flatten(), cleaned_regridded_data.flatten())[0]
        np.save(out_dir / str('gpm_imerg_instantaneous_stats_' + year + month ), [rmse, bias, corr])
        del regridded_data 
        print('GPM processing done.', flush = True)



