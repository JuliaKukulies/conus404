"""
This script regrids the mask files various atmospheric features identified with the MOAAP algorithm to the Stage IV grid. For each atmospheric feature (i.e. atmospheric rivers, tropical and extratropical cyclones), some bulk statistics are calculated based on observed precipitationa as well as ice water path.
-------------------------------------------------------------------------------------------
Contact: kukulies@ucar.edu

MOAAP algorithm and publication:
https://github.com/AndreasPrein/MOAAP
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EF003534

"""
import datetime 
import utils
from pathlib import Path 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import xarray as xr
import pickle 
import tobac
import sys
import ccic
import s3fs
print('Using tobac version ', tobac.__version__, flush = True)
import warnings
warnings.filterwarnings('ignore')

############# input parameters to script ##########################

year = str(sys.argv[2])
weather_type = str(sys.argv[1])

############# read in the data ####################################
savedir = Path( str( '/glade/work/kukulies/pe_conus404/moaap_features/obs/') + str(weather_type) )  
for mon in np.arange(1,13):
    month = str(mon).zfill(2)
    # check if this month has already been processed 
    output_fname = savedir / str('features_' + weather_type+ '_'+ year + '_' + month + '.nc') 

    if not output_fname.is_file():
    # read in feature object mask files 
        path = Path('/glade/campaign/mmm/c3we/prein/Papers/2022_CONUS404-Features/data/V1/ERA-StageIV/')
        ds = xr.open_dataset( path / str(year + '01_ERA-StageIV_ObjectMasks__dt-1h_MOAAP-masks.nc'))
        monthly_ds= ds.sel(time=ds.time.dt.month.isin([int(month)]))
        atmospheric_feature = monthly_ds[ str(weather_type + '_Objects') ].where(monthly_ds.PR>=0)
        era_lats = ds.lat
        era_lons = ds.lon

        # read in corresponding month with OBS: Stage-IV (Precip), CCIC (cloud ice)
        print('reading in observational data for precip and cloud ice....', flush = True)
        stageIV= Path('/glade/campaign/mmm/c3we/prein/observations/STAGE_II_and_IV/data/')
        monthly_file_prec = stageIV / str('LEVEL_2-4_hourly_precipitation_' + year + month +'.nc') 
        ds_prec = xr.open_dataset(monthly_file_prec)
        precip = ds_prec.Precipitation
        # get coordinates for Stage IV dataset 
        stage_iv_conus = Path('/glade/campaign/mmm/c3we/prein/observations/STAGE_II_and_IV/DEM_STAGE-IV/STAGE4_A.nc') 
        ds_coords = xr.open_dataset(stage_iv_conus, decode_times = False)
        # swap dimensions such that time dimension is last
        precip = precip.transpose("rlat", "rlon", "time")
        assert precip.dims[-1] == 'time'

        #### CCIC ####
        # Create a filesystem for S3
        s3 = s3fs.S3FileSystem(anon=True)
        # Lazy load a CCIC file
        aws_path = Path('chalmerscloudiceclimatology/record/cpcir/')
        fnames = s3.glob(str(aws_path) + '/'+ str(year) +  '/*'+ year +month +'*zarr') 
        fnames.sort()
        
        # read in all monthly files
        for fname in fnames:
            # read in one file 
            ds_ccic = xr.open_zarr(s3.get_mapper(fname))
            # Load `tiwp` into memory
            tiwp = ds_ccic.tiwp.load().mean('time') #  hourly average 

            # crop region 
            tiwp_cropped = utils.subset_data_to_conus(tiwp, 'latitude', 'longitude')

            # regrid
            regridded_tiwp = utils.regrid_merggrid(tiwp_cropped, 'latitude', 'longitude')
            if fname == fnames[0]:
                tiwp_ds = regridded_tiwp 
                continue
            else: 
                # concatenate on time dim 
                tiwp_ds  = np.dstack([ tiwp_ds, regridded_tiwp])
                del ds_ccic, tiwp

        # make xarray dataarray for tiwp
        tiwp = precip.copy()
        tiwp.data =tiwp_ds.astype(np.int64)
    
        # quick check that obs data and atmospheric feature mask have same size of time dimension
        if precip.shape[-1] != atmospheric_feature.shape[0] == tiwp.shape[-1]:
            print(year, month, 'Precip:', precip.shape, flush = True)
            print('CCIC: ', tiwp.shape, flush = True)
            print('MOAAP feature mask:', atmospheric_feature.shape, flush = True)
            continue
        
        timedim  = precip.shape[-1]

        # read in pickle files that contain the track meta data
        if weather_type == 'AR':
            weather_str = weather_type + 's'
        elif weather_type == 'CY_z500':
            weather_str = 'CY-z500'
        else:
            weather_str = weather_type

        pickle_file = str( path / str(weather_str + '_'+year+'01__dt-1h_MOAAP-masks.pkl'))
        with open(pickle_file, 'rb') as f:
            feature_dict = pickle.load(f)

        # get features in right format  
        feature_df = utils.get_feature_dataframe(feature_dict)
        
            
        ################################ Regridding, reformatting, bulk statistics ############
        
        # initiate empty matrix 
        atmospheric_feature_4km = np.zeros((tiwp.shape[0], tiwp.shape[1],timedim))

        # perform regridding for each timestep
        print(datetime.datetime.now(), flush = True)
        print('regridding the input feature mask for ', year, month , flush = True)
    
        for idx, tt in enumerate(precip.time.values):
            era_var = atmospheric_feature.data[idx].flatten()
            atmospheric_feature_4km_t = utils.regrid_era_to_stageIV(era_var,era_lons, era_lats, ds_coords)
            atmospheric_feature_4km[:,:,idx] =  atmospheric_feature_4km_t

        # get mask in right format 
        segments = precip.copy()
        segments.data = atmospheric_feature_4km.astype(np.int64)
        print(np.unique(segments.data), 'are the unique feature labels in month ', month, flush = True)

        # compute the bulk statistics for one month
        print('compute bulk statistics...')
        print(datetime.datetime.now(), flush = True)
        stats_df = utils.get_statistics_obs(feature_df, segments, precip, tiwp)

        # save output
        stats_df.to_xarray().to_netcdf(output_fname) 
        print(output_fname, ' saved.', flush = True )
        print(datetime.datetime.now(), flush = True) 

    else:
        print(output_fname, ' already exists.', flush = True )
        continue
