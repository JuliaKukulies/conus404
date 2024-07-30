"""
This script regrids the mask files various atmospheric features identified with the MOAAP algorithm to the CONUS404 grid. For each atmospheric feature (i.e. atmospheric rivers, tropical and extratropical cyclones), some bulk statistics are calculated based on the CONUS404 fields precipitation, condensation, as well as liquid and ice water path.


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
print('Using tobac version ', tobac.__version__, flush = True)
import warnings
warnings.filterwarnings('ignore')

############# input parameters to script ##########################

#year = '2020'
#month = '01'
#weather_type = 'AR'
#month = str(sys.argv[2]).zfill(2)

year = str(sys.argv[2])
weather_type = str(sys.argv[1])

############# read in the data ####################################
savedir = Path('/glade/work/kukulies/pe_conus404/moaap_features/')

for mon in np.arange(1,13):
    month = str(mon).zfill(2)
    # check if this month has already been processed 
    fname = savedir / str('features_' + weather_type+ '_'+ year + '_' + month + '.nc') 
    if not fname.is_file():
    # read in feature object mask files 
        path = Path('/glade/campaign/mmm/c3we/prein/Papers/2022_CONUS404-Features/data/V1/CONUS404')
        ds = xr.open_dataset( path / str(year + '01_CONUS404_ObjectMasks__dt-1h_MOAAP-masks.nc'))
        monthly_ds= ds.sel(time=ds.time.dt.month.isin([int(month)]))
        atmospheric_feature = monthly_ds[ str(weather_type + '_Objects') ]
        era_lats = ds.lat
        era_lons = ds.lon

        # read in corresponding month with conus data with relevant 2d variables
        data2d = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/')
        monthly_file = data2d / str('conus404_'+ year + month +'.nc' ) 
        conus = xr.open_dataset(monthly_file)

        # quick check that CONUS data and atmospheric feature mask have same size of time dimension
        assert conus.surface_precip.shape[-1] == atmospheric_feature.shape[0]
        timedim = atmospheric_feature.shape[0]

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

        ################################ Regridding, reformatting, bulk statistics ###################

        # initiate empty matrix 
        atmospheric_feature_4km = np.zeros((conus.surface_precip.shape[0], conus.surface_precip.shape[1],timedim))

        # perform regridding for each timestep
        print(datetime.datetime.now(), flush = True)
        print('regridding the data for ', year, month , flush = True)
        for idx, tt in enumerate(conus.time.values): 
            era_var = atmospheric_feature.data[idx].flatten()
            atmospheric_feature_4km_t = utils.regrid_data(era_var,era_lats, era_lons, conus)
            atmospheric_feature_4km[:,:,idx] =  atmospheric_feature_4km_t

        # get mask in right format 
        segments = conus.surface_precip.copy()
        segments.data = atmospheric_feature_4km.astype(np.int64)
        # get features in right format 
        feature_df = utils.get_feature_dataframe(feature_dict)
        print(np.unique(segments.data), 'are the unique feature labels in month ', month, flush = True)

        # compute the bulk statistics for one month
        print('compute bulk statistics...')
        print(datetime.datetime.now(), flush = True)
        stats_df = utils.get_statistics_conus(feature_df, segments, conus)

        # save output
        stats_df.to_xarray().to_netcdf(fname) 
        print(fname, ' saved.', flush = True )
        print(datetime.datetime.now(), flush = True) 

    else:
        print(fname, ' alreadu exists.', flush = True )
        continue
