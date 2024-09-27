"""
This script produces monthly files of IWP, LWP, condensation rates, surface precipitation and brightness temperatures from standard 3D model WRF output for the C404 PGW dataset.

kukulies@ucar.edu

"""
from pathlib import Path 
import xarray as xr 
import numpy as np 
import wrf 
from microphysics import microphysics_functions as micro 
from netCDF4 import Dataset
import datetime 
import sys

########################### Directory with model output ######################################
path = Path('/glade/campaign/ncar/USGS_Water/CONUS404_PGW/') 
out = Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/PGW/')

year_in = int(sys.argv[1])
mon_in = int(sys.argv[2])
years = np.arange(year_in, year_in  + 1 )
################################## Function for OLR-Tb conversion ###########################

def get_tb(olr):
    """                                                                                                       
    This function converts outgoing longwave radiation to brightness temperatures.                             
    using the Planck constant.                                                                                 
                                                                                                              
    Args:                                                                                                       
        olr(xr.DataArray or numpy array): 2D field of model output with OLR                                      
    Returns:                                                                                                    
        tb(xr.DataArray or numpy array): 2D field with estimated brightness temperatures                        
    """
    # constants                                                                                                
    aa = 1.228
    bb = -1.106 * 10**(-3) # K−1                                                                              
    # Planck constant                                                                                          
    sig = 5.670374419 * 10**(-8) # W⋅m−2⋅K−4                                                                    
    # flux equivalent brightness temperature                                                                   
    Tf = (abs(olr)/sig) ** (1./4)
    tb = (((aa ** 2 + 4 * bb *Tf ) ** (1./2)) - aa)/(2*bb)
    return tb 

def wstagger_to_mass(W):
    """                                                                                                         
    W are the data on the top and bottom of a grid box                                                          
    A simple conversion of the stagger grid to the mass points.                                                 
                                                                                                                
    (row_j1+row_j2)/2 = masspoint_inrow                                                                         
                                                                                                                
    Input:                                                                                                      
        Wgrid with size (##+1)                                                                                  
    Output:                                                                                                     
        W on mass points with size (##)                                                                         
    """
    # create the first column manually to initialize the array with correct dimensions                          
    W_masspoint = (W[0, :, :]+W[1, :, :])/2. # average of first and second column    
    W_masspoint = np.expand_dims(W_masspoint, 0)

    W_num_levels = int(W.shape[0])-1 # we want one less level than we have                                      

    # Loop through the rest of the rows                                                                         
    # We want the same number of rows as we have columns.                                                       
    # Take the first and second row, average them, and store in first row in V_masspoint                        
    for lev in range(1,W_num_levels):
        lev_avg = (W[lev, :, :]+W[lev+1, :, :])/2.
        lev_avg = np.expand_dims(lev_avg, 0)
        W_masspoint = np.vstack((W_masspoint,lev_avg))
    return W_masspoint

##############################################################################################
water_year = year_in
month = mon_in
print('input received for water year:', str(water_year), str(month), flush = True)

if int(month) >= 10:
    # define actual year
    year = water_year - 1
    print('water year ', str(water_year), ',but month is in year:', str(year), flush = True )
else:
    year = water_year

# check first if month has already been processed 
out_file = out / ( 'conus404_' + str(year) + str(month).zfill(2) +  '.nc')
if out_file.is_file():
    print(out_file, ' already processed.',  flush = True)
else:
    monthly_path = path / ('WY' + str(water_year)) 
    hourly_files_3d = list(monthly_path.glob(str('wrf3d*'+str(year) +'-' + str(month).zfill(2) + '*') )) 
    hourly_files_2d = list(monthly_path.glob( str('wrf2d*'+str(year) +'-' + str(month).zfill(2) + '*') ))
    hourly_files_3d.sort()
    hourly_files_2d.sort()
    print(len(hourly_files_3d), flush = True)
    assert len(hourly_files_3d) == len(hourly_files_3d)

    # open file and extract necessary variables 
    print('processing variables for', str(year) +' '+ str(month), ' over ',len(hourly_files_2d), 'files', flush = True)
    print(datetime.datetime.now(), flush = True)
    for file_idx, fname in enumerate(hourly_files_2d):
        ds2d = xr.open_dataset(fname)
        ds3d = xr.open_dataset(hourly_files_3d[file_idx])
        wrfin = Dataset(hourly_files_3d[file_idx])

        # 3d variables 
        iwc = ds3d.QSNOW.squeeze() + ds3d.QGRAUP.squeeze() + ds3d.QICE.squeeze()
        lwc = ds3d.QRAIN.squeeze() + ds3d.QCLOUD.squeeze()

        # get variables that are necessary for calculation of condensation rates
        w_staggered = ds3d.W.squeeze()
        vertical_velocity = wstagger_to_mass(w_staggered)
        temp = ds3d.TK.squeeze()
        pressure = ds3d.P.squeeze()
        qcloud = ds3d.QCLOUD.squeeze()

        # 2d variables 
        precip = ds2d.PREC_ACC_NC.squeeze().data 
        snow = ds2d.SNOW_ACC_NC.squeeze().data 
        graup = ds2d.GRAUPEL_ACC_NC.squeeze().data
        olr = ds2d.OLR.squeeze().data

        ############ calculate quantities from standard output: LWP, IWP, C, P, Tb ###################    
        brightness_temperature = get_tb(olr)
        # integate iwc and lwp over pressure                                                                    
        iwp = micro.pressure_integration(iwc.data, -pressure.data)
        lwp = micro.pressure_integration(lwc.data, -pressure.data)
        # condensation rate from w,P,T,qcloud
        condensation_rate_t = micro.get_condensation_rate(vertical_velocity, temp, pressure)
        condensation_cloud = condensation_rate_t.where(qcloud > 0, 0 )
        condensation_masked = condensation_cloud.where(condensation_cloud > 0, 0 ).data
        condensation_integrated = micro.pressure_integration(condensation_masked,-pressure.data)
        # concatenate processed variables along time axis 
        if file_idx == 0:
            tiwp = iwp
            tlwp = lwp
            condensation_rate = condensation_integrated
            surface_precip = precip
            snow_rate = snow
            graup_rate = graup
            tb = brightness_temperature
            time = np.array(ds2d.Times.values) 
        else:
            surface_precip = np.dstack((surface_precip, precip))
            snow_rate = np.dstack((snow_rate, snow))
            graup_rate = np.dstack((graup_rate, graup))
            tiwp  = np.dstack((tiwp, iwp ))
            tlwp = np.dstack((tlwp, lwp))
            condensation_rate = np.dstack((condensation_rate, condensation_integrated))
            tb = np.dstack((tb, brightness_temperature))
            time = np.append(time, ds2d.Times.values)


        # get coords
        #lats = ds2d.XLAT.values
        #lons = ds2d.XLONG.values
        south_north = ds2d.south_north.values
        west_east = ds2d.west_east.values 

        # close datasets and delete variables
        ds3d.close()
        ds2d.close()
        wrfin.close()

        del iwc, iwp, lwc, lwp
        del temp, pressure, qcloud, vertical_velocity
        del precip, olr, snow, graup 
        del condensation_cloud, condensation_integrated, condensation_masked

    # write new netCDF file for month 
    data_vars = dict(tiwp=(["south_north", "west_east", "time"], tiwp),
                     tlwp=(["south_north", "west_east", "time"], tlwp),
                     surface_precip=(["south_north", "west_east", "time"],surface_precip),
                     condensation_rate=(["south_north", "west_east", "time"], condensation_rate),
                     tb=(["south_north", "west_east", "time"], tb),
                     snow=(["south_north", "west_east", "time"], snow_rate),
                     graup=(["south_north", "west_east", "time"], graup_rate),)

    print(datetime.datetime.now(), flush = True)
    coords = dict(south_north=south_north, west_east=west_east, time = time)
    data = xr.Dataset(data_vars=data_vars, coords=coords)
    data.to_netcdf( out / ( 'conus404_' + str(year) + str(month).zfill(2) +  '.nc') )  

    # close datasets and delete variables     
    del tiwp, tlwp
    del surface_precip, condensation_rate
    del tb, time
    del snow_rate, graup_rate
    data.close()





