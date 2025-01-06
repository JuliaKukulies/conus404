"""
Some utilities and methods for the project: Precipitation efficiency of convective storms and other atmospheric features in CONUS404 and observations.

Contact: kukulies@ucar.edu

"""
import numpy.ma as ma 
import numpy as np 
from pathlib import Path 
import xarray as xr 
import pandas as pd 
import tobac
from tobac.utils.periodic_boundaries import weighted_circmean

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

def regrid_data(era_var, era_lons, era_lats, conus): 
    """
    Regrids the feature object mask on 0.25 x 0.25 degree grid to the 4km CONUS grid. 

    era_var: ERA5 variable at a specific time point. 

    conus: Postprocessed CONUS404 data with 2D lon-lat coordinates.

    Returns: 
        The regridded object mask 

    """
    from scipy.interpolate import griddata
    coords = np.array([ era_lons.values.flatten(), era_lats.values.flatten()]).T
    regridded = griddata(coords, era_var , (conus.lons.values, conus.lats.values), method='nearest')    
    return regridded


def crop_gpm_to_conus(xarray_data, lons, lats):
    conus_lon_min = -134.02492
    conus_lon_max = -59.963787
    conus_lat_min = 19.80349
    conus_lat_max = 57.837646

    col_start = np.where(lons > conus_lon_min)[0][0]
    col_end = np.where(lons < conus_lon_max)[0][-1]
    
    row_start = np.where(lats  > conus_lat_min)[0][0]
    row_end = np.where(lats  < conus_lat_max)[0][-1]
    
    cropped_gpm = xarray_data[{'lat': slice(row_start, row_end), 'lon': slice(col_start, col_end)}]

    return cropped_gpm



def regrid_to_conus(input, latname, lonname, ds = 'StageIV'): 
    '''                                                                                            
    regrids CCIC, GPM IMERG, or Stage IV to CONUS404 grid. 
    ''' 
    from scipy.interpolate import griddata

    if ds == 'StageIV':
        stage_iv_conus = Path('/glade/campaign/mmm/c3we/prein/observations/STAGE_II_and_IV/DEM_STAGE-IV/STAGE4_A.nc')
        ds_coords = xr.open_dataset(stage_iv_conus, decode_times = False)
        stageIV_lon = ds_coords.lon.values
        stageIV_lat = ds_coords.lat.values
        input_lons, input_lats = stageIV_lon, stageIV_lat
    elif ds == 'CCIC':
        input_lons, input_lats = np.meshgrid(input[lonname].compute().data, input[latname].compute().data)
    else: 
        input_lons, input_lats = np.meshgrid(input[lonname].compute().data, input[latname].compute().data)

    # values to regrid as flat array 
    values = input.compute().data.flatten()
    points = np.array([ input_lons.flatten(), input_lats.flatten()]).T

    # target grid
    conus_data = xr.open_dataset(Path('/glade/campaign/mmm/c3we/CPTP_kukulies/conus404/processed/conus404_201010.nc'))
    target_lons = conus_data.lons.values
    target_lats = conus_data.lats.values
    
    regridded = griddata(points, values, (target_lons, target_lats), method = 'nearest') 
    return regridded 



def regrid_merggrid(ds_tb, latname, lonname):
    '''
    regrids the MERG grid (CPC-IR or CCIC) from a regularly spaced
    to the irregularly spaced grid of Stage IV 

    '''
    from scipy.interpolate import griddata
    stage_iv_conus = Path('/glade/campaign/mmm/c3we/prein/observations/STAGE_II_and_IV/DEM_STAGE-IV/STAGE4_A.nc') 
    ds_coords = xr.open_dataset(stage_iv_conus, decode_times = False)
    stageIV_lon = ds_coords.lon
    stageIV_lat = ds_coords.lat

    target_lons, target_lats = stageIV_lon, stageIV_lat
    values = ds_tb.data.flatten()

    lons, lats = np.meshgrid(ds_tb[lonname].values, ds_tb[latname].values) 
    points = np.array([ lons.flatten(), lats.flatten()]).T
    regridded = griddata(points, values, (target_lons, target_lats), method = 'nearest')
    return regridded


def filter_data_for_valid_stageIV(data, precip):
    '''
    Sets regridded Tb or CCIC data to NaN where there is no Stage IV data
    
    '''
    mask= np.ma.masked_invalid(precip[:,:,0].data)
    data[mask.mask == True] = np.nan
    return data 


def subset_data_to_conus(dataset: xr.Dataset, latname, lonname) -> xr.Dataset:
    """
            Subset CPCIR data to CONUS.

            Args:
                dataset: An xarray.Dataset to subset.

            Return:
                The dataset restricted to CONUS.
    """

    from pyresample import create_area_def
    
    MERGIR_GRID = create_area_def(
        "cpcir_area",
        {"proj": "longlat", "datum": "WGS84"},
        area_extent=[-180.0, -60.0, 180.0, 60.0],
        resolution=(0.03637833468067906, 0.036385688295936934),
        units="degrees",
        description="MERG IR grid",)
    
    # lat min/max, lon min/max for CONUS404
    #17.647308 57.34342 -138.73135 -57.068634

    # using the corner points of the Stage IV dataset 
    conus_lon_min = -134.02492
    conus_lon_max = -59.963787
    conus_lat_min = 19.80349
    conus_lat_max = 57.837646

    lons_cpcir, lats_cpcir = MERGIR_GRID.get_lonlats()
        
    lons_cpcir = lons_cpcir[0]
    lats_cpcir = lats_cpcir[:, 0]

    col_start = np.where(lons_cpcir > conus_lon_min)[0][0]
    col_end = np.where(lons_cpcir < conus_lon_max)[0][-1]
    row_start = np.where(lats_cpcir < conus_lat_max)[0][0]
    row_end = np.where(lats_cpcir > conus_lat_min)[0][-1] 
    
    return dataset[{latname: slice(row_start, row_end), lonname: slice(col_start, col_end)}]


def get_iwp_tendency(tiwp, timedim = 'time', seconds = 3600):
    '''
    Calculate the positive IWP tendency from CCIC over CONUS.

        Args:
        -----
          tiwp: xr.DataArray with IWP (in kg/m2) regridded to match features and cropped over CONUS
          timedim: name of time dimension
          seconds: seconds in timestep (default = 3600, if hourly data is used)
        
        Returns: 
        --------
           tiwp_ten: positive IWP tendency (dIWP[>0]/dt)
    '''

    tiwp_diff = tiwp.diff(dim = timedim) / seconds # to get the units to kg/m2/s
    tiwp_ten = tiwp_diff.where(tiwp_diff > 0 )

    return tiwp_ten


def get_stats_iwp_tendency(features, segments, tiwp, inplace=False):
    """    
    Calculate only the IWP tendency for each feature and append it to feature dataframe (based on C404 data).
    
    """
    if not inplace:
        features = features.copy()

    # initiate variables
    features["max_iwpten"] = np.nan    
    features["total_iwpten"] = np.nan
    features["mean_iwpten"] = np.nan
    # get positive IWP tendency from CCIC dataset 
    tiwp_ten = get_iwp_tendency(tiwp)
    
    #### get statistics for each detected feature ####
    
    # POSITIVE ICE WATER PATH TENDENCY 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(features, segments, tiwp_ten, statistic=dict(total_iwpten=np.nansum), default=np.nan)
    
    # MAX IWP TENDENCY
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(features, segments, tiwp_ten, statistic=dict(max_iwpten=np.nanmax), default=np.nan)

    # MEAN IWP TENDENCY
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, tiwp_ten, statistic=dict(mean_iwpten=np.nanmean), default=np.nan
    )
    return features

def get_statistics_obs(features, segments, precip, tiwp, timedim = 0, inplace=False):
    """
    Calculate the area, maximum precip rate and total precip volume for each feature
    """
    
    if not inplace:
        features = features.copy()

    # get matrix area (16 km2)
    area = precip.copy()
    areas = np.full(area.shape, 4*4)
    area.data = areas

    # initiate variables 
    features["area"] = np.nan
    
    features["max_precip"] = np.nan
    features["max_iwpten"] = np.nan
    features["max_iwp"] = np.nan
    
    features["total_precip"] = np.nan
    features["total_iwpten"] = np.nan
    features["total_iwp"] = np.nan
    features["mean_iwpten"] = np.nan
   
    # get positive IWP tendency from CCIC dataset 
    tiwp_ten = get_iwp_tendency(tiwp)
    
    if timedim == 0:
        segments_ten = segments[1:,:,:]
    elif timedim == 2:
        segments_ten = segments[:,:,1:]

    print(segments.time.dtype, features.time.dtype, tiwp_ten.time.dtype, flush = True)
    
    #### get statistics for each detected feature ####
    # AREA 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
    features, segments, area, statistic=dict(area=np.nansum), default=np.nan
    )
    # MAX PRECIP 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, precip, statistic=dict(max_precip=np.nanmax), default=np.nan
    )

    # POSITIVE ICE WATER PATH TENDENCY
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(features, segments_ten, tiwp_ten, statistic=dict(max_iwpten=np.nanmax), default=np.nan)


    # POSITIVE ICE WATER PATH TENDENCY 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(features, segments_ten, tiwp_ten, statistic=dict(mean_iwpten=np.nanmean), default=np.nan)
    
    # MAX IWP
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(features, segments, tiwp, statistic=dict(max_iwp=np.nanmax), default=np.nan)

    # TOTAL PRECIP
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, precip, statistic=dict(total_precip=np.nansum), default=np.nan
    )
    # POSTITIVE ICE WATER PATH TENDENCY 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(features, segments_ten, tiwp_ten, statistic=dict(total_iwpten=np.nansum), default=np.nan)

    # TOTAL IWP 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(features, segments, tiwp, statistic=dict(total_iwp=np.nansum), default=np.nan)
    return features 


def get_statistics_conus(features, segments, ds, inplace=False): 
    """
    Calculate the area, precipitation, condensation and other statistics for each feature
    detected in CONUS404.

    features: pd.DataFrame (output from tobac) with the tracked features
    segments: segmentation mask (output from tobac)
    ds: xr.DataSet containing the 2D CONUS404 postprocessed data.

    Returns:
       features: the feature dataframe with the statistics
    
    """
    if not inplace:
        features = features.copy() 
        
    # get matrix with area for every grid cell (16 km2)
    area = ds.surface_precip.copy()

    # get matrix area (16 km2)
    area = ds.surface_precip.copy()
    areas = np.full(area.shape, 4*4)
    area.data = areas
    #area.data[:] = 4*4

    # initiate variables 
    features["area"] = np.nan
    
    features["max_precip"] = np.nan
    features["max_con"] = np.nan
    features["max_iwp"] = np.nan
    features["max_lwp"] = np.nan
    
    features["total_precip"] = np.nan
    features["total_con"] = np.nan
    features["total_iwp"] = np.nan
    features["total_lwp"] = np.nan 

    features["min_tb"] = np.nan
    
    #### get statistics for each detected feature ####

    # AREA 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
    features, segments, area, statistic=dict(area=np.nansum), default=np.nan
    )
    # MAX PRECIP 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.surface_precip, statistic=dict(max_precip=np.nanmax), default=np.nan
    )

    # MAX CONDENSATION
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.condensation_rate, statistic=dict(max_con=np.nanmax), default=np.nan
    )
    # MAX IWP
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tiwp, statistic=dict(max_iwp=np.nanmax), default=np.nan
    )

    # MAX LWP
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tlwp, statistic=dict(max_lwp=np.nanmax), default=np.nan
    )
    
    # TOTAL PRECIP
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.surface_precip, statistic=dict(total_precip=np.nansum), default=np.nan
    )
    # TOTAL CONDENSATION
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.condensation_rate, statistic=dict(total_con=np.nansum), default=np.nan
    )
    # TOTAL IWP 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tiwp, statistic=dict(total_iwp=np.nansum), default=np.nan
    )
    
    # TOTAL LWP 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tlwp, statistic=dict(total_lwp=np.nansum), default=np.nan
    )
    
    # MIN TB
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tb, statistic=dict(min_tb=np.nanmin), default=np.nan)
    return features

def get_stats_conus(features, segments, ds, timedim = 0, inplace=False): 
    """
    Updated statistics calculation - include tendencies and histograms. 
    
    Calculate the area, precipitation, condensation and other statistics for each feature
    detected in CONUS404.

    features: pd.DataFrame (output from tobac) with the tracked features
    segments: segmentation mask (output from tobac)
    ds: xr.DataSet containing the 2D CONUS404 postprocessed data.

    Returns:
       features: the feature dataframe with the statistics
    
    """
    if not inplace:
        features = features.copy() 
        
    # get matrix with area for every grid cell (16 km2)
    area = ds.surface_precip.copy()

    # get matrix area (16 km2)
    area = ds.surface_precip.copy()
    areas = np.full(area.shape, 4*4)
    area.data = areas
    #area.data[:] = 4*4

    # initiate variables 
    features["area"] = np.nan
    features["max_precip"] = np.nan
    features["max_con"] = np.nan
    features["max_iwp"] = np.nan
    features["max_lwp"] = np.nan
    features["total_precip"] = np.nan
    features["total_con"] = np.nan
    features["total_iwp"] = np.nan
    features["total_lwp"] = np.nan 
    features["min_tb"] = np.nan
    # distributions
    features['iwp_hist'] = np.nan
    features['precip_hist'] = np.nan
    features['condensation_hist'] = np.nan
    
    # IWP tendency  
    features["total_iwpten"] = np.nan
    # get positive IWP tendency from CCIC dataset 
    tiwp_ten = get_iwp_tendency(ds.tiwp)
    tiwp_ten["time"] = segments.time.values[1:]
    
    # POSITIVE ICE WATER PATH TENDENCY

    if timedim == 0:
        segments_ten = segments[1:,:,:]
    elif timedim == 2:
        segments_ten = segments[:,:,1:]

    features = tobac.utils.bulk_statistics.get_statistics_from_mask(features, segments_ten, tiwp_ten, statistic=dict(total_iwpten=np.nansum), default=np.nan)

    ### get histogram statistics for each detected feature ###
    rain_rate_bins = np.linspace(1,200,200)
    condensation_bins = np.linspace(1, 200,200)
    iwp_bins = np.linspace(1,100,200)
    #iwp_bins = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.surface_precip, statistic = dict(precip_hist= (np.histogram, {'bins': rain_rate_bins})), default = np.nan)
    
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tiwp, statistic = dict(iwp_hist= (np.histogram, {'bins': iwp_bins})), default = np.nan)
    
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
    features, segments, ds.condensation_rate*3600, statistic = dict(condensation_hist=(np.histogram, {'bins': condensation_bins})), default = np.nan)

    #### get statistics for each detected feature ####

    # AREA 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
    features, segments, area, statistic=dict(area=np.nansum), default=np.nan
    )
    # MAX PRECIP 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.surface_precip, statistic=dict(max_precip=np.nanmax), default=np.nan
    )

    # MAX CONDENSATION
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.condensation_rate, statistic=dict(max_con=np.nanmax), default=np.nan
    )
    # MAX IWP
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tiwp, statistic=dict(max_iwp=np.nanmax), default=np.nan
    )

    # MAX LWP
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tlwp, statistic=dict(max_lwp=np.nanmax), default=np.nan
    )
    
    # TOTAL PRECIP
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.surface_precip, statistic=dict(total_precip=np.nansum), default=np.nan
    )
    # TOTAL CONDENSATION
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.condensation_rate, statistic=dict(total_con=np.nansum), default=np.nan
    )
    # TOTAL IWP 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tiwp, statistic=dict(total_iwp=np.nansum), default=np.nan
    )
    
    # TOTAL LWP 
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tlwp, statistic=dict(total_lwp=np.nansum), default=np.nan
    )
    
    # MIN TB
    features = tobac.utils.bulk_statistics.get_statistics_from_mask(
        features, segments, ds.tb, statistic=dict(min_tb=np.nanmin), default=np.nan)
    
    return features 





def max_consecutive_true(condition: np.ndarray[bool]) -> int:
    """
    Return the maximum number of consecutive True values in 'condition'

    Parameters
    ----------
    condition : np.ndarray[bool]
        numpy array of boolean values

    Returns
    -------
    int
        the maximum number of consecutive True values in 'condition'
    """
    if isinstance(condition, pd.Series):
        condition = condition.to_numpy()
    if np.any(condition):
        return np.max(
            np.diff(
                np.where(
                    np.concatenate(
                        ([condition[0]], condition[:-1] != condition[1:], [True])
                    )
                )[0]
            )[::2],
            initial=0,
        )
    else:
        return 0

    
def is_track_mcs(tracks: pd.DataFrame) -> pd.DataFrame:
    """Test whether each track in features meets the condtions for an MCS

    Parameters
    ----------
    features : pd.Dataframe
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    consecutive_precip_max = tracks.groupby(["cell"]).max_precip.apply(lambda x:max_consecutive_true(x>=10))#, include_groups=False)
    
    consecutive_area_max = tracks.groupby(["cell"]).area.apply(lambda x:max_consecutive_true(x>=4e4))#, include_groups=False)
    
    max_total_precip = tracks.groupby(["cell"]).total_precip.max()
    
    is_mcs = np.logical_and.reduce(
        [
            consecutive_precip_max >= 4,
            consecutive_area_max >= 4,
            max_total_precip.to_numpy() >= 2e4,
        ]
    )
    mcs_tracks =  pd.Series(data=is_mcs, index=consecutive_precip_max.index)
    mcs_tracks.index.name="cell_id"
    return mcs_tracks



### only if merging and splitting is used ###
def process_clusters(tracks, lonname, latname):
    groupby_order = ["frame", "track"]
    tracks["cluster"] = (tracks.groupby(groupby_order).feature.cumcount()[tracks.sort_values(groupby_order).index]==0).cumsum().sort_index()
    
    gb_clusters = tracks.groupby("cluster")
    
    clusters = gb_clusters.track.first().to_frame().rename(columns=dict(track="cluster_track_id"))
    
    clusters["cluster_time"] = gb_clusters.time.first().to_numpy()
    clusters["cluster_longitude"] = gb_clusters.apply(lambda x:weighted_circmean(x[lonname].to_numpy(), x.area.to_numpy(), low=0, high=360))#, include_groups=False)
    clusters["cluster_latitude"] = gb_clusters.apply(lambda x:np.average(x[latname].to_numpy(), weights=x.area.to_numpy()))#, include_groups=False)
    
    clusters["cluster_area"] = gb_clusters.area.sum().to_numpy()
    clusters["cluster_max_precip"] = gb_clusters.max_precip.max().to_numpy()
    clusters["cluster_total_precip"] = gb_clusters.total_precip.sum().to_numpy()
    clusters["cluster_total_precip_volume"] = gb_clusters.total_precip.sum().to_numpy() * gb_clusters.area.sum().to_numpy()
    return tracks, clusters


def is_track_mcs_cluster(clusters: pd.DataFrame) -> pd.DataFrame:
    """Test whether each track in features meets the conditions for an MCS

    Parameters
    ----------
    features : pd.Dataframe
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    consecutive_precip_max = clusters.groupby(["cluster_track_id"]).cluster_max_precip.apply(lambda x:max_consecutive_true(x>=10))#, include_groups=False)
    
    consecutive_area_max = clusters.groupby(["cluster_track_id"]).cluster_area.apply(lambda x:max_consecutive_true(x>=4e4))#, include_groups=False)
    
    max_total_precip = clusters.groupby(["cluster_track_id"]).cluster_total_precip_volume.max() 
    
    is_mcs = np.logical_and.reduce(
        [
            consecutive_precip_max >= 4,
            consecutive_area_max >= 4,
            max_total_precip.to_numpy() >= 2e4,
        ]
    )
    mcs_tracks =  pd.Series(data=is_mcs, index=consecutive_precip_max.index)
    mcs_tracks.index.name="track"
    return mcs_tracks


def regrid_era_to_stageIV(era_var,era_lons, era_lats, target): 
    """
    Regrids the feature object mask on 0.25 x 0.25 degree grid to the 4km CONUS grid. 

    era_var: ERA5 variable at a specific time point. 

    target: Stage IV data with lats and lons 

    Returns: 
        The regridded object mask
    
    """
    from scipy.interpolate import griddata    
    coords = np.array([ era_lons.values.flatten(), era_lats.values.flatten()]).T   
    regridded = griddata(coords, era_var , (target.lon.values, target.lat.values), method='nearest')
    
    return regridded


def get_feature_dataframe(feature_dict): 

    """
    Function to convert the dict information from the MOAAP atmospheric feature dataset to pandas dataframe structure that is the same for the convective tracking.
    
    """
    feature_ids = np.array(list(feature_dict.keys()))
    columns = ['feature', 'time', 'lon', 'lat', 'area'] 
    feature_df = pd.DataFrame( columns = columns) 
    # loop through each feature ID 
    for id in feature_ids:
        # initiate empty dataframe 
        df = pd.DataFrame( columns = columns) 
        # populate dataframe with values from dict 
        df['area'] = feature_dict[id]['size'] / 1e6
        df['time'] = feature_dict[id]['times']
        df['min_tracking_var'] = feature_dict[id]['min']
        df['max_tracking_var'] = feature_dict[id]['max']
        df['mean_tracking_var'] = feature_dict[id]['mean']
        df['feature'] = int(id)
        df['lon'] = feature_dict[id]['track'][:, 1]
        df['lat'] = feature_dict[id]['track'][:,0]
        
        # append to overall dataframe for atmospheric feature
        feature_df = pd.concat([feature_df, df], ignore_index = True)
                              
    # make sure the IDs are integers 
    feature_df['feature'] = feature_df['feature'].values.astype(np.int64) 
    # adjust time format 
    times = np.array([x.to_datetime64() for x in feature_df.time])
    feature_df['time'] = times
    # check if feature ID is right type 
    assert feature_df['feature'].dtype == 'int64'
    # check that all unique feature labels are included 
    assert feature_df.feature.unique().size == feature_ids.size

    return feature_df






