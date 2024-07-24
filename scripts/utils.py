"""
1;95;0cSome utilities and methods for the project: Precipitation efficiency of convective storms and other atmospheric features in CONUS404 and observations.

Contact: kukulies@ucar.edu

"""

import numpy as np 
from pathlib import Path 
import xarray as xr 
import pandas as pd 
import tobac

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
    area.data[:] = 4*4

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

def is_track_mcs(clusters: pd.DataFrame) -> pd.DataFrame:
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
    consecutive_precip_max = clusters.groupby(["cell"]).max_precip.apply(lambda x:max_consecutive_true(x>=10))#, include_groups=False)
    
    consecutive_area_max = clusters.groupby(["cell"]).area.apply(lambda x:max_consecutive_true(x>=4e4))#, include_groups=False)
    
    max_total_precip = clusters.groupby(["cell"]).total_precip.max()
    
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

### only if merging and splitting is used ###
def process_clusters(tracks):
    groupby_order = ["frame", "track"]
    tracks["cluster"] = (tracks.groupby(groupby_order).feature.cumcount()[tracks.sort_values(groupby_order).index]==0).cumsum().sort_index()
    
    gb_clusters = tracks.groupby("cluster")
    
    clusters = gb_clusters.track.first().to_frame().rename(columns=dict(track="cluster_track_id"))
    clusters["cluster_time"] = gb_clusters.time.first().to_numpy()
    clusters["cluster_longitude"] = gb_clusters.apply(lambda x:weighted_circmean(x.lon.to_numpy(), x.area.to_numpy(), low=0, high=360))
    clusters["cluster_latitude"] = gb_clusters.apply(lambda x:np.average(x.lat.to_numpy(), weights=x.area.to_numpy()))
    
    clusters["cluster_area"] = gb_clusters.area.sum().to_numpy()
    clusters["cluster_max_precip"] = gb_clusters.max_precip.max().to_numpy()
    clusters["cluster_total_precip"] = gb_clusters.total_precip.sum().to_numpy()
    
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
    consecutive_precip_max = clusters.groupby(["cluster_track_id"]).cluster_max_precip.apply(lambda x:max_consecutive_true(x>=10))
    
    consecutive_area_max = clusters.groupby(["cluster_track_id"]).cluster_area.apply(lambda x:max_consecutive_true(x>=4e4))
    
    max_total_precip_volume = clusters.groupby(["cluster_track_id"]).cluster_total_precip.max() * 16
    
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


def regrid_data(era_var, conus): 
    """
    Regrids the feature object mask on 0.25 x 0.25 degree grid to the 4km CONUS grid. 

    era_var: ERA5 variable at a specific time point. 

    conus: Postprocessed CONUS404 data with 2D lon-lat coordinates.

    Returns: 
        The regridded object mask 

    """
    from scipy.interpolate import griddata
    coords = np.array([ ds.lon.values.flatten(), ds.lat.values.flatten()]).T
    regridded = griddata(coords, era_var , (conus.lons.values, conus.lats.values), method='nearest')
    
    return regridded

def get_feature_dataframe(feature_dict): 

    """
    Function to convert the dict information from the MOAAP atmospheric feature dataset to pandas dataframe structure that is the same for the convective tracking.
    
    """

    feature_ids = np.array(list(ar_dict.keys()))
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






