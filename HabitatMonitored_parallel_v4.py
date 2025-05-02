# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 20:24:37 2025

@author: kaity
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"   # or 1, 8 … anything ≤ 24


import numpy as np
from geopy.distance import geodesic
from geopy.point import Point
import matplotlib.pyplot as plt
from geopy.point import Point
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
import arlpy.uwapm as pm
import arlpy.plot as arlplt
import matplotlib.tri as tri
from pyproj import Geod
import h5py
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import time




#---------------------------------------------------------------------------
# Create a Geod instance for vectorized geodesic computations.
geod = Geod(ellps='WGS84')
bathy_full = None        # set once in main
subset_df = None
subsetBathy= None

def save_dive_frequency(
        h5_path: str,
        drift_id: str,
        dive_id: str,
        freq_khz: int,
        metadata: dict,
        grid_results: list,            # list of dicts from your thread pool
        gzip_level: int = 4):
    """
    grid_results = [
        {'lat': float,
         'lon': float,
         'transmission_loss': 1‑D np.array,
         'tl_depths':         1‑D np.array},
        ...
    ]
    """
    # Number of grid pints 
    n = len(grid_results)
    max_N = max(len(g['tl_depths']) for g in grid_results)

    # pre‑allocate and pad with NaNs
    lat  = np.empty(n,              dtype=np.float32)
    lon  = np.empty(n,              dtype=np.float32)
    dmat = np.full((n, max_N), np.nan, dtype=np.float32)
    tlmat= np.full((n, max_N), np.nan, dtype=np.float32)
    vlen = np.empty(n, dtype=np.uint16)
    
    
    for i, g in enumerate(grid_results):
        k = len(g['tl_depths'])
        lat[i] = g['lat']
        lon[i] = g['lon']
        dmat[i, :k]  = g['tl_depths']
        tlmat[i, :k] = np.squeeze(np.round(g['transmission_loss'],2)) 
        vlen[i] = k
    
    if not os.path.exists(h5_path):
        print(f"[save_dive_frequency] creating new HDF5 file {h5_path}")
    
    with h5py.File(h5_path, "a") as hf:
        fgrp = hf\
            .require_group(f"drift_{drift_id}")\
            .require_group(f"dive_{dive_id}")\
            .require_group(f"frequency_{freq_khz}")

        # save dive‑level attrs once
        for k, v in metadata.items():
            fgrp.parent.attrs[k] = v

        # create or overwrite datasets
        def _dset(name, data, **kw):
            if name in fgrp:
                del fgrp[name]
            fgrp.create_dataset(
                name,
                data=data,
                compression="gzip",
                compression_opts=gzip_level,
                chunks=kw.get("chunks"))

        _dset("lat",  lat)
        _dset("lon",  lon)
        _dset("valid_len", vlen)
        
        # --- choose chunk dims: each <= data dims ---------------------
        row_chunk = min(256, n)        # never exceed #rows
        col_chunk = max_N              # always valid: max_N == depth.shape[1]
        
        _dset("depth", dmat, chunks=(row_chunk, col_chunk))
        _dset("tl",    tlmat,    chunks=(row_chunk, col_chunk))

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) using the haversine formula.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculate the initial compass bearing in degrees between two points.
    """
    lat1, lon1 = map(np.radians, pointA)
    lat2, lon2 = map(np.radians, pointB)
    diffLong = lon2 - lon1
    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(diffLong)
    initial_bearing = np.degrees(np.arctan2(x, y))
    return (initial_bearing + 360) % 360

def extract_bathymetry_from_subset(subset_df, start_lat, start_lon, stop_lat, stop_lon, interval):
    """
    Extracts interpolated bathymetry along the great‐circle path between two points
    using only the data from a scattered subset DataFrame.
    """
    start_point = Point(start_lat, start_lon)
    stop_point  = Point(stop_lat, stop_lon)
    total_distance_km = geodesic(start_point, stop_point).kilometers
    interval_km       = interval / 1000.0
    num_points        = max(int(total_distance_km / interval_km), 1)
    bearing           = calculate_initial_compass_bearing((start_lat, start_lon), (stop_lat, stop_lon))
    path_lats = np.zeros(num_points + 1)
    path_lons = np.zeros(num_points + 1)
    for i in range(num_points + 1):
        current_distance = min(i * interval_km, total_distance_km)
        new_point = geodesic(kilometers=current_distance).destination(start_point, bearing)
        path_lats[i] = new_point.latitude
        path_lons[i] = new_point.longitude
    subset_points = subset_df[['lat', 'lon']].values
    subset_depths = subset_df['depth'].values
    path_points   = np.vstack((path_lats, path_lons)).T
    bathymetry_values = griddata(subset_points, subset_depths, path_points, method='linear')
    range_km = np.array([
        geodesic(start_point, (path_lats[i], path_lons[i])).kilometers
        for i in range(len(path_lats))
    ])
    return bathymetry_values, path_lons, path_lats, range_km

def extract_bathymetry_from_subset_vectorized(
    subset_df: pd.DataFrame,
    start_lat: float,
    start_lon: float,
    stop_lat: float,
    stop_lon: float,
    interval: float):
    """
    Compute bathymetry along the path using vectorized geodesic.
    """
    
    total_distance_km = geodesic((start_lat, start_lon), (stop_lat, stop_lon)).kilometers
    interval_km       = interval / 1000.0
    actual_distance =total_distance_km
    # We need a minimum of 1.1 kms for bellhop
    if total_distance_km<1.1:
        total_distance_km = 1.1

    num_points        = max(int(total_distance_km / interval_km), 1)
    bearing           = calculate_initial_compass_bearing((start_lat, start_lon), (stop_lat, stop_lon))
    distances_m = np.linspace(0, total_distance_km * 1000, num_points + 1)
    
    
    
    lons, lats, _ = geod.fwd(
        np.full_like(distances_m, start_lon),
        np.full_like(distances_m, start_lat),
        np.full_like(distances_m, bearing),
        distances_m
    )
    start_point = Point(start_lat, start_lon)
    range_km = np.array([geodesic(start_point, (lat, lon)).kilometers for lat, lon in zip(lats, lons)])
    subset_points = subset_df[['lat', 'lon']].values
    subset_depths = subset_df['depth'].values
    path_points   = np.column_stack((lats, lons))
    bathymetry_values = griddata(subset_points, subset_depths, path_points, method='linear')
    if np.any(np.isnan(bathymetry_values)):
        nan_mask = np.isnan(bathymetry_values)
        bathymetry_values[nan_mask] = griddata(
            subset_points, subset_depths, path_points[nan_mask], method='nearest'
        )
    return bathymetry_values, lons, lats, range_km, actual_distance

def interpolate_sound_speed(dive_data, maxDepth, plot=False):
    dive_data_sorted = dive_data.sort_values('Depth_m')
    dive_data_sorted.dropna(inplace=True, subset=['SoundSpeed_m_s'])
    depth_range = np.arange(0, maxDepth)
    sound_speed_interp = np.interp(
        depth_range,
        dive_data_sorted['Depth_m'],
        dive_data_sorted['SoundSpeed_m_s']
    )
    return pd.DataFrame({'Depth_m': depth_range, 'SoundSpeed_m_s': sound_speed_interp})


def checkEnv(ii, subset_df, drifter_lat, drifter_lon, freq_hz, ssp, bathy_interval =100):
    '''Function to check that all of the enviornmental parameters are set up
    correctly
    '''
    bathy_vals, path_lon, path_lat, cumulative_distance = extract_bathymetry_from_subset_vectorized(
        subset_df=subset_df,
        start_lat=drifter_lat,
        start_lon=drifter_lon,
        stop_lat=subset_df['lat'].iloc[ii],
        stop_lon=subset_df['lon'].iloc[ii],
        interval=bathy_interval
    )
    bathy_grid = pd.DataFrame({'range': cumulative_distance * 1000, 
                               'depth_m': -bathy_vals})
    bathy_grid.drop_duplicates(inplace=True)
    bathy_grid.sort_values('range', inplace=True)
    bathy_grid.loc[0, 'range'] = 0
    bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()

    # Create the enviornment
    env = pm.create_env2d(
        depth=bathy,
        soundspeed=ssp,
        bottom_density=2700,    # kg/m^3
        bottom_absorption=0.1,
        bottom_soundspeed=5250,
        tx_depth=100,
        frequency=freq_hz,
        nbeams=0,
        max_angle=90,
        min_angle=-90
    )
    env['rx_range'] = bathy_grid['range'].iloc[-1]
    env['rx_depth'] = np.arange(0, bathy_grid['depth_m'].iloc[-1], 50)
    pm.check_env2d(env)

def calcTL(ii, subset_df, bathy_full, drifter_lat, drifter_lon, freq_hz, ssp):
    # If the minimum distane is less than 1km we need a different function that
    
    print(f'Starting index {ii}')
    bathy_vals, path_lon, path_lat, cumulative_distance, actual_distance = extract_bathymetry_from_subset_vectorized(
        subset_df=bathy_full,
        start_lat=drifter_lat,
        start_lon=drifter_lon,
        stop_lat=subset_df['lat'].iloc[ii],
        stop_lon=subset_df['lon'].iloc[ii],
        interval=200
    )
    bathy_grid = pd.DataFrame({'range': cumulative_distance * 1000, 
                               'depth_m': -bathy_vals})
    bathy_grid.drop_duplicates(inplace=True)
    bathy_grid.sort_values('range', inplace=True)
    bathy_grid.loc[0, 'range'] = 0
    bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()
    
    # Create the enviornment
    env = pm.create_env2d(
        depth=bathy,
        soundspeed=ssp,
        bottom_density=2700,    # kg/m^3
        bottom_absorption=0.1,
        bottom_soundspeed=5250,
        tx_depth=250,
        frequency=freq_hz,
        nbeams=0,
        max_angle=90,
        min_angle=-90
    )
    
    # We need to set the receiver range to either the last location or the 
    # location nearest to the actual range (in cases where the receiver is 
    # closer than 1.1kms
    if actual_distance<1.1:
        indexer =  bathy_grid['depth_m'].index.get_indexer([actual_distance], method='nearest')
        env['rx_range'] = actual_distance*1000
        env['rx_depth'] = np.arange(0, indexer[0], 50)
    else:
        env['rx_range'] = bathy_grid['range'].iloc[-1]    
        env['rx_depth'] = np.arange(0, bathy_grid['depth_m'].iloc[-1], 50)
    
    # Check that the bathymetry doesn't have a seamount in front of it
    # Check that there isn't a seamount in the way
    minVal = bathy_grid['depth_m'].min()
    if minVal <200:
        tlosDb =np.zeros(len(env['rx_depth']))/0
    else:
        tloss = pm.compute_transmission_loss(env, mode='incoherent')
        tlosDb = 20 * np.log10(np.abs(tloss))
        print(f'done! {ii}')
    return(tlosDb)

# Worker for multiprocessing
def _worker(task):
    ii, subset_df, bathy_full, drifter_lat, drifter_lon, freq_hz, ssp = task
    
    
    # If the minimum distane is less than 1km we need a different function that
    
    print(f'Starting index {ii}')
    bathy_vals, path_lon, path_lat, cumulative_distance, actual_distance = extract_bathymetry_from_subset_vectorized(
        subset_df=bathy_full,
        start_lat=drifter_lat,
        start_lon=drifter_lon,
        stop_lat=subset_df['lat'].iloc[ii],
        stop_lon=subset_df['lon'].iloc[ii],
        interval=200
    )
    bathy_grid = pd.DataFrame({'range': cumulative_distance * 1000, 
                               'depth_m': -bathy_vals})
    bathy_grid.drop_duplicates(inplace=True)
    bathy_grid.sort_values('range', inplace=True)
    bathy_grid.loc[0, 'range'] = 0
    bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()
    
    # Create the enviornment
    env = pm.create_env2d(
        depth=bathy,
        soundspeed=ssp,
        bottom_density=2700,    # kg/m^3
        bottom_absorption=0.1,
        bottom_soundspeed=5250,
        tx_depth=250,
        frequency=freq_hz,
        nbeams=0,
        max_angle=90,
        min_angle=-90
    )
    
    # We need to set the receiver range to either the last location or the 
    # location nearest to the actual range (in cases where the receiver is 
    # closer than 1.1kms
    if actual_distance<1.1:
       indexer =  bathy_grid['depth_m'].index.get_indexer([actual_distance], method='nearest')
       env['rx_range'] = actual_distance*1000
       env['rx_depth'] = np.arange(0, indexer[0], 50)
    else:
        env['rx_range'] = bathy_grid['range'].iloc[-1]    
        env['rx_depth'] = np.arange(0, bathy_grid['depth_m'].iloc[-1], 50)
    
    # Check that the bathymetry doesn't have a seamount in front of it
    # Check that there isn't a seamount in the way
    minVal = bathy_grid['depth_m'].min()
    if minVal <200:
        tlosDb =np.zeros(len(env['rx_depth']))/0
    else:
        tloss = pm.compute_transmission_loss(env, mode='incoherent')
        tlosDb = 20 * np.log10(np.abs(tloss))
    
    # Env checking
    #pm.check_env2d(env)
    

    #print(f'done! {ii}')
    return ii, tlosDb, env['rx_depth']


from multiprocessing.pool import ThreadPool
import traceback   # optional, if you want full stack traces

def _safe_worker(args):
    """
    Run _worker(args) but never let an exception kill the pool.
    If _worker succeeds     → return ('ok',   ii, result_tuple)
    If _worker raises error → return ('fail', ii, exc)
    """
    ii = args[0]           # first element is your index
    try:
        # _worker should return (ii, tlosDb, rx_depths)
        res = _worker(args)
        return ('ok', ii, res)
    except Exception as exc:
        # Uncomment next line if you want the full traceback printed
        traceback.print_exc()
        return ('fail', ii, exc)


############################################################################
#%% Run the analysis

# Determine the number of workers on the machine. Leave 2 cpus for sanity.
nWorkers = multiprocessing.cpu_count()-3


# Load a drift
driftCTD = pd.read_csv("C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\modelling\\sg639_MHI_Apr2023_CTD.csv")

# Determine if the glider is ascending or descending
depth_diff = np.diff(driftCTD['Depth_m'], prepend=np.nan)
driftCTD['Direction'] = np.where(depth_diff > 0, 'dec', 'asc')
if depth_diff[1] > 0:
    driftCTD.at[0, 'Direction'] = 'dec'
else:
    driftCTD.at[0, 'Direction'] = 'asc'

# Define DiveID
driftCTD['DiveID'] = driftCTD['DiveNumber'].astype(str) + '_' + driftCTD['Direction']

# Bathymetry data from NCEI
nc_file = 'C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\bathymetry\\GEBCO_28_Mar_2025_ade9db365e34\\gebco_2024_n23.5_s18.5_w-160.0_e-154.0.nc'
ds = xr.open_dataset(nc_file)
latvec = ds['lat'].values
lonvec = ds['lon'].values
depth2d = ds['elevation'].values
lon_mesh, lat_mesh = np.meshgrid(lonvec, latvec)
bathymetry_df = pd.DataFrame({
    'depth': depth2d.flatten(),
    'lat': lat_mesh.flatten(),
    'lon': lon_mesh.flatten()})

freq_hz =10000


# Existing partially written HF5 file
#hf = h5py.File('Spacious_Hawaii_250m_v1.h5', 'r')
unique_ids = driftCTD['DiveID'].drop_duplicates().to_numpy()

# 2) loop from the third element onward (index 2, because Python is zero‑based)
for driftId in unique_ids[0:]:
    # 3) pull out the corresponding group “on the fly”
    group = driftCTD[driftCTD['DiveID'] == driftId]
    print(driftId)
    
    #if driftId not in hf['drift_01'].keys():
    #if driftId == '1_dec':
    if (1+1) ==2:   

        drifter_lat = group['Latitude'].iloc[0]
        drifter_lon = group['Longitude'].iloc[0]
        
      
        
        # Create the SSP profile
        profile = pd.DataFrame({
            'depth': group['Depth_m'],
            'ss':    group['SoundSpeed_m_s']
        })
        profile.sort_values('depth', inplace=True)
        profile.dropna(inplace=True)
        profile.reset_index(drop=True, inplace=True)
        profile.loc[0, 'depth'] = 0
        
        # Only use the dive if the profile depth is more than 200m
        if np.max(profile['depth'])>200:
            results = {}
            results[driftId] = []
            
            bathymetry_df['distance_km'] = haversine(
                drifter_lon, drifter_lat,
                bathymetry_df['lon'], bathymetry_df['lat']
            )
            
            # Pull out datapoints within 15k of the sensor and deeper than 20m
            subset_df = bathymetry_df[
                (bathymetry_df['distance_km'] <= 40) &
                #(bathymetry_df['distance_km'] > 1.1) &
                (bathymetry_df['depth'] < -200) 
                
            ]
            
            
            # This is where we will get the propagation data from  
            bathy_full = subset_df        # set once in main
    
            # Downsample the datapoints by 1/20th
            subset_df = subset_df[subset_df.index % 20 != 0] 
            subset_df.reset_index(drop=True, inplace=True)
            
        
            total_rows = len(subset_df)
        
            print(f'Running dive Id {driftId}  at {freq_hz} kHz')
            max_depth = np.max(np.abs(subset_df['depth']))
            last_ss = profile.iloc[-1]['ss']
            
            
            expanedProfile = pd.DataFrame(
                {'depth': np.arange(profile.iloc[-1]['depth']+1, max_depth+50, step =50),
                    'ss': np.repeat(last_ss,
                                    len(np.arange(profile.iloc[-1]['depth']+10, max_depth+50, step =50)))})
    
    
            profile = pd.concat([profile, expanedProfile])
            profile['ss'] = np.abs(profile['ss'])
            profile.sort_values('depth', inplace=True)
            ssp = profile.apply(lambda row: [row['depth'], row['ss']], axis=1).tolist()
        
        
            # Dictionary with keys 'start_lat', 'start_lon', and 'drifter_depth'.
            metadata = {'start_lat': drifter_lat,
                            'start_lon': drifter_lon,
                            'drifter_depth': 100}
            
            # for ii in np.arange(0, len(subset_df)):
            #    calcTL(ii, subset_df, bathy_full, drifter_lat, drifter_lon, freq_hz, ssp)
                
    
            # for ii in np.arange(807, len(subset_df)):        
            #     checkEnv(ii, subset_df, drifter_lat, drifter_lon, freq_hz, ssp)
            #     print(f"sucess {ii} dive {driftId}" )
    
            # Parallelize the Bellhop TL computations
            tasks = [
                (ii, subset_df, bathy_full, drifter_lat, drifter_lon, freq_hz, ssp)
                for ii in np.arange(0, len(subset_df))]
            t = time.time()
            with ThreadPool(processes=nWorkers) as pool:
                for status, ii, payload in pool.imap_unordered(_safe_worker,
                                                               tasks,
                                                               chunksize=5):
                    if status == 'fail':
                        #                         ↓ or logging.warning(...)
                        print(f"❌  error at index {ii}: {payload}")
                        continue                 # skip to next task
            
                    # success path ------------------------------------------------
                    _, tlosDb, rx_depths = payload   # unpack the worker result
                    results[driftId].append({
                        'lat':               subset_df['lat'].iloc[ii],
                        'lon':               subset_df['lon'].iloc[ii],
                        'transmission_loss': tlosDb,
                        'tl_depths':         rx_depths
                    })
                    print(f"Processed {ii} of {total_rows} points.")   
            
            elapsed = time.time() - t
            print(f'Dive {driftId} completed in {elapsed}')
            # # Inputs vectorized, run the analysis
            # with ThreadPool(processes= nWorkers) as pool:
            #     for ii, tlosDb, rx_depths in pool.imap_unordered(_worker, 
            #                                                      tasks, 
            #                                                      chunksize=5):
            #         results[driftId].append({
            #             'lat':               drifter_lat,
            #             'lon':               drifter_lon,
            #             'transmission_loss': tlosDb,
            #             'tl_depths':         rx_depths
            #         })
            #         print(f"Processed {ii} of {total_rows} points.")
        
            save_dive_frequency(
            h5_path      = "Spacious_Hawaii_100m_v0.h5",
            drift_id     = "01",
            dive_id      = driftId,
            freq_khz     = freq_hz,
            metadata     = metadata,
            grid_results = results[driftId])


