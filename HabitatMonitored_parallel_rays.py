# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 20:24:37 2025

@author: kaity
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"   # or 1, 8 … anything ≤ 24
import numpy as np
from geopy.distance import geodesic
from geopy.point import Point
import matplotlib.pyplot as plt
#from geopy.point import Point
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
import arlpy.uwapm as pm
#import arlpy.plot as arlplt
#import matplotlib.tri as tri
from pyproj import Geod
import h5py
import multiprocessing
#from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.pool import ThreadPool
import traceback   # optional, if you want full stack traces
import pandas as pd
from pyproj import Geod
import time


# ------------------------------------------------------------------ #
# Globals shared by all threads (ThreadPool → threads share memory)
# ------------------------------------------------------------------ #
bathy_full = None        # set once in main
geod       = Geod(ellps='WGS84')

def variable_rx_range(max_km, s0_km=0.2, s1_km=0.1, target_km=40):
    """
    s0_km- spacing at source (km)
    s1_km-spacing at maximum range (km)
    target_km - maximum distance 
    
    Returns a 1-D numpy array of hydrophone ranges (km) in which
      • spacing is 0.5 km at r = 0 km,
      • spacing shrinks linearly to 0.1 km at r = 10 km,
      • spacing stays 0.1 km beyond 10 km (if max_km > 10).
    """
    k = (s0_km - s1_km) / target_km       # slope of spacing-vs-range line
    a = 1.0 - k                            # recurrence factor (= 0.96 here)

    # ---------- inner segment: 0 km → min(target_km, max_km) -------------
    inner_end = min(target_km, max_km)
    # How many points are needed to reach that distance?
    n_inner = int(np.ceil(np.log(max(1e-12, 1 - inner_end * k / s0_km))
                          / np.log(a)))
    r_inner = (s0_km / k) * (1 - a**np.arange(n_inner + 1))
    r_inner = r_inner[r_inner <= inner_end]      # clip any tiny overshoot

    # ---------- outer tail: constant 0.1 km spacing ----------------------
    if max_km <= target_km:
        return r_inner

    tail = np.arange(r_inner[-1] + s1_km, max_km + s1_km * 0.5, s1_km)
    tail = tail[tail <= max_km]

    return np.concatenate((r_inner, tail))


def save_dive_frequency(
        h5_path: str,
        drift_id: str,
        dive_id: str,
        freq_khz: int,
        metadata: dict,
        grid_results: list,      # one dict per bearing / angle
        gzip_level: int = 4):
    """
    ------------------------------------------------------------------
    Expected structure of *grid_results* (one element per bearing):

        {
            'angle_deg'        : float                        # bearing or azimuth
            'rx_depths'        : 1-D np.ndarray (Nzᵢ,)        # positive-down (m)
            'rx_ranges'        : 1-D np.ndarray (Nrᵢ,)        # positive-out  (m)
            'transmission_loss': 2-D np.ndarray (Nzᵢ, Nrᵢ)    # TL(dB)
        }

    Nzᵢ and Nrᵢ can differ from one angle to the next because you trimmed
    rays behind atolls.  Everything is padded with NaNs to the **maximum
    depth and range** so we can keep a single 3-D HDF5 dataset:
        tl(angle, depth, range)
    ------------------------------------------------------------------
    """
    # -------------------------------------------------- 1.  derive global sizes
    n_ang   = len(grid_results)
    max_Nz  = max(g['depth_m'].size  for g in grid_results)
    max_Nr  = max(g['range_m'].size  for g in grid_results)

    # -------------------------------------------------- 2.  pre-allocate
    angle_deg       = np.empty(n_ang,               dtype=np.float32)
    valid_Nz        = np.empty(n_ang,               dtype=np.uint16)
    valid_Nr        = np.empty(n_ang,               dtype=np.uint16)

    depth_mat       = np.full((n_ang, max_Nz),      np.nan, dtype=np.float32)
    range_mat       = np.full((n_ang, max_Nr),      np.nan, dtype=np.float32)
    tl_mat          = np.full((n_ang, max_Nz, max_Nr),
                              np.nan, dtype=np.float32)

    # -------------------------------------------------- 3.  fill in the data
    for i, g in enumerate(grid_results):
        d, r, tl = g['depth_m'], g['range_m'], g['transmission_loss']

        assert tl.shape == (d.size, r.size), \
            f"TL grid shape {tl.shape} != ({d.size}, {r.size})"

        angle_deg[i]      = g['angle_deg']
        valid_Nz[i]       = d.size
        valid_Nr[i]       = r.size

        depth_mat[i, :d.size]               = d
        range_mat[i, :r.size]               = r
        tl_mat  [i, :d.size, :r.size]       = np.round(tl, 2)

    # -------------------------------------------------- 4.  write / overwrite HDF5
    if not os.path.exists(h5_path):
        print(f"[save_dive_frequency] creating {h5_path}")

    with h5py.File(h5_path, "a") as hf:
        fgrp = (
            hf.require_group(f"drift_{drift_id}")
              .require_group(f"dive_{dive_id}")
              .require_group(f"frequency_{freq_khz}")
        )

        # ---------- dive-level metadata -------------
        for k, v in metadata.items():
            fgrp.parent.attrs[k] = v

        # ---------- helper to replace or create -----
        def _dset(name, data, **kw):
            if name in fgrp:
                del fgrp[name]
            fgrp.create_dataset(
                name,
                data=data,
                compression="gzip",
                compression_opts=gzip_level,
                **kw)

        _dset("angle_deg", angle_deg)
        _dset("valid_depth_len", valid_Nz)
        _dset("valid_range_len", valid_Nr)
        _dset("depth_m", depth_mat,
              chunks=(min(32, n_ang), max_Nz))
        _dset("range_m", range_mat,
              chunks=(min(32, n_ang), max_Nr))

        # choose manageable 3-D chunk sizes
        row_chunk   = min(16, n_ang)
        depth_chunk = min(64, max_Nz)
        range_chunk = min(256, max_Nr)

        _dset("transmission_loss", tl_mat,
              chunks=(row_chunk, depth_chunk, range_chunk))

    print(f"[save_dive_frequency] wrote TL for {n_ang} angles "
          f"(Nz ≤ {max_Nz}, Nr ≤ {max_Nr}) → {h5_path}")


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

def checkEnv(ii, subset_df, drifter_lat, drifter_lon, freq_hz, ssp):
    '''Function to check that all of the enviornmental parameters are set up
    correctly
    '''
    bathy_vals, path_lon, path_lat, cumulative_distance = extract_bathymetry_from_subset_vectorized(
        subset_df=subset_df,
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
        tx_depth=100,
        frequency=freq_hz,
        nbeams=0,
        max_angle=90,
        min_angle=-90
    )
    env['rx_range'] = bathy_grid['range'].iloc[-1]
    env['rx_depth'] = np.arange(0, bathy_grid['depth_m'].iloc[-1], 50)
    #pm.check_env2d(env)

def calcTL(subset_df, drifter_lat, drifter_lon, freq_hz, bearing_deg,
           interval, hydVertSpacing, hydHorzSpacing, max_distance_km, ssp):
    ''' In line version of transmission loss calculation for 
    testing and evaluation'''
    
    
    
    bath_vals, lons, lats, r_km = extract_bathymetry_along_ray_vectorized(
    subset_df     = subset_df,   # your pre-filtered bathy points
    start_lat     = drifter_lat,
    start_lon     = drifter_lon,
    bearing_deg   = bearing_deg,            # NE-ish
    max_distance_km = max_distance_km,          # 20 km ray
    interval      = interval)            # sample every 50 m)
    
    
    bathy_grid = pd.DataFrame({'range': r_km * 1000, 
                               'depth_m': -bath_vals})
    
    bathy_grid.drop_duplicates(inplace=True)
    bathy_grid.sort_values('range', inplace=True)
    bathy_grid.loc[0, 'range'] = 0
    #plt.plot(bathy_grid['range'], -bathy_grid['depth_m'])
    
    # Trim the bathymetry grid to the first location that is less than 100m
    lastRow = np.where(bathy_grid['depth_m']<100)
    
    if lastRow[0].size>0:
        lastRow = np.min(lastRow)
        bathy_grid = bathy_grid.iloc[:lastRow]
        #plt.plot(bathy_grid['range'], -bathy_grid['depth_m'])

    
    bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()
    
    # Check if the bearing line crosses an atol
   

    # Create the enviornment -770 seconds to run 40km enviornment file, 77 hours
    # per dive. 177 seconds for 15km, 17 hours per dive.
    
    # If the recievers are only at 40km 181 sec. 25156 rows within 40km 1
    # 1194.91 hours per dive
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
    
    
    # Set hydrophone range, linearly decreasing space and depth linearly spaced
    env['rx_range'] = variable_rx_range((bathy_grid['range'].iloc[-1])/1000)*1000
    env['rx_depth'] = np.arange(0, np.max(bathy_grid['depth_m']), hydVertSpacing)
    
    # Check the enviornment and set up hydrophones
    #pm.check_env2d(env)
    #pm.plot_ssp(env)
    
    t = time.time()
    tloss = pm.compute_transmission_loss(env, mode='coherent')
    elapsed = time.time() - t
    tlosDb = 20 * np.log10(np.abs(tloss)+.01)
    #pm.plot_transmission_loss(tloss, env=env, clim=[-90,-30], width=900)
    
   
    
    # ------------------------------------------
    # 1.  Interpolate seabed to the hydrophone ranges
    #     – convert km → m so units match
    # ------------------------------------------

    rx_range_km  = env['rx_range']           # length = 369
    rx_depth_m   = env['rx_depth']           # length = 34   (positive-down)
    seabed_depth_neg = np.interp(env['rx_range'],
                                 bathy_grid['range'].to_numpy(),     # x-coordinate (m)
                                 bathy_grid['depth_m'].to_numpy(),   # y-coordinate (-m)
                                 left=np.nan, right=np.nan)        # NaN if out of bounds
    

    # ------------------------------------------
    # 2.  Build a depth-vs-range mask
    # ------------------------------------------
    mask = rx_depth_m[:, None] > seabed_depth_neg[None, :]
    
    # Mask the first column
    mask[:,0]=True
    
    # ------------------------------------------
    # 3.  Apply the mask (choose your favourite style)
    # ------------------------------------------
    tloss_masked = np.where(mask, np.nan, tlosDb)      # NaN-fill
    tloss_masked = pd.DataFrame(tloss_masked)
    #pm.plot_transmission_loss(10**(tloss_masked/20), env=env, clim=[-60,-30], width=900)
   
 
    print(f'done! {ii}')
    return(tloss_masked,env['rx_range'],  env['rx_depth'])

def extract_bathymetry_along_ray_vectorized(
    subset_df: pd.DataFrame,
    start_lat: float,
    start_lon: float,
    bearing_deg: float,
    max_distance_km: float,
    interval: float = 100.0):
    """
    Sample bathymetry values along a ray that starts at (start_lat, start_lon),
    follows `bearing_deg` degrees, and extends `max_distance_km`.

    Parameters
    ----------
    subset_df : pd.DataFrame
        Must contain columns 'lat', 'lon', and 'depth'.
    start_lat, start_lon : float
        Ray origin in decimal degrees (WGS-84).
    bearing_deg : float
        Compass bearing *clockwise from true north* (0° = due N, 90° = due E).
    max_distance_km : float
        Total length of the ray in kilometres.
    interval : float, default 100.0
        Spacing between successive bathymetry samples **in metres**.

    Returns
    -------
    bathymetry_values : np.ndarray
        Interpolated (or nearest-filled) seabed depths along the ray.
    lons, lats : np.ndarray
        Longitude and latitude of each sampled point.
    range_km : np.ndarray
        1-D array of cumulative ground ranges (km) from the start point.
        
    Example usage
    -------
    bath_vals, lons, lats, r_km = extract_bathymetry_along_ray_vectorized(
    subset_df     = bathymetry_df,   # your pre-filtered bathy points
    start_lat     = 50.123,
    start_lon     = -128.456,
    bearing_deg   = 42.0,            # NE-ish
    max_distance_km = 20.0,          # 20 km ray
    interval      = 50.0             # sample every 50 m)
    """
    # ------------------------------------------------ 1. RANGE VECTOR
    interval_km = interval / 1_000.0
    num_points  = max(int(max_distance_km / interval_km), 1)

    # Distances along the ray, in **metres**
    distances_m = np.linspace(0.0, max_distance_km * 1_000.0, num_points + 1)

    # ------------------------------------------------ 2. FORWARD GEODESIC
    lons, lats, _ = geod.fwd(
        np.full_like(distances_m, start_lon),
        np.full_like(distances_m, start_lat),
        np.full_like(distances_m, bearing_deg),
        distances_m,
    )

    # ------------------------------------------------ 3. CUMULATIVE RANGE (km)
    start_point = Point(start_lat, start_lon)
    range_km = np.array(
        [geodesic(start_point, (lat, lon)).kilometers for lat, lon in zip(lats, lons)]
    )

    # ------------------------------------------------ 4. INTERPOLATE BATHYMETRY
    subset_points  = subset_df[["lat", "lon"]].values
    subset_depths  = subset_df["depth"].values
    path_points    = np.column_stack((lats, lons))

    bathymetry_values = griddata(
        subset_points, subset_depths, path_points, method="linear"
    )

    # Fallback to nearest neighbour where linear interpolation fails
    if np.any(np.isnan(bathymetry_values)):
        nan_mask = np.isnan(bathymetry_values)
        bathymetry_values[nan_mask] = griddata(
            subset_points,
            subset_depths,
            path_points[nan_mask],
            method="nearest",
        )

    return bathymetry_values, lons, lats, range_km

# Worker for multiprocessing
def _worker(args):
    (angle_deg,              # <- was “ii”, keeps the bearing / task-ID
     subset_df,
     drifter_lat,
     drifter_lon,
     freq_hz,
     ssp,
     tx_depth,
     max_distance_km,
     interval_m,
     hydVertSpacing) = args
    

        
    bath_vals, lons, lats, r_km = extract_bathymetry_along_ray_vectorized(
        subset_df       = subset_df,
        start_lat       = drifter_lat,
        start_lon       = drifter_lon,
        bearing_deg     = angle_deg,
        max_distance_km = max_distance_km,
        interval        = interval_m)
     
    bathy_grid = pd.DataFrame({'range': r_km * 1000, 
                                'depth_m': -bath_vals})
     
    bathy_grid.drop_duplicates(inplace=True)
    bathy_grid.sort_values('range', inplace=True)
    bathy_grid.loc[0, 'range'] = 0
    #plt.plot(bathy_grid['range'], -bathy_grid['depth_m'])
     
    # Trim the bathymetry grid to the first location that is less than 100m
    lastRow = np.where(bathy_grid['depth_m']<100)
     
    if lastRow[0].size>0:
         lastRow = np.min(lastRow)
         bathy_grid = bathy_grid.iloc[:lastRow]
         #plt.plot(bathy_grid['range'], -bathy_grid['depth_m'])
    
     
    bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()
     
    # Check if the bearing line crosses an atol
    
    
    # Create the enviornment -770 seconds to run 40km enviornment file, 77 hours
    # per dive. 177 seconds for 15km, 17 hours per dive.
     
    # If the recievers are only at 40km 181 sec. 25156 rows within 40km 1
    # 1194.91 hours per dive
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
     
     
    # Set hydrophone range, linearly decreasing space and depth linearly spaced
    env['rx_range'] = variable_rx_range((bathy_grid['range'].iloc[-1])/1000)*1000
    env['rx_depth'] = np.arange(0, np.max(bathy_grid['depth_m']), hydVertSpacing)
     
    # Check the enviornment and set up hydrophones
    #pm.check_env2d(env)
    #pm.plot_ssp(env)
     
    t = time.time()
    tloss = pm.compute_transmission_loss(env, mode='coherent')
    elapsed = time.time() - t
    tlosDb = 20 * np.log10(np.abs(tloss)+.01)
    #pm.plot_transmission_loss(tloss, env=env, clim=[-90,-30], width=900)
     
    
     
    # ------------------------------------------
    # 1.  Interpolate seabed to the hydrophone ranges
    #     – convert km → m so units match
    # ------------------------------------------
   
    rx_range_km  = env['rx_range']           # length = 369
    rx_depth_m   = env['rx_depth']           # length = 34   (positive-down)
    seabed_depth_neg = np.interp(env['rx_range'],
                                 bathy_grid['range'].to_numpy(),     # x-coordinate (m)
                                 bathy_grid['depth_m'].to_numpy(),   # y-coordinate (-m)
                                 left=np.nan, right=np.nan)        # NaN if out of bounds
    
   
    # ------------------------------------------
    # 2.  Build a depth-vs-range mask
    # ------------------------------------------
    mask = rx_depth_m[:, None] > seabed_depth_neg[None, :]
    
    # Mask the first column
    mask[:,0]=True
    
    # ------------------------------------------
    # 3.  Apply the mask (choose your favourite style)
    # ------------------------------------------
    tloss_masked = np.where(mask, np.nan, tlosDb)      # NaN-fill
    tloss_masked = pd.DataFrame(tloss_masked)
    #pm.plot_transmission_loss(10**(tloss_masked/20), env=env, clim=[-60,-30], width=900)
   
   
    print(f'done! angle {angle_deg}')
    return (angle_deg, tloss_masked,rx_range_km,rx_depth_m )


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

if __name__ == "__main__":
    # Determine the number of workers on the machine. Leave 2 cpus for sanity.
    nWorkers = multiprocessing.cpu_count()-3
    
    maxRangekm = 40
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
    bathy_full = pd.DataFrame({
        'depth': depth2d.flatten(),
        'lat': lat_mesh.flatten(),
        'lon': lon_mesh.flatten()})
    
    
    freq_hz =10000
    
    
    # Existing partially written HF5 file
    unique_ids = driftCTD['DiveID'].drop_duplicates().to_numpy()
    
    
    # 2) loop from the third element onward (index 2, because Python is zero‑based)
    for driftId in unique_ids[0:16]:
        # 3) pull out the corresponding group “on the fly”
        group = driftCTD[driftCTD['DiveID'] == driftId]
        print(driftId)
        
        #if driftId not in hf['drift_01'].keys():
        #if driftId == '1_dec':
        if (1+1) ==2:   
    
            drifter_lat = group['Latitude'].iloc[0]
            drifter_lon = group['Longitude'].iloc[0]
            
           
            # Create a distance metric
            bathy_full['distance_km'] = haversine(
                drifter_lon, drifter_lat,
                bathy_full['lon'], 
                bathy_full['lat']
            )
            
            # Pull out datapoints within 15k of the sensor 
            subset_df = bathy_full[
                (bathy_full['distance_km'] <= 40)]
            
            # # Downsample the datapoints by 1/3
            # subset_df = subset_df[subset_df.index % 3 != 0]
            # total_rows = len(subset_df)
        
    
            # Create the SSP profile
            profile = pd.DataFrame({
                'depth': group['Depth_m'],
                'ss':    group['SoundSpeed_m_s'] })
            ()
            profile.sort_values('depth', inplace=True)
            profile.dropna(inplace=True)
            profile.reset_index(drop=True, inplace=True)
            profile.loc[0, 'depth'] = 0
            
            # Only use the dive if the profile depth is more than 200m
            if np.max(profile['depth'])>200:
                results = {}
                results[driftId] = []
    
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
                txDepth = metadata['drifter_depth']
        

                max_distance_km = 40 
                hydVertSpacing = 75
                hydHorzSpacing = 100
                interval = 100 #bathymetry spacing
        
                # # For each bearing angle 
                # for ii in np.arange(90,359, step =90):
                #     print(f'starting angle {ii}')
                #     tlGrid, range_m, depth_m = calcTL(subset_df, drifter_lat, drifter_lon, freq_hz, 
                #            ii, interval, hydVertSpacing, 
                #            hydHorzSpacing, max_distance_km, ssp)
                    
                #     results[driftId].append({
                #         'range_m':              range_m,
                #         'depth_m':              depth_m,
                #         'transmission_loss':    tlGrid,
                #         'angle':ii
                #     })
                # print(f"Processed {ii} of 360 {driftId}")

                nSteps = 360
                # Parallelize the Bellhop TL computations
                tasks = [
                    (angle, subset_df, drifter_lat, drifter_lon, freq_hz, ssp, txDepth,
                     maxRangekm, interval, hydVertSpacing)
                    for angle in np.arange(0, 359, step = 1)]
                                
                with ThreadPool(processes=nWorkers) as pool:
                    for status, ii, payload in pool.imap_unordered(_safe_worker,
                                                                   tasks,
                                                                   chunksize=5):
                        if status == 'fail':
                            #                         ↓ or logging.warning(...)
                            print(f"❌  error at index {ii}: {payload}")
                            continue                 # skip to next task
                
                        # success path ------------------------------------------------
                        angle_deg, tloss_masked,rx_range_km,rx_depth_m = payload   # unpack the worker result
                        results[driftId].append({
                            'angle_deg'        : angle_deg,   # or whatever variable you use
                            'range_m'          : rx_range_km,    # 1-D np.ndarray
                            'depth_m'          : rx_depth_m,   # 1-D np.ndarray
                            'transmission_loss': tloss_masked      # 2-D array (depth × range)
                        })
                        print(f"Processed {ii} of {nSteps} rays.")    

            
                save_dive_frequency(
                 h5_path      = "Spacious_Hawaii_100m_rays.h5",
                 drift_id     = "01",
                 dive_id      = driftId,
                 freq_khz     = freq_hz,
                 metadata     = metadata,
                 grid_results = results[driftId])

