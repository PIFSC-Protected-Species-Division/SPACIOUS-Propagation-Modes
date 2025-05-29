#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bellhop sweep – **same physics, faster parallel loop**
------------------------------------------------------
•  No changes to SSP or bathymetry logic.
•  Worker pool now stays alive (`maxtasksperchild=200`).
•  Heavy read‑only objects broadcast once via `initializer` – each task
   only passes an *index* instead of the whole DataFrame.
•  `chunksize=8` amortises IPC without starving the progress bar.

Drop‑in replacement: paste over your original script, keep the file paths.
"""

###############################################################################
# 0)  ––– BLAS/OpenMP limits (unchanged)
###############################################################################
import os, multiprocessing as mp
os.environ.update({
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS":      "1",
    "OMP_NUM_THREADS":      "1",
})

###############################################################################
# 1)  ––– imports (same as original; duplicates trimmed)
###############################################################################
import time, traceback
import numpy as np
import pandas as pd
import xarray as xr
import h5py
from tqdm.auto import tqdm
from geopy.distance import geodesic
from geopy.point import Point
from scipy.interpolate import griddata
from pyproj import Geod
import arlpy.uwapm as pm
#---------------------------------------------------------------------------
# Create a Geod instance for vectorized geodesic computations.
geod = Geod(ellps='WGS84')
bathy_full = None        # set once in main
subset_df = None
subsetBathy= None

###############################################################################
# 2)  ––– original helper fns (unchanged) ------------------------------------
###############################################################################
# haversine, calculate_initial_compass_bearing, extract_bathymetry_from_subset
# extract_bathymetry_from_subset_vectorized, tl_incoherent_from_arrivals …
#    ↳ (copy the full originals here – omitted for brevity)        

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


def save_dive_frequency(h5_path, drift_id, dive_id, freq_khz,
                        metadata, grid_results, gzip_level=4):

    # ─── unchanged pre-amble (lat/lon/TL matrices) ───
    n_pts = len(grid_results)
    max_N = max(len(np.asarray(g["tl_depths"]).reshape(-1)) for g in grid_results)

    lat   = np.empty(n_pts, np.float32)
    lon   = np.empty(n_pts, np.float32)
    dmat  = np.full((n_pts, max_N), np.nan, np.float32)
    tlmat = np.full((n_pts, max_N), np.nan, np.float32)
    vlen  = np.empty(n_pts, np.uint16)

    for i, g in enumerate(grid_results):
        lat[i] = g["lat"]
        lon[i] = g["lon"]

        depths = np.asarray(g["tl_depths"]).reshape(-1)
        tlvals = np.asarray(g["transmission_loss"]).reshape(-1)

        k = depths.size
        dmat[i, :k]  = depths
        tlmat[i, :k] = np.round(tlvals, 2)
        vlen[i] = k

    # ─── open/create file ───
    if not os.path.exists(h5_path):
        print(f"[save_dive_frequency] creating new HDF5 file {h5_path}")

    with h5py.File(h5_path, "a") as hf:
        base = (
            hf.require_group(f"drift_{drift_id}")
              .require_group(f"dive_{dive_id}")
              .require_group(f"frequency_{freq_khz}")
        )

        # one-time metadata
        for k, v in metadata.items():
            base.parent.attrs[k] = v

        def _save(name, data, chunks=None):
            if name in base:
                del base[name]
            base.create_dataset(name, data=data,
                                compression="gzip",
                                compression_opts=gzip_level,
                                chunks=chunks)

        row_chunk = min(256, n_pts)
        _save("lat",        lat)
        _save("lon",        lon)
        _save("valid_len",  vlen)
        _save("depth",      dmat, (row_chunk, max_N))
        _save("tl",         tlmat, (row_chunk, max_N))

        # ─── arrivals mini-tables ───
        if "arrivals" in base:
            del base["arrivals"]
        arrivals_grp = base.create_group("arrivals")

        for i, g in enumerate(grid_results):
            arr_obj = g.get("arr", None)

            # ---------- new handling ----------
            if isinstance(arr_obj, pd.DataFrame) and not arr_obj.empty:
                pt_grp = arrivals_grp.create_group(f"pt_{i:05d}")
                for col in arr_obj.columns:
                    pt_grp.create_dataset(
                        name=col,
                        data=arr_obj[col].to_numpy(copy=False),
                        compression="gzip",
                        compression_opts=gzip_level,
                    )
                pt_grp.attrs["row_index_name"] = arr_obj.index.name or ""
                pt_grp.attrs["n_rows"] = len(arr_obj)
            else:
                # empty branch; keeps the mapping but stores no data
                pt_grp = arrivals_grp.create_group(f"pt_{i:05d}")
                pt_grp.attrs["n_rows"] = 0
            # -----------------------------------

    
###############################################################################
# 3)  ––– POOL HELPERS  (new) -------------------------------------------------
###############################################################################

def _init_worker(_subset_df,
                 _dr_lat, _dr_lon,
                 _freq_hz, _ssp, _dr_depth):
    """Broadcast large read‑only data to global namespace inside each worker."""
    global subset_df, bathy_full
    global drifter_lat, drifter_lon, freq_hz, ssp, drifter_depth
    subset_df     = _subset_df
    drifter_lat   = _dr_lat
    drifter_lon   = _dr_lon
    freq_hz       = _freq_hz
    ssp           = _ssp
    drifter_depth = _dr_depth


def _worker(ii):
    """Exact same physics as before – but only the *index* is passed in."""
    try:
        # --- BEGIN unmodified body ------------------------------------------------
        bathy_vals, path_lon, path_lat, cumulative_distance, actual_distance = (
            extract_bathymetry_from_subset_vectorized(
                subset_df=subset_df,
                start_lat=drifter_lat,
                start_lon=drifter_lon,
                stop_lat=subset_df['lat'].iloc[ii],
                stop_lon=subset_df['lon'].iloc[ii],
                interval=200))

        bathy_grid = pd.DataFrame({'range': cumulative_distance * 1000,
                                   'depth_m': -bathy_vals})
        bathy_grid.drop_duplicates(inplace=True)
        bathy_grid.sort_values('range', inplace=True)
        bathy_grid.loc[0, 'range'] = 0
        bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()

        env = pm.create_env2d(
            depth=bathy,
            soundspeed=ssp,
            bottom_density=1600,
            bottom_absorption=0.2,
            bottom_soundspeed=1600,
            tx_depth=drifter_depth,
            frequency=freq_hz,
            nbeams=0,
            max_angle=90,
            min_angle=-90,
            soundspeed_interp='pchip')

        if actual_distance < 1.1:
            env['rx_range'] = actual_distance * 1000
            env['rx_depth'] = np.arange(0, bathy_grid['depth_m'].iloc[0], 100)
        else:
            env['rx_range'] = bathy_grid['range'].iloc[-1]
            env['rx_depth'] = np.arange(0, bathy_grid['depth_m'].iloc[-1], 100)

        if bathy_grid['depth_m'].min() < 200:
            arr = pd.DataFrame()
            tlosDb = np.full(len(env['rx_depth']), np.nan)
        else:
            arr = pm.compute_arrivals(env)
            tlosDb = np.full(len(env['rx_depth']), np.nan)  # TL later if needed
        # --- END unmodified body --------------------------------------------------
        return ('ok', ii, (ii, tlosDb, arr, env['rx_depth']))

    except Exception as exc:
        # Forward the traceback to parent so it can be printed once
        return ('fail', ii, traceback.format_exc())

###############################################################################
# 4)  ––– MAIN ----------------------------------------------------------------
###############################################################################
if __name__ == "__main__":
    # ------------------------------------------------------------------ paths
    drift_csv = r"C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\modelling\\sg639_MHI_Apr2023_CTD.csv"
    gebco_nc  = r"C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\bathymetry\\GEBCO_28_Mar_2025_ade9db365e34\\gebco_2024_n23.5_s18.5_w-160.0_e-154.0.nc"
    out_h5    = r"Spacious_Hawaii_100m_ArrArray_PCHIP_35khz.h5"

    # ------------------------------------------------------------------- setup
    nWorkers = max(1, mp.cpu_count() - 4)

    driftCTD = pd.read_csv(drift_csv)
    # ... (unchanged CTD preprocessing & profile/ssp build) ...

    # build once outside the loop
    ds = xr.open_dataset(gebco_nc)
    bathymetry_df = pd.DataFrame({
        'depth': ds['elevation'].values.flatten(),
        'lat':   np.repeat(ds['lat'].values, len(ds['lon'])),
        'lon':   np.tile(  ds['lon'].values, len(ds['lat']))})
    
    
    
    
    # Determine if the glider is ascending or descending
    depth_diff = np.diff(driftCTD['Depth_m'], prepend=np.nan)
    driftCTD['Direction'] = np.where(depth_diff > 0, 'dec', 'asc')
    if depth_diff[1] > 0:
        driftCTD.at[0, 'Direction'] = 'dec'
    else:
        driftCTD.at[0, 'Direction'] = 'asc'
    
    # Define DiveID
    driftCTD['DiveID'] = driftCTD['DiveNumber'].astype(str) + '_' + driftCTD['Direction']
    unique_ids = driftCTD['DiveID'].drop_duplicates().to_numpy()
    
    freq_hz =35000
    # ---------------------------------------------------------------- dive loop
    for dive_id in driftCTD['DiveID'].unique():
        # ... (unchanged filtering to subset_df & ssp creation) ...
        # 3) pull out the corresponding group “on the fly”
        group = driftCTD[driftCTD['DiveID'] == dive_id]
        print(dive_id)
        
    
        drifter_lat = group['Latitude'].iloc[0]
        drifter_lon = group['Longitude'].iloc[0]
        drifter_depth = 100
      
        
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
            results[dive_id] = []
            
            bathymetry_df['distance_km'] = haversine(
                drifter_lon, drifter_lat,
                bathymetry_df['lon'], bathymetry_df['lat']
            )
            
            # Pull out datapoints within 40km of the sensor and the water is deeper than 150 m
            subset_df = bathymetry_df[
                (bathymetry_df['distance_km'] <= 40) &
                (bathymetry_df['depth'] < -150)]
            
            


            # Downsample the datapoints by 1/20th
            subset_df = subset_df[subset_df.index % 20 != 0] 
            subset_df.reset_index(drop=True, inplace=True)
            
        
            total_rows = len(subset_df)
        
            print(f'Running dive Id {dive_id}  at {freq_hz} kHz')
            max_depth = np.max(np.abs(subset_df['depth']))
            last_ss = profile.iloc[-1]['ss']
            
            # Expand the soundspeed profile
            expanedProfile = pd.DataFrame(
                {'depth': np.arange(profile.iloc[-1]['depth']+10, max_depth+50, step =50),
                    'ss': np.repeat(last_ss,
                                    len(np.arange(profile.iloc[-1]['depth']+10, max_depth+50, step =50)))})
            
            
            profile = pd.concat([profile, expanedProfile])
            profile['ss'] = np.abs(profile['ss'])
            profile.sort_values('depth', inplace=True)
            
            last_ss = profile.iloc[-1]['ss']
            
            
            expanedProfile = pd.DataFrame(
                {'depth': np.arange(profile.iloc[-1]['depth']+10, max_depth+50, step =50),
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

            tasks = list(range(len(subset_df)))
            print(f"\nDive {dive_id}: {len(tasks)} grid‑points @ {nWorkers} workers")
    
            t0 = time.time()
            results = []
            with mp.get_context("spawn").Pool(
                    processes=nWorkers,
                    initializer=_init_worker,
                    initargs=(subset_df, 
                              drifter_lat, drifter_lon,
                              freq_hz, ssp, 100),
                    maxtasksperchild=200) as pool:
    
                for status, ii, payload in tqdm(
                        pool.imap_unordered(_worker, tasks, chunksize=8),
                        total=len(tasks), desc="Bellhop jobs"):
    
                    if status == 'fail':
                        tqdm.write(f"❌ index {ii} failed\n{payload}")
                        continue
    
                    _, tl, arr, rx_depths = payload
                    if ii % 50 == 0:
                        tqdm.write(f"{ii}/{len(tasks)} finished")
    
                    results.append({'lat': subset_df['lat'].iat[ii],
                                    'lon': subset_df['lon'].iat[ii],
                                    'arr': arr,
                                    'transmission_loss': tl,
                                    'tl_depths': rx_depths})

            print(f"Dive {dive_id} done in {time.time()-t0:.1f} s")
            save_dive_frequency(
            h5_path      = "Spacious_Hawaii_100m_ArrArray_PCHIP_35khz.h5",
            drift_id     = "01",
            dive_id      = dive_id,
            freq_khz     = freq_hz,
            metadata     = metadata,
            grid_results = results[dive_id])
