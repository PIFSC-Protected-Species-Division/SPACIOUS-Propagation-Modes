import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from geopy.distance import geodesic
from geopy.point import Point
from scipy.interpolate import griddata
import arlpy.uwapm as pm
import arlpy.plot as arlplt
import concurrent.futures

# Helper functions (unchanged from your current code)
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth's radius in kilometers
    return c * r

def calculate_initial_compass_bearing(pointA, pointB):
    lat1, lon1 = map(np.radians, pointA)
    lat2, lon2 = map(np.radians, pointB)
    diffLong = lon2 - lon1
    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(diffLong)
    initial_bearing = np.degrees(np.arctan2(x, y))
    return (initial_bearing + 360) % 360

def extract_bathymetry_from_subset(subset_df, start_lat, start_lon, stop_lat, stop_lon, interval):
    start_point = Point(start_lat, start_lon)
    stop_point = Point(stop_lat, stop_lon)
    total_distance_km = geodesic(start_point, stop_point).kilometers
    interval_km = interval / 1000.0
    num_points = max(int(total_distance_km / interval_km), 1)
    bearing = calculate_initial_compass_bearing((start_lat, start_lon), (stop_lat, stop_lon))
    path_lats = np.zeros(num_points + 1)
    path_lons = np.zeros(num_points + 1)
    for i in range(num_points + 1):
        current_distance = min(i * interval_km, total_distance_km)
        new_point = geodesic(kilometers=current_distance).destination(start_point, bearing)
        path_lats[i] = new_point.latitude
        path_lons[i] = new_point.longitude

    subset_points = subset_df[['lat', 'lon']].values
    subset_depths = subset_df['depth'].values
    path_points = np.vstack((path_lats, path_lons)).T
    bathymetry_values = griddata(subset_points, subset_depths, path_points, method='linear')
    range_km = np.array([geodesic(start_point, (path_lats[i], path_lons[i])).kilometers for i in range(len(path_lats))])
    return bathymetry_values, path_lons, path_lats, range_km

#-----------------------------------------------------------------------
# This helper function does the heavy computation for one subset grid point.
# It is designed to be called in parallel.
def process_subset_point(args):
    """
    Process one grid point (a row from subset_df) to compute transmission loss.
    args is a tuple containing:
      ii: the index (for progress printing)
      drifter_lat, drifter_lon: source coordinates.
      point_lat, point_lon: target (receiver) coordinates.
      subset_df: the entire subset DataFrame.
      ssp: the sound speed profile (list of [depth, ss]).
      interval: sampling interval in meters.
    """
    (ii, drifter_lat, drifter_lon, point_lat, point_lon, subset_df, ssp, interval) = args

    # 1. Extract the bathymetry along the path using the subset data.
    bathy_vals, path_lon, path_lat, cumulative_distance = extract_bathymetry_from_subset(
        subset_df, drifter_lat, drifter_lon, point_lat, point_lon, interval
    )
    bathy_grid = pd.DataFrame({'range': cumulative_distance*1000, 'depth_m': -bathy_vals})
    bathy_grid.drop_duplicates(inplace=True)
    bathy_grid.sort_values('range', inplace=True)
    bathy_grid.reset_index(drop=True, inplace=True)
    bathy_grid.loc[0, 'range'] = 0
    bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()
    
    # 2. Create environment for Bellhop model.
    env = pm.create_env2d(
        depth=bathy,
        soundspeed=ssp,
        bottom_density=2700,  # kg/m^3
        bottom_absorption=0.1,
        bottom_soundspeed=5250,
        tx_depth=250,
        frequency=3000,
        nbeams=0,
        max_angle=90,
        min_angle=-90
    )
    
    # 3. Set receiver locations
    env['rx_range'] = bathy_grid['range'].iloc[-1]
    env['rx_depth'] = np.arange(0, bathy_grid['depth_m'].iloc[-1], 50)
    
    # 4. Compute transmission loss.
    tloss = pm.compute_transmission_loss(env, mode='incoherent')
    tlosDb = 20 * np.log10(np.abs(tloss))
    
    # For progress, include the index in the return (optionally, can print from main loop)
    return (ii, {
        'lat': drifter_lat,
        'lon': drifter_lon,
        'transmission_loss': tlosDb,
        'tl_depths': env['rx_depth']
    })
#-----------------------------------------------------------------------
# Main pipeline code
if __name__ == "__main__":
    import sys
    import multiprocessing
    import concurrent.futures
    multiprocessing.set_executable(sys.executable)
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()  # Needed on Windows
    
    # Load a drift
    driftCTD = pd.read_csv("C:\\Users\\kaity\\Downloads\\sg639_MHI_Apr2023_CTD.csv")
    depth_diff = np.diff(driftCTD['Depth_m'], prepend=np.nan)
    driftCTD['Direction'] = np.where(depth_diff > 0, 'dec', 'asc')
    if depth_diff[1] > 0:
        driftCTD.at[0, 'Direction'] = 'dec'
    else:
        driftCTD.at[0, 'Direction'] = 'asc'
    driftCTD['DiveID'] = driftCTD['DiveNumber'].astype(str) + '_' + driftCTD['Direction']
    
    # Load bathymetry data from NCEI
    nc_file = 'C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\bathymetry\\GEBCO_28_Mar_2025_ade9db365e34\\gebco_2024_n23.5_s18.5_w-160.0_e-154.0.nc'
    ds = xr.open_dataset(nc_file)
    depth = ds['elevation'].values
    latvec = ds['lat'].values
    lonvec = ds['lon'].values
    lon_mesh, lat_mesh = np.meshgrid(lonvec, latvec)
    depth_flat = depth.flatten()
    lat_flat = lat_mesh.flatten()
    lon_flat = lon_mesh.flatten()
    
    bathymetry_df = pd.DataFrame({
        'depth': depth_flat,
        'lat': lat_flat,
        'lon': lon_flat
    })
    
    # Dictionary to hold results for each driftID
    results = {}
    
    for driftId, group in driftCTD.groupby('DiveID'):
        drifter_lat = group['Latitude'].iloc[0]
        drifter_lon = group['Longitude'].iloc[0]
        
        # Compute distances from drifter location and filter subset
        bathymetry_df['distance_km'] = haversine(drifter_lon, drifter_lat,
                                                 bathymetry_df['lon'], bathymetry_df['lat'])
        subset_df = bathymetry_df[bathymetry_df['distance_km'] <= 10]
        total_rows = len(subset_df)
        results[driftId] = []
        
        # Build the sound speed profile (ssp) from your CTD data.
        profile = pd.DataFrame({
            'depth': group['Depth_m'][group['Direction'] == 'asc'],
            'ss': group['SoundSpeed_m_s'][group['Direction'] == 'asc']
        })
        profile.sort_values('depth', inplace=True)
        profile.dropna(axis=0, inplace=True)
        profile.reset_index(drop=True, inplace=True)
        profile.loc[0, 'depth'] = 0
        max_depth = np.max(np.abs(subset_df['depth']))
        last_ss = profile.iloc[-1]['ss']
        profile.loc[profile.index.max() + 1] = [max_depth, last_ss]
        profile['ss'] = np.abs(profile['ss'])
        ssp = profile.apply(lambda row: [row['depth'], row['ss']], axis=1).tolist()
        
        # Prepare arguments for each subset point.
        # For each row in subset_df, we want to compute the TL from the drifter
        # location to that subset point.
        args_list = []
        for ii in range(total_rows):
            point_lat = subset_df['lat'].iloc[ii]
            point_lon = subset_df['lon'].iloc[ii]
            args = (ii, drifter_lat, drifter_lon, point_lat, point_lon, subset_df, ssp, 200)
            args_list.append(args)
        
        # Use ProcessPoolExecutor to run the slow function in parallel.
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for i, result_data in executor.map(process_subset_point, args_list):
                results[driftId].append(result_data)
                print(f"DriftID {driftId}: Processed {i+1} of {total_rows} points.")
    
    # Now, results is a dictionary keyed by driftID.
    # Each value is a list of dictionaries with keys: 'lat', 'lon', 'transmission_loss', and 'tl_depths'.
    # You can now save or further process the results as needed.
