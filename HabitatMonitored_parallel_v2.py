import sys
import multiprocessing
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from geopy.distance import geodesic
from geopy.point import Point
from scipy.interpolate import griddata
from pyproj import Geod

import arlpy.uwapm as pm
import arlpy.plot as arlplt

# =============================================================================
# Global Data for Workers
# =============================================================================
# These globals are set once per worker using the initializer.
global_subset_df = None
global_ssp = None

# Create a Geod instance for vectorized geodesic computations.
geod = Geod(ellps='WGS84')

# =============================================================================
# Utility Functions
# =============================================================================
def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in kilometers)
    using the haversine formula.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth's radius in kilometers
    return c * r


def calculate_initial_compass_bearing(pointA: tuple, pointB: tuple) -> float:
    """
    Calculate the initial bearing (in degrees) from pointA to pointB.
    Points are provided as tuples in the form (latitude, longitude).
    """
    lat1, lon1 = map(np.radians, pointA)
    lat2, lon2 = map(np.radians, pointB)
    diffLong = lon2 - lon1
    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(diffLong)
    initial_bearing = np.degrees(np.arctan2(x, y))
    return (initial_bearing + 360) % 360


def extract_bathymetry_from_subset_vectorized(
    subset_df: pd.DataFrame,
    start_lat: float,
    start_lon: float,
    stop_lat: float,
    stop_lon: float,
    interval: float
):
    """
    Compute bathymetry along the path from (start_lat, start_lon) to (stop_lat, stop_lon)
    at a specified interval (in meters) using a vectorized computation of destination 
    coordinates and a geopy-based calculation of cumulative distances.
    
    Returns:
      - bathymetry_values: interpolated depth values along the path.
      - path_lons: array of longitudes along the path.
      - path_lats: array of latitudes along the path.
      - range_km: array of cumulative distances (in kilometers) from the start.
      
    This function uses pyproj.fwd for speed but calculates cumulative distances with geopy.
    """
    # Calculate total distance in kilometers.
    total_distance_km = geodesic((start_lat, start_lon), (stop_lat, stop_lon)).kilometers

    # Convert interval to kilometers and determine number of points.
    interval_km = interval / 1000.0
    num_points = max(int(total_distance_km / interval_km), 1)
    
    # Compute the constant bearing.
    bearing = calculate_initial_compass_bearing((start_lat, start_lon), (stop_lat, stop_lon))
    
    # Generate an array of distances (in meters) along the path.
    distances_m = np.linspace(0, total_distance_km * 1000, num_points + 1)
    
    # Use pyproj.fwd to compute destination points in one vectorized call.
    lons, lats, _ = geod.fwd(
        np.full_like(distances_m, start_lon),
        np.full_like(distances_m, start_lat),
        np.full_like(distances_m, bearing),
        distances_m
    )
    
    # Compute cumulative distance (range) using geopy for accuracy.
    start_point = Point(start_lat, start_lon)
    range_km = np.array([geodesic(start_point, (lat, lon)).kilometers for lat, lon in zip(lats, lons)])
    
    # Interpolate the bathymetry values along the computed path.
    subset_points = subset_df[['lat', 'lon']].values
    subset_depths = subset_df['depth'].values
    path_points = np.column_stack((lats, lons))
    bathymetry_values = griddata(subset_points, subset_depths, path_points, method='linear')
    
    # --- New fix: Replace any NaN values (from extrapolation issues) using nearest-neighbor interpolation.
    if np.any(np.isnan(bathymetry_values)):
        nan_mask = np.isnan(bathymetry_values)
        bathymetry_values[nan_mask] = griddata(
            subset_points, subset_depths, path_points[nan_mask], method='nearest'
        )
    
    return bathymetry_values, lons, lats, range_km


# =============================================================================
# Multiprocessing Setup
# =============================================================================
def initializer(subset_df: pd.DataFrame, ssp: list):
    """
    Initializer function for worker processes.
    Sets the global subset dataframe and sound speed profile (ssp).
    """
    global global_subset_df, global_ssp
    global_subset_df = subset_df
    global_ssp = ssp
    # Uncomment the print below to verify that each worker is receiving the correct SSP.
    # print("Worker initialized with global_ssp:", global_ssp, flush=True)


def process_subset_point_worker(args: tuple):
    """
    Worker function to process a single grid point.
    
    Args:
        args: tuple containing (ii, drifter_lat, drifter_lon, point_lat, point_lon, interval)
    
    Returns:
        A tuple of the point index and a dictionary with results (including TL and associated data).
    """
    (ii, drifter_lat, drifter_lon, point_lat, point_lon, interval) = args
    
    # Compute the bathymetry along the path using the vectorized function.
    bathy_vals, path_lon, path_lat, cumulative_distance = extract_bathymetry_from_subset_vectorized(
        global_subset_df, drifter_lat, drifter_lon, point_lat, point_lon, interval
    )
    
    # Create a DataFrame for the bathymetry grid.
    bathy_grid = pd.DataFrame({
        'range': cumulative_distance * 1000,  # convert km to meters
        'depth_m': -bathy_vals                 # negative depth (as used in propagation)
    })
    bathy_grid.drop_duplicates(inplace=True)
    bathy_grid.sort_values('range', inplace=True)
    bathy_grid.reset_index(drop=True, inplace=True)
    # Ensure the path begins at range 0.
    bathy_grid.loc[0, 'range'] = 0
    
    # Convert grid data into a list of [range, depth] pairs.
    bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()
    
    # Create the environment using the Fortran-based propagation model.
    env = pm.create_env2d(
        depth=bathy,
        soundspeed=global_ssp,
        bottom_density=2700,       # in kg/m^3
        bottom_absorption=0.1,
        bottom_soundspeed=5250,
        tx_depth=250,
        frequency=3000,
        nbeams=0,
        max_angle=90,
        min_angle=-90
    )
    
    # Set receiver locations.
    env['rx_range'] = bathy_grid['range'].iloc[-1]
    env['rx_depth'] = np.arange(0, bathy_grid['depth_m'].iloc[-1], 50)
    
    # Compute transmission loss.
    tloss = pm.compute_transmission_loss(env, mode='incoherent')
    tlosDb = 20 * np.log10(np.abs(tloss))
    
    return (ii, {
        'lat': drifter_lat,
        'lon': drifter_lon,
        'transmission_loss': tlosDb,
        'tl_depths': env['rx_depth']
    })


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def main():
    """
    Main function encapsulating the entire processing pipeline.
    Loads drift CTD data, bathymetry data, constructs processing tasks, and
    dispatches them in parallel.
    """
    # Setup multiprocessing parameters.
    multiprocessing.set_executable(sys.executable)
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()  # Needed on Windows.
    
    # ---------------------------
    # Load CTD Data for the Drift
    # ---------------------------
    drift_ctd = pd.read_csv("C:\\Users\\kaity\\Downloads\\sg639_MHI_Apr2023_CTD.csv")
    depth_diff = np.diff(drift_ctd['Depth_m'], prepend=np.nan)
    drift_ctd['Direction'] = np.where(depth_diff > 0, 'dec', 'asc')
    drift_ctd.at[0, 'Direction'] = 'dec' if depth_diff[1] > 0 else 'asc'
    drift_ctd['DiveID'] = drift_ctd['DiveNumber'].astype(str) + '_' + drift_ctd['Direction']
    
    # ---------------------------
    # Load Bathymetry Data from NetCDF
    # ---------------------------
    nc_file = ("C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\bathymetry\\"
               "GEBCO_28_Mar_2025_ade9db365e34\\gebco_2024_n23.5_s18.5_w-160.0_e-154.0.nc")
    ds = xr.open_dataset(nc_file)
    depth = ds['elevation'].values
    latvec = ds['lat'].values
    lonvec = ds['lon'].values
    lon_mesh, lat_mesh = np.meshgrid(lonvec, latvec)
    
    bathymetry_df = pd.DataFrame({
        'depth': depth.flatten(),
        'lat': lat_mesh.flatten(),
        'lon': lon_mesh.flatten()
    })
    
    # Dictionary to store results for each dive.
    results = {}
    
    # Process each dive separately.
    for dive_id, group in drift_ctd.groupby('DiveID'):
        drifter_lat = group['Latitude'].iloc[0]
        drifter_lon = group['Longitude'].iloc[0]
        
        # Filter bathymetry points within a 10 km radius (and optionally >1 km to avoid too-close points).
        bathymetry_df['distance_km'] = haversine(drifter_lon, drifter_lat,
                                                 bathymetry_df['lon'], bathymetry_df['lat'])
        subset_df = bathymetry_df[(bathymetry_df['distance_km'] <= 10) & 
                                  (bathymetry_df['distance_km'] > 1)]
        total_points = len(subset_df)
        results[dive_id] = []
        
        print(f"Dive {dive_id}: {total_points} bathymetry points found within 10 km.", flush=True)
        
        # ---------------------------
        # Build the Sound Speed Profile (SSP) Using CTD Data
        # ---------------------------
        profile = pd.DataFrame({
            'depth': group.loc[group['Direction'] == 'asc', 'Depth_m'],
            'ss': group.loc[group['Direction'] == 'asc', 'SoundSpeed_m_s']
        })
        profile.dropna(inplace=True)
        profile.sort_values('depth', inplace=True)
        profile.reset_index(drop=True, inplace=True)
        # Ensure the profile starts at depth 0.
        profile.loc[0, 'depth'] = 0
        
        # Get the maximum water depth from the bathymetry data.
        max_depth = np.max(np.abs(bathymetry_df['depth']))
        # Determine the deepest measured CTD depth and corresponding sound speed.
        deepest_ctd_depth = profile['depth'].max()
        deepest_ss = profile.loc[profile['depth'].idxmax(), 'ss']
        print(f"Dive {dive_id}: max_depth = {max_depth}, deepest_ctd_depth = {deepest_ctd_depth}, deepest_ss = {deepest_ss}", flush=True)
        # If the water bottom is deeper than the deepest measurement, extend the profile.
        if max_depth > deepest_ctd_depth:
            profile.loc[profile.index.max() + 1] = [max_depth + 1, deepest_ss]
        
        # Make sure sound speed values are positive.
        profile['ss'] = profile['ss'].abs()
        # Convert the profile into a list of lists.
        ssp = profile.apply(lambda row: [row['depth'], row['ss']], axis=1).tolist()
        #print(f"Dive {dive_id}: Computed SSP: {ssp}", flush=True)
        
        # ---------------------------
        # Prepare Arguments for Each Grid Point
        # ---------------------------
        task_args = []
        for idx in range(total_points):
            point_lat = subset_df['lat'].iloc[idx]
            point_lon = subset_df['lon'].iloc[idx]
            task_args.append((idx, drifter_lat, drifter_lon, point_lat, point_lon, 200))
        
        print(f"Dive {dive_id}: Preparing {len(task_args)} tasks to process.", flush=True)
        # Execute processing tasks in parallel.
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=8,
                initializer=initializer,
                initargs=(subset_df, ssp)
        ) as executor:
            for idx, result_data in executor.map(process_subset_point_worker, task_args):
                results[dive_id].append(result_data)
                print(f"Dive {dive_id}: Processed {idx+1} of {total_points} points.", flush=True)
    
    print("Processing complete.", flush=True)

if __name__ == "__main__":
    main()
