# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 21:30:37 2025

@author: kaity
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load a drift
driftCTD = pd.read_csv("C:\\Users\\kaity\\Downloads\\sg639_MHI_Apr2023_CTD.csv")

# Determine if the glider is ascending or descending
depth_diff = np.diff(driftCTD['Depth_m'], prepend=np.nan)

# Define 'asc' for ascending and 'dec' for descending
driftCTD['Direction'] = np.where(depth_diff > 0, 'dec', 'asc')

# Correcting the first entry if needed
if depth_diff[1] > 0:
    driftCTD.at[0, 'Direction'] = 'dec'
else:
    driftCTD.at[0, 'Direction'] = 'asc'



# Define 'asc' for ascending and 'dec' for descending
driftCTD['DiveID'] = driftCTD['DiveNumber'].astype(str) + '_' + driftCTD['Direction']


# Giggle test the dives
# Choose a specific dive number to plot
dive_number = "37"  # You can change this to any dive number you want to inspect

# Filter the DataFrame for the chosen dive and both ascending and descending
dive_data = driftCTD[driftCTD['DiveNumber'].astype(str) == dive_number]

# Plotting
plt.figure(figsize=(3, 10))
for direction, color in [('asc', 'blue'), ('dec', 'red')]:
    segment_data = dive_data[dive_data['DiveID'].str.contains(direction)]
    plt.plot(segment_data['SoundSpeed_m_s'], segment_data['Depth_m'], label=f'Dive {dive_number} {direction}', color=color, marker='o', linestyle='-')

plt.gca().invert_yaxis()  # Inverts the y-axis so depth increases downwards
plt.xlabel('Sound Speed (m/s)')
plt.ylabel('Depth (m)')
plt.title(f'Sound Speed Profile for Dive {dive_number}')
plt.legend()
plt.show()


# Use splines to interpolate the dive data
from scipy.interpolate import UnivariateSpline


# Function to perform spline interpolation and plot
def plot_spline_interpolation(dive_data, dive_id, plot = False):
    # Sorting by depth might be necessary if not already sorted
    dive_data_sorted = dive_data.sort_values('Depth_m')

    # Drop the NA values
    dive_data_sorted.dropna(inplace = True, subset = ['SoundSpeed_m_s'])

    # Set up the spline with sorted data
    spline = UnivariateSpline(dive_data_sorted['Depth_m'], 
                              dive_data_sorted['SoundSpeed_m_s'])

    
    # Create an array of depths at 1m intervals
    depth_range = np.arange(dive_data_sorted['Depth_m'].min(), dive_data_sorted['Depth_m'].max())
    
    # Predict sound speed at these depths using the spline
    sound_speed_interp = spline(depth_range)

    if plot:
        # Plotting
        plt.figure(figsize=(3, 10))
        plt.plot(dive_data_sorted['SoundSpeed_m_s'],dive_data_sorted['Depth_m'],  'ro', label=f'Original Data ({dive_id})')
        plt.plot(sound_speed_interp, depth_range,  'b-', label=f'Interpolated Spline ({dive_id})')
        plt.gca().invert_yaxis()  # Inverts the y-axis so depth increases downwards
        plt.xlabel('Depth (m)')
        plt.ylabel('Sound Speed (m/s)')
        plt.title(f'Sound Speed Profile for {dive_id}')
        plt.legend()
        plt.show()

    return depth_range, sound_speed_interp

# Example usage for a specific DiveID
dive_id = '1_asc'  # Replace this with any DiveID you want to analyze
selected_dive_data = driftCTD[driftCTD['DiveID'] == dive_id]
depths, interp_speeds = plot_spline_interpolation(selected_dive_data, dive_id)



# Function to interpolate sound speed for each meter of depth
def interpolate_sound_speed(dive_data, maxDepth, plot =False):
    # Sorting by depth might be necessary if not already sorted
    dive_data_sorted = dive_data.sort_values('Depth_m')
    
    # Drop the NA values
    dive_data_sorted.dropna(inplace = True, subset = ['SoundSpeed_m_s'])

    # Create an array of depths at 1m intervals
    depth_range = np.arange(0, maxDepth)
    
    # Predict sound speed at these depths using the spline
    sound_speed_interp = np.interp(depth_range, dive_data_sorted['Depth_m'],
                                   dive_data_sorted['SoundSpeed_m_s'])
    if plot:
        # Plotting
        plt.figure(figsize=(3, 10))
        plt.plot(dive_data_sorted['SoundSpeed_m_s'], 
                 dive_data_sorted['Depth_m'],  'ro', 
                 label=f'Original Data ({dive_id})')
        plt.plot(sound_speed_interp, depth_range,  'b-', 
                 label=f'Interpolated Spline ({dive_id})')
        plt.gca().invert_yaxis()  # Inverts the y-axis so depth increases downwards
        plt.xlabel('Depth (m)')
        plt.ylabel('Sound Speed (m/s)')
        plt.title(f'Sound Speed Profile for {dive_id}')
        plt.xlim(1480, 1540)
        plt.legend()
        plt.show()
        
    return pd.DataFrame({'Depth_m': depth_range, 'SoundSpeed_m_s': sound_speed_interp})


# Collect all interpolated data
all_interpolated_data = pd.DataFrame()

# Use just the descending dives
decDives = driftCTD[driftCTD['Direction']=='dec']

for dive_id, group in decDives.groupby('DiveID'):
    if len(group)>200:
        interpolated_data = interpolate_sound_speed(group,1000, plot =False)
        interpolated_data['DiveID']=dive_id
        all_interpolated_data = pd.concat([all_interpolated_data, 
                                       interpolated_data], ignore_index=True)


# Calculate percentiles for each depth across all dives
percentiles_by_depth = all_interpolated_data.groupby('Depth_m')['SoundSpeed_m_s'].quantile([0.05, 0.95]).unstack()
percentiles_by_depth.columns = ['5th_percentile', '95th_percentile']



# plot these results
plt.figure(figsize=(3, 10))
for dive_id, group in all_interpolated_data.groupby('DiveID'):
    plt.plot( group['SoundSpeed_m_s'],group['Depth_m'], 
             color='lightgray', label=f'{dive_id}' if dive_id == list(all_interpolated_data['DiveID'].unique())[0] else "")


plt.plot(percentiles_by_depth['5th_percentile'], percentiles_by_depth.index,  label='5th Percentile')
plt.plot(percentiles_by_depth['95th_percentile'], percentiles_by_depth.index, label='95th Percentile')
plt.gca().invert_yaxis()  # Depth increases downwards
plt.ylabel('Depth (m)')
plt.xlabel('Sound Speed (m/s)')
plt.title('5th and 95th Percentiles of Sound Speed by Depth Across All Dives')
plt.legend()
plt.show()

percentiles_by_depth['Diff'] = percentiles_by_depth['95th_percentile']-percentiles_by_depth['5th_percentile']

np.max(percentiles_by_depth['Diff'])

########################################################################
#%% Download Bathymetry 
###########################################################################




import numpy as np
from geopy.distance import geodesic
from geopy.point import Point


import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from geopy.point import Point


import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from geopy.point import Point
from scipy.interpolate import interpn
from scipy.interpolate import RegularGridInterpolator


def download_and_plot_bathymetry(nc_file):
    """
    Loads a NetCDF file and plots the bathymetry data.
    
    Parameters:
    - nc_file: Path to the NetCDF file containing bathymetry data.
    """
    # Load the dataset
    ds = xr.open_dataset(nc_file)
    
    # Print dataset info to understand the structure
    print(ds.info())
    
    # Access the bathymetry data; adjust variables names as per the dataset specifics
    bathymetry = ds.elevation  # or ds['elevation'] depending on the structure
    
    # Plot the data using a simple pcolormesh
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(bathymetry.longitude, bathymetry.latitude, bathymetry, shading='auto')
    plt.colorbar(label='Elevation (m)')
    plt.title('GEBCO Bathymetry')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    
    return bathymetry.longitude, bathymetry.latitude, bathymetry

def extract_bathymetry(data, latvec, lonvec, lat, lon, bearing, distance, interval):
    """
    Extracts and plots interpolated bathymetry data along a line defined by an initial point, a bearing, and distance.
    
    Parameters:
    - data: 2D numpy array of bathymetry data.
    - latvec: 1D array of latitudes corresponding to the rows of 'data'.
    - lonvec: 1D array of longitudes corresponding to the columns of 'data'.
    - lat, lon: Latitude and longitude of the starting point.
    - bearing: Bearing in degrees from the north (clockwise).
    - distance: Total distance along the bearing to sample.
    - interval: Distance between sampling points.

    Returns:
    - bathymetry_values: Array of interpolated bathymetry values along the path.
    """
    # Starting point
    start_point = Point(lat, lon)
    
    # Calculate the number of points
    num_points = int(distance / interval)
    
    # Arrays to store the coordinates of the path and bathymetry values
    path_lats = np.zeros(num_points + 1)
    path_lons = np.zeros(num_points + 1)
    bathymetry_values = np.zeros(num_points + 1)
    
    # Loop to calculate each point
    for i in range(num_points + 1):
        # Calculate the new point using the bearing and interval
        new_point = geodesic(kilometers=i * interval / 1000).destination(start_point, bearing)
        path_lats[i] = new_point.latitude
        path_lons[i] = new_point.longitude

    # Create an interpolator instance
    grid_interpolator = RegularGridInterpolator((latvec, lonvec), data, method='linear', bounds_error=False, fill_value=None)
    
    # Prepare points for interpolation
    points = np.vstack((path_lats, path_lons)).T

    # Interpolate bathymetry using bilinear interpolation
    bathymetry_values = grid_interpolator(points)

    # Calculate distances
    range_km = np.array([geodesic(start_point, (path_lats[i], path_lons[i])).kilometers for i in range(len(path_lats))])
    
    # Plot the bathymetry grid
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(lonvec, latvec, data, shading='auto')
    plt.colorbar(label='Depth (m)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Bathymetry and Sampling Path')

    # Plot the path
    plt.plot(path_lons, path_lats, 'r-', linewidth=2, label='Sampling Path')
    plt.scatter(path_lons, path_lats, color='yellow', s=50, zorder=5, label='Sample Points')
    plt.legend()

    # Show the plot
    plt.show()
    
    return bathymetry_values, path_lons, path_lats, range_km


# Bathymetry data from NCEI
nc_file = 'C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\bathymetry\\GEBCO_28_Mar_2025_ade9db365e34\\gebco_2024_n23.5_s18.5_w-160.0_e-154.0.nc'
ds = xr.open_dataset(nc_file)
depth = ds['elevation'].values

# Access the latitude and longitude vectors
latvec = ds['lat'].values  # Extracting latitude values
lonvec = ds['lon'].values  # Extracting longitude values
depth = ds['elevation'].values  # 2D array

# Create meshgrid and flatten the arrays
lon_mesh, lat_mesh = np.meshgrid(lonvec, latvec)
depth_flat = depth.flatten()
lat_flat = lat_mesh.flatten()
lon_flat = lon_mesh.flatten()

# Create a DataFrame from the flattened arrays
bathymetry_df = pd.DataFrame({
    'depth': depth_flat,
    'lat': lat_flat,
    'lon': lon_flat
})


bathyLine, path_lons, path_lats, range_km  =  extract_bathymetry(depth, latvec, 
                                lonvec,
                                driftCTD.Latitude[446752], 
                                driftCTD.Longitude[446752], 
                                 88, 20000, 10)


# Plot the bathymetry grid

bathy_grid = pd.DataFrame({'range':range_km*1000, 'depth_m': -bathyLine })
bathy_grid.drop_duplicates(inplace=True)
bathy_grid.sort_values('range')
bathy_grid.reindex()

bathy_grid.loc[0, 'range'] = 0

plt.figure(figsize=(10, 8))
plt.plot(bathy_grid['range'], bathy_grid['depth_m'], 'r-',
         linewidth=2, label='Sampling Path')
plt.ylim(0, np.max(-bathyLine))
plt.xlim(0, np.max(range_km*1000))
plt.gca().invert_yaxis()  # Depth increases downwards
plt.show()



############################################################################
#%% Build out the structore for Area monitored 


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) using the haversine formula.

    Parameters:
        lon1, lat1: float or np.array, longitude and latitude of the first point.
        lon2, lat2: float or np.array, longitude and latitude of the second point.

    Returns:
        distance in kilometers.
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Compute differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Haversine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371  # Earth's radius in kilometers
    return c * r


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy import Point
from geopy.distance import geodesic
from scipy.interpolate import griddata

import arlpy.uwapm as pm
import arlpy.plot as plt
import numpy as np


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculate the initial compass bearing in degrees between two points.
    
    Parameters:
      pointA: (lat, lon) tuple for the start.
      pointB: (lat, lon) tuple for the end.
      
    Returns:
      Bearing in degrees.
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
    
    Parameters:
      subset_df: Pandas DataFrame with columns 'depth', 'lat', and 'lon'. Only the points within 
                 a given region (for example, within 10 km) should be included.
      start_lat, start_lon: Coordinates for the source (e.g., drifter location).
      stop_lat, stop_lon: Coordinates for the receiver (a point from subset_df).
      interval: Sampling interval along the path in meters.
      
    Returns:
      bathymetry_values: Array of interpolated depths along the path.
      path_lons: Array of longitudes along the path.
      path_lats: Array of latitudes along the path.
      range_km: Array of cumulative distances along the path in kilometers.
    """
    
    # Define start and stop points
    start_point = Point(start_lat, start_lon)
    stop_point = Point(stop_lat, stop_lon)
    
    # Determine total distance in kilometers between start and stop
    total_distance_km = geodesic(start_point, stop_point).kilometers
    
    # Convert interval from meters to kilometers
    interval_km = interval / 1000.0
    
    # Compute the number of points (ensure at least two endpoints)
    num_points = max(int(total_distance_km / interval_km), 1)
    
    # Calculate initial bearing between start and stop
    bearing = calculate_initial_compass_bearing((start_lat, start_lon), (stop_lat, stop_lon))
    
    # Create arrays for storing the path coordinates
    path_lats = np.zeros(num_points + 1)
    path_lons = np.zeros(num_points + 1)
    
    # Generate intermediate points along the great-circle path
    for i in range(num_points + 1):
        # Determine current distance (do not exceed total_distance)
        current_distance = min(i * interval_km, total_distance_km)
        new_point = geodesic(kilometers=current_distance).destination(start_point, bearing)
        path_lats[i] = new_point.latitude
        path_lons[i] = new_point.longitude
    
    # Now, interpolate the depth along these points using only the subset_df data.
    # Create a scattered points array and corresponding depth values from subset_df.
    subset_points = subset_df[['lat', 'lon']].values
    subset_depths = subset_df['depth'].values
    
    # Prepare the intermediate points for interpolation
    path_points = np.vstack((path_lats, path_lons)).T
    
    # Interpolate using griddata. You can choose a method (e.g., 'linear' or 'nearest').
    bathymetry_values = griddata(subset_points, subset_depths, path_points, method='linear')
    
    # Calculate cumulative distances along the path (in kilometers)
    range_km = np.array([
        geodesic(start_point, (path_lats[i], path_lons[i])).kilometers
        for i in range(len(path_lats))
    ])
    
    # # Plot for validation
    # plt.figure()
    
    # # Plot the scattered bathymetry points (using a scatter plot)
    # plt.scatter(subset_df['lon'], subset_df['lat'], color=subset_df['depth'], 
    #             cmap='viridis', s=20, label='Subset Points')
    # plt.colorbar(label='Depth (m)')
    
    # # Overlay the interpolated sampling path
    # plt.plot(path_lons, path_lats, 'r-', linewidth=2, label='Sampling Path')
    # plt.scatter(path_lons, path_lats, color='yellow', s=50, zorder=5, label='Sample Points')
    
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Interpolated Bathymetry Along the Path (Subset Data)')
    # plt.legend()
    # plt.show()
    
    return bathymetry_values, path_lons, path_lats, range_km
# Figure out the distance between the lat/lon grid and the location of the 
# drifter

results = {}
for driftId, group in driftCTD.groupby('DiveID'): 
    
    drifter_lat =group['Latitude'].iloc[0]
    drifter_lon = group['Longitude'].iloc[0]
    
    # Get distance between drifter location and all bathymetry points
    # Compute distances in a vectorized manner and add as a new column
    bathymetry_df['distance_km'] = haversine(drifter_lon, drifter_lat, 
                                             bathymetry_df['lon'], bathymetry_df['lat'])

    # Filter to only points within 10 km
    subset_df = bathymetry_df[bathymetry_df['distance_km'] <= 10]
    total_rows = len(subset_df)
    
    results[driftId] = []
    # Triangulate the scatter data
    #triangulation = tri.Triangulation(bathymetry_df['lon'], bathymetry_df['lat'])
    
    # Create a filled contour plot using the depth values
    #contour = plt.tricontourf(triangulation, bathymetry_df['depth'], levels=100, cmap='viridis')
    # plt.colorbar(contour, label='Depth')
    
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Bathymetry and Drifter Subset Points')
    
    # # Overlay the subset points (ignoring depth)
    # plt.scatter(subset_df['lon'], subset_df['lat'], color='red',
    #             label='Within 10 km')
    
    # # Overlay the drifter Loc
    # plt.scatter(drifter_lon, drifter_lat, color='yellow')
    
    # plt.legend()
    # plt.show()
    
 
    # Create a DataFrame for the SSP
    profile = pd.DataFrame({
        'depth': group['Depth_m'][group['Direction']== 'asc'],
        'ss': group['SoundSpeed_m_s'][group['Direction']== 'asc']
    })

    # Sort them by 'depth'
    profile.sort_values('depth', inplace=True)
    profile.dropna(axis=0, inplace=True)
    profile.reset_index(drop=True, inplace=True)  # Reset index after sorting

    # Change the depth value at the first row to 0
    profile.loc[0, 'depth'] = 0
    
   
    # Adding a new row without creating a new DataFrame explicitly
    max_depth =np.max(np.abs(subset_df['depth']))   # Define your maximum depth
    last_ss = profile.iloc[-1]['ss']
    profile.loc[profile.index.max() + 1] = [max_depth, last_ss]
    
    # Set the ssp values to be positive
    profile.ss = np.abs(profile.ss)

    
    # Convert the profile DataFrame into the BELLHOP list of lists format
    ssp = profile.apply(lambda row: [row['depth'], row['ss']], axis=1).tolist()
    
    
    # Now, for each grid location create the bellhop model
    for ii in  range(len(subset_df)):
        # Extract the bathymetry along the path using only the subset points
        bathy_vals, path_lon, path_lat, cumulative_distance = extract_bathymetry_from_subset(
            subset_df, drifter_lat, drifter_lon, subset_df['lat'].iloc[ii],
            subset_df['lon'].iloc[ii], 200
        )
        
        bathy_grid = pd.DataFrame({'range':cumulative_distance*1000, 'depth_m': -bathy_vals })
        bathy_grid.drop_duplicates(inplace=True)
        bathy_grid.sort_values('range')
        bathy_grid.reindex()

        bathy_grid.loc[0, 'range'] = 0
        bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()
        
        
        # Appending ssp and bathy to existing env file
        env = pm.create_env2d(
            depth=bathy,
            soundspeed=ssp,
            bottom_density=2700,# kg/m^3
            bottom_absorption=0.1,
            bottom_soundspeed=5250,
            tx_depth=250,
            frequency= 3000,
            nbeams = 0,
            max_angle = 90,
            min_angle = -90
        )
        
        # Receiver locations
        env['rx_range'] =bathy_grid['range'].iloc[-1]
        env['rx_depth'] =np.arange(0, bathy_grid['depth_m'].iloc[-1], 50)
        tloss = pm.compute_transmission_loss(env, mode='incoherent')
        tlosDb = 20*np.log10(np.abs(tloss))
        
        

        # Append a dictionary containing all the relevant data for this point.
        results[driftId].append({
            'lat': drifter_lat,
            'lon': drifter_lon,
            'transmission_loss': tlosDb,
            'tl_depths':  env['rx_depth']  # The additional depths array
        })
        print(f"Processed {ii} of {total_rows} points.")

        


        
        
        
        
    
    





###################################################################
#%% Run Bellhop



import arlpy.uwapm as pm
import arlpy.plot as plt
import numpy as np

# Pull the soundspeed profile
# get the dive number
diveNum = driftCTD.DiveNumber[446752] 


depth = driftCTD['Depth_m'][driftCTD['DiveNumber'] == diveNum][driftCTD['Direction']== 'asc']
sp = driftCTD['SoundSpeed_m_s'][driftCTD['DiveNumber'] == diveNum][driftCTD['Direction']== 'asc']


# Create a DataFrame
profile = pd.DataFrame({
    'depth': depth,
    'ss': sp
})

# Sort them by 'depth'
profile.sort_values('depth', inplace=True)
profile.dropna(axis=0, inplace=True)
profile.reset_index(drop=True, inplace=True)  # Reset index after sorting




# Change the depth value at the first row to 0
profile.loc[0, 'depth'] = 0

max_depth =np.abs(bathyLine[0])   # Define your maximum depth
last_ss = profile.iloc[-1]['ss']

# Adding a new row without creating a new DataFrame explicitly
profile.loc[profile.index.max() + 1] = [max_depth, last_ss]

# Set the ssp values to be positive
profile.ss = np.abs(profile.ss)

# Convert the profile DataFrame into the BELLHOP list of lists format
ssp = profile.apply(lambda row: [row['depth'], row['ss']], axis=1).tolist()
bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()



# Appending ssp and bathy to existing env file
env = pm.create_env2d(
    depth=bathy,
    soundspeed=ssp,
    bottom_density=2700,# kg/m^3
    bottom_absorption=0.1,
    bottom_soundspeed=5250,
    tx_depth=2500,
    frequency= 1000,
    nbeams = 0,
    max_angle = 90,
    min_angle = -90
)

pm.print_env(env)
pm.plot_ssp(env, width=500)
#pm.plot_env(env, width=900)

rays = pm.compute_rays(env)
#pm.plot_rays(rays, env=env, width=500)

env['rx_range'] = np.linspace(0, 19999, 501)
env['rx_depth'] = np.linspace(0, 3500, 151)


tloss = pm.compute_transmission_loss(env, mode='incoherent')
tlosDb = 20*np.log10(np.abs(tloss))
pm.plot_transmission_loss(tloss, env=env, width=900)

plt.figure()
plt.pcolormesh(tlosDb)
