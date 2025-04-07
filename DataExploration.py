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
data = ds['elevation'].values
# Access the latitude and longitude vectors
latvec = ds['lat'].values  # Extracting latitude values
lonvec = ds['lon'].values  # Extracting longitude values

bathyLine, path_lons, path_lats, range_km  =  extract_bathymetry(data, latvec, 
                                lonvec,
                                driftCTD.Latitude[446752], 
                                driftCTD.Longitude[446752], 
                                 88, 20000, 10)


# Plot the bathymetry grid




bathyLine, path_lons, path_lats, range_km  =  extract_bathymetry(data, latvec, 
                                lonvec,
                                driftCTD.Latitude[446752], 
                                driftCTD.Longitude[446752], 
                                 88, 20000, 10)



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
