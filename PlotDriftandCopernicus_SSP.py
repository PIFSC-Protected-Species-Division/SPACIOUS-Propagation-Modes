# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:19:06 2025

@author: kaity

Plot the copernicus SSP along with the drift SSPs

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use splines to interpolate the dive data
from scipy.interpolate import UnivariateSpline


# Load the SSPs from the drift and from the modelled data
driftCTD = pd.read_csv("C:\\Users\\kaity\\Downloads\\sg639_MHI_Apr2023_CTD.csv")
copernicusSSP = pd.read_csv('C:/Users\\kaity\\Downloads\\sg639_MHI_Apr2023_final_targets_distances_wSS.csv')

#%% Process the drift data 
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


#%% Process the copernicus data

# Rename the copernicus colums to match
copernicusSSP.rename(columns ={'ss': 'SoundSpeed_m_s', 'depth':'Depth_m'}, inplace= True)
copernicusSSP['DiveID'] = pd.factorize(list(zip(copernicusSSP['Latitude'], 
                                                copernicusSSP['Longitude'])))[0]

# Calculate percentiles for Copernicus data
copernicus_percentiles = copernicusSSP.groupby('Depth_m')['SoundSpeed_m_s'].quantile([0.05, 0.95]).unstack()
copernicus_percentiles.columns = ['5th_percentile', '95th_percentile']



#%% Plotting




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
        plt.plot(sound_speed_interp, depth_range,  'b-')
        plt.gca().invert_yaxis()  # Inverts the y-axis so depth increases downwards
        plt.xlabel('Depth (m)')
        plt.ylabel('Sound Speed (m/s)')
        plt.title(f'Sound Speed Profile for {dive_id}')
        plt.legend()
        plt.show()

    return depth_range, sound_speed_interp



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
plt.title('sg639_MHI_Apr2023 CTD data')
plt.legend()
plt.show()

percentiles_by_depth['Diff'] = percentiles_by_depth['95th_percentile']-percentiles_by_depth['5th_percentile']

np.max(percentiles_by_depth['Diff'])


# Plot 2: Copernicus-only SSPs
plt.figure(figsize=(4, 10))
for dive_id, group in copernicusSSP.groupby('DiveID'):
    plt.plot(group['SoundSpeed_m_s'], group['Depth_m'], color='skyblue', alpha=0.4)

plt.plot(copernicus_percentiles['5th_percentile'], copernicus_percentiles.index, label='5th Percentile')
plt.plot(copernicus_percentiles['95th_percentile'], copernicus_percentiles.index, label='95th Percentile')
plt.gca().invert_yaxis()
plt.xlabel('Sound Speed (m/s)')
plt.ylabel('Depth (m)')
plt.title('Copernicus SSP Profiles')
plt.legend()
plt.tight_layout()
plt.show()



# Plot 3: Combined Glider + Copernicus
plt.figure(figsize=(4, 10))
# Copernicus
for dive_id, group in copernicusSSP.groupby('DiveID'):
    plt.plot(group['SoundSpeed_m_s'], group['Depth_m'], color='skyblue', alpha=0.1)

# Glider
for dive_id, group in all_interpolated_data.groupby('DiveID'):
    plt.plot(group['SoundSpeed_m_s'], group['Depth_m'], color='gray', alpha=0.1)
    
plt.plot(copernicus_percentiles['5th_percentile'], copernicus_percentiles.index,
         label='Copernicus 5th–95th')    

plt.gca().invert_yaxis()
plt.xlabel('Sound Speed (m/s)')
plt.ylabel('Depth (m)')
plt.title('CTD vs Copernicus SSPs')
plt.legend()
plt.tight_layout()
plt.show()
