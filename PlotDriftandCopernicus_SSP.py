# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 13:20:15 2025

@author: kaity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def mackenzie_sound_speed(temp_C, salinity_PSU, depth_m):
    """
    Compute sound speed in seawater using the Mackenzie (1981) formula.
    
    Parameters:
        temp_C : array-like or float
            Temperature in degrees Celsius
        salinity_PSU : array-like or float
            Salinity in PSU
        depth_m : array-like or float
            Depth in meters

    Returns:
        Sound speed in m/s
    """
    T = np.asarray(temp_C)
    S = np.asarray(salinity_PSU)
    D = np.asarray(depth_m)

    return (
        1448.96
        + 4.591 * T
        - 5.304e-2 * T**2
        + 2.374e-4 * T**3
        + 1.340 * (S - 35)
        + 1.630e-2 * D
        + 1.675e-7 * D**2
        - 1.025e-2 * T * (S - 35)
        - 7.139e-13 * T * D**3
    )


def load_data():
    drift = pd.read_csv("C:/Users/kaity/Downloads/sg639_MHI_Apr2023_CTD.csv")
    drift['MacKenzieSSP'] = mackenzie_sound_speed(drift['Temperature_C'],
                                                  drift['Salinity_PSU'],
                                                  drift['Depth_m'])
    
    
    model = pd.read_csv("C:/Users/kaity/Downloads/sg639_MHI_Apr2023_final_targets_distances_wSS.csv")
    model.rename(columns={'ss': 'SoundSpeed_m_s', 
                          'depth': 'Depth_m', 
                          'so':'Salinity_PSU', 
                          'thetao':'Temperature_C'}, inplace=True)
    print("Model columns after renaming:", model.columns.tolist())
    model['DiveID'] = pd.factorize(model[['Latitude', 'Longitude']].apply(tuple, axis=1))[0]
    model['MacKenzieSSP'] = mackenzie_sound_speed(model['Temperature_C'],
                                                  model['Salinity_PSU'],
                                                  model['Depth_m'])
    
    return drift, model

def process_drift_data(df):
    depth_diff = np.diff(df['Depth_m'], prepend=np.nan)
    df['Direction'] = np.where(depth_diff > 0, 'dec', 'asc')
    df.at[0, 'Direction'] = 'dec' if depth_diff[1] > 0 else 'asc'
    df['DiveID'] = df['DiveNumber'].astype(str) + '_' + df['Direction']
    return df



def interpolate_profiles(df, value_col, max_depth=None, min_points=20):
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    result = pd.DataFrame()
    for dive_id, group in df.groupby('DiveID'):
        if len(group) > min_points:
            group_sorted = group.sort_values('Depth_m')
            group_sorted = group_sorted.dropna(subset=[value_col])

            # Use max observed depth if not specified
            this_max_depth = max_depth if max_depth is not None else group_sorted['Depth_m'].max()
            depth_range = np.arange(0, int(this_max_depth) + 1)

            interp_vals = np.interp(depth_range, group_sorted['Depth_m'], group_sorted[value_col])
            temp = pd.DataFrame({
                'Depth_m': depth_range,
                value_col: interp_vals,
                'DiveID': dive_id
            })
            result = pd.concat([result, temp], ignore_index=True)
    return result


# def calculate_percentiles(df, value_col):
#     percentiles = df.groupby('Depth_m')[value_col].quantile([0.05, 0.95]).unstack()
#     percentiles.columns = ['5th_percentile', '95th_percentile']
#     return percentiles

def calculate_percentiles(df, value_col, min_profiles=5):
    # Count how many profiles contributed at each depth
    profile_counts = df.groupby('Depth_m')['DiveID'].nunique()
    
    # Keep only depths with enough dives
    valid_depths = profile_counts[profile_counts >= min_profiles].index

    # Filter the DataFrame to those depths
    df_filtered = df[df['Depth_m'].isin(valid_depths)]

    # Compute percentiles on filtered data
    percentiles = df_filtered.groupby('Depth_m')[value_col].quantile([0.05, 0.95]).unstack()
    percentiles.columns = ['5th_percentile', '95th_percentile']
    
    return percentiles



def plot_profiles(observed, modeled, obs_perc, mod_perc, value_col, label, units):
    plt.figure(figsize=(4, 10))

    for _, group in observed.groupby('DiveID'):
        plt.plot(group[value_col], group['Depth_m'], color='gray', alpha=0.05)

    for _, group in modeled.groupby('DiveID'):
        plt.plot(group[value_col], group['Depth_m'], color='skyblue', alpha=0.2)

    plt.plot(mod_perc['5th_percentile'], mod_perc.index, 'b--', label='Modeled 5th–95th')
    plt.plot(mod_perc['95th_percentile'], mod_perc.index, 'b--')

    plt.plot(obs_perc['5th_percentile'], obs_perc.index, 'r--', label='Observed 5th–95th')
    plt.plot(obs_perc['95th_percentile'], obs_perc.index, 'r--')

    plt.gca().invert_yaxis()
    plt.xlabel(f'{label} ({units})')
    plt.ylabel('Depth (m)')
    plt.title(f'{label} Profiles: CTD vs Copernicus')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(value_col):
    label_map = {
        'SoundSpeed_m_s': ('Sound Speed', 'm/s'),
        'Temperature_C': ('Temperature', '°C'),
        'Salinity_PSU': ('Salinity', 'PSU'),
        'MacKenzieSSP': ('Sound Speed', 'm/s')
    }

    if value_col not in label_map:
        raise ValueError(f"Invalid variable: {value_col}. Choose from {list(label_map.keys())}")

    drift, model = load_data()
    drift = process_drift_data(drift)

    print("Drift columns:", drift.columns.tolist())
    print("Model columns:", model.columns.tolist())

    # drift_interp = interpolate_profiles(drift, value_col)
    # model_interp = interpolate_profiles(model, value_col)
    
    drift_interp = interpolate_profiles(drift, value_col, max_depth=None)
    model_interp = interpolate_profiles(model, value_col, max_depth=1000)

    print("Interpolated model columns:", model_interp.columns.tolist())

    if 'Depth_m' not in model_interp.columns:
        raise KeyError("Interpolated model data is missing 'Depth_m'. Check source columns.")

    # drift_perc = calculate_percentiles(drift_interp, value_col)
    # model_perc = calculate_percentiles(model_interp, value_col)
    
    drift_perc = calculate_percentiles(drift_interp, value_col, min_profiles=5)
    model_perc = calculate_percentiles(model_interp, value_col, min_profiles=5)


    label, units = label_map[value_col]
    plot_profiles(drift_interp, model_interp, drift_perc, model_perc, value_col, label, units)

# To run, call main with 'SoundSpeed_m_s', 'Temperature', or 'Salinity':
main('MacKenzieSSP')
#main('SoundSpeed_m_s')
main('Temperature_C')
main('Salinity_PSU')

