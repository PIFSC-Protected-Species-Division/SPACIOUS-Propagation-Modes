# -*- coding: utf-8 -*-
"""
Created on Fri May  9 09:25:05 2025

@author: kaity
"""

import pathlib
import pyhydrophone as pyhy
import pypam
import pandas as pd
import numpy as np


from netCDF4 import Dataset
calFileLoc = 'C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\sg679_CalCurCEAS_Aug2024_sensitivity_2025-05-19.nc'



#%% Try end to end calibration again
# Modified version

import matplotlib as mpl
import matplotlib.pyplot as plt

# 1. Reset everything to the default Matplotlib style:
mpl.rcdefaults()     # → clears out any custom rcParams/style sheets

# 2. Make sure usetex is off (in case something later turns it on again):
mpl.rcParams['text.usetex'] = False

# 3. (Optionally) choose the default style explicitly:
plt.style.use('default')

# Set up the Hydrophone/recording system
model = "RandomPhone"
name = "Bob"
serial_number = 67416073
calFileLoc = 'C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\sg679_CalCurCEAS_Aug2024_sensitivity_2025-05-19.nc'

# Custom or generic hydrophone class with options for name, model, sensitivity, 
# serial number etc
drifter_Cal = pyhy.custom(
    name=name, 
    model=model, 
    sensitivity=0, 
    serial_number=serial_number,
    preamp_gain= 1,
    Vpp =1,
    calibration_file = calFileLoc)

# Get the end-to-end calibration values in the time domain, only frequency
# independent values such as gain and voltage
drifter_Cal.end_to_end_calibration()

# Set up the acoustic study
# First, decide band to study. The top frequency should not be higher than the nyquist frequency (sampling rate/2)
band = [0, 90000]

# Then, set the nfft to double the sampling rate. 
nfft = band[1] * 2  # or nfft = 8000

# Set the band to 1 minute
binsize = 60.0

# Select features of the files
include_dirs = False
zipped_files = False
dc_subtract = False # subtract dc offset, useful for soundtraps

asa_cal = pypam.ASA(
    hydrophone=drifter_Cal,
    folder_path="C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\GliderData\\",
    binsize=binsize,
    nfft=nfft,
    timezone="UTC",
    include_dirs=include_dirs,
    zipped=zipped_files,
    dc_subtract=dc_subtract,
    calibration=-1) # set to negative one to pull from hudrophoen info)


# Compute the hybrid millidecade bands
# You can choose 'density' or 'spectrum' as a method
milli_psd_cal = asa_cal.hybrid_millidecade_bands(
    db=True, method="density", band=band, percentiles=None)

# Pull the frequency values from the hybrid milidecade bands
frequencies = milli_psd_cal.frequency.values
frequency_increment = drifter_Cal.freq_cal_inc(frequencies)

frequencies = milli_psd_cal.frequency_bins.values
frequency_increment = drifter_Cal.freq_cal_inc(frequencies)


# Plot the incement -giggle test
plt.plot(frequency_increment['frequency'][1:], -frequency_increment['inc_value'][1:])
plt.xlabel('Frequency (Hz)')
plt.ylabel('dB')
plt.title('End-to-End Sensitivity (hyd, preamp, voltage)')

# 1) grab the PSD DataArray (shape = (9, 2752))
psd = milli_psd_cal["millidecade_bands"]

# 2) subtract the 1-D increment (length 2752). xarray will broadcast it along the first axis:
calibrated = frequency_increment["inc_value"].values+psd  

# 3) put it back into a copy of the original Dataset:
milli_psd_cal_dB_upa = milli_psd_cal.copy()
milli_psd_cal_dB_upa["millidecade_bands"] = calibrated


# Plot the spectrum mean with the standard deviation
pypam.plots.plot_spectrum_median(
    milli_psd_cal, data_var="millidecade_bands", frequency_coord="frequency_bins")

# Plot the spectrum mean with the standard deviation
calibratedPlot = pypam.plots.plot_spectrum_median(
    milli_psd_cal_dB_upa, data_var="millidecade_bands", frequency_coord="frequency_bins")


# Note that the last chunk of the hybridmili decades band is wrong likely that
# the system is dividing by the lenght of the chunk when only it may represent
# only a few seconds

##################################################################
#%% Load the NC file with the calibration
###################################################################

from netCDF4 import Dataset
calFileLoc = 'C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\sg679_CalCurCEAS_Aug2024_sensitivity_2025-05-19.nc'

data = Dataset(calFileLoc, 'r')  # 'r' means read mode

print(data.variables.keys())
print(data.dimensions.keys())

# Frequency and sensitivity response
freq = data.variables['frequency'][:]
sendb = data.variables['sensitivity'][:]

plt.plot(freq, sendb)
plt.xlabel('frequency hz')
plt.ylabel('Sensitivity dB')


with Dataset(calFileLoc, 'r') as ds:
    print("File loaded successfully.")
    print("Global attributes:", ds.ncattrs())
    print("Number of dimensions:", len(ds.dimensions))
    print("Number of variables:", len(ds.variables))
    print("Number of groups:", len(ds.groups))

    if ds.groups:
        print("Groups present:", list(ds.groups.keys()))
        for group_name, group in ds.groups.items():
            print(f"\nGroup: {group_name}")
            print("  Dimensions:", list(group.dimensions.keys()))
            print("  Variables:", list(group.variables.keys()))










