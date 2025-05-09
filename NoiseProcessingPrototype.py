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

# Soundtrap
model = "ST300HF"
name = "SoundTrap"
serial_number = 67424266
drifter = pyhy.custom(
    name=name, 
    model=model, 
    sensitivity=-172.8, 
    serial_number=serial_number,
    preamp_gain= 10,
    Vpp =2
)

# First, decide band to study. The top frequency should not be higher than the nyquist frequency (sampling rate/2)
band = [0, 4000]

# Then, set the nfft to double the sampling rate. If you want to pass None to your band, that is also an option, but
# then you need to know the sampling frequency to choose the nfft.
nfft = band[1] * 2  # or nfft = 8000

# Set the band to 1 minute
binsize = 60.0

include_dirs = False
zipped_files = False
dc_subtract = True
asa = pypam.ASA(
    hydrophone=drifter,
    folder_path="C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData",
    binsize=binsize,
    nfft=nfft,
    timezone="UTC",
    include_dirs=include_dirs,
    zipped=zipped_files,
    dc_subtract=dc_subtract)

# Compute the hybrid millidecade bands
# You can choose 'density' or 'spectrum' as a method
milli_psd = asa.hybrid_millidecade_bands(
    db=True, method="density", band=band, percentiles=None)
print(milli_psd["millidecade_bands"])


aa = milli_psd.band_density.to_dataframe()
bb = 20*np.log10(milli_psd.band_density.values)

