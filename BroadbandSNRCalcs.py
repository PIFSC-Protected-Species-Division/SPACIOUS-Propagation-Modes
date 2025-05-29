# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:06:25 2025

@author: kaity
"""

import numpy as np
import pandas as pd
from SourceLevel import tukeyClick, synthesize_received_broadband_incoh, plotTLSigs, synthesize_received_broadband

import matplotlib.pyplot as plt
import h5py 

# Create habitat monitored with broadband clciks

#%% 1) Generate a broadband signal and plot it
fs=200000
clickSig, tt = tukeyClick(fs=fs,  dur_s=0.25, dbPtP=202, padding_s=.1)

# Time-domain plot
plt.subplot(2, 1, 1)
plt.plot(tt, clickSig)
plt.title('Time-Domain Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μPa)')

# Frequency-domain plot
plt.subplot(2, 1, 2)
frequencies, power_spectral_density = plt.psd(clickSig, NFFT=1024, Fs=fs)
plt.title('Frequency-Domain Spectral Content')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)')

plt.tight_layout()
plt.show()
#%% 2) Load the noise levels






#%% 3) Load the arrivals grid and create the SNR at each location



hf =  h5py.File('ExampleData/Spacious_Hawaii_100m_ArrArray_PCHIP_35khz.h5', 'r')
diveIds = list(hf['drift_01'].keys())
dive_grp = hf['drift_01'][diveIds[0]]['frequency_35000']
arrivs = dive_grp['arrivals']
# Index of the verticle array 
ii= 9800


# All of the arrivals for all of the points
arrIdx =  list(arrivs.keys())
group = arrivs[arrIdx[ii]]
arr = pd.DataFrame({k: group[k][()] for k in group.keys()})

# Sensor depth, range, lat and lon
sen_depths = np.unique(arr['rx_depth'])
sen_range = np.unique(arr['rx_range'])
print(ii, sen_range.tolist())

# lat/lon of the verticle array
lat = np.array(dive_grp['lat'][ii])
lon = np.array(dive_grp['lon'][ii])

# Drifter (source) position
d_lat= dive_grp.parent.attrs['start_lat'] 
d_lon= dive_grp.parent.attrs['start_lon'] 
d_depth = dive_grp.parent.attrs['drifter_depth']

# For each unique depth calculate the arrival
for depth in sen_depths:
    
    
    arrSub = arr[arr['rx_depth']==depth]
    t_rcv, rcv   = synthesize_received_broadband(arrSub= arrSub, 
                                                 source = clickSig, 
                                                 tx_depth = depth,  
                                                 fs = fs,
                                                 td_thresh=.5)
    # # Strip off the 0s
    # first_idx = np.argmax(np.abs(rcv) > 0)
    # rcv = rcv[first_idx : first_idx + len(clickSig)]
    # t_rcv = t_rcv[first_idx : first_idx + len(clickSig)]

    # # now compute your PSD on `sig` only:
    # from scipy.signal import welch
    # f, Pxx_received = welch(rcv, fs=fs, nperseg=512, noverlap=256)
    # f, Pxx_original = welch(clickSig, fs=fs, nperseg=512, noverlap=256)
    
    # # convert to dB/Hz (power spectral density)
    # Srec_dB = 10*np.log10(Pxx_received)
    # Ssrc_dB = 10*np.log10(Pxx_original)
    
    # TL_f = Ssrc_dB - Srec_dB   # this is your transmission loss vs frequency

    t_rcv, rcv
    plotTLSigs(clickSig, tt, rcv, t_rcv)

















