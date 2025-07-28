# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:25:26 2025

@author: kaity
Run from 'NewPropagation' enviornment- has ARLPY with pchip interpolation and
I broke the base enviornment. Oops.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401


#from skimage import measure
import h5py
import arlpy.uwapm as pm
import pandas as pd
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import hilbert, welch
# Function to input signal and arrival array and return SNR of the convolved
# signal with an option to plot


# Modified version of arrivals to impulse response for speed
import numpy as _np
import pandas as pd

def arrivals_to_impulse_response(arrivals, fs, abs_time=False):
    """Convert arrival times and coefficients to an impulse response.

    :param arrivals: a Pandas DataFrame or dict with 'time_of_arrival' and 'arrival_amplitude'
    :param fs: sampling rate (Hz)
    :param abs_time: use absolute time if True, otherwise relative to first arrival
    :returns: 1D complex128 impulse response
    """
    # Handle dicts or DataFrames
    is_dict = isinstance(arrivals, dict)
    toa = arrivals['time_of_arrival'] if is_dict else arrivals.time_of_arrival
    amp = arrivals['arrival_amplitude'] if is_dict else arrivals.arrival_amplitude

    t0 = 0 if abs_time else _np.min(toa)
    irlen = int(_np.ceil((_np.max(toa) - t0) * fs)) + 1
    ir = _np.zeros(irlen, dtype=_np.complex128)

    for i in range(len(toa)):
        ndx = int(_np.round((toa[i].real - t0) * fs))
        if 0 <= ndx < irlen:
            ir[ndx] = amp[i]

    return ir

def p2pArrivalSNR(arrivals, segment, fs, r, plotPSD = False, plotTime= False):
    
    
    ir = arrivals_to_impulse_response(arrivals, fs=fs, abs_time=True)
    
    # for plotting define time vector and box on 'good' data
    
    
    outputSig=np.convolve(segment, ir)[:len(ir)]
    
    outputSig_real = np.real(outputSig)
    att_conv_p2p = 20*np.log10(np.ptp(outputSig_real)) 
    
    if plotPSD:
        
        time_ir=np.arange(len(ir))/fs
        to=np.min(arrivals['time_of_arrival'])
        stir=(time_ir>to-0.01) & (time_ir<to+0.5)
        
        # Plot PSD
        plt.figure()
        plt.psd(segment, 128, 1/fs, label='Source')
        plt.psd(outputSig_real, 128, 1/fs, label='Convolved Source')

        
        plt.ylabel('PSD(db)')
        plt.xlabel('Frequency')
        plt.legend()
        plt.title(f"PSD of signal at {np.round(r/1000,1)} km",
                  fontsize = 14, fontweight ='bold')
        
        plt.figure()
        plt.figure()
        
        # Capture the Line2D objects 
        line1, = plt.plot(time_ir[stir], 
                          (np.abs(ir[stir])), 
                          label = 'Impulse response')
        
        line2, = plt.plot(time_ir[stir], 
                          outputSig_real[stir], 
                          label='Convolved signal')
        
        
        
    
    return np.round(att_conv_p2p,1), outputSig_real
    




#%%
# --- Main Evaluation Block ---
if __name__ == "__main__":
   
    #%% Set up the signal 
    # Use the click from Barkley et al
    wav_path = "C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\LF_1705_20171028_010934_441.wav"
    samplerate, audiodata = wavfile.read(wav_path)
    
    # start time, end time, channel
    t_start = 32.58
    t_end   = 32.60
    chan = 4

    segment = audiodata[int(round(t_start * samplerate)):int(round(t_end   * samplerate)), chan] 
    tt = np.arange(0, len(segment))/ samplerate
    plt.figure()
    plt.plot(tt, segment)
    plt.plot(tt, segment/np.mean(audiodata)) # don't think this actuallhelps
    plt.show()
    
    # Scale the segment so that it's 220 dB re 1upa p2p
    outP2P = 220
    init_p2pdB = 20*np.log10(np.ptp(segment))
    addP2P_linear = 10**(((outP2P-init_p2pdB)/20))
    segment= segment*addP2P_linear
    
    
    
#%% Get the environmental data    
    # Columns needed by the arrivals to impulse function
    required_cols = [
        'time_of_arrival', 'arrival_amplitude',
        'tx_depth_ndx', 'rx_depth_ndx', 'rx_range_ndx'
    ]
    
    with h5py.File('Spacious_Hawaii_diveDepth_ArrArray_PCHIP_35khz_20km - Copy.h5', 'r') as hf:
        diveIds = list(hf['drift_01'].keys())
        dive_grp = hf['drift_01'][diveIds[0]]['frequency_35000']
    
        run_ids = list(dive_grp['arrivals'].keys())
        depthGrid = np.array(dive_grp['depth'])
        lat = np.array(dive_grp['lat'])
        lon = np.array(dive_grp['lon'])
    
        d_lat = dive_grp.parent.attrs['start_lat']
        d_lon = dive_grp.parent.attrs['start_lon']
        d_depth = dive_grp.parent.attrs['drifter_depth']
    
        p2p_grid = np.full_like(depthGrid, np.nan, dtype=np.float64)  # make sure it's float for assigning p2p
    
        for runIndex in range(len(run_ids)):
            arr0 = dive_grp['arrivals'][run_ids[runIndex]]
            
            #range m 
            r = np.array(arr0['rx_range'])
    
            # Extract fields once into dict
            data = {}

            for name, ds in arr0.items():
                if name not in required_cols:
                    continue  # skip unused fields
            
                arr = ds[()]
                if isinstance(arr, np.ma.MaskedArray):
                    arr = arr.data
                if arr.ndim > 1:
                    arr = arr.ravel()
                data[name] = arr  # Let pandas or arlpy handle dtype
            
            arrivals_df = pd.DataFrame(data)
            
            # Cast only what's needed
            arrivals_df['rx_depth_ndx'] = arrivals_df['rx_depth_ndx'].astype(np.int64)
    
            # Only create the DataFrame if needed
            arrivals_df = pd.DataFrame(data)
    
            # Check if this runIndex exists in depthGrid (avoid shape mismatch errors)
            if runIndex >= depthGrid.shape[0]:
                continue
    
            # Only look at valid depth indices (column dimension)
            valid_depths = depthGrid[runIndex, :] > 0
            depth_idxs = np.where(valid_depths)[0]
            
            
    
            for depthIdx in depth_idxs:
                
                depth_idx = data['rx_depth_ndx']
                mask = (depth_idx == depthIdx)
                hyd_range = np.unique(mask*r)[1]
                
                hydData = {k: v[mask] for k, v in data.items()}
               
                # if hydData.empty:
                #     continue
    
                ptpOut, _ = p2pArrivalSNR(hydData, segment, samplerate, hyd_range)
                p2p_grid[runIndex, depthIdx] = ptpOut
    
            print(runIndex)
            
            
        
    
    # # initial time and impulse response
    # ir = pm.arrivals_to_impulse_response(hydData, fs=samplerate, abs_time=True)

    # # for plotting define time vector and box on 'good' data
    # to=np.min(hydData['time_of_arrival'])
    # time_ir=np.arange(len(ir))/samplerate
    # stir=(time_ir>to-0.01) & (time_ir<to+0.05)
        
    # # figure attenuation assuming spherical
    # r=hydData['rx_range'].iloc[0] # m
    # print(r)
    
    # # Attenuation
    # att0=9.55 # dB/km # absorption for 35 kHz
    # att_molecular = att0*(r/1000) #total dB molecular
    # att_spherical = 18*np.log10(r) # total dB spherical
    
    # # Total attenuation in dB from shperical and cylindrical spreading
    # totalDb = np.round(att_molecular+att_spherical,1)
    
    # # combine spherical and molecular spreading, convert to linear. 
    # att = 10**((att_molecular+att_spherical)/18)

    
    # # Create the attenuated signal from spherical/molecular loss and the
    # # by the convolution
    # outputSig=np.convolve(segment,np.real(ir))[:len(ir)]
    # convolvedRaw = np.real(outputSig)
    # attenuatedSig = segment/att # no phase involved
    
    # # Convolve the attenuated signal with the impulse respones
    # outputSig=np.convolve(segment/att,np.real(ir))[:len(ir)]
    # convolved_attenuated = np.real(outputSig)
    
    
    # # plt.figure()
    # # # Capture the Line2D objects 
    # # line1, = plt.plot(time_ir[stir], 
    # #                   (np.abs(ir[stir])), 
    # #                   label = 'Impulse response')
    # # line3, = plt.plot(to + tt/1000, 
    # #                   attenuatedSig, 
    # #                   label='Attenuated signal')
    # # line2, = plt.plot(time_ir[stir], 
    # #                   convolvedRaw[stir], 
    # #                   label='Convolved signal')
    

    
    # # # line4, = plt.plot(to + tt/1000, 
    # # #                   segment, label='Origional signal')

    
    # # plt.legend()

    # # plt.title(f'Origional and Convolved Signals Run at {40000/1000} kHz')
    # # plt.xlabel('Time (s)')
    # # plt.ylabel('Amplitude (units)')


    # # peak to peak calculations for the source
    # ref_p2p = 20*np.log10(np.ptp(attenuatedSig)) # Ref: Source- sperhical and molecular
    # rec_p2p = 20*np.log10(np.ptp(convolvedRaw[stir])) # Convolved ref
    
    # att_conv_p2p = 20*np.log10(np.ptp(convolved_attenuated[stir])) 

    # # Attenuation using the convolution method 
    # convolvAtt_ptp = np.round(src_lvl_ptp-rec_p2p,1) #79 dB

    # # Peak to peak difference between the origional signal and the reference signal
    # # (attenuated with frequency and spherical spreading)
    # simpleAtt_ptp = np.round(src_lvl_ptp-ref_p2p, 1) #77 dB

    # print(f"Observed attenuation {simpleAtt_ptp} expected attenuation {totalDb} ptp"+
    #       f" at {np.round(r/1000,1)} km")
    # print(f"Observed attenuation {convolvAtt_ptp} using convolution method ptp"+
    #       f" at {np.round(r/1000,1)} km")

  
    
    # # Plot PSD
    # plt.figure()
    # plt.psd(segment, 256, 1/samplerate, label='Source')
    # plt.psd(convolvedRaw[stir], 256, 1/samplerate, label='Convolved Source')
    # plt.psd(attenuatedSig, 256, 1/samplerate, label='Attenuated signal')
    
    # plt.ylabel('PSD(db)')
    # plt.xlabel('Frequency')
    # plt.legend()
    # plt.title(f"PSD of signal at {np.round(r/1000,1)} km",
    #           fontsize = 14, fontweight ='bold')
    
