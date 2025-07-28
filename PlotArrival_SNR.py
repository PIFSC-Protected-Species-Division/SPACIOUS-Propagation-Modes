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


#%%
# --- Main Evaluation Block ---
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert, welch
    
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
    
    
    # Add phase to the recording 
    # 1) get analytic recording & its phase
    segment_analytic = hilbert(segment)
    phase           = np.angle(segment_analytic)

    # 2) define your new envelope (same length as segment)
    #    for example, env = np.abs(some_synthetic_signal)
    env = np.abs(segment)    # or whatever real‐valued envelope you like
    
    # Add phase information
    phase = np.angle(segment_analytic)
    # 2) define your new envelope (same length as segment)
    #    for example, env = np.abs(some_synthetic_signal)
    env = np.abs(segment)    # or whatever real‐valued envelope you like

    # 3) graft the phase on
    synthetic_with_phase = env * np.exp(1j * phase)

    # 4) normalize if needed
    synthetic_with_phase /= np.max(np.abs(synthetic_with_phase))
    
    # Compute PSD using Welch's method

    plt.psd(segment, 256, 1/samplerate)
    plt.ylabel('PSD(db)')
    plt.xlabel('Frequency')
    plt.title('matplotlib.pyplot.psd() Example\n',
              fontsize = 14, fontweight ='bold')
    
    # Source p2p and SEL dB
    src_lvl_ptp = 20 * np.log10(np.ptp(np.real(segment)))
    src_lvl_ptp = np.round(src_lvl_ptp,1)

    src_lvl_rms = 20 * np.log10(np.sqrt(np.mean(np.real(segment)**2)))
    src_lvl_rms = np.round(src_lvl_rms,1)
    
#%% Get the environmental data    
    # Load the hdf5 file
    hf = h5py.File('Spacious_Hawaii_diveDepth_ArrArray_PCHIP_35khz_20km - Copy.h5', 'r')
    diveIds = list(hf['drift_01'].keys())
    dive_grp = hf['drift_01'][diveIds[0]]['frequency_35000']
    run_ids = np.array(dive_grp['arrivals'])
    depth = np.array(dive_grp['depth'])
    lat = np.array(dive_grp['lat'])
    lon = np.array(dive_grp['lon'])
    
    # sensor position
    d_lat= dive_grp.parent.attrs['start_lat'] 
    d_lon= dive_grp.parent.attrs['start_lon'] 
    d_depth = dive_grp.parent.attrs['drifter_depth']
    

    # Pull out an example arrival and convert to database 
    runIndex =3300
    arr0 = dive_grp['arrivals'][run_ids[runIndex]]
    
    # convert the group to a dataframe
    data = {}
    for name in arr0.keys():
        # grab raw array
        raw = arr0[name][()]            # for netCDF4 groups use arr0.variables[name][:]
        # unmask if it’s a masked array
        arr = raw.data if isinstance(raw, np.ma.MaskedArray) else raw
        arr = arr.flatten()
    
        # cast to the dtype your working DF uses
        if name in ('arrival_amplitude', 'complex_time_of_arrival'):
            arr = arr.astype(np.complex128)
        elif name in (
            'tx_depth_ndx', 'rx_depth_ndx', 'rx_range_ndx',
            'arrival_number', 'surface_bounces', 'bottom_bounces'
        ):
            arr = arr.astype(np.int64)
        else:
            # everything else is float64
            arr = arr.astype(np.float64)
    
        data[name] = arr
        
    # build the DataFrame
    arrivals_df = pd.DataFrame(data)
        
    # time of first arrival
    depth_idxs = np.unique(arrivals_df['rx_depth'])
    hydData = arrivals_df[arrivals_df['rx_depth']==depth_idxs[-4]]
    
    # initial time and impulse response
    ir = pm.arrivals_to_impulse_response(hydData, fs=samplerate, abs_time=True)

    # for plotting define time vector and box on 'good' data
    to=np.min(hydData['time_of_arrival'])
    time_ir=np.arange(len(ir))/samplerate
    stir=(time_ir>to-0.01) & (time_ir<to+0.05)
        
    # figure attenuation assuming spherical
    r=hydData['rx_range'].iloc[0] # m
    print(r)
    
    # Attenuation
    att0=9.55 # dB/km # absorption for 35 kHz
    att_molecular = att0*(r/1000) #total dB molecular
    att_spherical = 18*np.log10(r) # total dB spherical
    
    # Total attenuation in dB from shperical and cylindrical spreading
    totalDb = np.round(att_molecular+att_spherical,1)
    
    # combine spherical and molecular spreading, convert to linear. 
    att = 10**((att_molecular+att_spherical)/18)

    
    # Create the attenuated signal from spherical/molecular loss and the
    # by the convolution
    outputSig=np.convolve(segment,np.real(ir))[:len(ir)]
    convolvedRaw = np.real(outputSig)
    attenuatedSig = segment/att # no phase involved
    
    # Convolve the attenuated signal with the impulse respones
    outputSig=np.convolve(segment/att,np.real(ir))[:len(ir)]
    convolved_attenuated = np.real(outputSig)
    
    
    # plt.figure()
    # # Capture the Line2D objects 
    # line1, = plt.plot(time_ir[stir], 
    #                   (np.abs(ir[stir])), 
    #                   label = 'Impulse response')
    # line3, = plt.plot(to + tt/1000, 
    #                   attenuatedSig, 
    #                   label='Attenuated signal')
    # line2, = plt.plot(time_ir[stir], 
    #                   convolvedRaw[stir], 
    #                   label='Convolved signal')
    

    
    # # line4, = plt.plot(to + tt/1000, 
    # #                   segment, label='Origional signal')

    
    # plt.legend()

    # plt.title(f'Origional and Convolved Signals Run at {40000/1000} kHz')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude (units)')


    # peak to peak calculations for the source
    ref_p2p = 20*np.log10(np.ptp(attenuatedSig)) # Ref: Source- sperhical and molecular
    rec_p2p = 20*np.log10(np.ptp(convolvedRaw[stir])) # Convolved ref
    
    att_conv_p2p = 20*np.log10(np.ptp(convolved_attenuated[stir])) 

    # Attenuation using the convolution method 
    convolvAtt_ptp = np.round(src_lvl_ptp-rec_p2p,1) #79 dB

    # Peak to peak difference between the origional signal and the reference signal
    # (attenuated with frequency and spherical spreading)
    simpleAtt_ptp = np.round(src_lvl_ptp-ref_p2p, 1) #77 dB

    print(f"Observed attenuation {simpleAtt_ptp} expected attenuation {totalDb} ptp"+
          f" at {np.round(r/1000,1)} km")
    print(f"Observed attenuation {convolvAtt_ptp} using convolution method ptp"+
          f" at {np.round(r/1000,1)} km")

  
    
    # Plot PSD
    plt.figure()
    plt.psd(segment, 256, 1/samplerate, label='Source')
    plt.psd(convolvedRaw[stir], 256, 1/samplerate, label='Convolved Source')
    plt.psd(attenuatedSig, 256, 1/samplerate, label='Attenuated signal')
    
    plt.ylabel('PSD(db)')
    plt.xlabel('Frequency')
    plt.legend()
    plt.title(f"PSD of signal at {np.round(r/1000,1)} km",
              fontsize = 14, fontweight ='bold')
    
