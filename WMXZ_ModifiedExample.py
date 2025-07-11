import numpy as np
import arlpy.uwapm as pm
import arlpy.plot as arlplt
import matplotlib.pyplot as plt


def seawater_absorption(freq =1500,Z=0, T=5, S=35, pH=8):
    '''
    Following Kinsler, et al "Fundamentals of Acoustics, Fourth Edition" p. 226-228.

    Parameters
    ----------
    freq : float
        frequency in Hz. The default is 1500.
    Z : float, optional
        depth in km. The default is 0.
    T : float, optional
        temperature in C. The default is 5.
    S : float, optional
        Salinity in ppt. The default is 35.
    pH : float, optional
        Water ph. The default is 8.

    Returns
    -------
    alpha, absorption coeficient in dB/km.

    '''
        
    f_1 = 780*np.exp(T/29)
    f_2 = 42000*np.exp(T/18)
    A = 0.083*(S/35)*np.exp(T/31 - Z/91 + 1.8*(pH-8))
    B = 22*(S/35)*np.exp(T/14-Z/6)
    C = 4.9E-10*np.exp(-T/26 - Z/25)
    boric_acid = A/(f_1**2+freq**2) # contribution from boric acid
    MgSO4 = B/(f_2**2+freq**2) # contribution from MgSO4
    hydrostatic = C # contribution from hydrostatic pressure
    alpha = (boric_acid + MgSO4 + hydrostatic)*freq**2
        
    return alpha

# Frequncy and hydrophone range
freq = 30000
HydRange = 20000

# Just keep it scaled
bathy = [
    [0, 2500],    # 30 m water depth at the transmitter
    [HydRange/3, 1800],  # 20 m water depth 300 m away
    [HydRange, 2300]  # 25 m water depth at 1 km
]
if 0:
    ssp=1500
    Title='Received sound for Isovelocity'
else:
    ssp = [
        [ 0, 1540],  # 1540 m/s at the surface
        [40, 1530],  # 1530 m/s at 10 m depth
        [80, 1532],  # 1532 m/s at 20 m depth
        [100, 1533],  # 1533 m/s at 25 m depth
        [500, 1535],   # 1535 m/s at the seabed
        [2501, 1535]
    ]
    Title='Received sound for "realistic" Sound Speed Profile'

env = pm.create_env2d(
    depth=bathy,
    soundspeed=ssp,
    bottom_soundspeed=1250,
    bottom_density=2700,
    bottom_absorption=0.1,
    frequency=freq,
    tx_depth=150,
    nbeams=0)


env['rx_range'] = HydRange
env['rx_depth'] = 900 # again just let things scale

# for visualization of the geometry
rays_eigen = pm.compute_eigenrays(env)
rays_all = pm.compute_rays(env)
pm.pyplot_rays(rays_eigen)
#pm.pyplot_rays(rays_all)

arrivals = pm.compute_arrivals(env)
pm.pyplot_arrivals(arrivals, dB=True)

# time of first arrival
to=np.min(arrivals['time_of_arrival'])

# simulate Zc beaked whale click
def zcSig(tt,f0,fm,aa,bb,cc):
    return (aa*tt)**bb * np.exp(-(aa*tt)**cc + 2*np.pi*1j*(f0+fm*tt)*tt)
#
fs=192 #kHz
ts=0.3 #ms
tt=np.arange(0,ts,1/fs)

f0=freq/1000
fm=60
aa=13
bb=1.5
cc=1.5
ss=zcSig(tt,f0,fm,aa,bb,cc)
ss=ss/np.max(np.abs(ss)) # source

# Source p2p and SEL dB
src_lvl_ptp = 20 * np.log10(np.ptp(np.real(ss)))
src_lvl_ptp = np.round(src_lvl_ptp,1)

src_lvl_rms = 20 * np.log10(np.sqrt(np.mean(np.real(ss)**2)))
src_lvl_rms = np.round(src_lvl_rms,1)

print(f"Peak to peak source level of simulated Zc click {src_lvl_ptp} dB re 1upa")
print(f"RMS source level of simulated Zc click {src_lvl_ptp} dB re 1upa")

# simulated received signal (convolve emitted signal with impuse response)
fsamp=fs*1000

ir = pm.arrivals_to_impulse_response(arrivals, fs=fsamp,abs_time=True)

# for plotting define time vector and box on 'good' data
time_ir=np.arange(len(ir))/fsamp
stir=(time_ir>to-0.01) & (time_ir<to+0.005)


att0=seawater_absorption(freq) # dB/km # absorption for 40 kHz
att_molecular = att0*(HydRange/1000) #total dB molecular
att_spherical = 20*np.log10(HydRange) # total dB spherical

# Total attenuation in dB from shperical and cylindrical spreading
totalDb = np.round(att_molecular+att_spherical,1)

print(f"Total attenuation from spherical and molecular relaxation {totalDb} dB re...")

# combine spherical and molecular spreading, convert to linear. 
att = 10**((att_molecular+att_spherical)/20)


dat=np.convolve(ss,ir)[:len(ir)]
xlim=[to-0.00025,to+0.0015]
plt.plot(time_ir[stir], np.abs(ir[stir]))


plt.figure()
# Capture the Line2D objects 
line1, = plt.plot(time_ir[stir], 
                  (np.abs(ir[stir])), 
                  label = 'Impulse response')
# line2, = plt.plot(time_ir[stir][2000] + np.arange(0, len(ss))/(fs*1000),
#                  (np.real(ss)), 
#                   label = 'Source') # 
line3, = plt.plot(to + tt/1000, 
                  (np.real(ss)/att), 
                  label='Attenuated signal')
line4, = plt.plot(time_ir[stir], 
                 (np.real(dat[stir])), 
                  label='Convolved signal')


plt.legend()

plt.title(f'Origional and Convolved Signals Run at {freq/1000} kHz')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (units)')
plt.xlim(xlim)


# peak to peak calculations for the source
ref_p2p = 20*np.log10(np.ptp(np.real(ss)/att)) # Ref: Source- sperhical and molecular
rec_p2p = 20*np.log10(np.ptp(np.real(dat[stir]))) # Convolved ref

# Attenuation using the convolution method 
convolvAtt_ptp = np.round(src_lvl_ptp-rec_p2p,1) #79 dB

# Peak to peak difference between the origional signal and the reference signal
# (attenuated with frequency and spherical spreading)
simpleAtt_ptp = np.round(src_lvl_ptp-ref_p2p, 1) #77 dB

print(f"Observed attenuation {simpleAtt_ptp} expected attenuation {totalDb} ptp"+
      f" at {HydRange/1000} km")
print(f"Observed attenuation {convolvAtt_ptp} using convolution method ptp"+
      f" at {HydRange/1000} km")

# Try RMS
# peak to peak calculations for the source

ref_rms = 20*np.log10(np.sqrt(np.mean((np.real(ss)/att)**2))) # Ref: Source- sperhical and molecular
rec_rms= 20*np.log10(np.sqrt(np.mean(np.real(dat[stir])**2))) # Convolved ref

# rms difference between the origional signal and the convolved signal
convolvAtt_rms= np.round(src_lvl_rms-rec_rms,1) # 56 dB

# rms difference between the origional signal and the simply attenuated signal
simpleAtt_rms = np.round(src_lvl_rms-ref_rms,1) #28 dB


print(f"Observed attenuation {simpleAtt_rms} expected attenuation {totalDb} rms"+
      f" at {HydRange/1000} km")
print(f"Observed attenuation {convolvAtt_rms} using convolution method rms"+
      f" at {HydRange/1000} km")



