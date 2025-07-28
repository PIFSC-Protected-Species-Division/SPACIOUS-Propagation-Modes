# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:01:33 2025

@author: kaity
"""


#%%  <- This is how we create a section of code This section will be the

# This code has examples of code for estimating geometric spreading, absorption,
# bellhop propagation models and parabolic equations. Using code blocks allows
# you to run sections one at a time with the 'run current cell'  button above

#%% Geometric spreading and Absorption


# =============================================================================
# 
# # Geometric spreading using your noggin (and the lecture notes)
# 
# 
# For the first section we will create a function called "geom_spreading' that
# takes in the waterdepth (H) in meters and the range between the source and
# receiver. This can be a single value or an array. The structure of the function
# is laid out below, fill in the blank to finish the function
# =============================================================================



import matplotlib.pyplot as plt
import numpy as np

def geom_spreading(H=1, r=1):
    """
    

    Parameters
    ----------
    H : Waterdepth in meters
    r : range between source and receiver(s)

    Returns
    -------
    TL : Transmission loss in dB.

    """
    TL = H*r
    
    return TL


# Test the function

r = np.linspace(0, 10000, num =200)
TL = geom_spreading(r=r)

plt.plot(r, TL)
plt.xlabel('Range m')
plt.ylabel('Transmission loss dB re 1upa')



# Now create a function for the absoprption of sound as a funciton of 
# the frequency and enviornmental characteristics. There are lots of ways
# to do that several are summarized here: 
# http://resource.npl.co.uk/acoustics/techguides/seaabsorption/physics.html
# 
# Speed of sound in seawater
# http://resource.npl.co.uk/acoustics/techguides/soundseawater/underlying-phys.html


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

# Set frequency values for which you want to calculate the absorption 
# coefficient and call it f. Then calculate the absoprtion coefficient

f = np.linspace(1, 100000, num =500)
absorpCoef = seawater_absorption(freq= f)


# Plot alpha as a function of f

plt.plot(f, absorpCoef) 
ax = plt.gca()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Absorption Coeficient (dB/km)")
ax.set_yscale('log')
ax.set_xscale('log')
plt.grid(True, which="both", ls="-")


# Create a plot of freshwater for the same freqeuncy range by setting the 
# salinity to 0

absorpCoef_fresh = seawater_absorption(freq= f, S =0)

# add it to the above plot



plt.plot(f, absorpCoef, 'k',f, absorpCoef_fresh, 'b') 
ax = plt.gca()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Absorption Coeficient (dB/km)")
plt.title('Absorption coefficient')
ax.set_yscale('log')
ax.set_xscale('log')
plt.grid(True, which="both", ls="-")


#%% Normal Mode Propagation

# In this section we will explore normal modes in an ideal enviornment

import numpy as np
import matplotlib.pyplot as plt

# Define waveguide parameters
depth = 100  # m
c = 1500  # m/s, constant sound speed
frequencies = [50, 200, 800]  # Example frequencies to compare
mode_numbers = np.arange(1, 6)  # First 5 normal modes
x_max = 1000  # max range in meters
x_vals = np.linspace(0, x_max, 500)  # Range values

# Create figure for ray tracing interpretation
plt.figure(figsize=(8, 5))

for mode in mode_numbers:
    launch_angle = np.arcsin((mode * c) / (2 * depth * frequencies[1]))  # Approximate angle
    for sign in [-1, 1]:  # Upward and downward paths
        y_vals = depth / 2 * (1 + sign * np.sin(2 * np.pi * x_vals / (x_max / mode)))
        plt.plot(x_vals, y_vals, label=f'Mode {mode}' if sign == 1 else None)

plt.xlabel("Range (m)")
plt.ylabel("Depth (m)")
plt.title("Ray Path Representation of Normal Modes")
plt.legend()
plt.gca().invert_yaxis()
plt.grid()

# Create figure for waveguide mode shapes
plt.figure(figsize=(8, 5))
z_vals = np.linspace(0, depth, 200)  # Depth values
for f in frequencies:
    plt.plot(np.cos(mode_numbers[1] * np.pi * z_vals / depth), z_vals, label=f'{f} Hz')

plt.xlabel("Mode Amplitude")
plt.ylabel("Depth (m)")
plt.title("Normal Mode Shapes in a 100m Waveguide")
plt.legend()
plt.gca().invert_yaxis()
plt.grid()

plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Define waveguide parameters
depth = 100  # m
c = 1500  # m/s, constant sound speed
frequency = 200  # Hz (choosing one frequency for TL plot)
wavelength = c / frequency  # m
mode_numbers = np.arange(1, 11)  # First 11 normal modes
x_max = 1000  # max range in meters
x_vals = np.linspace(0, x_max, 500)  # Range values
z_vals = np.linspace(0, depth, 200)  # Depth values

# --- Ray Tracing Interpretation ---
plt.figure(figsize=(8, 5))

for mode in mode_numbers:
    launch_angle = np.arcsin((mode * c) / (2 * depth * frequency))  # Approximate angle
    for sign in [-1, 1]:  # Upward and downward paths
        y_vals = depth / 2 * (1 + sign * np.sin(2 * np.pi * x_vals / (x_max / mode)))
        plt.plot(x_vals, y_vals, label=f'Mode {mode}' if sign == 1 else None)

plt.xlabel("Range (m)")
plt.ylabel("Depth (m)")
plt.title("Ray Path Representation of Normal Modes")
plt.legend()
plt.gca().invert_yaxis()
plt.grid()

# --- Waveguide Mode Shapes ---
plt.figure(figsize=(8, 5))
for mode in mode_numbers:
    plt.plot(np.cos(mode * np.pi * z_vals / depth), z_vals, label=f'Mode {mode}')

plt.xlabel("Mode Amplitude")
plt.ylabel("Depth (m)")
plt.title("Normal Mode Shapes in a 100m Waveguide")
plt.legend()
plt.gca().invert_yaxis()
plt.grid()

# --- Transmission Loss Field Calculation ---
X, Z = np.meshgrid(x_vals, z_vals)
pressure_field = np.zeros_like(X, dtype=np.complex_)

for mode in mode_numbers:
    kz = mode * np.pi / depth  # Vertical wavenumber
    kr = np.sqrt((2 * np.pi * frequency / c)**2 - kz**2)  # Horizontal wavenumber
    mode_shape = np.cos(kz * Z)  # Mode shape
    pressure_field += mode_shape * np.exp(1j * kr * X)  # Sum over modes

TL = -20 * np.log10(np.abs(pressure_field) + 1e-6)  # Convert to dB (avoid log(0))

# --- Plot Transmission Loss Field ---
plt.figure(figsize=(8, 5))
plt.contourf(X, Z, TL, levels=30, cmap="viridis")
plt.colorbar(label="Transmission Loss (dB)")
plt.xlabel("Range (m)")
plt.ylabel("Depth (m)")
plt.title("Transmission Loss Field in a 100m Waveguide")
plt.gca().invert_yaxis()
plt.show()




#%% Bellhop Propagation Model using ARLPY

# In ths section we will create ray tracing models using the Bellhop model 
# developed by Mike Porter in the early 2000's. The model uses a fortran 
# back end that others have built Matlab, Pyhton, and potentially R interfaces
# to. Here we are using the Acoustic Research Laboratory version of the code
# and an interfacce to arlpy written by Jay Patel https://patel999jay.github.io/ 


import arlpy.uwapm as pm #https://github.com/org-arl/arlpy
import arlpy.plot as plt
import numpy as np

# Here we create the 'enviornment' which is a dictionary (thing containing
# lists, numbers, characters etc.).This is what bellhop needs to run.
# Open the dictionary in the variable explorer and note it has many default 
# values including soundspeed. .
env = pm.create_env2d()

# What is the maximum water depth? 
# What is the source and receiver depths? (tx, rx)
# What frequency are we running the simulation at?


# Plot the enviornment. Open in the variable explorer and note the 
pm.plot_env(env, width=900)

# Well there we are then, but we can also use our own soundspeed profiles
ssp = [
    [ 0, 1540],  # 1540 m/s at the surface
    [10, 1530],  # 1530 m/s at 10 m depth
    [20, 1532],  # 1532 m/s at 20 m depth
    [25, 1533],  # 1533 m/s at 25 m depth
    [30, 1535]   # 1535 m/s at the seabed
]

# Overwrite the default soundspeed profile with our new one and plot again
env = pm.create_env2d(soundspeed=ssp)


# Create eigenrays using ARLPY
env = pm.create_env2d()
rays = pm.compute_eigenrays(env)
pm.plot_rays(rays, env=env, width=900)

# Now compute the arrival times of the eigenrays and plot
arrivals = pm.compute_arrivals(env)
pm.plot_arrivals(arrivals, width=900)

arrivals[arrivals.arrival_number < 10][['time_of_arrival', 'angle_of_arrival',
                                        'surface_bounces', 'bottom_bounces']]


# Lets make the bathymetry a tad more complicated
bathy = [
    [0, 30],    # 30 m water depth at the transmitter
    [300, 20],  # 20 m water depth 300 m away
    [1000, 25]  # 25 m water depth at 1 km
]

# Create a new enviornment with the bathymetry and the ssp
env = pm.create_env2d(
    depth=bathy,
    soundspeed=ssp)

# Giggle test the enviornment
pm.plot_env(env, width=900)

# Recompute the eigenrays and plot
rays = pm.compute_eigenrays(env)
pm.plot_rays(rays, env=env, width=900)

# To calculate the transmission los grids lets update the eniornment to include
# a bit more information

# Appending ssp and bathy to existing env file
env = pm.create_env2d(
    depth=bathy,
    soundspeed=ssp,
    bottom_soundspeed=1450,
    bottom_density=1200,
    bottom_absorption=1.0,
    tx_depth=15
)

# If we want to create a transmission loss grid, we need to put 'recivers' 
# across our model space so we add receiver ranges and depths to the environment


env['rx_range'] = np.linspace(0, 1000, 1001)
env['rx_depth'] = np.linspace(0, 30, 301)

rays = pm.compute_eigenrays(env)



# Coherent Transmission Loss

#     Preserves phase information across all contributing ray paths.
#     Interference effects (constructive & destructive) are fully included.
#     Results in fine-scale variations (rapid fluctuations) due to phase interactions.
#     Useful for studying detailed wave interference, beamforming, or coherent signal processing.
#     Best for: Predicting structured arrival patterns and applications where phase coherence matters (e.g., sonar, array processing).


tloss = pm.compute_transmission_loss(env)
pm.plot_transmission_loss(tloss, env=env, clim=[-60,-30], width=900)


# Incoherent Transmission Loss

#     Only considers ray amplitudes; phase information is discarded.
#     Avoids rapid interference fluctuations by averaging intensities.
#     Produces a smoother transmission loss field.
#     Useful for environments where phase relationships are unpredictable or not needed.
#     Best for: Estimating general transmission loss trends, long-range propagation modeling, and energy-based analyses.

tloss = pm.compute_transmission_loss(env, mode='incoherent')
pm.plot_transmission_loss(tloss, env=env, clim=[-60,-30], width=900)

#%% Parabolic equations with KADLU
# This probably won't work, but I'm working on it so at some point it will.
# The Kadlu sofware pacakage was developed by meridian but has not been 
# maintained. Some of the code is untested with various updates and it was
# written in packages that do not work on windows. Some of those issues were
# addressed in the updates on canvas but some remain outstanding. 



import kadlu
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from kadlu.geospatial.ocean import Ocean
from kadlu.kadlu.plot_util import plot2D
from kadlu.kadlu.sound.geophony import transmission_loss

# ocean boundaries
bounds = dict(
               south=43.53, north=44.29, west=-59.84, east=-58.48,
               start=datetime(2019,1,1), end=datetime(2019,1,10), 
               top=0, bottom=10000
             )

# data sources#
data_sources = dict(load_bathymetry='gebco', load_temperature='hycom',
                    load_salinity='hycom')

data_sources = dict(load_bathymetry='gebco')


# initialize Ocean instance
gully = Ocean(**bounds, **data_sources)

kadlu.plot_util.plot2D(var='bathymetry', source='gebco', **bounds)

# specify bottom acoustic properties
seafloor = {'sound_speed':1700,'density':1.5,'attenuation':0.5}

sound_source = {'freq': 200, 'lat': 43.9, 'lon': -59.2, 'source_depth': 12}



# initialize transmission loss object
transm_loss = transmission_loss(seafloor=seafloor, 
                                propagation_range=30, 
                                **sound_source,
                                **bounds, 
                                **data_sources)

#calculate transmission loss (be patient, this may take a while ...)
_ = transm_loss.calc(rec_depth=np.linspace(1, 1000, 100), vertical=True, nz_max=1000)

#transmission loss with bathy and ssp superimposed
fig = transm_loss.plot_vert(angle=20, max_depth=1770, show_ssp=True)

fig = transm_loss.plot_horiz(rec_depth_idx=99)  # transmission loss at 30 m depth (receiver no. 2)





