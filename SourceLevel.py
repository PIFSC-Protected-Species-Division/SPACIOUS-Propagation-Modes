import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
from scipy.signal import welch
from scipy.interpolate import interp1d
import matplotlib as mpl



def sperm_whale_click_psd(source_level_dbpp, freq, plot=False,
                           sample_rate=None, click_duration=None,
                           click_type='usual'):
    """
    Generate synthetic sperm whale click (usual, creak, or slow) based on
    published spectral shapes from Madsen et al. (2002).
    """

    # Define click-specific parameters from Table 1
    click_specs = {
        'usual': {'fc': 15000, 'bw_10db': 19000, 'sr': 200000, 'dur': 0.0012},
        'creak': {'fc': 15000, 'bw_10db': 17000, 'sr': 200000, 'dur': 0.001},
        'slow':  {'fc': 3000,  'bw_10db': 4000,  'sr': 48000,  'dur': 0.005}
    }

    if click_type not in click_specs:
        raise ValueError("click_type must be 'usual', 'creak', or 'slow'")

    # Extract parameters
    fc = click_specs[click_type]['fc']
    bw = click_specs[click_type]['bw_10db']
    sr = sample_rate or click_specs[click_type]['sr']
    dur = click_duration or click_specs[click_type]['dur']

    # Gaussian std to match -10 dB bandwidth
    sigma = (bw / 2) / (np.sqrt(2 * np.log(10)))

    N = int(sr * dur)
    if N % 2: N += 1  # make even
    f = np.fft.rfftfreq(N, d=1/sr)

    # Symmetric Gaussian envelope
    envelope = np.exp(-0.5 * ((f - fc) / sigma)**2)

    # Optional: shape tweaks for slow clicks (slightly skewed or non-Gaussian)
    if click_type == 'slow':
        envelope *= (1 - 0.1 * ((f - fc) / fc))  # taper slightly to low side

    # # Random phase- Option 1
    # phase = np.exp(2j * np.pi * np.random.rand(len(f)))
    # spectrum = envelope * phase
    ## Time-domain waveform
    #click = np.fft.irfft(spectrum, n=N)
    # from scipy.signal import minimum_phase
    
    # Use real-valued spectrum and convert to minimum-phase -option 2
    from scipy.signal import minimum_phase
    spectrum = envelope
    impulse_response = np.fft.irfft(spectrum, n=N)
    click = minimum_phase(impulse_response, method='hilbert')

    
    
    # ## Use real-valued spectrum for smoother, repeatable PSD- option 3
    # spectrum = envelope
    # click = np.fft.irfft(spectrum, n=N)

    

    # Scale to match source level
    p_pp = 10 ** (source_level_dbpp / 20)
    p_rms = p_pp / (2 * np.sqrt(2))
    click *= p_rms / np.sqrt(np.mean(click**2))

    # Compute Welch PSD
    from scipy.signal import welch
    f_welch, Pxx = welch(click, fs=sr, nperseg=min(1024, N), scaling='density')
    Pxx_db = 10 * np.log10(Pxx)

    # Interpolate PSD
    from scipy.interpolate import interp1d
    freq = np.atleast_1d(freq)
    interp_func = interp1d(f_welch, Pxx_db, bounds_error=False, fill_value="extrapolate")
    psd_values = interp_func(freq)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(f_welch, Pxx_db)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (dB re 1 µPa²/Hz)')
        plt.title(f'Power Spectral Density')
        plt.xlim(0, sr // 2)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

    return psd_values, click

# Example frequencies to evaluate
frequencies = [1000, 5000, 10000, 15000, 20000]
frequencies = np.arange(0,48000,1000)

# Compute PSD
freqs = np.linspace(1000, 30000, 100)
psd_usual = sperm_whale_click_psd(0, freqs, sample_rate =96000, plot=False, click_type='usual')
psd_creak = sperm_whale_click_psd(0, freqs, sample_rate =96000, plot=True, click_type='creak')
psd_slow  = sperm_whale_click_psd(0, freqs, sample_rate =96000, plot=True, click_type='slow')


##############################################################################
#%% Different click simulation approach

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def sperm_whale_click_psd_time_domain(source_level_dbpp, freq, plot=False,
                                      sample_rate=None, click_duration=None,
                                      click_type='usual'):
    """
    Generate a synthetic sperm whale click using the time-domain first derivative
    of a Gaussian. This version avoids phase artifacts and produces a smooth PSD.
    """

    # Click type parameters
    click_specs = {
        'usual': {'fc': 15000, 'bw_10db': 19000, 'sr': 200000, 'dur': 0.0012},
        'creak': {'fc': 15000, 'bw_10db': 17000, 'sr': 200000, 'dur': 0.001},
        'slow':  {'fc': 3000,  'bw_10db': 4000,  'sr': 48000,  'dur': 0.005}
    }

    if click_type not in click_specs:
        raise ValueError("click_type must be 'usual', 'creak', or 'slow'")

    fc = click_specs[click_type]['fc']
    bw = click_specs[click_type]['bw_10db']
    sr = sample_rate or click_specs[click_type]['sr']
    dur = click_duration or click_specs[click_type]['dur']

    N = int(sr * dur)
    if N % 2: N += 1
    t = np.linspace(-dur/2, dur/2, N, endpoint=False)

    # Gaussian derivative
    sigma = 1 / (2 * np.pi * (bw / 2))  # approx relationship between BW and std dev
    gauss = np.exp(-t**2 / (2 * sigma**2))
    click = -t * gauss  # First derivative (antisymmetric)

    # Normalize to RMS pressure
    p_pp = 10 ** (source_level_dbpp / 20)     # peak pressure from dBpp
    p_rms = p_pp / (2 * np.sqrt(2))           # RMS of sine wave with same peak
    click *= p_rms / np.sqrt(np.mean(click**2))

    # PSD estimation
    f_welch, Pxx = welch(click, fs=sr, nperseg=min(1024, N), scaling='density')
    Pxx_db = 10 * np.log10(Pxx)

    # Interpolate PSD to match requested frequencies
    from scipy.interpolate import interp1d
    freq = np.atleast_1d(freq)
    interp_func = interp1d(f_welch, Pxx_db, bounds_error=False, fill_value="extrapolate")
    psd_values = interp_func(freq)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(f_welch, Pxx_db, label='Click PSD')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (dB re 1 µPa²/Hz)')
        plt.title(f'Power Spectral Density ({click_type} click)')
        plt.xlim(0, sr // 2)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return psd_values, click

freqs = np.linspace(1000, 30000, 100)
psd_usual, click_usual = sperm_whale_click_psd_time_domain(
    source_level_dbpp=220,
    freq=freqs,
    sample_rate=96000,
    click_type='usual',
    plot=True)

psd_usual = sperm_whale_click_psd_time_domain(0, freqs, sample_rate =96000, plot=False, click_type='usual')
psd_creak = sperm_whale_click_psd_time_domain(0, freqs, sample_rate =96000, plot=True, click_type='creak')
psd_slow  = sperm_whale_click_psd_time_domain(0, freqs, sample_rate =96000, plot=True, click_type='slow')


##########################################################################
#%% Attenuate the clicks, simulation
#############################################################################
# Simulate TL values at coarse frequency intervals (e.g., from Bellhop)
tl_freqs = np.arange(5000, 35001, 5000)  # Hz
tl_values_db = np.array([20, 25, 30, 35, 40, 45, 50])  # dB TL at each freq

# Define target frequency array (e.g., output of sperm_whale_click_psd)
target_freqs = np.linspace(1000, 30000, 100)  # Hz

# Get PSD for a 'usual' click at these freqs (you already had this)
psd_values_db, _ = sperm_whale_click_psd(
    source_level_dbpp=220,  # or whatever value you're simulating
    freq=target_freqs,
    sample_rate=96000,
    plot=False,
    click_type='usual'
)

# Interpolate TL to match the PSD frequency bins
tl_interp_func = interp1d(
    tl_freqs, tl_values_db,
    bounds_error=False,
    fill_value=(tl_values_db[0], tl_values_db[-1])  # extend TL flat outside bounds
)
tl_interp_values = tl_interp_func(target_freqs)

# Apply attenuation in dB
received_psd_db = psd_values_db - tl_interp_values

# Optional: Plot both PSD and attenuated spectrum
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(target_freqs, psd_values_db, label="Source PSD")
plt.plot(target_freqs, received_psd_db, label="Received PSD (after TL)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (dB re 1 µPa²/Hz)")
plt.title("Sperm Whale Click PSD Before and After Transmission Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



