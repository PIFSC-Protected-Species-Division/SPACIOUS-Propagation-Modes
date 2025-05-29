
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal.windows import tukey


def tukeyClick(fs= 200000, dur_s=2.5, dbPtP = 205, padding_s =1, lfButter=4000,
               hfButter = 60000, order =3,
               reference_pressure =1e-6):
    '''
    Create a broadband signal following/approximating kusel et al 2016

    Parameters
    ----------
    fs : TYPE, Sample rate hz
        DESCRIPTION. The default is 200000.
    dur_s : TYPE, Signal duration seconds
        DESCRIPTION. The default is 2.5.
    dbPtP : TYPE, Peak to Peak signal amplitude
        DESCRIPTION. The default is 205.
    padding_s : TYPE, Seconds of padding
        DESCRIPTION. The default is 1.
    lfButter : TYPE, Low frequency band of butterworth filter
        DESCRIPTION. The default is 1000.
    hfButter : TYPE, High frequency band of butterworth filter
        DESCRIPTION. The default is 60000.
    order : TYPE, filter order
        DESCRIPTION. The default is 3.
    reference_pressure : TYPE, reference pressure
        DESCRIPTION. The default is 1e-6.

    Returns
    -------
    signal_filtered : TYPE
        DESCRIPTION.

    '''
    
    n_samples = int(fs * dur_s)  # Number of samples
    tukey_window = tukey(n_samples)
    
    # Apply inverse FFT to get time-domain signal
    signal_time_domain = np.fft.ifft(tukey_window).real
    
    # Zero padding: 1 ms before and after the signal
    padding_samples = int(fs * padding_s)  
    signal_padded = np.pad(signal_time_domain, 
                           (padding_samples, padding_samples), 'constant')
    
    

    # Apply Butterworth bandpass filter (10-60 kHz)
    nyquist = 0.5 * fs
    low = lfButter / nyquist
    high = hfButter / nyquist
    b, a = butter(order, [low, high], btype='band')
    signal_filtered = filtfilt(b, a, signal_padded)
    
    # Scale to 205 dB re 1 μPa
    peak_to_peak_linear = 10 ** (dbPtP / 20) * reference_pressure
    signal_scaled = signal_filtered / np.max(np.abs(signal_filtered)) * peak_to_peak_linear
    
    tt = np.arange(len(signal_scaled)) / fs   # Convert to milliseconds
    
    return signal_filtered, tt





import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
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
def thorp_alpha_dbkm(f_khz):
    """Thorp absorption [dB/km] for f in kHz."""
    f2 = f_khz**2
    return (
        0.11 * f2 / (1 + f2)           +        # boric acid
        44   * f2 / (4100 + f2)         +        # MgSO4
        2.75e-4 * f2                   +        # pure water (HF)
        0.003                                    # relaxation floor
    )

def apply_absorption(source, r_m, fs):
    """
    Return `source` after frequency-dependent absorption over distance r_m.
    Works on real time-series; keeps length unchanged.
    """
    N   = len(source)
    freqs = rfftfreq(N, d=1/fs) / 1000          # kHz
    #alpha = thorp_alpha_dbkm(freqs) / 1000      # dB/m
    alpha = seawater_absorption(freqs)/1000
    A     = 10**(-alpha * r_m / 20)             # linear gain

    S     = rfft(source)
    return irfft(S * A, n=N)                    # real, same length

def synthesize_received_broadband(arrSub, source, tx_depth, fs, td_thresh =1):
    '''
    

    Parameters
    ----------
    arrSub : dataframe
        arrival array for a single hydrohpone location.
    source : array
        Time domain signal to be convolved with the arrival array
    tx_depth : float
        Tranceiver depth in meters
    fs : int
        sample rate in hz.
    fs: float
        Time delay threshold (seconds)

    Returns
    -------
    TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    '''
    # Sort by time of arrival in ascending order
    arrSub = arrSub.sort_values(by='time_of_arrival')
    
    tau = arrSub['time_of_arrival'].to_numpy()
    
    # Only use arrivals that fall within td_thres of the first arrival
    arrSub = arrSub[tau - tau[0] < td_thresh]
    
    tau = arrSub['time_of_arrival'].to_numpy()
    a   = arrSub['arrival_amplitude'].to_numpy()
    r = np.sqrt(arrSub['rx_range'].to_numpy()**2 + 
                (tx_depth - arrSub['rx_depth'].to_numpy())**2)


    k   = np.round(tau * fs).astype(int)
    n_out = k.max() + len(source)
    y = np.zeros(n_out, dtype=float)

    # --- pre-compute unique distances to avoid doing FFT for every ray -----
    cache = {}
    for idx, amp, dist in zip(k, a, r):
        if dist not in cache:
            # note: use the *relative* version here:
            cache[dist] = apply_absorption(source, dist, fs)
        y[idx:idx+len(source)] += np.real(amp * cache[dist])

    return np.arange(n_out)/fs, y

def tl_incoherent_from_arrivals(arrivals,
                                freqs_hz,
                                fc_design=None,
                                alpha_dbkm=None):
    """
    Incoherent transmission-loss (TL) from Bellhop `arrivals`.

    Parameters
    ----------
    arrivals : pandas.DataFrame
        Output of `pm.compute_arrivals()`.
        Must contain columns:
            'arrival_amplitude' (complex),
            'time_of_arrival'   (s),
            'rx_depth', 'rx_range' (m).
    freqs_hz : 1-D iterable
        Design frequencies (Hz) at which to compute TL.
    fc_design : float or None
        If you fed Bellhop a *design* frequency (the one that controls
        phase in the impulse response) pass it here so the phase term
        uses the correct ω.  If None, uses each `freqs_hz` in turn.
    alpha_dbkm : callable or None
        Absorption curve ­– function that takes `f_khz` and returns
        α(f) in dB/km (e.g. `thorp_alpha_dbkm`).  If None, no
        absorption is applied (geometric spreading only).

    Returns
    -------
    tl_dB : list of 2-D np.ndarrays
        One array per `freqs_hz`, shaped (n_depths, n_ranges).
    z_unique : 1-D np.ndarray
        Sorted list of unique receiver depths (m).
    r_unique : 1-D np.ndarray
        Sorted list of unique receiver ranges (m).
    """
    # --- cache receiver grid ----------------------------------------------
    z_unique = np.sort(arrivals['rx_depth'].unique())
    r_unique = np.sort(arrivals['rx_range'].unique())
    nz, nr   = len(z_unique), len(r_unique)

    # group rays by receiver cell for fast access
    grp = arrivals.groupby(['rx_depth', 'rx_range'])

    tl_maps = []

    for f_hz in freqs_hz:
        omega = 2 * np.pi * (fc_design or f_hz)
        # container for this frequency
        tl = np.empty((nz, nr), dtype=float)

        for (z, r), g in grp:
            # phasor pressure for each ray
            Ph = g['arrival_amplitude'].values * \
                 np.exp(-1j * omega * g['time_of_arrival'].values)

            # incoherent sum → intensities
            I = np.sum(np.abs(Ph)**2)

            if alpha_dbkm is not None:
                alpha_db_per_m = alpha_dbkm(f_hz / 1000.0) / 1000.0
                I *= 10**(-alpha_db_per_m * r  / 10)     # two-way loss

            # 10*log10(I) is level relative to 1 µPa; TL = −Level
            tl_val = -10.0 * np.log10(I + 1e-300)           # avoid log(0)

            # write into grid
            tl[z_unique == z, r_unique == r] = tl_val

        tl_maps.append(tl)

    return tl_maps, z_unique, r_unique

def plotTLSigs(sig, tt, rcv, t_rcv, NFFT =1024, fs = 200000):
    
    
    
    # Frequency-domain plot
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(tt, sig)
    plt.title('Time Domain Click ')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (units)')
    
    
    plt.subplot(2,2,3)
    frequencies, power_spectral_density = plt.psd(sig, NFFT=NFFT, Fs=fs)
    plt.title('Frequency-Domain Spectral Content')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density Initial (dB/Hz)')
    
    plt.subplot(2,2,2)
    plt.plot(t_rcv, rcv)
    plt.title('Time Domain Click Transformed')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (units)')
    
    
    plt.subplot(2,2,4)
    frequencies, power_spectral_density = plt.psd(rcv, NFFT=NFFT, Fs=fs)
    plt.title('Frequency-Domain Spectral Content')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density Transformed (dB/Hz)')
    
    plt.tight_layout()
    plt.show()

def apply_rel_absorption(source, r_m, fs, f0_khz=35.0):
    """
    Apply the *difference* in Thorp absorption between each freq and f0_khz.

    - r_m: path length in meters
    - source: time-series array
    - fs:    sampling rate in Hz
    - f0_khz: the Bellhop design frequency (35 kHz by default)
    """
    N     = len(source)
    # freq vector in kHz
    freqs = rfftfreq(N, 1/fs) / 1000.0  
    
    # absolute absorption [dB/m] at each freq and at f0
    #alpha    = thorp_alpha_dbkm(freqs) / 1000.0      # dB/m
    #alpha0   = thorp_alpha_dbkm(f0_khz) / 1000.0     # dB/m (scalar)
    
    alpha = seawater_absorption(freqs)/1000 # dB/m
    alpha0   = seawater_absorption(f0_khz) / 1000.0 
    
    # relative attenuation: zero at f0, negative for f < f0 (i.e. a relative boost)
    A_rel = 10**( -(alpha - alpha0) * r_m / 20.0 )
    
    S = rfft(source)
    return irfft(S * A_rel, n=N)


import numpy as np
from scipy.fft import rfft, irfft, rfftfreq

def synthesize_received_broadband_incoh(arrSub, source, tx_depth, fs, td_thresh):
    """
    Build the received signal by incoherently summing each ray’s energy.
    
    Returns
    -------
    t : 1D array of time stamps
    y_incoh : 1D array of sqrt-summed energy (pressure envelope)
    """
    # sort & threshold as before
    arr = arrSub.sort_values('time_of_arrival')
    tau = arr['time_of_arrival'].to_numpy()
    arr = arr[tau - tau[0] < td_thresh]  # or your td_thresh
    tau = arr['time_of_arrival'].to_numpy()
    
    rx_range = arr['rx_range'].to_numpy()
    rx_depth = arr['rx_depth'].to_numpy()
    rx_depth= rx_depth[0]
    
    # Arrivals, ranges, time delays
    a = arr['arrival_amplitude'].to_numpy()
    r = np.array(np.sqrt(rx_range**2 +  (tx_depth - rx_depth**2)))
    k = np.round(tau * fs).astype(int)
    
    Nsrc = len(source)
    Nout = k.max() + Nsrc
    
    # precompute per-distance filtered pulses
    cache = {}
    for dist in np.unique(r):
        cache[dist] = apply_rel_absorption(source, dist, fs)  # or apply_rel_absorption
    
    # allocate energy accumulator
    energy = np.zeros(Nout, dtype=float)
    
    # for each ray: pad its filtered pulse into the full-length buffer, square & add
    for idx, amp, dist in zip(k, a, r):
        pulse = amp * cache[dist]
        # place it at offset idx and square
        energy[idx:idx+Nsrc] += np.abs(pulse)**2
    
    # convert back to pressure envelope
    y_incoh = np.sqrt(energy)
    t = np.arange(Nout) / fs
    return t, y_incoh



############################################################################
#%% Do the thing

if __name__ == "__main__":
    clickSig,tt = tukeyClick(fs=200000)

    # Time-domain plot
    plt.subplot(2, 1, 1)
    plt.plot(tt, clickSig)
    plt.title('Time-Domain Waveform')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (μPa)')

    # Frequency-domain plot
    plt.subplot(2, 1, 2)
    frequencies, power_spectral_density = plt.psd(clickSig, NFFT=1024, Fs=200000)
    plt.title('Frequency-Domain Spectral Content')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')

    plt.tight_layout()
    plt.show()

    from scipy.io import wavfile
    
    wav_path = "C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\LF_1705_20171028_010934_441.wav"
    
    samplerate, audiodata = wavfile.read(wav_path)
    
    t_start = 32.58
    t_end   = 32.60
    
    start_idx = int(round(t_start * samplerate))
    end_idx   = int(round(t_end   * samplerate))
    
    
    segment = audiodata[start_idx:end_idx, 4] 
    
    from scipy.signal import medfilt
    filtered_signal = medfilt(segment, kernel_size=7)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(segment)
    plt.subplot(2,1,2)
    plt.plot(filtered_signal)
    
    
    tt = np.arange(0, t_end-t_start-(1/samplerate), step =1/samplerate)
    
    arrSub = arr[176:228]
    tx_depth=100
    t_rcv, rcv    = synthesize_received_broadband(arrSub, filtered_signal, tx_depth,  samplerate)
    
    
    # Frequency-domain plot
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(tt, segment)
    plt.title('Time Domain Click ')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (units)')
    
    
    plt.subplot(2,2,3)
    frequencies, power_spectral_density = plt.psd(segment, NFFT=1024, Fs=samplerate)
    plt.title('Frequency-Domain Spectral Content')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density Initial (dB/Hz)')
    
    plt.subplot(2,2,2)
    plt.plot(t_rcv, rcv)
    plt.title('Time Domain Click Transformed')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (units)')
    
    
    plt.subplot(2,2,4)
    frequencies, power_spectral_density = plt.psd(rcv, NFFT=1024, Fs=samplerate)
    plt.title('Frequency-Domain Spectral Content')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density Transformed (dB/Hz)')
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    # Synthesized click
    clickSig,tt = tukeyClick(fs=200000)
    tx_depth =100
    t_rcv, rcv    = synthesize_received_broadband(arrSub, clickSig, env,  200000)
    
    
    # Frequency-domain plot
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(tt, clickSig)
    plt.title('Time Domain Click ')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (units)')
    
    
    plt.subplot(2,2,3)
    frequencies, power_spectral_density = plt.psd(clickSig, NFFT=1024, Fs=200000)
    plt.title('Frequency-Domain Spectral Content')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density Initial (dB/Hz)')
    
    plt.subplot(2,2,2)
    plt.plot(t_rcv, rcv)
    plt.title('Time Domain Click Transformed')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (units)')
    
    
    plt.subplot(2,2,4)
    frequencies, power_spectral_density = plt.psd(rcv, NFFT=1024, Fs=200000)
    plt.title('Frequency-Domain Spectral Content')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density Transformed (dB/Hz)')
    
    plt.tight_layout()
    plt.show()

#%% Calculate TL from arrival arrays
    
    freqs = [1000, 10000, 20000]        # any list of Hz
    tl_maps, z, r = tl_incoherent_from_arrivals(
        arrivals    = arrSub,
        freqs_hz    = freqs,
        fc_design   = 10000,            # if Bellhop ran at 10 kHz
        alpha_dbkm  = thorp_alpha_dbkm  # or None
    )
    
    r_m   = 19_893.          # m
    TL_ss = 20*np.log10(r_m)               # spherical spreading, one-way
    alpha = thorp_alpha_dbkm(1.0) / 1000   # dB/m   (≈6.9 × 10-5)
    TL_abs= alpha * r_m                    # one-way absorption
    print(f"SS - only : {TL_ss:5.1f} dB")
    print(f"+ absorption: {TL_ss+TL_abs:5.1f} dB")
    
    direct = arrSub.query("surface_bounces==0 & bottom_bounces==0")
    print(direct.head())