# -*- coding: utf-8 -*-
"""
Created on Sun May 25 14:42:22 2025

@author: kaity
"""

import numpy as np

# ---------- user inputs -------------------------------------------------
f_hz         = 1_000              # frequency you care about
target_range = 19_893             # metres  (≈ 19.9 km)
target_depth = 250                 # metres  (pick one of your receiver depths)
tolerance    = 100                # ±1 m range “bin” so we actually pick up rows
# ------------------------------------------------------------------------

# 1) pull out arrivals that land in that depth-range “cell”
mask = (
    (np.abs(arrSub.rx_range - target_range) < tolerance) &
    (np.abs(arrSub.rx_depth - target_depth) < 1e-6)      # exact depth match
)
cell = arrSub[mask]

if cell.empty:
    raise ValueError("No arrivals hit that range/depth cell!")

# 2) incoherent intensity = sum |A|^2  (no phases needed)
I = np.sum(np.abs(cell.arrival_amplitude.values)**2)

# ---- optional: add one-way absorption for each ray’s horizontal range ----
# comment out if you want *just* geometric spreading
alpha_dBpm = thorp_alpha_dbkm(f_hz/1000) / 1000           # dB per metre
I *= 10**(-alpha_dBpm * target_range / 10)

# 3) transmission loss in dB  (reference = 1 µPa)
TL_model = -10*np.log10(I)

# 4) spherical spreading (+ same absorption) for comparison
TL_ss   = 20*np.log10(target_range)
TL_base = TL_ss + alpha_dBpm*target_range

print(f"Model TL @ {f_hz/1000:.1f} kHz, {target_range/1000:.1f} km: {TL_model:6.2f} dB")
print(f"Spherical spreading (1/r)      : {TL_ss:6.2f} dB")
print(f"+ one-way absorption           : {TL_base:6.2f} dB")
