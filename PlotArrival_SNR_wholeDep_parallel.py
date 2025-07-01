# parallel_p2p_eval.py
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from scipy.io import wavfile
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- Signal Setup ---
wav_path = "C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\LF_1705_20171028_010934_441.wav"
samplerate, audiodata = wavfile.read(wav_path)
t_start, t_end, chan = 32.58, 32.60, 4
segment = audiodata[int(round(t_start * samplerate)):int(round(t_end * samplerate)), chan]

outP2P = 220
init_p2pdB = 20 * np.log10(np.ptp(segment))
addP2P_linear = 10 ** ((outP2P - init_p2pdB) / 20)
segment = segment * addP2P_linear

# --- Custom Impulse Response Function ---
def arrivals_to_impulse_response(arrivals, fs, abs_time=False):
    toa = arrivals['time_of_arrival']
    amp = arrivals['arrival_amplitude']
    t0 = 0 if abs_time else np.min(toa)
    irlen = int(np.ceil((np.max(toa) - t0) * fs)) + 1
    ir = np.zeros(irlen, dtype=np.complex128)
    for i in range(len(toa)):
        ndx = int(np.round((toa[i].real - t0) * fs))
        if 0 <= ndx < irlen:
            ir[ndx] = amp[i]
    return ir

# --- SNR Estimation ---
def p2pArrivalSNR(arrivals, segment, fs):
    
    ir = arrivals_to_impulse_response(arrivals, fs=fs, abs_time=False)
    outputSig = np.convolve(segment, ir)[:len(segment)]
    outputSig_real = np.real(outputSig)
    att_conv_p2p = 20 * np.log10(np.ptp(outputSig_real))
    
    
    return np.round(att_conv_p2p, 1), outputSig_real

# --- Worker Function ---
def process_run(run_id, runIndex, depth_row, segment, fs, h5_path, dive_id):
    required_cols = [
        'time_of_arrival', 'arrival_amplitude',
        'tx_depth_ndx', 'rx_depth_ndx', 'rx_range_ndx'
    ]
    results = []
    with h5py.File(h5_path, 'r') as hf:
        path = f'drift_01/{dive_id}/frequency_35000/arrivals/{run_id}'
        arr0 = hf[path]
        data = {}
        for name in required_cols:
            if name not in arr0:
                continue
            arr = arr0[name][()]
            arr = arr.data if isinstance(arr, np.ma.MaskedArray) else arr
            arr = arr.ravel() if arr.ndim > 1 else arr
            data[name] = arr
        if len(data['rx_depth_ndx']) == 0:
            return runIndex, []
        depth_idxs = np.where(depth_row > 0)[0]
        for depthIdx in depth_idxs:
            mask = data['rx_depth_ndx'] == depthIdx
            if not np.any(mask):
                results.append((depthIdx, np.nan))
                continue
            hydData = {k: v[mask] for k, v in data.items()}
            ptpOut, _ = p2pArrivalSNR(hydData, segment, fs)
            results.append((depthIdx, ptpOut))
    return runIndex, results


##################
# Plotting 
######################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pyproj import Transformer
import xarray as xr

def scatter_peak2peak_3d(
        lat, lon, depth_grid, pp_grid,
        drifter_lat, drifter_lon,
        cmap=cm.viridis,
        vmin=None, vmax=None,
        max_points=None,
        include_seabed=True,
        seabed_source="data",        # "data" | "netcdf"
        bathy_nc=None,               # path to .nc if seabed_source == "netcdf"
        bathy_var="elevation",       # var name in the .nc file
        seabed_color="0.5",          # grey
        seabed_alpha=0.6,
        elev=30, azim=-60
):
    """
    3-D scatter of peak-to-peak values in dB plus optional seabed surface.
    """
    # ------------------------------------------------------------------
    # 1. Project lat/lon → UTM metres, drifter at (0,0)
    # ------------------------------------------------------------------
    utm_zone   = int((drifter_lon + 180) // 6) + 1
    hemisphere = 'north' if drifter_lat >= 0 else 'south'
    transformer = Transformer.from_crs(
        "epsg:4326",
        f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84",
        always_xy=True
    )
    x, y = transformer.transform(lon.ravel(), lat.ravel())
    x0, y0 = transformer.transform(drifter_lon, drifter_lat)
    x, y   = x - x0, y - y0

    # ------------------------------------------------------------------
    # 2. Flatten valid samples
    # ------------------------------------------------------------------
    N, Z = pp_grid.shape
    xx = np.repeat(x[:, None], Z, axis=1).ravel()
    yy = np.repeat(y[:, None], Z, axis=1).ravel()
    zz = depth_grid.ravel()
    pp = pp_grid.ravel()

    valid = np.isfinite(pp) & np.isfinite(zz)
    if valid.sum() == 0:
        raise RuntimeError("No finite peak-to-peak/depth values to plot")

    xx, yy, zz, pp = xx[valid], yy[valid], zz[valid], pp[valid]

    # Optional random down-sampling
    if max_points is not None and valid.sum() > max_points:
        idx = np.random.choice(valid.sum(), max_points, replace=False)
        xx, yy, zz, pp = xx[idx], yy[idx], zz[idx], pp[idx]

    # ------------------------------------------------------------------
    # 3. Prepare seabed surface (optional)
    # ------------------------------------------------------------------
    seabed_mesh = None
    if include_seabed:
        if seabed_source == "data":
            # Deepest finite depth at each column
            seabed_depth = np.nanmax(depth_grid, axis=1)
            seabed_mesh = dict(x=x, y=y, z=seabed_depth)

        elif seabed_source == "netcdf":
            if bathy_nc is None:
                raise ValueError("bathy_nc must be provided for seabed_source='netcdf'")
            ds = xr.open_dataset(bathy_nc)
            lon_b = ds['lon'].values
            lat_b = ds['lat'].values
            Lon2d, Lat2d = np.meshgrid(lon_b, lat_b)
            lon_vec = Lon2d.ravel()
            lat_vec = Lat2d.ravel()
            depth_vec = ds[bathy_var].values.ravel().astype(float)

            if depth_vec.max() <= 0:
                depth_vec = -depth_vec  # flip sign if bathy is negative up

            xb, yb = transformer.transform(lon_vec, lat_vec)
            xb, yb = xb - x0, yb - y0

            seabed_mesh = dict(x=xb, y=yb, z=depth_vec)

    # ------------------------------------------------------------------
    # 4. Plot scatter
    # ------------------------------------------------------------------
    if vmin is None:
        vmin = np.nanpercentile(pp, 5)
    if vmax is None:
        vmax = np.nanpercentile(pp, 95)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(xx, yy, zz,
                    c=pp, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    s=6, marker='o', linewidths=0, alpha=0.9)

    # ------------------------------------------------------------------
    # 5. Seabed trisurf (if requested)
    # ------------------------------------------------------------------
    if seabed_mesh is not None:
        ax.plot_trisurf(seabed_mesh['x'],
                        seabed_mesh['y'],
                        seabed_mesh['z'],
                        color=seabed_color,
                        alpha=seabed_alpha,
                        linewidth=0,
                        antialiased=False)

    # ------------------------------------------------------------------
    # 6. Labels, view, colorbar
    # ------------------------------------------------------------------
    ax.set_xlabel('East–West range (m)')
    ax.set_ylabel('North–South range (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_zlim(np.nanmax(zz), 0)
    ax.set_title('3-D Peak-to-Peak Level Scatter with Seabed')

    ax.view_init(elev=elev, azim=azim)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel('Peak-to-Peak Level (dB re 1 µPa p-p)')

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pyproj import Transformer
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata, NearestNDInterpolator
from skimage import measure

def plot_peak2peak_isosurfaces(
        lat, lon, depth_grid, pp_grid,
        drifter_lat, drifter_lon,
        iso_levels=(90,),             # peak-to-peak dB levels to draw
        xy_res=200,
        cmap=cm.viridis,
        seabed_color='0.6',
        elev=25, azim=-45
):
    # ------------------------------------------------------------------
    # 1. lat/lon → UTM metres (drifter = 0,0)
    # ------------------------------------------------------------------
    utm_zone   = int((drifter_lon + 180) // 6) + 1
    hemisphere = 'north' if drifter_lat >= 0 else 'south'
    transformer = Transformer.from_crs(
        "epsg:4326",
        f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84",
        always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    x0, y0 = transformer.transform(drifter_lon, drifter_lat)
    x, y   = x - x0, y - y0

    # ------------------------------------------------------------------
    # 2. Interpolate XY slices at each depth
    # ------------------------------------------------------------------
    xi = np.linspace(x.min(), x.max(), xy_res)
    yi = np.linspace(y.min(), y.max(), xy_res)
    X2d, Y2d = np.meshgrid(xi, yi)

    z_vec = depth_grid[0, :]
    Z = len(z_vec)
    PP_vol = np.full((Z, xy_res, xy_res), np.nan, dtype=float)

    for k in range(Z):
        valid = np.isfinite(pp_grid[:, k])
        if valid.sum() < 3:
            continue

        pts  = np.column_stack((x[valid], y[valid]))
        vals = pp_grid[valid, k]

        try:
            PP_slice = griddata(pts, vals, (X2d, Y2d), method='cubic')
        except Exception:
            PP_slice = NearestNDInterpolator(pts, vals)(X2d, Y2d)

        PP_vol[k] = PP_slice

    # ------------------------------------------------------------------
    # 3. TL range & marching-cubes prep
    # ------------------------------------------------------------------
    finite_vals = PP_vol[np.isfinite(PP_vol)]
    if finite_vals.size == 0:
        raise RuntimeError("No finite peak-to-peak values – check inputs")

    real_min, real_max = finite_vals.min(), finite_vals.max()
    nan_fill = real_min - 1.0

    # ------------------------------------------------------------------
    # 4. draw iso-surfaces
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    
    for level in iso_levels:
        if not (real_min < level < real_max):
            print(f"⚠️  Skipping {level} dB – outside value range "
                  f"[{real_min:.1f}, {real_max:.1f}] dB")
            continue

        try:
            verts, faces, _, _ = measure.marching_cubes(
                np.nan_to_num(PP_vol, nan=nan_fill),
                level=level
            )
        except RuntimeError as e:
            print(f"⚠️  marching_cubes failed for {level} dB: {e}")
            continue

        idx_x = np.clip(np.round(verts[:, 2]).astype(int), 0, xi.size - 1)
        idx_y = np.clip(np.round(verts[:, 1]).astype(int), 0, yi.size - 1)
        idx_z = np.clip(np.round(verts[:, 0]).astype(int), 0, Z - 1)

        verts_xyz = np.empty_like(verts)
        verts_xyz[:, 0] = xi[idx_x]
        verts_xyz[:, 1] = yi[idx_y]
        verts_xyz[:, 2] = z_vec[idx_z]

        face_color = cmap((level - real_min) / (real_max - real_min))

        ax.add_collection3d(
            Poly3DCollection(verts_xyz[faces],
                             facecolor=face_color,
                             edgecolor='none',
                             alpha=0.4)
        )

    # ------------------------------------------------------------------
    # 5. Seabed surface
    # ------------------------------------------------------------------
    seabed_raw = np.nanmax(depth_grid, axis=1)
    valid_cols = np.isfinite(seabed_raw)

    seabed_grid = griddata(
        (x[valid_cols], y[valid_cols]), seabed_raw[valid_cols],
        (X2d, Y2d),
        method='linear',
        fill_value=np.nan
    )

    R2d = np.sqrt(X2d**2 + Y2d**2)
    seabed_grid[R2d > 15000] = np.nan
    seabed_mask = np.ma.masked_invalid(seabed_grid)

    ax.plot_surface(
        X2d, Y2d, seabed_mask,
        color=seabed_color,
        alpha=0.6,
        linewidth=0,
        antialiased=False
    )

    # ------------------------------------------------------------------
    # 6. Axes & layout
    # ------------------------------------------------------------------
    ax.set_xlabel('East–West range (m)')
    ax.set_ylabel('North–South range (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_zlim(np.nanmax(seabed_raw), 0)
    ax.set_title(f'Peak-to-Peak Iso-Surfaces ({iso_levels} dB re 1 µPa p-p)')
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()

from geopy.distance import geodesic
import numpy as np

def apply_alpha_correction(my_data, lat, lon, depth_grid, 
                           drifter_lat, drifter_lon, drifter_depth,
                           alpha_db_per_km):
    """
    Applies frequency-dependent attenuation based on a constant alpha value
    (in dB/km) to the peak-to-peak grid using slant range from the drifter.

    Parameters:
        my_data: ndarray (N_locations x N_depths)
        lat, lon: 1D arrays (N_locations,) matching rows in my_data
        depth_grid: ndarray (N_locations x N_depths) of receiver depths (m)
        drifter_lat, drifter_lon: float, drifter coordinates
        drifter_depth: float, drifter depth (m)
        alpha_db_per_km: float, attenuation coefficient in dB/km

    Returns:
        corrected_data: ndarray, same shape as my_data, adjusted for attenuation
    """
    N_locations, N_depths = my_data.shape
    corrected = my_data.copy()

    for i in range(N_locations):
        horiz_dist_km = geodesic((drifter_lat, drifter_lon), (lat[i], lon[i])).km
        for j in range(N_depths):
            if my_data[i, j] == -500 or not np.isfinite(my_data[i, j]):
                continue  # skip land or invalid
            receiver_depth = depth_grid[i, j]
            vertical_dist_km = abs(receiver_depth - drifter_depth) / 1000.0
            slant_km = np.hypot(horiz_dist_km, vertical_dist_km)
            correction = alpha_db_per_km * slant_km
            corrected[i, j] += correction

    return corrected
import matplotlib.pyplot as plt
from pyproj import Transformer
import numpy as np

def plot_detection_probability(
    my_data, lat, lon, threshold_db,
    drifter_lat, drifter_lon,
    cmap='viridis', vmin=0, vmax=1, 
    title=None, s=40
):
    """
    Computes and plots detection probability (proportion of depths > threshold)
    for each lat/lon point in my_data.

    Parameters:
        my_data: 2D ndarray (N_locations x N_depths)
        lat, lon: 1D arrays of lat/lon positions (length N_locations)
        threshold_db: float, dB threshold for detection
        drifter_lat, drifter_lon: float, used for local projection
        cmap: str or colormap, optional
        vmin, vmax: float, colorbar limits
        title: str, optional plot title
        s: int, marker size
    """
    # 1. Compute detection probability per location
    is_valid = my_data > -500
    is_detected = my_data > threshold_db
    detection_prob = (is_detected & is_valid).sum(axis=1) / is_valid.sum(axis=1)
    detection_prob = np.nan_to_num(detection_prob, nan=0.0)

    # 2. Project lat/lon → UTM (relative to drifter)
    utm_zone   = int((drifter_lon + 180) // 6) + 1
    hemisphere = 'north' if drifter_lat >= 0 else 'south'
    transformer = Transformer.from_crs(
        "epsg:4326",
        f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84",
        always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    x0, y0 = transformer.transform(drifter_lon, drifter_lat)
    x, y = x - x0, y - y0

    # 3. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=detection_prob,
                    cmap=cmap, vmin=vmin, vmax=vmax, s=s)
    
    cbar = fig.colorbar(sc, ax=ax, label='Detection Probability')
    ax.set_xlabel('East–West Range (m)')
    ax.set_ylabel('North–South Range (m)')
    ax.set_title(title or f'Detection Probability (Threshold: {threshold_db} dB)')
    ax.grid(True)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

    return detection_prob

from geopy.distance import geodesic
import numpy as np
import matplotlib.pyplot as plt

def plot_detection_vs_range(
    my_data, lat, lon, drifter_lat, drifter_lon,
    threshold_db=120, bin_width_km=1.0,
    min_range_km=0, max_range_km=None,
    color='tab:blue', title=None
):
    """
    Plots median detection probability vs range, with 95% CI shading.

    Parameters:
        my_data: 2D ndarray (N_locations x N_depths)
        lat, lon: 1D arrays matching rows of my_data
        drifter_lat, drifter_lon: float
        threshold_db: float, detection threshold
        bin_width_km: float, width of range bins (km)
        min_range_km: float, minimum range to plot
        max_range_km: float or None, maximum range to plot (None = auto)
        color: str, matplotlib color
        title: str, optional title

    Returns:
        stats_df: pd.DataFrame with range bin centers, medians, and CI bounds
    """
    import pandas as pd

    # 1. Compute detection probability for each lat/lon
    is_valid = my_data > -500
    is_detected = my_data > threshold_db
    detection_prob = (is_detected & is_valid).sum(axis=1) / is_valid.sum(axis=1)
    detection_prob = np.nan_to_num(detection_prob, nan=0.0)

    # 2. Compute horizontal distance to each point
    ranges_km = np.array([
        geodesic((drifter_lat, drifter_lon), (lat[i], lon[i])).km
        for i in range(len(lat))
    ])

    # 3. Bin and compute statistics
    if max_range_km is None:
        max_range_km = np.nanmax(ranges_km)

    bins = np.arange(min_range_km, max_range_km + bin_width_km, bin_width_km)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    medians = []
    lower_95 = []
    upper_95 = []

    for i in range(len(bins) - 1):
        in_bin = (ranges_km >= bins[i]) & (ranges_km < bins[i+1])
        values = detection_prob[in_bin]
        if len(values) == 0:
            medians.append(np.nan)
            lower_95.append(np.nan)
            upper_95.append(np.nan)
        else:
            medians.append(np.nanmedian(values))
            lower_95.append(np.nanpercentile(values, 2.5))
            upper_95.append(np.nanpercentile(values, 97.5))

    # 4. Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(bin_centers, medians, color=color, label='Median')
    ax.fill_between(bin_centers, lower_95, upper_95, 
                    color=color, alpha=0.3, label='95% CI')

    ax.set_xlabel('Range from Drifter (km)')
    ax.set_ylabel('Detection Probability')
    ax.set_ylim(0, 1)
    ax.set_xlim(min_range_km, max_range_km)
    ax.grid(True)
    ax.legend()
    ax.set_title(title or f'Detection Probability vs Range (Threshold: {threshold_db} dB)')
    plt.tight_layout()
    plt.show()

    # 5. Return stats as DataFrame
    stats_df = pd.DataFrame({
        'range_km': bin_centers,
        'median': medians,
        'lower_95': lower_95,
        'upper_95': upper_95
    })

    return stats_df

from geopy.distance import geodesic
from pyproj import Geod
import numpy as np
import matplotlib.pyplot as plt

def plot_detection_by_bearing(
    my_data, lat, lon, drifter_lat, drifter_lon,
    threshold_db=120,
    bearing_bin_width=10,
    range_bin_width_km=0.2,
    min_range_km=0, max_range_km=None,
    cmap='turbo', alpha=0.6, title=None
):
    """
    Plots detection probability vs range for each bearing direction.

    Parameters:
        my_data: ndarray (N_locations x N_depths)
        lat, lon: 1D arrays of receiver positions
        drifter_lat, drifter_lon: float
        threshold_db: float, RL detection threshold
        bearing_bin_width: float, bearing bin width in degrees
        range_bin_width_km: float, range bin width in km
        min_range_km, max_range_km: optional bounds
        cmap: colormap or name
        alpha: float, line transparency
        title: str, optional title

    Returns:
        stats_by_bearing: dict of {bearing: DataFrame} for each slice
    """
    import pandas as pd
    from matplotlib.cm import get_cmap

    N = len(lat)
    geod = Geod(ellps="WGS84")

    # 1. Detection probability at each location
    is_valid = my_data > -500
    is_detected = my_data > threshold_db
    detection_prob = (is_detected & is_valid).sum(axis=1) / is_valid.sum(axis=1)
    detection_prob = np.nan_to_num(detection_prob, nan=0.0)

    # 2. Compute bearings and ranges
    bearings = np.zeros(N)
    ranges_km = np.zeros(N)
    for i in range(N):
        az12, _, dist_m = geod.inv(drifter_lon, drifter_lat, lon[i], lat[i])
        bearings[i] = az12 % 360
        ranges_km[i] = dist_m / 1000

    # 3. Define bins
    if max_range_km is None:
        max_range_km = np.nanmax(ranges_km)

    bearing_bins = np.arange(0, 360, bearing_bin_width)
    range_bins = np.arange(min_range_km, max_range_km + range_bin_width_km, range_bin_width_km)
    range_centers = (range_bins[:-1] + range_bins[1:]) / 2

    # 4. Group and compute stats
    stats_by_bearing = {}
    cmap_obj = get_cmap(cmap, len(bearing_bins))
    
    fig, ax = plt.subplots(figsize=(10, 6))

    for b_idx, b0 in enumerate(bearing_bins):
        b1 = b0 + bearing_bin_width
        in_slice = (bearings >= b0) & (bearings < b1)
        if np.sum(in_slice) < 5:
            continue

        dp_slice = detection_prob[in_slice]
        r_slice  = ranges_km[in_slice]

        dp_binned = []
        for i in range(len(range_bins) - 1):
            in_bin = (r_slice >= range_bins[i]) & (r_slice < range_bins[i+1])
            values = dp_slice[in_bin]
            dp_binned.append(np.nanmedian(values) if len(values) > 0 else np.nan)

        stats_df = pd.DataFrame({
            'range_km': range_centers,
            'detection_prob': dp_binned
        })

        stats_by_bearing[b0] = stats_df

        ax.plot(stats_df['range_km'], stats_df['detection_prob'],
                label=f'{b0:.0f}–{b1:.0f}°',
                color=cmap_obj(b_idx),
                alpha=alpha)

    # 5. Finalize plot
    ax.set_xlabel('Range from Drifter (km)')
    ax.set_ylabel('Detection Probability')
    ax.set_ylim(0, 1)
    ax.set_xlim(min_range_km, max_range_km)
    ax.set_title(title or f'Detection Probability vs Range by Bearing\n(Threshold: {threshold_db} dB)')
    ax.grid(True)
    #ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Bearing')
    plt.tight_layout()
    plt.show()

    return stats_by_bearing

import numpy as np
from geopy.distance import geodesic
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

def hazard_rate(r, sigma, b):
    return 1 - np.exp(-(r / sigma) ** -b)

def hazard_rate_log(r, sigma, b):
    r = np.maximum(r, 0.01)  # avoids log(0)
    return 1 - np.exp(-(np.log(r) / sigma) ** -b)

def fit_and_plot_hazard_rate_by_location(
    my_data, lat, lon, depth_grid,
    drifter_lat, drifter_lon,
    threshold_db=120,
    n_boot=500,
    smooth_range=np.linspace(0.05, 20, 200),
    ci_alpha=0.3,
    use_log_scale=True,
    seed=42
):
    """
    Fits a hazard-rate model to detection proportions (1 per location),
    using horizontal range and bootstrap CI.

    Parameters:
        my_data: 2D array (N_locations x N_depths)
        lat, lon: 1D arrays (N_locations,)
        depth_grid: 2D array (same shape as my_data)
        drifter_lat, drifter_lon: float
        threshold_db: float
        n_boot: int, bootstrap iterations
        smooth_range: array-like, range values to evaluate model on
        ci_alpha: float, transparency of CI ribbon
        use_log_scale: bool, whether to log-transform range input to model
        seed: int, RNG seed

    Returns:
        model_df: pd.DataFrame with smooth_range, mean, lower_95, upper_95
    """
    np.random.seed(seed)
    N = len(lat)
    ranges_km = np.array([
        geodesic((drifter_lat, drifter_lon), (lat[i], lon[i])).km
        for i in range(N)
    ])

    # Compute proportion of detected depths at each location
    is_valid = my_data > -500
    is_detected = my_data > threshold_db
    num_valid = is_valid.sum(axis=1)
    num_detected = (is_valid & is_detected).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        proportions = num_detected / num_valid
    mask = (num_valid > 0) & np.isfinite(proportions)

    # Clip the data
    X = np.clip(ranges_km[mask], 2, None)
    Y = proportions[mask]

    # Select model function
    model_fn = hazard_rate_log if use_log_scale else hazard_rate

    # 2. Fit hazard-rate model
    try:
        popt, _ = curve_fit(
            model_fn, X, Y,
            p0=(0.5, 2.0),
            bounds=([0.01, 0.1], [10.0, 10.0]),
            maxfev=10000
        )
    except Exception as e:
        print(f"⚠️ Initial fit failed: {e}")
        return None
    
    y_fit = model_fn(smooth_range, *popt)
    
    # 3. Bootstrap confidence intervals
    # 3. Bootstrap confidence intervals (resample detections at binomial level)
    boot_preds = []
    X_orig = np.clip(ranges_km[mask], 2, None)
    Y_orig = proportions[mask]
    n_valid_orig = num_valid[mask]
    
    for _ in range(n_boot):
        # Resample detection counts for each location using binomial draw
        boot_detected = [
            np.random.binomial(n, p)
            for n, p in zip(n_valid_orig, Y_orig)
        ]
        Yb = np.array(boot_detected) / n_valid_orig
    
        try:
            bopt, _ = curve_fit(
                model_fn, X_orig, Yb,
                p0=(0.5, 2.0),
                bounds=([0.01, 0.1], [10.0, 10.0]),
                maxfev=10000
            )
            boot_preds.append(model_fn(smooth_range, *bopt))
        except:
            continue

    boot_preds = np.array(boot_preds)

    # Remove rows with all-NaN values (bad fits)
    boot_preds = boot_preds[~np.isnan(boot_preds).all(axis=1)]
    
    print(f"✅ {boot_preds.shape[0]} successful bootstrap fits")
    
    if boot_preds.shape[0] < 10:
        print("⚠️ Too few valid bootstrap fits — no CI computed")
        lower = np.full_like(smooth_range, np.nan)
        upper = np.full_like(smooth_range, np.nan)
    else:
        lower = np.nanpercentile(boot_preds, 2.5, axis=0)
        upper = np.nanpercentile(boot_preds, 97.5, axis=0)

        
    
    boot_preds = np.array(boot_preds)
    print(f"✅ {boot_preds.shape[0]} successful bootstrap fits")


    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(smooth_range, y_fit, label='Fitted Hazard-Rate', color='blue')
    
    # Only plot CI where both bounds are finite
    valid_ci = np.isfinite(lower) & np.isfinite(upper)
    if np.any(valid_ci):
        ax.fill_between(smooth_range[valid_ci], lower[valid_ci], upper[valid_ci],
                        color='blue', alpha=ci_alpha, label='95% CI')
    else:
        print("⚠️ No finite confidence intervals to plot.")

    
    
    ax.scatter(X, Y, color='black', s=20, zorder=3, label='Location-Level Proportions')

    ax.set_xlabel('Range from Drifter (km)')
    ax.set_ylabel('Detection Probability')
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Hazard-Rate Model (Threshold: {threshold_db} dB, {'log' if use_log_scale else 'linear'} scale)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"✅ Fitted params: σ = {popt[0]:.3f}, b = {popt[1]:.3f}")

    return pd.DataFrame({
        'range_km': smooth_range,
        'mean': y_fit,
        'lower_95': lower,
        'upper_95': upper
    })



#%%
# --- Main Block ---
if __name__ == "__main__":
    
    
    # Read in the CSV file
    import numpy as np
    from numpy import genfromtxt
    import h5py
    
    my_data = genfromtxt('PeakToPeakDive_dive_24_dec.csv', delimiter=',')
    np.nanmax(my_data)
    my_data[:, 1:4] = -500
    #my_data[:, 15:48] = -500
    
    # Now get the depths
    hf = h5py.File( 'Spacious_Hawaii_diveDepth_ArrArray_PCHIP_35khz_20km - Copy.h5', 'r')
    diveIds = list(hf['drift_01'].keys())
    dive_grp = hf[f'drift_01/{diveIds[0]}/frequency_35000']
    run_ids = list(dive_grp['arrivals'].keys())
    depthGrid = np.array(dive_grp['depth'])
    
    # Grid postions
    lat = np.array(dive_grp['lat'])
    lon = np.array(dive_grp['lon'])
    
    #sensor position
    d_lat= dive_grp.parent.attrs['start_lat'] 
    d_lon= dive_grp.parent.attrs['start_lon'] 
    d_depth = dive_grp.parent.attrs['drifter_depth']
    
    
    plot_peak2peak_isosurfaces(
    lat=lat.ravel(),
    lon=lon.ravel(),
    depth_grid=depthGrid,
    pp_grid=my_data,
    drifter_lat=d_lat,
    drifter_lon=d_lon,
    iso_levels=([20]))  # e.g., 3 levels of pp dB)
    
    # Add an alpha value to simulate running the TL model at lower frequencies
    alpha = 1.2  # dB/km for a hypothetical frequency
    corrected_data = apply_alpha_correction(
    my_data=my_data,
    lat=lat,
    lon=lon,
    depth_grid=depthGrid,
    drifter_lat=d_lat,
    drifter_lon=d_lon,
    drifter_depth=d_depth,
    alpha_db_per_km=alpha)
    
    # Plot the new iso-surface
    plot_peak2peak_isosurfaces(
    lat=lat.ravel(),
    lon=lon.ravel(),
    depth_grid=depthGrid,
    pp_grid=corrected_data,
    drifter_lat=d_lat,
    drifter_lon=d_lon,
    iso_levels=([20]))  # e.g., 3 levels of pp dB)
    
    # Create a scatter plot of the detection probability from a birds-eye-view
    threshold = -60 # dB
    probabilities = plot_detection_probability(
        my_data=my_data,
        lat=lat,
        lon=lon,
        threshold_db=threshold,
        drifter_lat=d_lat,
        drifter_lon=d_lon
    )
    
    
    
    stats_df = plot_detection_vs_range(
    my_data=my_data,
    lat=lat,
    lon=lon,
    drifter_lat=d_lat,
    drifter_lon=d_lon,
    threshold_db=5,
    bin_width_km= .1)
    
    stats_dict = plot_detection_by_bearing(
    my_data=corrected_data,
    lat=lat,
    lon=lon,
    drifter_lat=d_lat,
    drifter_lon=d_lon,
    threshold_db=5,
    bearing_bin_width=10,
    range_bin_width_km=.25)


    model_df = fit_and_plot_hazard_rate_by_location(
    my_data=my_data,
    lat=lat,
    lon=lon,
    depth_grid=depthGrid,
    drifter_lat=d_lat,
    drifter_lon=d_lon,
    threshold_db=0,                # your detection threshold
    use_log_scale=True             # log scale for fitting stability
    )

    
    
   
    
    
    
    
    
    
    # # Run the dive- takes about 15 hrs
    # h5_path = 'Spacious_Hawaii_diveDepth_ArrArray_PCHIP_35khz_20km - Copy.h5'
    # with h5py.File(h5_path, 'r') as hf:
    #     diveIds = list(hf['drift_01'].keys())

    # p2p_grid = None

    # with ProcessPoolExecutor(max_workers=8) as pool:
    #     futures = []
    #     all_depth_rows = []

    #     for dive_id in diveIds:
    #         with h5py.File(h5_path, 'r') as hf:
    #             dive_grp = hf[f'drift_01/{dive_id}/frequency_35000']
    #             run_ids = list(dive_grp['arrivals'].keys())
    #             depthGrid = np.array(dive_grp['depth'])
    #             if p2p_grid is None:
    #                 p2p_grid = np.full_like(depthGrid, np.nan, dtype=np.float64)

    #             for runIndex, run_id in enumerate(run_ids):
    #                 depth_row = depthGrid[runIndex, :]
    #                 futures.append(pool.submit(
    #                     process_run, run_id, runIndex, depth_row,
    #                     segment, samplerate, h5_path, dive_id
    #                 ))

    #     for future in tqdm(futures):
    #         runIndex, results = future.result()
    #         for depthIdx, ptpOut in results:
    #             p2p_grid[runIndex, depthIdx] = ptpOut

    # # Optional: Save or plot p2p_grid here
    # np.savetxt("PeakToPeakDive_dive_24_dec.csv", p2p_grid, delimiter=",")
    
    # # Add depth to 
    
    
    
    
    
    
    
    
    
    
