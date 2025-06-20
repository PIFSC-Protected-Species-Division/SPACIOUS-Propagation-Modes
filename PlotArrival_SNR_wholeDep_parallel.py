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
    ax.set_title('3-D Peak-to-Peak Iso-Surfaces (dB re 1 µPa p-p)')
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()



#%%
# --- Main Block ---
if __name__ == "__main__":
    
    
    # Read in the CSV file
    import numpy as np
    from numpy import genfromtxt
    import h5py
    
    my_data = genfromtxt('PeakToPeakDive_dive_24_dec.csv', delimiter=',')
    np.nanmax(my_data)
    
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
    iso_levels=(40,)  # e.g., 3 levels of pp dB
)
    
    
    # Plot the peak to peak values
    scatter_peak2peak_3d(
    lat=lat,
    lon=lon,
    depth_grid=depthGrid,
    pp_grid=my_data,
    drifter_lat=d_lat,
    drifter_lon=d_lon,
    seabed_source="data"  # or "netcdf", bathy_nc="path/to/bathy.nc"
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
    
    
    
    
    
    
    
    
    
    
