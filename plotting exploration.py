# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 21:18:59 2025

@author: pam_user
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyproj import Transformer
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.spatial import QhullError
from skimage import measure
import h5py

def plot_tl_isosurfaces(
        lat, lon, depth_grid, tl_grid,
        drifter_lat, drifter_lon,
        iso_levels=(-60,),          # TL values (dB) to draw
        xy_res=200,                 # horizontal grid resolution
        cmap=cm.viridis,
        seabed_color='0.6'          # grey
):
    # ------------------------------------------------------------------
    # 1. lat/lon → UTM metres, drifter at (0,0)
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
    # 2. XY mesh & slice‑by‑slice interpolation
    # ------------------------------------------------------------------
    xi = np.linspace(x.min(), x.max(), xy_res)
    yi = np.linspace(y.min(), y.max(), xy_res)
    X2d, Y2d = np.meshgrid(xi, yi)

    z_vec = depth_grid[0, :]
    Z     = len(z_vec)

    TL_vol = np.full((Z, xy_res, xy_res), np.nan, dtype=float)

    for k in range(Z):
        valid = ~np.isnan(tl_grid[:, k])
        if valid.sum() < 3:
            continue

        pts  = np.column_stack((x[valid], y[valid]))
        vals = tl_grid[valid, k]

        try:
            TL_slice = griddata(pts, vals, (X2d, Y2d), method='cubic')
        except (QhullError, ValueError):
            TL_slice = NearestNDInterpolator(pts, vals)(X2d, Y2d)

        TL_vol[k] = TL_slice

    # ------------------------------------------------------------------
    # 3. real TL range & NaN fill
    # ------------------------------------------------------------------
    finite_vals = TL_vol[np.isfinite(TL_vol)]
    if finite_vals.size == 0:
        raise RuntimeError("No finite TL values – check inputs")

    real_min, real_max = finite_vals.min(), finite_vals.max()
    nan_fill = real_min - 1.0

    # ------------------------------------------------------------------
    # 4. iso‑surfaces
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    for tl_iso in iso_levels:
        if not (real_min < tl_iso < real_max):
            print(f"⚠️  Skipping {tl_iso} dB – outside TL range "
                  f"[{real_min:.1f}, {real_max:.1f}] dB")
            continue

        try:
            verts, faces, _, _ = measure.marching_cubes(
                np.nan_to_num(TL_vol, nan=nan_fill),
                level=tl_iso
            )
        except RuntimeError as e:
            print(f"⚠️  marching_cubes failed for {tl_iso} dB: {e}")
            continue

        # ---- vertex indices → physical coordinates (rounded) ------------
        idx_x = np.clip(np.round(verts[:, 2]).astype(int), 0, xi.size - 1)
        idx_y = np.clip(np.round(verts[:, 1]).astype(int), 0, yi.size - 1)
        idx_z = np.clip(np.round(verts[:, 0]).astype(int), 0, Z - 1)

        verts_xyz            = np.empty_like(verts)
        verts_xyz[:, 0]      = xi[idx_x]
        verts_xyz[:, 1]      = yi[idx_y]
        verts_xyz[:, 2]      = z_vec[idx_z]

        norm_val   = (tl_iso - real_min) / (real_max - real_min)
        face_color = cmap(norm_val)

        mesh = Poly3DCollection(verts_xyz[faces],
                                facecolor=face_color,
                                edgecolor='none',
                                alpha=0.4)
        ax.add_collection3d(mesh)

    # ------------------------------------------------------------------
    # 5. seabed surface
    # ------------------------------------------------------------------
    seabed_idx   = np.nanmax(np.where(np.isnan(TL_vol), np.nan, Z-1), axis=0)
    seabed_depth = z_vec[np.clip(seabed_idx.astype(int), 0, Z-1)]

    ax.plot_surface(X2d, Y2d, seabed_depth,
                    color=seabed_color, alpha=0.7,
                    linewidth=0, antialiased=False)

    # ------------------------------------------------------------------
    # 6. axes & layout
    # ------------------------------------------------------------------
    ax.set_xlabel('East–West range (m)')
    ax.set_ylabel('North–South range (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_zlim(z_vec.max(), 0)
    ax.set_title('3‑D Transmission‑Loss Iso‑Surfaces')
    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401
from pyproj import Transformer


def scatter_tl_3d(
        lat, lon, depth_grid, tl_grid,
        drifter_lat, drifter_lon,
        cmap=cm.viridis,
        vmin=None, vmax=None,
        max_points=None        # None = plot all
):
    """
    3‑D scatter of transmission‑loss values.

    Parameters
    ----------
    lat, lon          : (N,)  column positions (deg)
    depth_grid, tl_grid : (N, Z) arrays
        depth (m, +down) and TL (dB); NaNs allowed.
    drifter_lat, drifter_lon : float
        reference position (range = 0, 0)
    cmap               : Matplotlib colormap
    vmin, vmax         : colour scale limits (dB).  None -> finite min/max.
    max_points         : int, optional.  Randomly subsample to this many
                         points for faster plotting.
    """
    # ------------------------------------------------------------------
    # 1. Project lat/lon → UTM metres, set drifter at (0,0)
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
    x, y   = x - x0, y - y0                       # metres relative to drifter

    # ------------------------------------------------------------------
    # 2. Flatten valid samples
    # ------------------------------------------------------------------
    N, Z = tl_grid.shape
    xx   = np.repeat(x[:, None], Z, axis=1).ravel()
    yy   = np.repeat(y[:, None], Z, axis=1).ravel()
    zz   = depth_grid.ravel()
    tt   = tl_grid.ravel()

    valid = np.isfinite(tt) & np.isfinite(zz)
    if valid.sum() == 0:
        raise RuntimeError("No finite TL/depth values to plot")

    xx, yy, zz, tt = xx[valid], yy[valid], zz[valid], tt[valid]

    # Optional random down‑sampling for speed
    if max_points is not None and valid.sum() > max_points:
        idx = np.random.choice(valid.sum(), max_points, replace=False)
        xx, yy, zz, tt = xx[idx], yy[idx], zz[idx], tt[idx]

    # ------------------------------------------------------------------
    # 3. Scatter plot
    # ------------------------------------------------------------------
    if vmin is None:
        vmin = np.nanpercentile(tt, 5)    # avoid outliers
    if vmax is None:
        vmax = np.nanpercentile(tt, 95)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(xx, yy, zz,
                     c=tt, cmap=cmap,
                     vmin=vmin, vmax=vmax,
                     s=6, marker='o', linewidths=0, alpha=0.8)

    ax.set_xlabel('East–West range (m)')
    ax.set_ylabel('North–South range (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_zlim(np.nanmax(zz), 0)
    ax.set_title('3‑D Transmission‑Loss Scatter')

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel('TL (dB)')

    plt.tight_layout()
    plt.show()

hf = h5py.File('Spacious_Hawaii_250m.h5', 'r')
diveIds = list(hf['drift_01'].keys())
dive_grp = hf['drift_01'][diveIds[0]]['frequency_10000']
tl = np.array(dive_grp['tl'])
depth = np.array(dive_grp['depth'])
lat = np.array(dive_grp['lat'])
lon = np.array(dive_grp['lon'])

#sensor position
d_lat= dive_grp.parent.attrs['start_lat'] 
d_lon= dive_grp.parent.attrs['start_lon'] 


scatter_tl_3d(
    lat, lon, depth, tl,
    drifter_lat=d_lat,
    drifter_lon=d_lon,
    #vmin=-80, vmax=-60,     # choose a sensible range if you like
    max_points=40000         # down‑sample if rendering is slow
)
