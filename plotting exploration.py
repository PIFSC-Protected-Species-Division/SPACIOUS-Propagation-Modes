# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 21:18:59 2025

@author: pam_user
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401
from pyproj import Transformer

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
        seabed_color='0.6', # grey
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
    # 2. XY mesh & slice-by-slice interpolation
    # ------------------------------------------------------------------
    xi = np.linspace(x.min(), x.max(), xy_res)
    yi = np.linspace(y.min(), y.max(), xy_res)
    X2d, Y2d = np.meshgrid(xi, yi)

    z_vec = depth_grid[0, :]                     # may contain NaNs!
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
    # 3. TL range & NaN-fill for marching-cubes
    # ------------------------------------------------------------------
    finite_vals = TL_vol[np.isfinite(TL_vol)]
    if finite_vals.size == 0:
        raise RuntimeError("No finite TL values – check inputs")

    real_min, real_max = finite_vals.min(), finite_vals.max()
    nan_fill = real_min - 1.0

    # ------------------------------------------------------------------
    # 4. draw iso-surfaces
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    for tl_iso in iso_levels:
        if not (real_min < tl_iso < real_max):
            print(f"⚠️  Skipping {tl_iso} dB – outside TL range "
                  f"[{real_min:.1f}, {real_max:.1f}] dB")
            continue

        try:
            verts, faces, _, _ = measure.marching_cubes(
                np.nan_to_num(TL_vol, nan=nan_fill),
                level=tl_iso
            )
        except RuntimeError as e:
            print(f"⚠️  marching_cubes failed for {tl_iso} dB: {e}")
            continue

        idx_x = np.clip(np.round(verts[:, 2]).astype(int), 0, xi.size - 1)
        idx_y = np.clip(np.round(verts[:, 1]).astype(int), 0, yi.size - 1)
        idx_z = np.clip(np.round(verts[:, 0]).astype(int), 0, Z - 1)

        verts_xyz = np.empty_like(verts)
        verts_xyz[:, 0] = xi[idx_x]
        verts_xyz[:, 1] = yi[idx_y]
        verts_xyz[:, 2] = z_vec[idx_z]

        face_color = cmap((tl_iso - real_min) / (real_max - real_min))

        ax.add_collection3d(
            Poly3DCollection(verts_xyz[faces],
                             facecolor=face_color,
                             edgecolor='none',
                             alpha=0.4)
        )

    # ------------------------------------------------------------------
    # 5. seabed surface  (independent of TL, supports 'nearest' | 'linear')
    # ------------------------------------------------------------------
    # CONFIG ------------------------------------------------------------
    seabed_method   = 'linear'        # 'linear' looks smooth, 'nearest' is blocky
    max_radius_km   = 15              # clip anything > 15 km from the drifter
    # ------------------------------------------------------------------

    # A. deepest finite depth in each column ---------------------------
    seabed_raw = np.nanmax(depth_grid, axis=1)        # shape (N,)
    valid_cols = np.isfinite(seabed_raw)

    # B. interpolate to the regular XY mesh ----------------------------
    fill_val = np.nan                                   # keep NaNs outside hull
    seabed_grid = griddata(
        (x[valid_cols], y[valid_cols]), seabed_raw[valid_cols],
        (X2d, Y2d),
        method=seabed_method,
        fill_value=fill_val
    )

    # C. apply 15 km range mask ---------------------------------------
    R2d = np.sqrt(X2d**2 + Y2d**2)          # radial distance (m)
    seabed_grid[R2d > max_radius_km * 1000] = np.nan

    # D. convert to a masked array so plot_surface leaves NaN holes ----
    from numpy import ma
    seabed_mask = ma.masked_invalid(seabed_grid)

    # E. draw the bathymetry *first* so iso-surfaces stay visible ------
    ax.plot_surface(
        X2d, Y2d, seabed_mask,
        color=seabed_color,
        alpha=0.6,                # light transparency so TL shells show through
        linewidth=0,
        antialiased=False
    )

    # deepest real depth for z-axis limit
    max_depth = np.nanmax(seabed_raw)

    # ------------------------------------------------------------------
    # 6. axes & layout
    # ------------------------------------------------------------------
    ax.set_xlabel('East–West range (m)')
    ax.set_ylabel('North–South range (m)')
    ax.set_zlabel('Depth (m)')

    # Use the deepest seabed value for the z-limit
    max_depth = np.nanmax(seabed_raw)
    ax.set_zlim(max_depth, 0)

    ax.set_title('3-D Transmission-Loss Iso-Surfaces')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


def scatter_tl_3d(
        lat, lon, depth_grid, tl_grid,
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
    3-D scatter of TL plus optional seabed surface.

    seabed_source="data"  → build seabed from deepest finite depth per column
    seabed_source="netcdf"→ load bathymetry NetCDF (lon/lat grid, depth +ve down)

    bathy_nc : str
        Path to NetCDF file if using the netcdf option.
    bathy_var : str
        Variable in NetCDF holding depth (metres, +ve down).  Flip sign if needed.
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
    x, y = transformer.transform(lon, lat)
    x0, y0 = transformer.transform(drifter_lon, drifter_lat)
    x, y   = x - x0, y - y0           # metres relative to drifter

    # ------------------------------------------------------------------
    # 2. Flatten valid TL samples
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

    # Optional random down-sampling
    if max_points is not None and valid.sum() > max_points:
        idx = np.random.choice(valid.sum(), max_points, replace=False)
        xx, yy, zz, tt = xx[idx], yy[idx], zz[idx], tt[idx]

    # ------------------------------------------------------------------
    # 3. Prepare seabed surface (optional)
    # ------------------------------------------------------------------
    seabed_mesh = None
    if include_seabed:
        if seabed_source == "data":
            # deepest finite depth at each column
            seabed_depth = np.nanmax(depth_grid, axis=1)   # (N,)
            # Build a triangular surface directly from scattered points
            seabed_mesh = dict(
                x=x,
                y=y,
                z=seabed_depth
            )

        elif seabed_source == "netcdf":
            if bathy_nc is None:
                raise ValueError("bathy_nc must be provided for seabed_source='netcdf'")
            ds = xr.open_dataset(bathy_nc)
            # Assumes variables named "lon", "lat", plus bathy_var (metres, +ve down or -ve up)
            lon_b = ds['lon'].values
            lat_b = ds['lat'].values
            # meshgrid to 1-D vectors
            Lon2d, Lat2d = np.meshgrid(lon_b, lat_b)
            lon_vec = Lon2d.ravel()
            lat_vec = Lat2d.ravel()
            depth_vec = ds[bathy_var].values.ravel().astype(float)

            # Flip sign if bathy is negative up
            if depth_vec.max() <= 0:
                depth_vec = -depth_vec

            xb, yb = transformer.transform(lon_vec, lat_vec)
            xb, yb = xb - x0, yb - y0

            seabed_mesh = dict(x=xb, y= yb, z=depth_vec)
        else:
            raise ValueError("seabed_source must be 'data' or 'netcdf'")

    # ------------------------------------------------------------------
    # 4. Scatter plot
    # ------------------------------------------------------------------
    if vmin is None:
        vmin = np.nanpercentile(tt, 5)
    if vmax is None:
        vmax = np.nanpercentile(tt, 95)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    sc  = ax.scatter(xx, yy, zz,
                     c=tt, cmap=cmap,
                     vmin=vmin, vmax=vmax,
                     s=6, marker='o', linewidths=0, alpha=0.9)

    # ------------------------------------------------------------------
    # 5. Add seabed as trisurf (if requested)
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
    # 6. Labels, limits, colour-bar
    # ------------------------------------------------------------------
    ax.set_xlabel('East–West range (m)')
    ax.set_ylabel('North–South range (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_zlim(np.nanmax(zz), 0)
    ax.set_title('3-D Transmission-Loss Scatter with Seabed')
    
    ax.view_init(elev=elev, azim=azim)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel('TL (dB)')

    plt.tight_layout()
    plt.show()




hf = h5py.File('Spacious_Hawaii_250m_v0.h5', 'r')
diveIds = list(hf['drift_01'].keys())
dive_grp = hf['drift_01'][diveIds[1]]['frequency_10000']
tl = np.array(dive_grp['tl'])
depth = np.array(dive_grp['depth'])
lat = np.array(dive_grp['lat'])
lon = np.array(dive_grp['lon'])

#sensor position
d_lat= dive_grp.parent.attrs['start_lat'] 
d_lon= dive_grp.parent.attrs['start_lon'] 
d_depth = dive_grp.parent.attrs['drifter_depth']


scatter_tl_3d(
    lat, lon, depth, tl,
    drifter_lat=d_lat,
    drifter_lon=d_lon,
    max_points=40000,           # keep the plot nimble
    include_seabed=True,        # default
    seabed_source="data",        # <- default
    elev=25, azim=-45
)

plot_tl_isosurfaces(
    lat, lon, depth, tl,
    drifter_lat=d_lat,
    drifter_lon=d_lon,
    iso_levels=(-85,-87),
    elev=25, azim=140
)