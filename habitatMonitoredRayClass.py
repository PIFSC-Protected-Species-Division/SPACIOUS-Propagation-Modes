# -*- coding: utf-8 -*-
"""
Created on Sat May  3 10:23:17 2025

@author: kaity
"""

# propagation_runner.py
# -*- coding: utf-8 -*-
"""
Vectorised & threaded 2‑D Bellhop propagation runner.

Example
-------
from propagation_runner import PropagationModelRunner

runner = PropagationModelRunner(
    gebco_nc     ='gebco_2024_n23.5_s18.5_w-160.0_e-154.0.nc',   # or Path(...)
    drift_csv    ='sg639_MHI_Apr2023_CTD.csv',
    h5_out       ='Spacious_Hawaii_100m_rays_v2.h5',
    cores        =58,      # leave None → auto‑detect‑2
    freq_khz     =range(5_000, 35_000, 5_000),  # 5,10,…30 kHz
)
runner.run()        # crunches every dive in the CSV
"""


import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

import numpy as np          # safe now
import pandas as pd

import time, traceback, multiprocessing as mp
from multiprocessing.pool import ThreadPool
from functools import partial
from typing import Iterable, List, Tuple

import xarray as xr
import h5py
from pyproj import Geod
from geopy.distance import geodesic
import arlpy.uwapm as pm

__all__ = ["PropagationModelRunner"]


class PropagationModelRunner:
    # ------------------------------------------------------------------ #
    # construction / configuration
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        gebco_nc:      str,
        drift_csv:     str,
        h5_out:        str                                      ="Spacious_Hawaii_100m_rays_v2.h5",
        freq_khz:      Iterable[int]                            =range(5_000, 35_000, 5_000),
        cores:         int | None                               =None,
        blas_threads:  int                                      =1,
        bathy_radius_km:   float                                =40.0,
        freq_dep_rad: bool                                      =False,
        n_bearings:    int                                      =360,
        hyd_vert_m:      int                                      =50,
        bathy_interval_m:int                                    =100,
        fixedDrifterDetph: float =                              None,
        debug:         bool                                     =False,
    ):
        """
        Parameters
        ----------
        gebco_nc       : path to GEBCO NetCDF‑4 bathymetry
        drift_csv      : glider CTD export with DiveID, Depth_m, SoundSpeed_m_s …
        h5_out         : output HDF‑5 file for TL grids
        freq_khz       : iterable of centre frequencies **in hertz**
        cores          : how many worker threads to spawn
        blas_threads   : how many threads each BLAS call may spawn (OMP/MKL/…)
        bathy_radius_km: radius around the glider for bathy subset
        freq_dep_rad   : bool, whether to change the maximum eval range as with frequency
        n_bearings     : how many rays per dive (0 – 359° inclusive)
        hyd_vert       : vertical spacing between rx depths  [m]
        bathy_interval_m: along‑track sampling interval      [m]
        fixedDrifterDetph: optional int. Set a fixed drifter depth [m]
        debug          : if True, prints extra diagnostics
        """
        self.geod             = Geod(ellps="WGS84")
        self.gebco_nc         = gebco_nc
        self.drift_csv        = drift_csv
        self.h5_out           = h5_out
        self.freq_khz         = list(freq_khz)
        self.bathy_radius_km  = bathy_radius_km
        self.freq_dep_rad     = freq_dep_rad
        self.n_bearings       = n_bearings
        self.hyd_vert_m         = hyd_vert_m
        self.bathy_interval_m = bathy_interval_m
        self.debug            = debug
        self.fixedDrifterDetph = fixedDrifterDetph

        # ------------------------- threading & BLAS limits -------------
        self.cores = cores or max(1, mp.cpu_count() - 2)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = str(blas_threads)

        # ------------------------- big static inputs -------------------
        self.bathy_full = self._load_bathymetry(self.gebco_nc)
       
        raw_drift   = pd.read_csv(drift_csv)
        self.drift_ctd =  self._preprocess_drift(raw_drift) 

    # ==================================================================
    # public driver
    # ==================================================================
    def run(self, drift_ids: Iterable[str] | None = None) -> None:
        """
        Crunch each dive and append its TL grids to `self.h5_out`.
        """
        drift_ids = (
            list(drift_ids)
            if drift_ids is not None
            else self.drift_ctd["DiveID"].drop_duplicates().to_list()
        )

        step   = 360 / self.n_bearings        # angular increment
        angles = np.int16(np.arange(0, 360, step))
        

        with ThreadPool(processes=self.cores) as pool:
            for dive_id in drift_ids:
                group = self.drift_ctd[self.drift_ctd["DiveID"] == dive_id]
                if group.empty:
                    print(f"[skip] DiveID {dive_id!r} not found")
                    continue
                
                # If the drifet depth is set, use that otherwise take the first 
                # drift detph in the valid range
                if self.fixedDrifterDetph is not None:
                    drifterDepth = self.fixedDrifterDetph
                else:
                    drifterDepth  =group['Depth_m'][group['Depth_m'].first_valid_index()] 

                # Pull out the subset of data around the drifter
                lat0, lon0 = group["Latitude"].iloc[0], group["Longitude"].iloc[0]
                subset_df  = self._subset_bathy(lat0, lon0)

                ssp_table  = self._build_ssp(group, subset_df["depth"].abs().max())
                metadata   = dict(start_lat=lat0, 
                                  start_lon=lon0, 
                                  drifter_depth=100)

                for f_hz in self.freq_khz:
                    # Remember the parameters...
                    # angle_deg: float,
                    # lat0: float,
                    # lon0: float,
                    # freq_hz: float,
                    # ssp: list[list[float]],
                    # tx_depth: float,
                    # max_distance_km: float,
                    # freq_dep_rad: bool,
                    # interval_m: float,
                    # hyd_vert_m: int,
                    
                    # Colate the tasks. User defined input for maximum propagation range
                    if self.freq_dep_rad ==True:
                        tasks = [
                        (ang, lat0, lon0, f_hz, ssp_table,
                         metadata["drifter_depth"],
                         (45_000 - f_hz) / 1_000,   # crude max‑range rule
                         self.bathy_interval_m,
                         self.hyd_vert_m)
                        for ang in angles]
                        
                    else:
                        tasks = [
                            (ang, lat0, lon0, f_hz, ssp_table,
                             metadata["drifter_depth"],
                             self.bathy_radius_km,   # single detection range
                             self.bathy_interval_m,
                             self.hyd_vert_m)
                            for ang in angles
                        ]
                        

                    t0 = time.time()
                    results: List[dict] = []
                    for status, ang, payload in pool.imap_unordered(
                        self._safe_worker, tasks, chunksize=1
                    ):
                        if status == "fail":
                            print(f"❌  bearing {ang:>3}°  →  {payload!r}")
                            continue
                        results.append(
                            dict(
                                angle_deg=payload[0],
                                range_m=payload[2],
                                depth_m=payload[3],
                                transmission_loss=payload[1],
                            )
                        )
                        if self.debug:
                            print(f"✓  {dive_id} {f_hz/1000:.1f} kHz  "
                                  f"bearing {ang:3d}° done")

                    self._save_dive_frequency(
                        drift_id="01",  # update if you have multiple drifts
                        dive_id=dive_id,
                        freq_khz=f_hz,
                        metadata=metadata,
                        grid_results=results,
                    )
                    print(f"[{dive_id}] {f_hz/1000:.1f} kHz → "
                          f"{len(results)}/{len(angles)} bearings "
                          f"in {(time.time()-t0):.1f}s")

    # ==================================================================
    # ----------------------  internal helpers  ------------------------
    # ==================================================================
    @staticmethod
    def _load_bathymetry(nc_path: str) -> pd.DataFrame:
        ds        = xr.open_dataset(nc_path)
        lon, lat  = np.meshgrid(ds["lon"].values, ds["lat"].values)
        depth2d   = ds["elevation"].values
        return pd.DataFrame(
            dict(depth=depth2d.ravel(), lat=lat.ravel(), lon=lon.ravel())
        )

    # ---------- spatial subset around a point -------------------------
    def _subset_bathy(self, lat0: float, lon0: float) -> pd.DataFrame:
        df = self.bathy_full.copy()
        df["distance_km"] = self._haversine(
            lon0, lat0, df["lon"].values, df["lat"].values
        )
        return df[df["distance_km"] <= self.bathy_radius_km].reset_index(drop=True)

    # ---------- sound‑speed profile -----------------------------------
    def _build_ssp(self, dive: pd.DataFrame, max_depth: float) -> list[list[float]]:
        prof = (
            dive[["Depth_m", "SoundSpeed_m_s"]]
            .rename(columns={"Depth_m": "depth", "SoundSpeed_m_s": "ss"})
            .dropna()
            .sort_values("depth")
            .reset_index(drop=True)
        )
        prof.loc[0, "depth"] = 0
        if prof["depth"].iloc[-1] < max_depth:
            extra = pd.DataFrame(
                dict(
                    depth=np.arange(prof["depth"].iloc[-1] + 10, max_depth + 50, 50),
                    ss=prof["ss"].iloc[-1],
                )
            )
            prof = pd.concat([prof, extra], ignore_index=True)
        prof["ss"] = prof["ss"].abs()
        return prof.apply(lambda r: [r.depth, r.ss], axis=1).tolist()

    # ---------- static utilities --------------------------------------
    @staticmethod
    def _preprocess_drift(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure 'Direction' (asc/dec) and 'DiveID' columns exist,
        exactly as in the original script.
        """
        if "Direction" not in df.columns:
            depth_diff = np.diff(df["Depth_m"], prepend=np.nan)
            df["Direction"] = np.where(depth_diff > 0, "dec", "asc")
            df.at[0, "Direction"] = "dec" if depth_diff[1] > 0 else "asc"

        if "DiveID" not in df.columns:
            df["DiveID"] = df["DiveNumber"].astype(str) + "_" + df["Direction"]

        return df
    # ------------------------------------------------------------------
    # (everything else identical to the previous v1 code)
    @staticmethod
    def _haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    @staticmethod
    def _variable_rx_range(max_km, s0_km=0.2, s1_km=0.1, target_km=40):
        k, a = (s0_km - s1_km) / target_km, 1.0 - (s0_km - s1_km) / target_km
        inner_end = min(target_km, max_km)
        n_inner = int(np.ceil(np.log(max(1e-12, 1 - inner_end * k / s0_km)) / np.log(a)))
        r_inner = (s0_km / k) * (1 - a ** np.arange(n_inner + 1))
        r_inner = r_inner[r_inner <= inner_end]
        if max_km <= target_km:
            return r_inner
        tail = np.arange(r_inner[-1] + s1_km, max_km + s1_km * 0.5, s1_km)
        return np.concatenate((r_inner, tail[tail <= max_km]))

    # ==================================================================
    # ---------------- worker & HDF‑5 IO  -------------------------------
    # ==================================================================
    def _safe_worker(self, args):
        ang = args[0]
        try:
            return ("ok", ang, self._worker(*args))
        except Exception as exc:
            if self.debug:
                traceback.print_exc()
            return ("fail", ang, exc)

    def _worker(
        self,
        angle_deg: float,
        lat0: float,
        lon0: float,
        freq_hz: float,
        ssp: list[list[float]],
        tx_depth: float,
        max_distance_km: float,
        interval_m: float,
        hyd_vert_m: int,
    ):
        bathy_val, lons, lats, r_km = self._extract_bathy_ray(
            lat0, lon0, angle_deg, max_distance_km, interval_m
        )
        bathy_grid = pd.DataFrame(
            dict(range=r_km * 1000, depth_m=-bathy_val)
        ).drop_duplicates()
        bathy_grid.sort_values("range", inplace=True)
        bathy_grid.loc[bathy_grid.index[0], "range"] = 0

        # truncate when seabed < 100 m
        shoal = np.where(bathy_grid["depth_m"] < 100)[0]
        if shoal.size:
            bathy_grid = bathy_grid.iloc[: shoal.min()]

        env = pm.create_env2d(
            depth=bathy_grid.apply(lambda r: [r.range, r.depth_m], axis=1).tolist(),
            soundspeed=ssp,
            bottom_density=2700,
            bottom_absorption=0.1,
            bottom_soundspeed=5250,
            tx_depth=tx_depth,
            frequency=freq_hz,
            nbeams=0,
            max_angle=90,
            min_angle=-90,
        )
        env["rx_range"] = (
            self._variable_rx_range(bathy_grid["range"].iloc[-1] / 1000) * 1000
        )
        env["rx_depth"] = np.arange(0, bathy_grid["depth_m"].max(), hyd_vert_m)

        tloss = pm.compute_transmission_loss(env, mode="incoherent")
        tl_db = 20 * np.log10(np.abs(tloss))

        # seabed mask
        seabed = np.interp(
            env["rx_range"],
            bathy_grid["range"].values,
            bathy_grid["depth_m"].values,
            left=np.nan,
            right=np.nan,
        )
        mask = env["rx_depth"][:, None] > seabed[None, :]
        tl_db = np.where(mask, np.nan, tl_db)

        return angle_deg, pd.DataFrame(np.round(tl_db, 2)), env["rx_range"], env[
            "rx_depth"
        ]

    # ---------- extract bathy along a single ray ----------------------
    def _extract_bathy_ray(
        self, lat0, lon0, bearing, max_km, interval_m
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        interval_km = interval_m / 1000.0
        n_pts       = max(int(max_km / interval_km), 1)
        dist_m      = np.linspace(0.0, max_km * 1000.0, n_pts + 1)

        lons, lats, _ = self.geod.fwd(
            np.full_like(dist_m, lon0),
            np.full_like(dist_m, lat0),
            np.full_like(dist_m, bearing),
            dist_m,
        )
        subset_pts  = self.bathy_full[["lat", "lon"]].values
        subset_z    = self.bathy_full["depth"].values
        path_pts    = np.column_stack((lats, lons))

        z = self._griddata(subset_pts, subset_z, path_pts)
        return z, lons, lats, dist_m / 1000.0

    # tiny wrapper that fills NaN with nearest‑neighbour
    @staticmethod
    def _griddata(points, values, xi):
        from scipy.interpolate import griddata as gd

        z = gd(points, values, xi, method="linear")
        if np.any(np.isnan(z)):
            z[np.isnan(z)] = gd(points, values, xi[np.isnan(z)], method="nearest")
        return z

    # ---------- save routine (unchanged structurally) -----------------
    def _save_dive_frequency(
        self,
        drift_id: str,
        dive_id: str,
        freq_khz: int,
        metadata: dict,
        grid_results: List[dict],
        gzip_level: int = 4,
    ):
        if not grid_results:
            print(f"[{dive_id}] no successful bearings – nothing saved.")
            return

        n_ang   = len(grid_results)
        max_Nz  = max(len(g["depth_m"])  for g in grid_results)
        max_Nr  = max(len(g["range_m"])  for g in grid_results)

        angle_deg  = np.empty(n_ang,               np.float32)
        valid_Nz   = np.empty(n_ang,               np.uint16)
        valid_Nr   = np.empty(n_ang,               np.uint16)
        depth_mat  = np.full((n_ang, max_Nz),      np.nan, np.float32)
        range_mat  = np.full((n_ang, max_Nr),      np.nan, np.float32)
        tl_mat     = np.full((n_ang, max_Nz, max_Nr), np.nan, np.float32)

        for i, g in enumerate(grid_results):
            d, r, tl = g["depth_m"], g["range_m"], g["transmission_loss"].values
            angle_deg[i], valid_Nz[i], valid_Nr[i] = g["angle_deg"], len(d), len(r)
            depth_mat [i, : len(d)] = d
            range_mat [i, : len(r)] = r
            tl_mat    [i, : len(d), : len(r)] = tl

        with h5py.File(self.h5_out, "a") as hf:
            fgrp = (
                hf.require_group(f"drift_{drift_id}")
                  .require_group(f"dive_{dive_id}")
                  .require_group(f"frequency_{freq_khz}")
            )
            for k, v in metadata.items():
                fgrp.parent.attrs[k] = v

            def _dset(name, data, **kw):
                if name in fgrp:
                    del fgrp[name]
                fgrp.create_dataset(
                    name, data=data, compression="gzip", compression_opts=gzip_level, **kw
                )

            _dset("angle_deg", angle_deg)
            _dset("valid_depth_len", valid_Nz)
            _dset("valid_range_len", valid_Nr)
            _dset("depth_m", depth_mat,  chunks=(min(32, n_ang), max_Nz))
            _dset("range_m", range_mat,  chunks=(min(32, n_ang), max_Nr))
            _dset(
                "transmission_loss",
                tl_mat,
                chunks=(min(16, n_ang), min(64, max_Nz), min(256, max_Nr)),
            )



#%% Run the analysis

if __name__ == "__main__":
    # Bathymetry data from NCEI
    nc_file_loc = 'C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\bathymetry\\GEBCO_28_Mar_2025_ade9db365e34\\gebco_2024_n23.5_s18.5_w-160.0_e-154.0.nc'
    
    # Raw csv of drift SSP data
    drift_csv_loc = "C:\\Users\\kaity\\Downloads\\sg639_MHI_Apr2023_CTD_test.csv"

    # Run with fixed drifter depth
    # RunClass =PropagationModelRunner( gebco_nc =nc_file_loc,
    #                        drift_csv =drift_csv_loc,
    #                        h5_out = 'testClass.h5py',
    #                        cores =2,
    #                        freq_khz = (5000,),
    #                        bathy_radius_km =3,
    #                        n_bearings =4,
    #                        fixedDrifterDetph = 100)
    
    # Run with starting depth
    RunClass =PropagationModelRunner( gebco_nc =nc_file_loc,
                           drift_csv =drift_csv_loc,
                           h5_out = 'testClass_variable_depth_360.h5py',
                           cores =4,
                           freq_khz = (2000,),
                           bathy_radius_km =1,
                           n_bearings =120,
                           hyd_vert_m =100)
    
    RunClass.run()
                           
        # """
        # Parameters
        # ----------
        # gebco_nc       : path to GEBCO NetCDF‑4 bathymetry
        # drift_csv      : glider CTD export with DiveID, Depth_m, SoundSpeed_m_s …
        # h5_out         : output HDF‑5 file for TL grids
        # freq_khz       : iterable of centre frequencies **in hertz**
        # cores          : how many worker threads to spawn
        # blas_threads   : how many threads each BLAS call may spawn (OMP/MKL/…)
        # bathy_radius_km: radius around the glider for bathy subset
        # freq_dep_rad   : bool, whether to change the maximum eval range as with frequency
        # n_bearings     : how many rays per dive (0 – 359° inclusive)
        # hyd_vert       : vertical spacing between rx depths  [m]
        # bathy_interval_m: along‑track sampling interval      [m]
        # fixedDrifterDetph: optional int. Set a fixed drifter depth [m]
        # debug          : if True, prints extra diagnostics
        # """