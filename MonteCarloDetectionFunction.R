suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(stringr)
  library(geosphere)
})

# ============================== KNOBS ==========================================
CF_dB      <- 15         # crest factor p-p -> RMS
NL_rms_dB  <- 60         # ambient RMS noise (dB re 1 µPa)
SNR_thr_dB <- 20         # detection if RL_rms - NL_rms >= SNR_thr_dB
SL_ref_pp  <- 220        # CSV RL is on-axis at this SL

# Zimmer piston (13 kHz, D=0.55 m typical for sperm whale)
f_kHz      <- 13
D_m        <- 0.55
c_ms       <- 1500
DL_cap_dB  <- 30         # hard cap (loss never worse than -30 dB)

# ==================== SHIFTED-SCALED BETA PROBABILITY SIMS =====================
# Helper: derive (alpha, beta) from mode and "concentration"
.beta_ab_from_mode <- function(lo, hi, mode, conc) {
  stopifnot(lo < mode, mode < hi, conc > 2)
  m <- (mode - lo) / (hi - lo)
  a <- 1 + m * (conc - 2)
  b <- 1 + (1 - m) * (conc - 2)
  c(a=a, b=b)
}

# 1) Source level simulator: returns probability over a grid of SL (dB p-p)
sim_SL_probs <- function(sl_grid_pp, lo=205, hi=235, mode=220, conc=25) {
  ab <- .beta_ab_from_mode(lo, hi, mode, conc)
  u  <- (sl_grid_pp - lo) / (hi - lo)
  u  <- pmin(pmax(u, 0), 1)
  dens <- dbeta(u, ab["a"], ab["b"]) / (hi - lo) # proper density on dB scale
  tibble(SL_pp = sl_grid_pp, prob = dens / sum(dens))
}

# 2) Pitch simulator: probability over pitch 0..180 deg (90 = level)
sim_pitch_probs <- function(pitch_deg_grid, lo=0, hi=180, mode=110, conc=20) {
  ab <- .beta_ab_from_mode(lo, hi, mode, conc)
  u  <- (pitch_deg_grid - lo) / (hi - lo)
  u  <- pmin(pmax(u, 0), 1)
  dens <- dbeta(u, ab["a"], ab["b"]) / (hi - lo)
  tibble(pitch_deg = pitch_deg_grid, prob = dens / sum(dens))
}

# 3) Yaw/bearing simulator: uniform 0..360 deg
sim_yaw_probs <- function(yaw_deg_grid) {
  n <- length(yaw_deg_grid)
  tibble(yaw_deg = yaw_deg_grid, prob = rep(1/n, n))
}

# =================== ZIMMER PISTON BEAM & ATTENUATION ==========================
# NOTE: No self-referencing defaults in these three; pass args explicitly.
piston_gain_lin <- function(theta_rad, f_kHz, D_m, c_ms) {
  ka <- 2*pi*(f_kHz*1e3)/c_ms * (D_m/2)
  x  <- ka * sin(theta_rad)
  A  <- rep(1, length(x)); nz <- abs(x) > 0
  A[nz] <- 2 * besselJ(x[nz], 1) / x[nz]    # pressure directivity
  pmax(A*A, 1e-12)                          # power gain; floor to avoid -Inf
}

beam_gain_dB <- function(theta_rad, f_kHz, D_m, c_ms) {
  10*log10(piston_gain_lin(theta_rad, f_kHz, D_m, c_ms))
}

beam_loss_dB <- function(theta_rad, f_kHz, D_m, c_ms) {
  -beam_gain_dB(theta_rad, f_kHz, D_m, c_ms)
}

# Plot beam (use different arg names to avoid recursive default lookups)
plot_beam <- function(show = TRUE, f_kHz0 = f_kHz, D_m0 = D_m, c_ms0 = c_ms) {
  if (!show) return(invisible(NULL))
  a  <- seq(-90, 90, by=0.25)
  th <- abs(a)*pi/180
  dl <- beam_loss_dB(th, f_kHz = f_kHz0, D_m = D_m0, c_ms = c_ms0)
  if (!is.null(DL_cap_dB)) dl <- pmin(dl, DL_cap_dB)
  ggplot(tibble(angle=a, DL_dB=dl), aes(angle, DL_dB)) +
    geom_hline(yintercept=3, linetype=2) +
    geom_line() +
    labs(x="Off-axis angle (deg)", y="Directivity loss (dB)",
         title=sprintf("Zimmer piston (f=%g kHz, D=%.2f m)", f_kHz0, D_m0)) +
    theme_minimal(base_size=13)
}

# ============= OFF-AXIS ANGLE BETWEEN GLIDER AND TL CELL ======================
# Inputs:
#   glat,glon,gz_m: glider pos (deg,deg, depth +down)
#   wlat, wlon, wz_m: whale/cell pos (deg,deg, depth +down)
#   pitch_deg (0..180; 90=level), yaw_deg (0..360; 0=N, CW)
# Returns: alpha (rad) = angle between whale beam and LOS whale->glider
off_axis_angle_rad <- function(glat, glon, gz_m, wlat, wlon, wz_m, pitch_deg, yaw_deg) {
  # Horizontal geo
  az_deg <- geosphere::bearing(c(wlon, wlat), c(glon, glat))   # 0°=N, CW
  h_m    <- geosphere::distHaversine(c(wlon, wlat), c(glon, glat))
  az     <- az_deg * pi/180
  # LOS (East, North, Up)
  dz_up  <- ( -gz_m ) - ( -wz_m )   # up-positive
  R      <- sqrt(h_m^2 + dz_up^2)
  u_e    <- sin(az) * (h_m/R)
  u_n    <- cos(az) * (h_m/R)
  u_u    <- dz_up / R
  u <- c(u_e, u_n, u_u)
  
  # Whale beam axis from pitch & yaw
  yaw    <- (yaw_deg %% 360) * pi/180
  elev   <- (90 - pitch_deg) * pi/180    # +up, 0 at level
  b_e    <- cos(elev) * sin(yaw)
  b_n    <- cos(elev) * cos(yaw)
  b_u    <- sin(elev)
  b <- c(b_e, b_n, b_u)
  
  # Angle
  ct <- sum(b * u); ct <- max(-1, min(1, ct))
  acos(ct)
}

# ======== 2-D GRID (pitch × yaw) OF DIRECTIVITY LOSS FOR A GIVEN LOS ==========
directivity_grid <- function(glat, glon, gz_m, wlat, wlon, wz_m,
                             pitch_seq = seq(0,180,by=2),
                             yaw_seq   = seq(0,358,by=2),
                             cap_dB = DL_cap_dB,
                             f_kHz0 = f_kHz, D_m0 = D_m, c_ms0 = c_ms,
                             plot = TRUE) {
  grid <- expand.grid(pitch_deg=pitch_seq, yaw_deg=yaw_seq)
  alpha <- mapply(off_axis_angle_rad,
                  pitch_deg=grid$pitch_deg, yaw_deg=grid$yaw_deg,
                  MoreArgs=list(glat=glat, glon=glon, gz_m=gz_m,
                                wlat=wlat, wlon=wlon, wz_m=wz_m))
  dl <- beam_loss_dB(alpha, f_kHz = f_kHz0, D_m = D_m0, c_ms = c_ms0)
  if (!is.null(cap_dB)) dl <- pmin(dl, cap_dB)
  grid$DL_dB <- dl
  if (plot) {
    ggplot(grid, aes(yaw_deg, pitch_deg, fill=DL_dB)) +
      geom_raster(interpolate=FALSE) +
      scale_fill_viridis_c(option="C") +
      labs(x="Yaw / Bearing (deg)", y="Pitch (deg; 90=level)", fill="DL (dB)",
           title="Directivity loss over (pitch, yaw) for this LOS") +
      theme_minimal(base_size=13)
  } else grid
}

# ==================== LOAD RL CSV + PER-ROW CALCS =============================
# CSV columns: lat, lon, drifterlat, drifterlon, range, depth_m, RL (p-p dB on-axis at SL_ref_pp)
load_rl_csv <- function(csv_path) {
  df <- readr::read_csv(csv_path, show_col_types=FALSE) |>
    clean_names() |>
    rename(range_m=range, rl_pp_db=rl) |>
    select(lat, lon, drifterlat, drifterlon, range_m, depth_m, rl_pp_db)
  
  # Parse receiver depth from filename
  z_rx_m <- as.integer(str_match(basename(csv_path), "GliderDepth_([0-9]+)m")[,2])
  
  # horizontal bearing (cell -> glider)
  brg_deg <- geosphere::bearing(df[,c("lon","lat")], df[,c("drifterlon","drifterlat")])
  brg_deg <- (brg_deg + 360) %% 360
  
  # on-axis TL (p-p dB)
  TL_pp <- SL_ref_pp - df$rl_pp_db
  
  df |>
    mutate(
      glat = drifterlat, glon = drifterlon, gz_m = z_rx_m,
      brg_deg = brg_deg,
      TL_pp = TL_pp
    )
}

# ===== INTEGRATION OVER SL × PITCH × YAW → AREA-WEIGHTED g(r) =================
# Integrates detection probs over SL, pitch, yaw distributions; optional depth weighting across rows.
compute_g_of_r <- function(csv_path,
                           sl_grid = seq(205,235,by=0.5),
                           pitch_grid = seq(0,180,by=2),
                           yaw_grid = seq(0,358,by=2),
                           depth_prior = NULL,        # function(depth_m) -> weight; if NULL, uniform
                           range_bins = NULL,         # edges for r (m). If NULL, infer from unique ranges
                           plot=TRUE) {
  
  dat <- load_rl_csv(csv_path)
  
  # cell areas (km^2) from lat/lon grid
  lon_u <- sort(unique(dat$lon)); lat_u <- sort(unique(dat$lat))
  mid <- function(v) (head(v,-1)+tail(v,-1))/2
  lon_edges <- c(lon_u[1] - (lon_u[2]-lon_u[1])/2, mid(lon_u), tail(lon_u,1)+(tail(lon_u,1)-lon_u[length(lon_u)-1])/2)
  lat_edges <- c(lat_u[1] - (lat_u[2]-lat_u[1])/2, mid(lat_u), tail(lat_u,1)+(tail(lat_u,1)-lat_u[length(lat_u)-1])/2)
  ii <- match(dat$lat, lat_u); jj <- match(dat$lon, lon_u)
  area_mat <- matrix(NA_real_, nrow=length(lat_u), ncol=length(lon_u))
  for (i in seq_along(lat_u)) for (j in seq_along(lon_u)) {
    poly <- matrix(c(lon_edges[j],   lat_edges[i],
                     lon_edges[j+1], lat_edges[i],
                     lon_edges[j+1], lat_edges[i+1],
                     lon_edges[j],   lat_edges[i+1],
                     lon_edges[j],   lat_edges[i]), ncol=2, byrow=TRUE)
    area_mat[i,j] <- geosphere::areaPolygon(poly)/1e6
  }
  dat$cell_area_km2 <- area_mat[cbind(ii, jj)]
  
  # Discrete priors (normalized)
  pSL    <- sim_SL_probs(sl_grid_pp = sl_grid)
  pPitch <- sim_pitch_probs(pitch_grid)
  pYaw   <- sim_yaw_probs(yaw_grid)
  
  # Optional depth prior weights per row (normalize across rows)
  w_depth <- if (is.null(depth_prior)) rep(1, nrow(dat)) else depth_prior(dat$depth_m)
  w_depth <- w_depth / sum(w_depth)
  
  # Monte Carlo integration from discrete priors
  set.seed(123)
  N_mc <- 2000
  SL_samp    <- sample(pSL$SL_pp,         N_mc, replace=TRUE, prob=pSL$prob)
  pitch_samp <- sample(pPitch$pitch_deg,  N_mc, replace=TRUE, prob=pPitch$prob)
  yaw_samp   <- sample(pYaw$yaw_deg,      N_mc, replace=TRUE, prob=pYaw$prob)
  
  # For each row: compute detection probability under MC draws
  P_row <- numeric(nrow(dat))
  for (i in seq_len(nrow(dat))) {
    alpha <- mapply(off_axis_angle_rad,
                    pitch_deg = pitch_samp, yaw_deg = yaw_samp,
                    MoreArgs = list(glat = dat$glat[i], glon = dat$glon[i], gz_m = dat$gz_m[i],
                                    wlat = dat$lat[i],  wlon = dat$lon[i],  wz_m = dat$depth_m[i]))
    GdB <- beam_gain_dB(alpha, f_kHz=f_kHz, D_m=D_m, c_ms=c_ms)
    if (!is.null(DL_cap_dB)) GdB <- pmax(GdB, -abs(DL_cap_dB))  # cap loss
    # Effective RL_rms per draw for this row:
    # CSV RL is on-axis at SL_ref_pp. Adjust for SL draw, convert to RMS, add off-axis gain.
    RL_rms_draw <- (dat$rl_pp_db[i] + (SL_samp - SL_ref_pp) - CF_dB) + GdB
    det <- RL_rms_draw >= (NL_rms_dB + SNR_thr_dB)
    P_row[i] <- mean(det)
  }
  
  # Area × depth weighting per row
  w_area <- dat$cell_area_km2
  w_row  <- w_area * w_depth
  w_row  <- w_row / sum(w_row)
  
  # Build g(r): area- and depth-weighted average of P_row within range bins
  r <- dat$range_m
  if (is.null(range_bins)) {
    u <- sort(unique(round(r, 3)))
    dr <- if (length(u) >= 2) median(diff(u)) else 100
    range_bins <- seq(0, max(r, na.rm=TRUE) + dr, by=dr)
  }
  bin <- cut(r, breaks=range_bins, include.lowest=TRUE)
  idx <- split(seq_along(P_row), bin)
  midbins <- head(range_bins,-1) + diff(range_bins)/2
  g   <- sapply(idx, function(ix) {
    if (length(ix)==0) return(NA_real_)
    sum(w_row[ix] * P_row[ix], na.rm=TRUE) / sum(w_row[ix], na.rm=TRUE)
  })
  
  df_g <- tibble(r_km = midbins/1000, g = as.numeric(g))
  
  if (plot) {
    p <- ggplot(df_g, aes(r_km, g)) +
      geom_line(linewidth=1) +
      labs(x="Range (km)", y="Detection probability",
           title="Area-weighted average detection function g(r)") +
      theme_minimal(base_size=13)
    print(p)
  }
  df_g
}

# =========================== EXAMPLE WORKFLOW =================================
# 1) Plot beam to sanity-check
print(plot_beam(TRUE))

# 2) Optional: compute g(r) if you provide a CSV path
csv_path <- "C:\\Users\\kaity\\Documents\\SpaciousData\\CSVs\\bellhop_long\\PeakToPeak_dive_86_GliderDepth_200m_1_20khz_long.csv"

if (file.exists(csv_path)) {
  df_g <- compute_g_of_r(csv_path, plot=TRUE)
  print(head(df_g, 10))
} else {
  message("Set 'csv_path' to a valid file if you want to compute g(r).")
}
