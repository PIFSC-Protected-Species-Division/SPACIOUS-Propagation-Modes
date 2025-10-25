################################################################################
# Sensitivity pipeline for acoustic detection (g(r) → p̄, EDR, Rmax)
# ------------------------------------------------------------------------------
# What this script does (end-to-end):
#   1) Defines priors for SL, pitch, yaw, and depth (with helpers to plot them).
#   2) Loads Bellhop RL grids, builds area weights for grid cells.
#   3) Runs a Monte Carlo to compute g(r) by range bin, keeping per-draw values.
#   4) From g(r) draws, computes:
#        - Unconditional p̄ (median + 90% CI)
#        - Effective Detection Radius (EDR) (median + 90% CI)
#        - Max modeled range Rmax, and optional R50/R25 (medians)
#   5) Discovers scenarios from your directory (bottom/dive/glider depth).
#   6) Sweeps priors (pitch × depth) across all scenarios.
#   7) Produces:
#        - `summaries`: absolute metrics per scenario/prior combo
#        - `summ_rel`: relative % change vs. a reference combo, per location
#   8) Draws two baseline heatmaps (absolute p̄ and absolute EDR).
#
# Notes:
#   * All “draw-aware” summaries (p̄, EDR) come from the predictive draws of g(r),
#     so CIs are coherent with the MC process (no resampling needed).
#   * Keep your RL CSVs organized as: <root>/<bottom>/PeakToPeak_dive_<ID>_GliderDepth_<Z>m_*.csv
################################################################################

# -------------------------- 0) CLEAN ENV & LIBS -------------------------------

rm(list = ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(stringr)
  library(geosphere)
  library(ggplot2)
})


# --------------------- 2) PRIOR DEFINITIONS & UTILITIES -----------------------
# All priors are implemented as small objects with:
#   - $lo, $hi : support range
#   - $rpdf(x) : density on the shifted/scaled support
#   - $rsamp(n): random sampler
# They can be defined as functions (factory) or as ready-made lists.

# Convert mode+concentration on [lo,hi] to Beta(alpha,beta) (utility, optional)
.beta_ab_from_mode <- function(lo, hi, mode, conc) {
  stopifnot(lo < mode, mode < hi, conc > 2)
  m <- (mode - lo) / (hi - lo)
  a <- 1 + m * (conc - 2)
  b <- 1 + (1 - m) * (conc - 2)
  c(a=a, b=b)
}

# Shifted & scaled Beta on [shift, shift+scale]
make_shifted_beta <- function(alpha, beta, scale, shift) {
  stopifnot(is.finite(alpha), is.finite(beta), alpha > 0, beta > 0)
  stopifnot(is.finite(scale), is.finite(shift), scale > 0)
  lo <- shift; hi <- shift + scale; a <- as.numeric(alpha); b <- as.numeric(beta)
  list(
    kind="beta", lo=lo, hi=hi, a=a, b=b,
    # PDF on shifted/scaled support (apply Jacobian 1/scale)
    rpdf  = function(x){
      u <- (x - lo) / (hi - lo)
      dens <- dbeta(u, a, b) / (hi - lo)
      dens[u < 0 | u > 1 | !is.finite(u)] <- 0
      dens
    },
    # Sampler on [lo,hi]
    rsamp = function(n) lo + (hi - lo) * rbeta(n, a, b)
  )
}

# Uniform prior on [lo,hi]
make_uniform <- function(lo, hi) {
  list(kind="uniform", lo=lo, hi=hi,
       rpdf=function(x) dunif(x, lo, hi),
       rsamp=function(n) runif(n, lo, hi))
}

# Quick base-R plot for any prior or list of priors (for sanity checks)
plot_prior <- function(prior, title=NULL, n=500, show_legend=TRUE) {
  to_prior_obj <- function(p) if (is.function(p)) p() else p
  
  # Single prior passed in
  if (!is.list(prior) || is.null(names(prior))) {
    p <- to_prior_obj(prior)
    x <- seq(p$lo, p$hi, length.out=n); y <- p$rpdf(x)
    main <- if (!is.null(title)) sprintf("Prior PDF: %s", title)
    else sprintf("Prior PDF: %s", deparse(substitute(prior)))
    plot(x, y, type="l", lwd=2, col="blue",
         xlab = if (!is.null(title)) title else deparse(substitute(prior)),
         ylab = "Density", main = main)
    polygon(x, y, col = adjustcolor("skyblue", 0.4), border = NA)
    return(invisible(data.frame(x=x, pdf=y)))
  }
  
  # List of priors passed in: overlay with distinct colors
  pri_list <- lapply(prior, to_prior_obj)
  los <- vapply(pri_list, \(p) p$lo, numeric(1))
  his <- vapply(pri_list, \(p) p$hi, numeric(1))
  xlim <- c(min(los), max(his))
  
  curves <- lapply(pri_list, function(p) {
    x <- seq(p$lo, p$hi, length.out=n); y <- p$rpdf(x); list(x=x, y=y)
  })
  ymax <- max(vapply(curves, \(c) max(c$y, na.rm=TRUE), 0))
  cols <- grDevices::hcl.colors(length(pri_list), "Set2")
  fills <- vapply(cols, \(cc) adjustcolor(cc, 0.25), "")
  
  main <- if (!is.null(title)) sprintf("Prior PDF: %s", title)
  else sprintf("Prior PDF: %s", deparse(substitute(prior)))
  xlab <- if (!is.null(title)) title else deparse(substitute(prior))
  plot(NA, NA, xlim=xlim, ylim=c(0, ymax*1.05), xlab=xlab, ylab="Density", main=main)
  
  nms <- names(pri_list); if (is.null(nms) || any(nms=="")) nms <- paste0("prior_", seq_along(pri_list))
  for (i in seq_along(pri_list)) {
    lines(curves[[i]]$x, curves[[i]]$y, col=cols[i], lwd=2)
    polygon(curves[[i]]$x, curves[[i]]$y, col=fills[i], border=NA)
  }
  if (isTRUE(show_legend)) legend("topright", legend=nms, col=cols, lwd=2, bty="n")
}

# Named prior factories used in the sweep
SL_prior <- function() make_shifted_beta(alpha=10, beta=3, scale=10,   shift=190)  # pp dB


# Construct a full prior set by name (used by run_single)
build_priors <- function(pitch_key="down_biased", depth_key="mid") {
  stopifnot(pitch_key %in% names(pitch_priors), depth_key %in% names(depth_priors))
  list(SL=SL_prior(), pitch=pitch_priors[[pitch_key]](), yaw=yaw_prior(), depth=depth_priors[[depth_key]]())
}

# --------------------- 3) BEAM MODEL (Zimmer piston) --------------------------
# Pressure directivity → power gain; then convert to dB. We clamp at -DL_cap_dB
# to avoid degenerate zeroes (helps numerical stability).
piston_gain_lin <- function(theta_rad, f_kHz, D_m, c_ms) {
  ka <- 2*pi*(f_kHz*1e3)/c_ms * (D_m/2)
  x  <- ka * sin(theta_rad)
  A  <- rep(1, length(x)); nz <- abs(x) > 0
  A[nz] <- 2 * besselJ(x[nz], 1) / x[nz]
  pmax(A*A, 1e-12)
}
beam_gain_dB <- function(theta_rad, f_kHz, D_m, c_ms) 10*log10(piston_gain_lin(theta_rad, f_kHz, D_m, c_ms))
beam_loss_dB <- function(theta_rad, f_kHz, D_m, c_ms) -beam_gain_dB(theta_rad, f_kHz, D_m, c_ms)

# Fast lookup: angle (rad) → beam gain (dB) with optional floor at -DL_cap_dB
.make_beam_lookup <- function(f_kHz, D_m, c_ms, DL_cap_dB, n=20001L) {
  th <- seq(0, pi, length.out=n)
  ka <- 2*pi*(f_kHz*1e3)/c_ms * (D_m/2)
  x  <- ka * sin(th)
  A  <- rep(1, length(x)); nz <- abs(x) > 0
  A[nz] <- 2 * besselJ(x[nz], 1) / x[nz]
  Glin <- pmax(A*A, 1e-12)
  GdB  <- 10*log10(Glin)
  if (!is.null(DL_cap_dB)) GdB <- pmax(GdB, -abs(DL_cap_dB))
  approxfun(th, GdB, rule=2)  # safe extrapolation at edges
}

# Quick diagnostic plot (optional)
plot_beam <- function(show=TRUE, f_kHz0=f_kHz, D_m0=D_m, c_ms0=c_ms) {
  if (!show) return(invisible(NULL))
  a  <- seq(-90, 90, by=0.25)
  th <- abs(a)*pi/180
  dl <- beam_loss_dB(th, f_kHz0, D_m0, c_ms0)
  if (!is.null(DL_cap_dB)) dl <- pmin(dl, DL_cap_dB)
  ggplot(tibble(angle=a, DL_dB=dl), aes(angle, DL_dB)) +
    geom_hline(yintercept=3, linetype=2) +
    geom_line() +
    labs(x="Off-axis angle (deg)", y="Directivity loss (dB)",
         title=sprintf("Zimmer piston (f=%g kHz, D=%.2f m)", f_kHz0, D_m0)) +
    theme_minimal(base_size=13)
}

# --------------------- 4) RL CSV LOADING & GRID WEIGHTS -----------------------
# Expected columns in CSV:
#   lat, lon, drifterlat, drifterlon, range, depth_m, RL
#   RL is pp dB on-axis at SL_ref_pp. We derive gz (glider depth) from filename.
load_rl_csv <- function(csv_path) {
  df <- readr::read_csv(csv_path, show_col_types=FALSE) |>
    clean_names() |>
    rename(range_m=range, rl_pp_db=rl) |>
    select(lat, lon, drifterlat, drifterlon, range_m, depth_m, rl_pp_db)
  
  # Receiver depth (glider depth) encoded in filename
  z_rx_m <- as.integer(str_match(basename(csv_path), "GliderDepth_([0-9]+)m")[,2])
  
  # Bearing from cell → glider; keep also a pp TL for completeness
  brg_deg <- geosphere::bearing(df[,c("lon","lat")], df[,c("drifterlon","drifterlat")])
  brg_deg <- (brg_deg + 360) %% 360
  TL_pp   <- SL_ref_pp - df$rl_pp_db
  
  df |>
    mutate(glat=drifterlat, glon=drifterlon, gz_m=z_rx_m,
           brg_deg=brg_deg, TL_pp=TL_pp)
}

# Build robust range bins (edges) from raw ranges
.make_range_bins <- function(r, target_nbins = 200) {
  r <- r[is.finite(r)]
  u <- sort(unique(r))
  if (length(u) <= 1L) {
    dr <- if (length(u) == 1L) max(1, u/10) else 1
    return(c(min(r, 0) - dr, max(r, 0) + dr))
  }
  du <- diff(u); du <- du[du > 0]
  dr <- if (length(du) == 0L) max(1, stats::sd(r, na.rm=TRUE)) else stats::median(du)
  span <- max(u) - min(u); dr_t <- span / target_nbins
  if (is.finite(dr_t) && dr_t > 0) dr <- dr * max(1L, round(dr_t/dr))
  seq(min(u) - 0.5*dr, max(u) + 0.5*dr, by=dr)
}

# --------------------- 5) CORE: Monte Carlo g(r) ------------------------------
# Given a CSV file and priors, computes detection probability vs range (g(r)).
# Returns a tibble with bin midpoints (km) and mean g, plus attributes:
#   attr(., "G_draws")  : matrix [bins × draws] with per-draw g(r) by bin
#   attr(., "r_km_vec") : numeric vector of bin midpoints (km)

# ------------------------------------------------------------------------------
# compute_g_of_r_fast()
#   - Monte Carlo over SL, pitch, yaw, depth
#   - Collapses cell detections into bins by PLAIN MEAN (no area in g)
#   - Returns g(r) + predictive draws and attaches per-bin area fractions f_bin
# ------------------------------------------------------------------------------
compute_g_of_r_fast <- function(
    csv_path,
    priors,
    N_mc = 2000,
    range_bins = NULL,
    return_draws = TRUE,
    depth_mode = c("cell-resample", "csv"),   # "cell-resample" = new behavior
    seed = 123
) {
  depth_mode <- match.arg(depth_mode)
  
  # ---- 1) Load RL grid; group rows into surface cells (lon,lat) ---------------
  RL <- load_rl_csv(csv_path)
  cell_key     <- paste(RL$lon, RL$lat, sep = " ")
  rows_by_cell <- split(seq_len(nrow(RL)), cell_key)
  n_cell       <- length(rows_by_cell)
  rep_row      <- vapply(rows_by_cell, `[`, integer(1), 1L)
  
  # Horizontal geometry (constant across depths)
  p_src_cell <- as.matrix(RL[rep_row, c("lon","lat")])
  p_rx_cell  <- as.matrix(RL[rep_row, c("glon","glat")])
  az_cell    <- geosphere::bearing(p_src_cell, p_rx_cell) * pi/180
  H_cell     <- geosphere::distHaversine(p_src_cell, p_rx_cell)   # m
  gz_cell    <- RL$gz_m[rep_row]                                  # m
  r_cell     <- RL$range_m[rep_row]                               # m
  
  # ---- 2) Cell areas (km^2) and bin edges ------------------------------------
  lon_u <- sort(unique(RL$lon)); lat_u <- sort(unique(RL$lat))
  mid   <- function(v) (head(v,-1) + tail(v,-1))/2
  lon_edges <- c(lon_u[1] - (lon_u[2]-lon_u[1])/2, mid(lon_u),
                 tail(lon_u,1) + (tail(lon_u,1)-lon_u[length(lon_u)-1])/2)
  lat_edges <- c(lat_u[1] - (lat_u[2]-lat_u[1])/2, mid(lat_u),
                 tail(lat_u,1) + (tail(lat_u,1)-lat_u[length(lat_u)-1])/2)
  
  ii <- match(RL$lat[rep_row], lat_u)
  jj <- match(RL$lon[rep_row], lon_u)
  
  area_mat <- matrix(NA_real_, nrow = length(lat_u), ncol = length(lon_u))
  for (i in seq_along(lat_u)) {
    for (j in seq_along(lon_u)) {
      poly <- matrix(
        c(lon_edges[j],   lat_edges[i],
          lon_edges[j+1], lat_edges[i],
          lon_edges[j+1], lat_edges[i+1],
          lon_edges[j],   lat_edges[i+1],
          lon_edges[j],   lat_edges[i]),
        ncol = 2, byrow = TRUE
      )
      area_mat[i, j] <- geosphere::areaPolygon(poly) / 1e6  # km^2
    }
  }
  area_cell <- area_mat[cbind(ii, jj)]     # raw cell areas (km^2)
  A_tot     <- sum(area_cell, na.rm = TRUE)
  
  # ---- 3) MC draws for SL, pitch, yaw, depth ---------------------------------
  set.seed(seed)
  pitch_deg <- priors$pitch$rsamp(N_mc)
  yaw_deg   <- priors$yaw$rsamp(N_mc)
  SL_pp     <- priors$SL$rsamp(N_mc)
  
  # Beam-axis unit vectors per draw
  yaw  <- (yaw_deg %% 360) * pi/180
  elev <- (90 - pitch_deg) * pi/180
  b_e <- cos(elev) * sin(yaw)
  b_n <- cos(elev) * cos(yaw)
  b_u <- sin(elev)
  Bm  <- rbind(b_e, b_n, b_u)   # 3 × N_mc
  
  # Per-row on-axis baseline (RMS dB) and threshold
  base_row_rms <- RL$rl_pp_db - SL_ref_pp - CF_dB
  thr_rms_dB   <- NL_rms_dB + SNR_thr_dB
  
  # Depth layers available per cell
  depth_lists <- lapply(rows_by_cell, function(ix) {
    d <- RL$depth_m[ix]; o <- order(d)
    list(depths = d[o], rows = ix[o])
  })
  
  # ---- 4) Choose source depth per cell×draw (CSV depth or rejection-resample) -
  Z_src_sel    <- matrix(NA_real_, nrow = n_cell, ncol = N_mc)
  base_rms_sel <- matrix(NA_real_, nrow = n_cell, ncol = N_mc)
  
  if (depth_mode == "csv") {
    Z_src_sel[,]    <- RL$depth_m[rep_row]
    base_rms_sel[,] <- base_row_rms[rep_row]
  } else {
    # Rejection sample from prior within each cell's available [min,max]
    Z_samp <- matrix(priors$depth$rsamp(n_cell * N_mc), nrow = n_cell, ncol = N_mc)
    min_d  <- vapply(depth_lists, function(d) min(d$depths, na.rm = TRUE), 0.0)
    max_d  <- vapply(depth_lists, function(d) max(d$depths, na.rm = TRUE), 0.0)
    
    max_iter <- 50L; iter <- 0L
    repeat {
      iter <- iter + 1L
      too_shallow <- Z_samp < matrix(min_d, nrow = n_cell, ncol = N_mc)
      too_deep    <- Z_samp > matrix(max_d, nrow = n_cell, ncol = N_mc)
      bad <- too_shallow | too_deep
      if (!any(bad, na.rm = TRUE)) break
      if (iter > max_iter) {
        warning("Depth rejection exceeded max_iter; filling remainder with Uniform[min,max].")
        Z_samp[bad] <- runif(sum(bad),
                             min = min_d[row(Z_samp)[bad]],
                             max = max_d[row(Z_samp)[bad]])
        break
      }
      Z_samp[bad] <- priors$depth$rsamp(sum(bad))
    }
    
    # Snap to nearest available CSV layer (vectorized per cell)
    for (c in seq_len(n_cell)) {
      d_av <- depth_lists[[c]]$depths  # sorted
      r_av <- depth_lists[[c]]$rows
      zc   <- Z_samp[c, ]
      pos  <- findInterval(zc, d_av); pos[pos < 1] <- 1; pos[pos >= length(d_av)] <- length(d_av)
      prev <- pmax(pos - 1, 1)
      pick <- ifelse(abs(d_av[prev] - zc) <= abs(d_av[pos] - zc), prev, pos)
      Z_src_sel[c, ]    <- d_av[pick]
      base_rms_sel[c, ] <- base_row_rms[r_av[pick]]
    }
  }
  
  # ---- 5) Geometry per cell×draw, beam loss, and detection --------------------
  H  <- matrix(H_cell,  nrow = n_cell, ncol = N_mc)
  az <- matrix(az_cell, nrow = n_cell, ncol = N_mc)
  gz <- matrix(gz_cell, nrow = n_cell, ncol = N_mc)
  
  dz_up <- Z_src_sel - gz
  R     <- sqrt(H^2 + dz_up^2)
  u_e   <- sin(az) * (H / R)
  u_n   <- cos(az) * (H / R)
  u_u   <- dz_up   / R
  
  cosA <- u_e %*% diag(Bm[1, ]) + u_n %*% diag(Bm[2, ]) + u_u %*% diag(Bm[3, ])
  cosA[cosA >  1] <- 1; cosA[cosA < -1] <- -1
  alpha <- acos(cosA)
  
  gdb_fun <- .make_beam_lookup(f_kHz, D_m, c_ms, DL_cap_dB)
  GdB     <- matrix(gdb_fun(abs(alpha)), nrow = nrow(alpha), ncol = ncol(alpha))
  
  RL_rms  <- GdB
  RL_rms  <- sweep(RL_rms, 2, SL_pp, `+`)
  RL_rms  <- RL_rms + base_rms_sel
  det_mat <- RL_rms >= thr_rms_dB            # (cell × draw) TRUE/FALSE
  
  # ---- 6) Range bins & collapse to g(r) WITHOUT area weighting ----------------
  if (is.null(range_bins)) {
    r <- r_cell
    u <- sort(unique(r[is.finite(r)]))
    if (length(u) <= 1L) {
      dr <- if (length(u) == 1L) max(1, u/10) else 1
      range_bins <- c(min(r, 0) - dr, max(r, 0) + dr)
    } else {
      du <- diff(u); du <- du[du > 0]
      dr <- if (length(du) == 0L) max(1, stats::sd(r, na.rm = TRUE)) else stats::median(du)
      span <- max(u) - min(u); dr_t <- span / 200
      if (is.finite(dr_t) && dr_t > 0) dr <- dr * max(1L, round(dr_t/dr))
      range_bins <- seq(min(u) - 0.5*dr, max(u) + 0.5*dr, by = dr)
    }
  }
  
  bin     <- cut(r_cell, breaks = range_bins, include.lowest = TRUE, right = TRUE)
  groups  <- split(seq_len(n_cell), bin)
  midbins <- head(range_bins, -1) + diff(range_bins) / 2
  
  # Per-bin area fraction f_b (for p̄ integration later)
  f_bin <- vapply(groups, function(ix) {
    if (!length(ix)) 0 else sum(area_cell[ix], na.rm = TRUE) / A_tot
  }, numeric(1))
  
  # g(r) per-draw: plain mean across cells in each bin (NO area here!)
  nb <- length(groups)
  G_draws <- matrix(NA_real_, nrow = nb, ncol = N_mc)
  for (b in seq_len(nb)) {
    ix <- groups[[b]]
    if (!length(ix)) next
    G_draws[b, ] <- colMeans(det_mat[ix, , drop = FALSE], na.rm = TRUE)
  }
  
  g_point <- rowMeans(G_draws, na.rm = TRUE)
  g_lo    <- apply(G_draws, 1, stats::quantile, probs = 0.025, na.rm = TRUE)
  g_hi    <- apply(G_draws, 1, stats::quantile, probs = 0.975, na.rm = TRUE)
  
  df_g <- tibble::tibble(
    r_km = midbins / 1000,
    g    = as.numeric(g_point),
    g_lo = as.numeric(g_lo),
    g_hi = as.numeric(g_hi)
  )
  
  # (Optional prettifying for the quick plot)
  linfill <- function(x, y) {
    ok <- is.finite(y) & is.finite(x)
    if (sum(ok) < 2L) return(y)
    stats::approx(x = x[ok], y = y[ok], xout = x, rule = 2)$y
  }
  df_plot <- df_g[order(df_g$r_km), , drop = FALSE]
  df_plot$g    <- linfill(df_plot$r_km, df_plot$g)
  df_plot$g_lo <- linfill(df_plot$r_km, df_plot$g_lo)
  df_plot$g_hi <- linfill(df_plot$r_km, df_plot$g_hi)
  
  p <- ggplot(df_plot, aes(x = r_km, y = g)) +
    geom_ribbon(aes(ymin = g_lo, ymax = g_hi), alpha = 0.30) +
    geom_line(linewidth = 1) +
    labs(x = "Range (km)", y = "Detection probability",
                  title = "Detection function g(r) with predictive bands") +
    theme_minimal(base_size = 13)
  print(p)
  
  if (isTRUE(return_draws)) {
    attr(df_g, "G_draws")  <- G_draws         # [bins × draws] g(r) per draw
    attr(df_g, "r_km_vec") <- df_g$r_km       # bin midpoints (km)
    attr(df_g, "f_bin")    <- f_bin           # area fraction per bin (∑ ≈ 1)
  }
  
  df_g
}



# --------------------- 6) RANGE SUMMARIES & METRICS ---------------------------
# Find the range where g(r) crosses probability p (per draw ⇒ summarize)
summarize_detection_range <- function(df_g, p=0.5, probs=c(0.05,0.5,0.95)) {
  G <- attr(df_g,"G_draws"); rvec <- attr(df_g,"r_km_vec"); stopifnot(!is.null(G), !is.null(rvec))
  r_per_draw <- apply(G, 2, function(gcol){
    ok <- is.finite(gcol) & is.finite(rvec); g <- gcol[ok]; r <- rvec[ok]
    if (length(g) < 2) return(NA_real_)
    o <- order(r); r <- r[o]; g <- g[o]
    if (all(g > p,  na.rm=TRUE)) return(max(r, na.rm=TRUE))
    if (all(g <= p, na.rm=TRUE)) return(min(r, na.rm=TRUE))
    i <- which(head(g,-1) > p & tail(g,-1) <= p)
    if (!length(i)) i <- which(head(g,-1) < p & tail(g,-1) >= p)
    if (!length(i)) return(NA_real_)
    i <- i[1]
    r[i] + (p - g[i]) * (r[i+1] - r[i]) / (g[i+1] - g[i])
  })
  tibble(
    p     = p,
    R_lo  = quantile(r_per_draw, probs[1], na.rm=TRUE),
    R_med = quantile(r_per_draw, probs[2], na.rm=TRUE),
    R_hi  = quantile(r_per_draw, probs[3], na.rm=TRUE)
  )
}

# Unconditional detection probability p̄ (deterministic midpoint discretization).
# We still use draws for CIs later; this is mainly for reference.
compute_pbar <- function(df_g) {
  r_m <- df_g$r_km * 1000; g <- df_g$g
  if (length(r_m) < 2 || all(!is.finite(g))) return(NA_real_)
  dr <- c(r_m[1], diff(r_m))              # approximate Δr per midpoint
  Rmax <- max(r_m, na.rm=TRUE)
  if (!is.finite(Rmax) || Rmax <= 0) return(NA_real_)
  (2 / Rmax^2) * sum(g * r_m * dr, na.rm=TRUE)
}

# Draw-aware summary for p̄ and EDR (medians + 90% CI) + Rmax + R50/R25 medians
# ------------------------------------------------------------------------------
# summarize_df_g()
#   - Uses the returned f_bin to compute p̄ = Σ f_b * g_b,draw  (per draw)
#   - EDR = sqrt(p̄) * Rmax  (per draw), then reports medians + 90% CI
# ------------------------------------------------------------------------------
summarize_df_g <- function(df_g, ci = 0.90) {
  G <- attr(df_g, "G_draws")
  r <- attr(df_g, "r_km_vec")
  f <- attr(df_g, "f_bin")
  stopifnot(!is.null(G), !is.null(r), !is.null(f))
  
  keep <- rowSums(is.finite(G)) > 0
  G <- G[keep, , drop = FALSE]
  r <- r[keep]
  f <- f[keep]
  
  # Unconditional p̄ per draw (all “area” is here, not in g)
  pbar_draw <- as.numeric(t(f) %*% G)   # 1 × N_draw
  
  Rmax_km      <- max(r, na.rm = TRUE)
  EDR_draw_km  <- sqrt(pbar_draw) * Rmax_km
  
  q <- c((1-ci)/2, 0.5, 1 - (1-ci)/2)
  
  r50 <- suppressWarnings(summarize_detection_range(df_g, p = 0.50, probs = q))
  r25 <- suppressWarnings(summarize_detection_range(df_g, p = 0.25, probs = q))
  
  tibble::tibble(
    pbar_med   = as.numeric(stats::quantile(pbar_draw,   q[2], na.rm = TRUE)),
    pbar_lo    = as.numeric(stats::quantile(pbar_draw,   q[1], na.rm = TRUE)),
    pbar_hi    = as.numeric(stats::quantile(pbar_draw,   q[3], na.rm = TRUE)),
    EDR_km_med = as.numeric(stats::quantile(EDR_draw_km, q[2], na.rm = TRUE)),
    EDR_km_lo  = as.numeric(stats::quantile(EDR_draw_km, q[1], na.rm = TRUE)),
    EDR_km_hi  = as.numeric(stats::quantile(EDR_draw_km, q[3], na.rm = TRUE)),
    Rmax_km    = Rmax_km,
    R50_km     = as.numeric(r50$R_med),
    R25_km     = as.numeric(r25$R_med)
  )
}

# --------------------- 7) SCENARIO DISCOVERY (files → meta) -------------------
# Parses your folder structure into a tibble of (bottom, dive, gdepth_m, csv_path)
list_scenarios <- function(root_dir) {
  files <- list.files(root_dir, recursive=TRUE, full.names=TRUE, pattern="\\.csv$")
  tibble(csv_path = files) |>
    mutate(
      rel      = str_remove(csv_path, fixed(paste0(root_dir, .Platform$file.sep))),
      bottom   = str_split(rel, .Platform$file.sep, simplify=TRUE)[,1],
      dive     = as.integer(str_match(rel, "dive_([0-9]+)")[,2]),
      gdepth_m = as.integer(str_match(rel, "GliderDepth_([0-9]+)m")[,2])
    ) |>
    arrange(bottom, dive, gdepth_m)
}

# --------------------- 8) SINGLE-RUN WRAPPER (guardrails) ---------------------
# Executes compute_g_of_r_fast() for one (file × priors) and returns both:
#   - df_g : the full g(r) object with draws
#   - summary : one-row tibble with p̄/EDR/Rmax/R50/R25
run_single <- function(csv_path, pitch_key="down_biased", depth_key="mid",
                       N_mc=300, DL_cap_dB=60, seed=123) {
  DL_cap_dB <<- DL_cap_dB                      # make local cap visible to beam lookup
  priors <- build_priors(pitch_key, depth_key) # assemble prior set
  set.seed(seed)
  df_g <- tryCatch(
    compute_g_of_r_fast(csv_path=csv_path, priors=priors,
                        N_mc=N_mc, range_bins=NULL, return_draws=TRUE),
    error = function(e){
      message("FAILED: ", basename(csv_path), " | ", pitch_key, "/", depth_key, " | ", e$message)
      return(NULL)
    }
  )
  if (is.null(df_g)) return(NULL)
  list(df_g=df_g, summary=summarize_df_g(df_g, ci=0.90))
}

# ========================= 1) Build relative table ============================
# Works whether your summaries have pbar_med/EDR_km_med (preferred) or pbar/EDR_km.
make_rel_tables <- function(summaries,
                            ref_bottom   = "silt",
                            ref_gdepth_m = 500,
                            ref_pitch    = "down_biased",
                            ref_depth    = "mid") {
  
  # pick column names that exist
  pbar_col <- if ("pbar_med"   %in% names(summaries)) "pbar_med"   else "pbar"
  edr_col  <- if ("EDR_km_med" %in% names(summaries)) "EDR_km_med" else "EDR_km"
  
  stopifnot(all(c(pbar_col, edr_col, "bottom","gdepth_m","pitch_key",
                  "depth_key","dive") %in% names(summaries)))
  
  # per-dive reference values
  ref_tbl <- summaries |>
    dplyr::filter(bottom == ref_bottom,
                  gdepth_m == ref_gdepth_m,
                  pitch_key == ref_pitch,
                  depth_key == ref_depth) |>
    dplyr::select(dive,
                  pbar_ref = dplyr::all_of(pbar_col),
                  EDR_ref  = dplyr::all_of(edr_col))
  
  # join & compute relative % deltas
  summ_rel <- summaries |>
    dplyr::left_join(ref_tbl, by = "dive") |>
    dplyr::mutate(
      gdepth      = factor(gdepth_m),
      facet_col   = interaction(pitch_key, depth_key, sep = "\n"),
      d_pbar_pct  = 100 * (.data[[pbar_col]] / pbar_ref - 1),
      d_EDR_pct   = 100 * (.data[[edr_col]]  / EDR_ref  - 1),
      is_ref      = (bottom == ref_bottom & gdepth_m == ref_gdepth_m &
                       pitch_key == ref_pitch & depth_key == ref_depth)
    )
  
  # for drawing the black outline
  outline_df <- summ_rel |>
    dplyr::filter(is_ref) |>
    dplyr::transmute(dive, facet_col, x = gdepth, y = bottom)
  
  list(summ_rel = summ_rel, outline_df = outline_df)
}

# ========================= 2) Plot helpers (per dive) =========================
# Same look as your example: rows=sediment; cols=pitch×depth; line+points;
# diverging scale; black ring on the reference cell.


plot_rel_points_by_dive <- function(df_rel, outline_df, dive_id,
                                    value = c("d_pbar_pct","d_EDR_pct"),
                                    limits = c(-20, 20),
                                    title_prefix = "Relative") {
  value <- match.arg(value)
  fill_lab <- if (value == "d_pbar_pct") "Δp̄ (%)" else "ΔEDR (%)"
  
  d   <- dplyr::filter(df_rel, dive == dive_id)
  out <- dplyr::filter(outline_df, dive == dive_id)
  
  ggplot(d, aes(x = factor(gdepth_m), y = .data[[value]])) +
    geom_hline(yintercept = 0, color = "grey90") +
    geom_line(aes(group = interaction(bottom, facet_col)),
                       color = "grey70", linewidth = 0.5) +
    geom_point(aes(fill = .data[[value]]),
                        shape = 21, size = 2.8, stroke = 0.3, color = "black") +
    geom_point(data = out, aes(x = x, y = 0),
                        inherit.aes = FALSE,
                        shape = 21, size = 4.2, stroke = 1.1, fill = NA, color = "black") +
    facet_grid(rows = vars(bottom),
                        cols  = vars(interaction(pitch_key, depth_key, sep = "\n"))) +
    scale_fill_gradient2(fill_lab, midpoint = 0,
                                  low = "#d73027", mid = "white", high = "#1a9850",
                                  limits = limits, oob = scales::squish) +
   labs(x = "Glider depth (m)", y = NULL,
                  title = sprintf("%s %s — dive %s",
                                  title_prefix,
                                  if (value == "d_pbar_pct") "p̄ vs reference" else "EDR vs reference",
                                  as.character(dive_id)),
                  subtitle = "Rows = sediment; columns = pitch × depth priors") +
    theme_minimal(base_size = 13) +
    theme(panel.grid.minor = element_blank(),
                   #panel.grid.major.y = element_blank(),
                   #axis.text.y  = element_blank(),
                   #axis.ticks.y = element_blank(),
                   strip.text.x = element_text(size = 10))
}


################################################################################
# --------------------- 9) DESIGN: which scenarios/priors to run ---------------

pitch_priors <- list(
  Diving = function() make_shifted_beta(7,4, 180, 0),  # degrees (more downward)
  Foraging       = function() make_shifted_beta(4,4, 180, 0),
  Ascending   = function() make_shifted_beta(4,7, 180, 0)
)
depth_priors <- list(
  shallow = function() make_shifted_beta(7,4, 600, 200),     # meters
  mid     = function() make_shifted_beta(7,4, 600, 400),
  deep    = function() make_shifted_beta(7,4, 600, 600)
)
yaw_prior <- function() make_uniform(0, 360)                 # degrees (uniform)


plot_prior(depth_priors)
plot_prior(pitch_priors)

# Acoustic / processing constants (adjust here, used everywhere)
CF_dB      <- 15   # Crest factor: convert peak-to-peak to RMS (pp dB -> RMS dB)
NL_rms_dB  <- 60   # Ambient noise level (RMS dB re 1 µPa)
SNR_thr_dB <- 5    # Detection threshold: RL_rms - NL_rms >= this SNR (dB)
SL_ref_pp  <- 220  # Reference SL (pp dB) that Bellhop on-axis RL corresponds to

# Piston (Zimmer) beam parameters (13 kHz sperm whale click)
f_kHz      <- 13
D_m        <- 0.55
c_ms       <- 1500
DL_cap_dB  <- 60   # Floor beam loss at -60 dB to avoid unreal side-lobes
N_mc = 10 # Number of draws per lat/lon


# Root folder containing all Bellhop CSVs (by sediment/dive/depth)
root_dir   <- "C:\\Users\\kaity\\Documents\\SpaciousData\\CSVs\\BotSensitivityCSVs"


scenarios_all <- list_scenarios(root_dir)
print(scenarios_all)

# Choose levels to sweep (full set shown; shrink if needed)
bottom_sel <- c("bassalt", "gravel", "silt")
dive_sel   <- c(42, 86, 167)
gdepth_sel <- c(200, 500, 800)
pitch_sel  <- c("Diving", "Foraging", "Ascending")
depth_sel  <- c("shallow", "mid", "deep")

# Build the scenario grid (files) then cross with prior choices
scen_grid <- scenarios_all |>
  filter(bottom %in% bottom_sel, dive %in% dive_sel, gdepth_m %in% gdepth_sel) |>
  arrange(bottom, dive, gdepth_m) |>
  mutate(scen_id = row_number())

design <- expand_grid(
  scen      = scen_grid,
  pitch_key = pitch_sel,
  depth_key = depth_sel
) |>
  mutate(scen_label = sprintf("%s | dive=%d | z=%.0fm | pitch=%s | depth=%s",
                              scen$bottom, scen$dive, scen$gdepth_m, pitch_key, depth_key))

# --------------------- 10) RUN ALL DESIGN POINTS ------------------------------
# For each (file × prior combo), compute g(r) and summarize.
results <- purrr::pmap(
  list(design$scen$csv_path, design$pitch_key, design$depth_key),
  ~ run_single(csv_path=..1, 
               pitch_key=..2, 
               depth_key=..3,
               N_mc=N_mc, DL_cap_dB=60)
)

# Collect per-run summaries (skip failed runs cleanly)
summaries <- purrr::map2_dfr(results, seq_along(results), function(res, i) {
  if (is.null(res)) return(NULL)
  des_row <- design[i,]
  tibble(
    scen_label = des_row$scen_label,
    pitch_key  = des_row$pitch_key,
    depth_key  = des_row$depth_key,
    bottom     = des_row$scen$bottom,
    dive       = des_row$scen$dive,
    gdepth_m   = des_row$scen$gdepth_m
  ) |> dplyr::bind_cols(res$summary)
})

# Sanity: required columns present?
stopifnot(all(c("pbar_med","EDR_km_med","Rmax_km") %in% names(summaries)))
glimpse(summaries)

# --------------------- 11) RELATIVE TABLE (vs per-dive reference) -------------
# ========================= 3) Usage ===========================================
# ---- Set your exact reference (case-sensitive) ----
ref_bottom    <- "silt"
ref_gdepth_m  <- 500
ref_pitch_key <- "Foraging"   # must match summaries$pitch_key exactly
ref_depth_key <- "mid"        # must match summaries$depth_key exactly

# ---- Make per-dive reference and relative deltas (no fallback logic) ----
stopifnot(all(c("pbar_med","EDR_km_med","bottom","dive","gdepth_m",
                "pitch_key","depth_key") %in% names(summaries)))

ref_tbl <- summaries %>%
  dplyr::filter(bottom   == ref_bottom,
                gdepth_m == ref_gdepth_m,
                pitch_key== ref_pitch_key,
                depth_key== ref_depth_key) %>%
  dplyr::select(dive,
                pbar_ref = pbar_med,
                EDR_ref  = EDR_km_med)

summ_rel <- summaries %>%
  dplyr::left_join(ref_tbl, by = "dive") %>%            # joins NA if that dive lacks a ref row
  dplyr::mutate(
    gdepth    = factor(gdepth_m),
    facet_col = interaction(pitch_key, depth_key, sep = "\n"),
    d_pbar_pct = 100 * (pbar_med   / pbar_ref - 1),
    d_EDR_pct  = 100 * (EDR_km_med / EDR_ref  - 1),
    is_ref = (bottom   == ref_bottom &
                gdepth_m == ref_gdepth_m &
                pitch_key== ref_pitch_key &
                depth_key== ref_depth_key)
  )

# For the black outline (only where the exact ref truly exists)
outline_df <- summ_rel %>%
  dplyr::filter(is_ref) %>%
  dplyr::transmute(dive, facet_col, x = gdepth, y = bottom)


dives <- sort(unique(summ_rel$dive))

# Δp̄ (%)
invisible(lapply(dives, function(dv)
  print(plot_rel_points_by_dive(summ_rel, outline_df, dv,
                                value = "d_pbar_pct",
                                limits = c(-20, 20),
                                title_prefix = "Relative p̄ vs reference"))))

# ΔEDR (%)
invisible(lapply(dives, function(dv)
  print(plot_rel_points_by_dive(summ_rel, outline_df, dv,
                                value = "d_EDR_pct",
                                limits = c(-20, 20),
                                title_prefix = "Relative EDR vs reference"))))


# --------------------- 12) BASELINE ABSOLUTE HEATMAPS -------------------------
# These mirror the earlier figures you liked; they’re handy “first looks.”

# a) Absolute p̄ (median)
p_abs_pbar <- summaries |>
  mutate(gdepth=factor(gdepth_m), facet_col=interaction(pitch_key, depth_key, sep="\n")) |>
  ggplot(aes(x=gdepth, y=bottom, fill=pbar_med)) +
  geom_tile(color="white") +
  facet_grid(dive ~ facet_col) +
  scale_fill_viridis_c(name="p̄ (median)", option="C") +
  labs(x="Glider depth (m)", y="Bottom",
       title="Unconditional p(detect) — by bottom × glider depth",
       subtitle="Rows = dive (location); columns = pitch × depth prior") +
  theme_minimal(base_size=12)

# b) Absolute EDR (median)
p_abs_edr <- summaries |>
  mutate(gdepth=factor(gdepth_m), facet_col=interaction(pitch_key, depth_key, sep="\n")) |>
  ggplot(aes(x=gdepth, y=bottom, fill=EDR_km_med)) +
  geom_tile(color="white") +
  facet_grid(dive ~ facet_col) +
  scale_fill_viridis_c(name="EDR (km, median)", option="C") +
  labs(x="Glider depth (m)", y="Bottom",
       title="Effective Detection Radius — by bottom × glider depth",
       subtitle="Rows = dive (location); columns = pitch × depth prior") +
  theme_minimal(base_size=12)

print(p_abs_pbar)
print(p_abs_edr)

# --------------------- 13) At this point you also have ------------------------
#   - `summaries` : absolute metrics (p̄, EDR, Rmax, R50, R25) per scenario/prior
#   - `summ_rel`  : relative % changes vs reference combo (per dive)
# From here you can re-create:
#   * faceted relative heatmaps with the black reference outline
#   * point/line vistas
#   * polar diagnostics
################################################################################


# ================== Build a long table of g(r) with metadata ==================
# Uses existing objects: `results` (list of run_single outputs) and `design`

library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(purrr)

# 1) Extract per-run df_g and attach scenario columns
g_long <- purrr::map2_dfr(results, seq_along(results), function(res, i){
  if (is.null(res) || is.null(res$df_g)) return(NULL)
  meta <- design[i, ]
  as_tibble(res$df_g) %>%
    mutate(
      bottom    = meta$scen$bottom,
      dive      = meta$scen$dive,
      gdepth_m  = meta$scen$gdepth_m,
      pitch_key = meta$pitch_key,
      depth_key = meta$depth_key
    )
})

# Safety checks
stopifnot(all(c("r_km","g","g_lo","g_hi","bottom","dive","gdepth_m",
                "pitch_key","depth_key") %in% names(g_long)))

# 2) Nice facet labels to match your figure
pitch_lab_map <- c(
  "Diving"   = "Diving",
  "Foraging" = "Feeding-horizontal",
  "Ascending"= "Surfacing"
)
depth_lab_map <- c(
  "deep" = "Deep Divers",
  "mid"  = "Mid Divers",
  "shallow" = "Shallow Divers"
)

g_plot <- g_long %>%
  mutate(
    # facet labels
    pitch_lab = factor(pitch_lab_map[pitch_key], levels = c("Diving","Feeding-horizontal","Surfacing")),
    depth_lab = factor(depth_lab_map[depth_key], levels = c("Deep Divers","Mid Divers","Shallow Divers")),
    # legend for glider depth (order 200, 500, 800)
    gdepth_f  = factor(gdepth_m, levels = c(200, 500, 800))
  )

# 3) Filter to the panel you want (e.g., Silt, Location 42)
g_panel <- g_plot %>% filter(bottom == "silt", dive == 42)

# 4) Colors to match your example (200=red, 500=green, 800=blue)
depth_cols  <- c("200" = "#e74c3c", "500" = "#2ecc71", "800" = "#3498db")
depth_fills <- c("200" = scales::alpha("#e74c3c", 0.25),
                 "500" = scales::alpha("#2ecc71", 0.25),
                 "800" = scales::alpha("#3498db", 0.25))

# 5) Plot
p_gr <- ggplot(g_panel, aes(x = r_km, y = g,
                            color = gdepth_f, fill = gdepth_f)) +
  geom_ribbon(aes(ymin = g_lo, ymax = g_hi), 
              linewidth = 0, alpha = 0.25, show.legend = FALSE) +
  geom_line(linewidth = 0.6) +
  facet_grid(rows = vars(depth_lab), cols = vars(pitch_lab)) +
  scale_color_manual(values = depth_cols, name = "Glider depth (m)") +
  scale_fill_manual(values = depth_fills, guide = "none") +
  coord_cartesian(xlim = c(0, 20), ylim = c(0, 1)) +
  labs(
    x = "Range (km)",
    y = "Detection probability g(r)",
    title = "Detection functions g(r): Silt, Location 42"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text.x = element_text(size = 11),
    strip.text.y = element_text(size = 11)
  )

print(p_gr)

# ---------------- Cap domain at 15 km and rebuild f(r) -----------------
r_cap <- 15      # km
bin_width <- 0.5 # km (tune if you like)

# Surface-cell range (km) and area (km^2) as before
cell_key     <- paste(RL$lon, RL$lat, sep=" ")
rows_by_cell <- split(seq_len(nrow(RL)), cell_key)
rep_row      <- vapply(rows_by_cell, function(ix) ix[1], integer(1))
r_cell_km    <- RL$range_m[rep_row] / 1000

# Per-cell area (km^2) as before
lon_u <- sort(unique(RL$lon)); lat_u <- sort(unique(RL$lat))
mid <- function(v) (head(v,-1) + tail(v,-1))/2
lon_edges <- c(lon_u[1] - (lon_u[2]-lon_u[1])/2, mid(lon_u),
               tail(lon_u,1) + (tail(lon_u,1) - lon_u[length(lon_u)-1])/2)
lat_edges <- c(lat_u[1] - (lat_u[2]-lat_u[1])/2, mid(lat_u),
               tail(lat_u,1) + (tail(lat_u,1) - lat_u[length(lat_u)-1])/2)
ii <- match(RL$lat[rep_row], lat_u); jj <- match(RL$lon[rep_row], lon_u)

area_mat <- matrix(NA_real_, nrow=length(lat_u), ncol=length(lon_u))
for (i in seq_along(lat_u)) for (j in seq_along(lon_u)) {
  poly <- matrix(c(lon_edges[j],   lat_edges[i],
                   lon_edges[j+1], lat_edges[i],
                   lon_edges[j+1], lat_edges[i+1],
                   lon_edges[j],   lat_edges[i+1],
                   lon_edges[j],   lat_edges[i]), ncol=2, byrow=TRUE)
  area_mat[i,j] <- geosphere::areaPolygon(poly) / 1e6  # km^2
}
area_cell <- area_mat[cbind(ii, jj)]

# Depth filter (keep cells with ANY layer ≤ 2000 m)
min_depth_cell <- vapply(rows_by_cell, function(ix) min(RL$depth_m[ix], na.rm=TRUE), numeric(1))
keep_depth <- (min_depth_cell <= 2000)

# Range cap filter and combined keep
keep_cap  <- (r_cell_km <= r_cap)
keep_cell <- keep_depth & keep_cap

r_cell_km <- r_cell_km[keep_cell]
area_cell <- area_cell[keep_cell]

# Area weights renormalized on [0, r_cap] with depth filter
w_cell <- area_cell / sum(area_cell, na.rm=TRUE)

# Regular bins on [0, r_cap]
edges_km <- seq(0, r_cap, by = bin_width)
if (tail(edges_km,1) < r_cap) edges_km <- c(edges_km, r_cap)  # ensure exact cap
mid_km   <- head(edges_km,-1) + diff(edges_km)/2

# Mass per bin (area-probability), then cumulative & smooth
bin_id <- cut(r_cell_km, breaks=edges_km, include.lowest=TRUE, right=TRUE)
mass_per_bin <- tapply(w_cell, bin_id, sum, na.rm=TRUE)
mass_per_bin <- ifelse(is.na(mass_per_bin), 0, mass_per_bin)
# Align in case of empty factor levels
if (length(mass_per_bin) != length(mid_km)) {
  tmp <- rep(0, length(mid_km)); names(tmp) <- levels(bin_id)
  tmp[names(mass_per_bin)] <- mass_per_bin
  mass_per_bin <- tmp
}
A_step <- cumsum(mass_per_bin)

# Smooth cumulative area then differentiate for a smooth, nonnegative f(r)
spl <- smooth.spline(x = mid_km, y = A_step, spar = 0.65)
A_smooth <- pmin(pmax(predict(spl, x = mid_km)$y, 0), 1)
f_smooth <- pmax(predict(spl, x = mid_km, deriv = 1)$y, 0)
# Renormalize f to integrate to 1 on [0, r_cap]
mass_from_f <- sum(f_smooth * diff(edges_km), na.rm=TRUE)
if (is.finite(mass_from_f) && mass_from_f > 0) f_smooth <- f_smooth / mass_from_f

# Interpolate g(r) to the same midpoints, and truncate plotting to r_cap
g_on_mid <- approx(x = df_g$r_km, y = df_g$g, xout = mid_km, rule = 2)$y
g_on_mid <- pmin(pmax(g_on_mid, 0), 1)

# Product mass per bin = f(r) * g(r) * Δr  (this sums to p̄ over [0, r_cap])
dr_km     <- diff(edges_km)                         # bin widths
prod_mass <- f_smooth * g_on_mid * dr_km
pbar_0_rCap <- sum(prod_mass, na.rm = TRUE)

message(sprintf("p̄ over [0, %.1f] km = %.5f", r_cap, pbar_0_rCap))

# ---------------- Plots ----------------

library(ggplot2)

# Overlay g(r) and f(r) (scaled to left axis) on [0, r_cap]
scale_factor <- max(g_on_mid, 1e-8) / max(f_smooth, 1e-8)
df_overlay <- tibble::tibble(
  r_km = mid_km,
  g    = g_on_mid,
  f    = f_smooth
)

ggplot(df_overlay, aes(r_km)) +
  geom_line(aes(y = g), linewidth = 1) +
  geom_line(aes(y = f * scale_factor), linetype = 2) +
  scale_x_continuous(limits = c(0, r_cap)) +
  scale_y_continuous(
    name = "g(r)  (probability of detection)",
    sec.axis = sec_axis(~ . / scale_factor,
                        name = "f(r)  (density, 1/km), depth ≤ 2000 m; renormalized on [0, 15] km")
  ) +
  labs(x = "Range (km)",
       title = "Overlay: g(r) and smooth f(r) on [0, 15] km",
       subtitle = "Solid = g(r); Dashed = f(r). Depth filter: cells with min depth ≤ 2000 m.") +
  theme_minimal(base_size = 13)

# Product f(r) * g(r) per bin (probability mass) on [0, r_cap]
df_prod <- tibble::tibble(
  r_km     = mid_km,
  mass_bin = prod_mass
)

ggplot(df_prod, aes(r_km, mass_bin)) +
  geom_col(width = bin_width) +
  scale_x_continuous(limits = c(0, r_cap)) +
  labs(x = "Range (km)",
       y = "f(r) × g(r)  (probability mass per bin)",
       title = "Product f(r) · g(r) on [0, 15] km",
       subtitle = sprintf("Bin width = %.2f km.  Sum over bins (p̄) = %.5f", bin_width, pbar_0_rCap)) +
  theme_minimal(base_size = 13)


# ---------------- Cap domain at 15 km and rebuild f(r) -----------------
r_cap <- 15      # km
bin_width <- 0.5 # km (tune if you like)

# Surface-cell range (km) and area (km^2) as before
cell_key     <- paste(RL$lon, RL$lat, sep=" ")
rows_by_cell <- split(seq_len(nrow(RL)), cell_key)
rep_row      <- vapply(rows_by_cell, function(ix) ix[1], integer(1))
r_cell_km    <- RL$range_m[rep_row] / 1000

# Per-cell area (km^2) as before
lon_u <- sort(unique(RL$lon)); lat_u <- sort(unique(RL$lat))
mid <- function(v) (head(v,-1) + tail(v,-1))/2
lon_edges <- c(lon_u[1] - (lon_u[2]-lon_u[1])/2, mid(lon_u),
               tail(lon_u,1) + (tail(lon_u,1) - lon_u[length(lon_u)-1])/2)
lat_edges <- c(lat_u[1] - (lat_u[2]-lat_u[1])/2, mid(lat_u),
               tail(lat_u,1) + (tail(lat_u,1) - lat_u[length(lat_u)-1])/2)
ii <- match(RL$lat[rep_row], lat_u); jj <- match(RL$lon[rep_row], lon_u)

area_mat <- matrix(NA_real_, nrow=length(lat_u), ncol=length(lon_u))
for (i in seq_along(lat_u)) for (j in seq_along(lon_u)) {
  poly <- matrix(c(lon_edges[j],   lat_edges[i],
                   lon_edges[j+1], lat_edges[i],
                   lon_edges[j+1], lat_edges[i+1],
                   lon_edges[j],   lat_edges[i+1],
                   lon_edges[j],   lat_edges[i]), ncol=2, byrow=TRUE)
  area_mat[i,j] <- geosphere::areaPolygon(poly) / 1e6  # km^2
}
area_cell <- area_mat[cbind(ii, jj)]

# Depth filter (keep cells with ANY layer ≤ 2000 m)
min_depth_cell <- vapply(rows_by_cell, function(ix) min(RL$depth_m[ix], na.rm=TRUE), numeric(1))
keep_depth <- (min_depth_cell <= 2000)

# Range cap filter and combined keep
keep_cap  <- (r_cell_km <= r_cap)
keep_cell <- keep_depth & keep_cap

r_cell_km <- r_cell_km[keep_cell]
area_cell <- area_cell[keep_cell]

# Area weights renormalized on [0, r_cap] with depth filter
w_cell <- area_cell / sum(area_cell, na.rm=TRUE)

# Regular bins on [0, r_cap]
edges_km <- seq(0, r_cap, by = bin_width)
if (tail(edges_km,1) < r_cap) edges_km <- c(edges_km, r_cap)  # ensure exact cap
mid_km   <- head(edges_km,-1) + diff(edges_km)/2

# Mass per bin (area-probability), then cumulative & smooth
bin_id <- cut(r_cell_km, breaks=edges_km, include.lowest=TRUE, right=TRUE)
mass_per_bin <- tapply(w_cell, bin_id, sum, na.rm=TRUE)
mass_per_bin <- ifelse(is.na(mass_per_bin), 0, mass_per_bin)
# Align in case of empty factor levels
if (length(mass_per_bin) != length(mid_km)) {
  tmp <- rep(0, length(mid_km)); names(tmp) <- levels(bin_id)
  tmp[names(mass_per_bin)] <- mass_per_bin
  mass_per_bin <- tmp
}
A_step <- cumsum(mass_per_bin)

# Smooth cumulative area then differentiate for a smooth, nonnegative f(r)
spl <- smooth.spline(x = mid_km, y = A_step, spar = 0.65)
A_smooth <- pmin(pmax(predict(spl, x = mid_km)$y, 0), 1)
f_smooth <- pmax(predict(spl, x = mid_km, deriv = 1)$y, 0)
# Renormalize f to integrate to 1 on [0, r_cap]
mass_from_f <- sum(f_smooth * diff(edges_km), na.rm=TRUE)
if (is.finite(mass_from_f) && mass_from_f > 0) f_smooth <- f_smooth / mass_from_f

# Interpolate g(r) to the same midpoints, and truncate plotting to r_cap
g_on_mid <- approx(x = df_g$r_km, y = df_g$g, xout = mid_km, rule = 2)$y
g_on_mid <- pmin(pmax(g_on_mid, 0), 1)

# Product mass per bin = f(r) * g(r) * Δr  (this sums to p̄ over [0, r_cap])
dr_km     <- diff(edges_km)                         # bin widths
prod_mass <- f_smooth * g_on_mid * dr_km
pbar_0_rCap <- sum(prod_mass, na.rm = TRUE)

message(sprintf("p̄ over [0, %.1f] km = %.5f", r_cap, pbar_0_rCap))

# ---------------- Plots ----------------

library(ggplot2)

# Overlay g(r) and f(r) (scaled to left axis) on [0, r_cap]
scale_factor <- max(g_on_mid, 1e-8) / max(f_smooth, 1e-8)
df_overlay <- tibble::tibble(
  r_km = mid_km,
  g    = g_on_mid,
  f    = f_smooth
)

ggplot(df_overlay, aes(r_km)) +
  geom_line(aes(y = g), linewidth = 1) +
  geom_line(aes(y = f * scale_factor), linetype = 2) +
  scale_x_continuous(limits = c(0, r_cap)) +
  scale_y_continuous(
    name = "g(r)  (probability of detection)",
    sec.axis = sec_axis(~ . / scale_factor,
                        name = "f(r)  (density, 1/km), depth ≤ 2000 m; renormalized on [0, 15] km")
  ) +
  labs(x = "Range (km)",
       title = "Overlay: g(r) and smooth f(r) on [0, 15] km",
       subtitle = "Solid = g(r); Dashed = f(r). Depth filter: cells with min depth ≤ 2000 m.") +
  theme_minimal(base_size = 13)

# Product f(r) * g(r) per bin (probability mass) on [0, r_cap]
df_prod <- tibble::tibble(
  r_km     = mid_km,
  mass_bin = prod_mass
)

ggplot(df_prod, aes(r_km, mass_bin)) +
  geom_col(width = bin_width) +
  scale_x_continuous(limits = c(0, r_cap)) +
  labs(x = "Range (km)",
       y = "f(r) × g(r)  (probability mass per bin)",
       title = "Product f(r) · g(r) on [0, 15] km",
       subtitle = sprintf("Bin width = %.2f km.  Sum over bins (p̄) = %.5f", bin_width, pbar_0_rCap)) +
  theme_minimal(base_size = 13)

#################################################################################
library(dplyr)
library(ggplot2)

# pick your base variable for detection probability
p_col <- if ("pbar_med" %in% names(summaries)) "pbar_med" else "pbar"

# Drop NAs and keep essentials
df <- summaries %>%
  select(dive, bottom, gdepth_m, pitch_key, depth_key, all_of(p_col)) %>%
  drop_na()

# Marginal mean ± range for each variable
marginal_depth  <- df %>% group_by(gdepth_m)       %>%
  summarise(mean_p = mean(.data[[p_col]]), range_p = diff(range(.data[[p_col]])))
marginal_bottom <- df %>% group_by(bottom)         %>%
  summarise(mean_p = mean(.data[[p_col]]), range_p = diff(range(.data[[p_col]])))
marginal_pitch  <- df %>% group_by(pitch_key)      %>%
  summarise(mean_p = mean(.data[[p_col]]), range_p = diff(range(.data[[p_col]])))
marginal_depthkey <- df %>% group_by(depth_key)    %>%
  summarise(mean_p = mean(.data[[p_col]]), range_p = diff(range(.data[[p_col]])))
marginal_dive   <- df %>% group_by(dive)           %>%
  summarise(mean_p = mean(.data[[p_col]]), range_p = diff(range(.data[[p_col]])))


# Stack all marginals
marginals <- bind_rows(
  mutate(marginal_depth,    variable="Glider depth",  level=as.character(gdepth_m)),
  mutate(marginal_bottom,   variable="Sediment",      level=bottom),
  mutate(marginal_pitch,    variable="Pitch",         level=pitch_key),
  mutate(marginal_depthkey, variable="Dive depth",    level=depth_key),
  mutate(marginal_dive,     variable="Location",      level=as.character(dive))
)

# Rank variables by overall range
effect_summary <- marginals %>%
  group_by(variable) %>%
  summarise(spread = max(mean_p) - min(mean_p)) %>%
  arrange(desc(spread))

ggplot(effect_summary, aes(x=reorder(variable, spread), y=spread)) +
  geom_col(fill="steelblue") +
  coord_flip() +
  labs(x=NULL, y="Δ mean p̄ (max – min)",
       title="Relative influence of factors on unconditional detection probability") +
  theme_minimal(base_size=13)

ggplot(df, aes(x=factor(gdepth_m), y=.data[[p_col]], fill=bottom)) +
  geom_boxplot(outlier.shape=NA) +
  facet_grid(rows = vars(pitch_key), cols = vars(depth_key)) +
  labs(x="Glider depth (m)", y="p̄ (unconditional detection probability)",
       title="Marginal effects of glider depth by sediment type and pitch/depth priors") +
  theme_minimal(base_size=13)


