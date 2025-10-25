# ============================================================
# PHASE 1 — TL-grid detection simulator (RMS, Zimmer-style beam)
# - Snap to TL grid
# - Depth-weighted Bernoulli to keep rows
# - Biologically-inspired composite beam (forward lobe + backward lobe + weak omni)
# - Add small angular jitter and per-click level scatter (≈ 3.5 dB)
# - Convert SL/RL p-p -> RMS via crest factor
# - Hard detector: SNR_rms >= threshold
# - Validation plots: beam patterns, detection function, plan view, θ vs DL, detections
# ============================================================

rm(list = ls())
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2); library(grid); library(tidyr)
})

# -------------------------- USER KNOBS ---------------------------------------

csv_path <- "C:\\Users\\kaity\\Documents\\SpaciousData\\CSVs\\bellhop_long\\PeakToPeak_dive_86_GliderDepth_200m_1_20khz_long.csv"

# Receiver
rx_depth_m <- 200

# TL/SL scale
SL_ref_pp_dB   <- 220          # p-p SL used to compute RL in CSV (so TL = 220 - RL_pp)

# p-p -> RMS conversion (in dB). 9.03 dB for a sinus; replace with measured crest factor if known.
crest_factor_dB <- 9.03

# Occurrence model (depth -> keep probability)
depth_max_m  <- 1200
depth_beta_a <- 10
depth_beta_b <- 3
KEEP_RATE    <- 1              # overall intensity (0..1)

# Source level distribution (draws in p-p dB; converted to RMS internally)
SL_mean_pp <- 220
SL_sd_dB   <- 6

# Detection (RMS domain)
noise_rms_dB <- 60
SNR_thr_dB   <- 5
noise_pp_dB  <- noise_rms_dB + crest_factor_dB

# Beam model (Zimmer-style composite)
# Forward (P1) one-sided HPBW ~ 10–12°, backward (P0) broader (~60°), LF weak omni.
hpbw_f_deg <- 11
hpbw_b_deg <- 60
w_f        <- 1.00             # relative power weight forward lobe
w_b        <- 0.12             # backward lobe ≈ -9 to -10 dB vs forward
w_lf       <- 0.01             # weak omni LF component
sd_scatter_dB <- 3.5           # per-click level scatter
sd_ang_jitter_deg <- 0.5       # tiny angular jitter to kill banding

# Visualization
SLICE_R_KM <- 5
N_PLOT_MAX <- 4000
BASE_SIZE  <- 13

# -------------------------- HELPERS ------------------------------------------

to_local_xy <- function(lat, lon, lat0, lon0) {
  m_per_deg_lat <- 111540
  m_per_deg_lon <- 111320 * cos((lat0 * pi / 180))
  x <- (lon - lon0) * m_per_deg_lon
  y <- (lat - lat0) * m_per_deg_lat
  cbind(x = x, y = y)
}

# ----- Biologically-inspired beam: von-Mises(-Fisher)-like lobes -----

vmf_lobe_power <- function(theta, kappa) {
  # Axisymmetric lobe with power ∝ exp(κ cos θ); normalize to 1 at θ=0
  exp(kappa * cos(theta)) / exp(kappa)
}

kappa_from_hpbw_deg <- function(hpbw_deg) {
  # Half-power (~ -3 dB) at one-sided angle hpbw_deg
  th <- hpbw_deg * pi/180
  log(2) / (1 - cos(th))
}

gain_linear_sw <- function(theta,
                           hpbw_f_deg = 11,
                           hpbw_b_deg = 60,
                           w_f = 1.0,
                           w_b = 0.12,
                           w_lf = 0.01) {
  kf <- kappa_from_hpbw_deg(hpbw_f_deg)
  kb <- kappa_from_hpbw_deg(hpbw_b_deg)
  Gf <- vmf_lobe_power(theta, kf)          # forward (center 0)
  Gb <- vmf_lobe_power(pi - theta, kb)     # backward (center π)
  Glf <- 1                                 # omni
  G  <- w_f*Gf + w_b*Gb + w_lf*Glf         # linear power gain
  pmax(G, 1e-12)
}

gain_dB_sw <- function(theta, ...) 10*log10(gain_linear_sw(theta, ...))

directivity_index_dB_generic <- function(Gfun, ..., n = 4001) {
  th <- seq(0, pi, length.out = n)
  G  <- Gfun(th, ...)
  I  <- sum(G * sin(th)) * (th[2] - th[1])  # ∫_0^π G(θ) sinθ dθ (axisym.)
  10*log10(2 / I)                            # DI = 10log10(4π / ∫ G dΩ) with symmetry
}

# -------------------------- LOAD + PREP --------------------------------------

df <- read_csv(csv_path, show_col_types = FALSE) %>%
  select(lat, lon, drifterlat, drifterlon, range, depth_m, RL) %>%
  mutate(
    TL_dB      = SL_ref_pp_dB - RL,       # TL (dB), metric-agnostic if SL/RL match
    depth_norm = pmin(pmax(depth_m / depth_max_m, 0), 1),
    depth_pdf  = dbeta(depth_norm, depth_beta_a, depth_beta_b)
  )
stopifnot(all(c("lat","lon","drifterlat","drifterlon","range","depth_m","TL_dB") %in% names(df)))

# Depth-weighted Bernoulli keep
w <- df$depth_pdf; w <- w / max(w, na.rm = TRUE)
p_keep <- pmin(pmax(KEEP_RATE * w, 0), 0.99)

set.seed(1)
keep <- as.logical(rbinom(nrow(df), 1, p_keep))

df_keep <- df[keep, , drop = FALSE]
if (nrow(df_keep) == 0) stop("No rows kept; increase KEEP_RATE.")
cat(sprintf("Kept rows: %d / %d (KEEP_RATE=%.2f)\n", nrow(df_keep), nrow(df), KEEP_RATE))

# -------------------------- GEOMETRY -----------------------------------------

xy_src <- to_local_xy(df_keep$lat, df_keep$lon, df_keep$drifterlat, df_keep$drifterlon)
xy_rx  <- to_local_xy(df_keep$drifterlat, df_keep$drifterlon, df_keep$drifterlat, df_keep$drifterlon)
x <- xy_src[,"x"]; y <- xy_src[,"y"]; x_rx <- xy_rx[,"x"]; y_rx <- xy_rx[,"y"]
stopifnot(all(abs(x_rx) < 1e-8), all(abs(y_rx) < 1e-8))

r_h     <- sqrt((x - x_rx)^2 + (y - y_rx)^2)
z_src   <- df_keep$depth_m
z_rx    <- rx_depth_m
r_slant <- df_keep$range

vx <- (x_rx - x); vy <- (y_rx - y); vz <- (z_rx - z_src)
r_eps <- pmax(r_slant, 1.0)
ux <- vx / r_eps; uy <- vy / r_eps; uz <- vz / r_eps

# ---------------------- SOURCE ORIENTATION -----------------------------------

set.seed(2)
yaw       <- runif(nrow(df_keep), 0, 2*pi)
pitch_deg <- qbeta(runif(nrow(df_keep)), 11, 7) * 180
pitch_rad <- pitch_deg * pi/180

bx <- sin(pitch_rad) * cos(yaw)
by <- sin(pitch_rad) * sin(yaw)
bz <- -cos(pitch_rad)

dot <- bx*ux + by*uy + bz*uz
dot <- pmin(pmax(dot, -1), 1)
theta_off_raw <- acos(dot)

# Small angular jitter to kill discretization banding
set.seed(21)
theta_off <- pmin(pi, pmax(0, theta_off_raw + rnorm(length(theta_off_raw), 0, sd_ang_jitter_deg*pi/180)))

# ---------------------- DIRECTIVITY + DETECTION ------------------------------

# Composite beam (Zimmer-style)
G_dB  <- gain_dB_sw(theta_off,
                    hpbw_f_deg = hpbw_f_deg, hpbw_b_deg = hpbw_b_deg,
                    w_f = w_f, w_b = w_b, w_lf = w_lf)

# Add per-click level scatter (angle-independent)
set.seed(42)
G_dB  <- G_dB + rnorm(length(G_dB), mean = 0, sd = sd_scatter_dB)

# Levels in RMS domain (consistent with noise_rms_dB)
set.seed(3)
SL_draw_pp  <- rnorm(nrow(df_keep), mean = SL_mean_pp, sd = SL_sd_dB)
SL_draw_rms <- SL_draw_pp - crest_factor_dB

TL_vals <- df_keep$TL_dB
RL_rms  <- SL_draw_rms + G_dB - TL_vals
SNR_rms <- RL_rms - noise_rms_dB
det     <- (SNR_rms >= SNR_thr_dB)

# ---------------------- VALIDATION: BEAM & DI --------------------------------

# Beam pattern (dB) vs angle
th_plot <- seq(0, pi, length.out = 721)
beam_df <- tibble(
  theta_deg = th_plot * 180/pi,
  G_dB = gain_dB_sw(th_plot,
                    hpbw_f_deg = hpbw_f_deg, hpbw_b_deg = hpbw_b_deg,
                    w_f = w_f, w_b = w_b, w_lf = w_lf)
)
DI_chk <- directivity_index_dB_generic(
  gain_linear_sw,
  hpbw_f_deg = hpbw_f_deg, hpbw_b_deg = hpbw_b_deg,
  w_f = w_f, w_b = w_b, w_lf = w_lf
)
cat(sprintf("Zimmer-style composite beam DI ≈ %.1f dB\n", DI_chk))

p_beam <- ggplot(beam_df, aes(theta_deg, G_dB)) +
  geom_line() +
  geom_vline(xintercept = c(hpbw_f_deg), linetype = 3) +
  scale_x_continuous(breaks = seq(0,180,20), limits = c(0,180)) +
  labs(x = "Off-axis angle (deg)", y = "Gain (dB)",
       title = "Composite beam pattern (Zimmer-style)",
       subtitle = sprintf("HPBW_f ≈ %g°, HPBW_b ≈ %g°, w_b=%g, w_lf=%g, DI≈%.1f dB",
                          hpbw_f_deg, hpbw_b_deg, w_b, w_lf, DI_chk)) +
  theme_minimal(base_size = BASE_SIZE)
print(p_beam)

# ---------------------- DETECTION FUNCTION -----------------------------------

n_bins <- 60
R_horiz <- max(r_h, na.rm = TRUE)
cuts <- seq(0, R_horiz, length.out = n_bins + 1)
mid  <- head(cuts, -1) + diff(cuts)/2
bin  <- cut(r_h, cuts, include.lowest = TRUE)

detfun <- tibble(r_h = r_h, det = det, bin = bin) %>%
  group_by(bin) %>%
  summarise(r_mid = mid[as.integer(first(bin))],
            P_detect = mean(det),
            .groups = "drop")

delta_r <- c(detfun$r_mid[1], diff(detfun$r_mid))
pbar <- (2 / R_horiz^2) * sum(detfun$P_detect * detfun$r_mid * delta_r)
miss_rate <- 1 - pbar
cat(sprintf("Area-weighted detection probability: %.3f | Miss rate: %.3f\n", pbar, miss_rate))

p_detfun <- ggplot(detfun, aes(x = r_mid/1000, y = P_detect)) +
  geom_line() + geom_point(size = 1) +
  labs(x = "Horizontal range (km)", y = "Detection probability",
       title = "Detection function (grid-snapped, RMS, Zimmer-style beam)",
       subtitle = sprintf("Noise(RMS)=%g dB, SNR_thr=%g dB", noise_rms_dB, SNR_thr_dB)) +
  theme_minimal(base_size = BASE_SIZE)
print(p_detfun)

# ------------------------------ PLOTS ----------------------------------------

viz <- tibble(
  x = x, y = y, r_h = r_h,
  z_src = z_src,
  yaw = yaw,
  pitch_deg = pitch_deg,
  theta_off = theta_off,
  DL_dB = -G_dB,
  det = det
)

# Horizontal projection unit arrows
bx_h <- sin(pitch_rad) * cos(yaw)
by_h <- sin(pitch_rad) * sin(yaw)
norm_h <- sqrt(bx_h^2 + by_h^2)
viz <- viz %>%
  mutate(
    bx_hu = ifelse(norm_h > 1e-6, bx_h / norm_h, NA_real_),
    by_hu = ifelse(norm_h > 1e-6, by_h / norm_h, NA_real_)
  )

# (1) Plan-view slice with arrows colored by DL
slice <- viz %>% filter(r_h <= SLICE_R_KM * 1000)
if (nrow(slice) > N_PLOT_MAX) slice <- dplyr::slice_sample(slice, n = N_PLOT_MAX)
ARROW_FRACTION <- 0.07
arrow_len <- ARROW_FRACTION * SLICE_R_KM * 1000
slice <- slice %>% mutate(xend = x + arrow_len * bx_hu, yend = y + arrow_len * by_hu)

p_slice <- ggplot(slice, aes(x = x/1000, y = y/1000)) +
  annotate("point", x = 0, y = 0, shape = 8, size = 4) +
  annotate("text", x = 0, y = 0, label = "Receiver", vjust = -1.1, size = 3) +
  geom_segment(aes(xend = xend/1000, yend = yend/1000, color = DL_dB),
               arrow = arrow(length = unit(2, "mm"), ends = "last"),
               linewidth = 0.4, na.rm = TRUE) +
  geom_point(aes(color = DL_dB), size = 0.6, alpha = 0.6) +
  scale_color_viridis_c(option = "C", direction = 1, name = "Directivity loss (dB)") +
  coord_fixed() +
  labs(x = "x (km)", y = "y (km)",
       title = sprintf("Beam orientation ≤ %.1f km (Zimmer-style), colored by DL", SLICE_R_KM)) +
  theme_minimal(base_size = BASE_SIZE)
print(p_slice)

# (2) DL vs off-axis angle
p_theta <- ggplot(viz, aes(x = theta_off * 180/pi, y = DL_dB)) +
  geom_bin2d(bins = 80) +
  scale_fill_viridis_c(option = "C", name = "Count") +
  labs(x = "Off-axis angle (deg)", y = "Directivity loss (dB)",
       title = "Directivity loss vs off-axis angle (Zimmer-style beam)") +
  theme_minimal(base_size = BASE_SIZE)
print(p_theta)

# (3) Detections near receiver
p_slice_det <- ggplot(slice, aes(x = x/1000, y = y/1000)) +
  annotate("point", x = 0, y = 0, shape = 8, size = 4) +
  geom_point(aes(color = det), alpha = 0.75, size = 0.9) +
  scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "black"), name = "Detected") +
  coord_fixed() +
  labs(x = "x (km)", y = "y (km)",
       title = sprintf("Detections ≤ %.1f km (Zimmer-style)", SLICE_R_KM)) +
  theme_minimal(base_size = BASE_SIZE)
print(p_slice_det)
