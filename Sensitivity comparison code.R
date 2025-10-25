# ==============================
# Sensitivity vs Threshold (R) Ray Tracing
# ==============================

library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(ggplot2)

# Folder with CSVs
data_dir <- "C:/Users/kaity/Documents/SpaciousData/CSVs"

# Thresholds to evaluate (x-axis)
threshold_seq <- seq(125, 175, by = 5)   # adjust as needed

# List all PeakToPeak CSVs
files <- list.files(data_dir, pattern = "^PeakToPeak_Bellhop.*\\.csv$", full.names = TRUE)

stopifnot(length(files) > 0)

# Helper to parse metadata from filename
parse_meta <- function(path) {
  fname <- basename(path)
  loc_id <- str_match(fname, "dive_(\\d+)")[, 2]
  depth  <- str_match(fname, "GliderDepth_(\\d+)m")[, 2]
  tibble(Location = loc_id, Depth_m = as.numeric(depth))
}

# Read a CSV to a numeric matrix (no headers assumed)
read_grid <- function(path) {
  # If your CSVs sometimes include row/col names, change col_names accordingly
  as.matrix(read_csv(path, col_names = FALSE, show_col_types = FALSE))
}

# Count cells > threshold
count_above <- function(mat, thr) {
  sum(mat > thr, na.rm = TRUE)  # strictly greater-than per your spec
}

# Build results: one row per (file, threshold)
results_bellhop <- lapply(files, function(f) {
  meta <- parse_meta(f)
  grid <- read_grid(f)
  
  # filter the depths between 300 and 1500
  depths = 100*(1:ncol(grid))
  grid[,which(depths>1500)] =NaN
  grid[,which(depths<300)] =NaN
  
  # Ensure numeric
  storage.mode(grid) <- "numeric"
  
  tibble(
    Threshold = threshold_seq,
    CountAbove = vapply(threshold_seq, function(t) count_above(grid, t), numeric(1))
  ) %>%
    bind_cols(meta)
}) %>% bind_rows()

library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(purrr)
library(tibble)

# ---- Helpers ----
read_grid <- function(path) as.matrix(read_csv(path, col_names = FALSE, show_col_types = FALSE))

parse_meta <- function(path) {
  fname <- basename(path)
  tibble(
    File     = fname,
    Model    = str_match(fname, "^PeakToPeak_([A-Za-z]+)")[,2],
    Location = str_match(fname, "dive_(\\d+)")[,2],
    Depth_m  = as.numeric(str_match(fname, "GliderDepth_(\\d+)m")[,2]),
    # Optional: parse band "…_1_10khz.csv" -> kHz → Hz
    f_low_hz  = as.numeric(str_match(fname, "_(\\d+)_([0-9]+)khz\\.csv$")[,2]) * 1000,
    f_high_hz = as.numeric(str_match(fname, "_(\\d+)_([0-9]+)khz\\.csv$")[,3]) * 1000
  )
}

# Convert peak-to-peak dB (re 1 µPa p-p) to RMS dB (re 1 µPa rms)
# For a sine wave, crest_db = 20*log10(2*sqrt(2)) ≈ 9.03 dB
pp_to_rms_db <- function(pp_db, crest_db = 9.03) pp_db - crest_db

# ---- Main function ----
snr_volume_from_dir <- function(
    data_dir,
    noise_csv_path,
    snr_thresholds = c(20),    # SNR thresholds in dB
    crest_db      = 9.03,      # p-p → RMS conversion (dB). Change if your pulse crest differs.
    cell_area_km2 = 0.152,     # surface area per grid cell (km^2)
    slice_thick_km = 0.1       # depth slice thickness (km)
) {
  stopifnot(dir.exists(data_dir), file.exists(noise_csv_path))
  
  # --- Load noise (band-integrated RMS dB) ---
  NL <- read.csv(noise_csv_path, check.names = FALSE)
  # Try common time column names; keep as POSIXct if present, else use row index
  time_col <- intersect(names(NL), c("time", "Time", "DateTime", "datetime", "timestamp"))
  time_vec <- if (length(time_col) > 0) as.POSIXct(NL[[time_col[1]]], tz = "UTC") else seq_len(nrow(NL))
  if (!("L_rms_db" %in% names(NL))) stop("Noise CSV must contain column 'L_rms_db' (band-integrated RMS dB).")
  nl_rms_db <- as.numeric(NL$L_rms_db)
  
  # File list
  files <- list.files(data_dir, pattern = "^PeakToPeak_(Bellhop|Spherical).*\\.csv$", full.names = TRUE)
  if (length(files) == 0) stop("No PeakToPeak_* CSVs found in ", data_dir)
  
  cell_vol_km3 <- cell_area_km2 * slice_thick_km
  
  # Process each file → results tibble
  results <- map_dfr(files, function(f) {
    meta <- parse_meta(f)
    
    # Read grid (dB peak-to-peak), convert to RMS dB once
    grid_pp <- read_grid(f)
    storage.mode(grid_pp) <- "numeric"
    grid_rms_db <- pp_to_rms_db(grid_pp, crest_db = crest_db)
    
    # For speed, flatten grid to a vector once (same for all timesteps)
    grid_vals <- as.numeric(grid_rms_db)
    
    # For each timestep, SNR > thr  <=>  grid_rms_db > nl_rms_db[t] + thr
    # We’ll compute counts for every threshold and timestep.
    counts_by_thr <- lapply(snr_thresholds, function(thr) {
      # Required RL_rms at each timestep
      req_rms <- nl_rms_db + thr  # vector length = nrow(NL)
      # Count cells > requirement for each timestep (loop over time; grid vectorized)
      cnt <- vapply(req_rms, function(req) sum(grid_vals > req, na.rm = TRUE), numeric(1))
      tibble(SNR_Thr = thr, CountAbove = cnt)
    }) %>% bind_rows()
    
    # Repeat/attach the time vector
    nT <- length(nl_rms_db)
    out <- counts_by_thr %>%
      group_by(SNR_Thr) %>%
      mutate(time = time_vec) %>%
      ungroup() %>%
      mutate(
        AreaKm3 = CountAbove * cell_vol_km3,
        Model    = meta$Model,
        Location = meta$Location,
        Depth_m  = meta$Depth_m,
        File     = meta$File
      ) %>%
      relocate(Model, Location, Depth_m, SNR_Thr, time, CountAbove, AreaKm3, File)
    
    # Optional: keep band info in output (if parsed)
    if (!is.na(meta$f_low_hz) && !is.na(meta$f_high_hz)) {
      out <- out %>% mutate(f_low_hz = meta$f_low_hz, f_high_hz = meta$f_high_hz)
    }
    out
  })
  
  results
}

snr_volume_from_dir <- function(
    data_dir,
    noise_csv_path,
    snr_thresholds = 20,     # vector OK
    crest_db = 9.03,         # p-p → RMS (dB); sine ≈ 9.03
    cell_area_km2 = 0.152,
    slice_thick_km = 0.1
) {
  stopifnot(dir.exists(data_dir), file.exists(noise_csv_path))
  cell_vol_km3 <- cell_area_km2 * slice_thick_km
  
  # Noise: parse time "YYYY-mm-dd HH:MM:SS UTC" and get L_rms_db
  NL <- read.csv(noise_csv_path, check.names = FALSE, stringsAsFactors = FALSE)
  tcol <- intersect(names(NL), c("time","Time","DateTime","datetime","timestamp"))[1]
  if (is.na(tcol)) stop("Noise CSV needs a time column (time/Time/DateTime/datetime/timestamp).")
  time_vec <- as.POSIXct(as.character(NL[[tcol]]), format = "%Y-%m-%d %H:%M:%S %Z", tz = "UTC")
  ok <- is.finite(as.numeric(time_vec))
  NL <- NL[ok, , drop = FALSE]; time_vec <- time_vec[ok]
  if (!"L_rms_db" %in% names(NL)) stop("Noise CSV must contain 'L_rms_db'.")
  nl <- as.numeric(NL$L_rms_db)
  ord <- order(time_vec); time_vec <- time_vec[ord]; nl <- nl[ord]
  
  files <- list.files(data_dir, pattern="^PeakToPeak_(Bellhop|Spherical).*\\.csv$", full.names=TRUE)
  if (!length(files)) stop("No PeakToPeak_* CSVs found.")
  
  out_list <- vector("list", length(files))
  for (i in seq_along(files)) {
    f <- files[i]; fn <- basename(f)
    model <- sub("^PeakToPeak_([A-Za-z]+).*", "\\1", fn)
    loc   <- sub(".*dive_(\\d+).*", "\\1", fn)
    depth <- as.numeric(sub(".*GliderDepth_(\\d+)m.*", "\\1", fn))
    
    # Read grid (dB p-p) → RMS dB
    grid_pp  <- as.matrix(read.csv(f, header = FALSE, check.names = FALSE))
    storage.mode(grid_pp) <- "numeric"
    grid_rms <- grid_pp - crest_db
    gv <- as.numeric(grid_rms)
    
    # Counts for each SNR threshold over all timesteps
    res_thr <- lapply(snr_thresholds, function(thr) {
      req <- nl + thr
      cnt <- vapply(req, function(r) sum(gv > r, na.rm = TRUE), numeric(1))
      data.frame(SNR_Thr = thr, time = time_vec, CountAbove = cnt)
    })
    res <- do.call(rbind, res_thr)
    res$AreaKm3  <- res$CountAbove * cell_vol_km3
    res$Model    <- model
    res$Location <- loc
    res$Depth_m  <- depth
    res$File     <- fn
    out_list[[i]] <- res
  }
  
  do.call(rbind, out_list)
}


# Clean up factor ordering (optional)
results_bellhop <- results_bellhop %>%
  mutate(
    Location = factor(Location, levels = sort(unique(Location))),
    Depth_m  = factor(Depth_m, levels = sort(unique(Depth_m)))
  )

results_bellhop$AreaKm3 = results_bellhop$CountAbove*0.152*(.1) # Approximate surface area *100m 

# ---- Plot: x = threshold, y = counts
# Lines colored by Location; point shapes by Depth; lines grouped by (Location, Depth)
p <- ggplot(results_bellhop,
            aes(x = Threshold, y = AreaKm3,
                color = Location, group = interaction(Location, Depth_m))) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = Depth_m), size = 2) +
  scale_y_log10()+
  labs(
    title = "Sensitivity of Monitored Area vs Threshold",
    x = "Threshold (dB, peak-to-peak)",
    y = "Volume Montiored (km^3)",
    color = "Location",
    shape = "Glider depth (m)"
  ) +
  theme_minimal(base_size = 12)

print(p)


p <- ggplot(subset(results_bellhop, Location =='86'),
            aes(x = Threshold, y = CountAbove,
                color = Location, group = interaction(Location, Depth_m))) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = Depth_m), size = 2) +
  scale_y_log10()+
  labs(
    title = "Sensitivity of Monitored Area vs Threshold",
    x = "Threshold (dB, peak-to-peak)",
    y = "Count of cells > threshold",
    color = "Location",
    shape = "Glider depth (m)"
  ) +
  theme_minimal(base_size = 12)

print(p)


library(scales)

# Data range (positive only for log scale)
yr <- subset(results_bellhop, Location == "86")$CountAbove
yr <- yr[is.finite(yr) & yr > 0]
emin <- floor(log10(min(yr)))
emax <- ceiling(log10(max(yr)))

major_breaks <- 10^(emin:emax)
minor_breaks <- as.numeric(outer(2:9, 10^(emin:emax), `*`))

p <- ggplot(results_bellhop,
            aes(x = Threshold, y = AreaKm3,
                color = Location, group = interaction(Location, Depth_m))) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = Depth_m), size = 2) +
  scale_y_log10(
    breaks = major_breaks,
    minor_breaks = minor_breaks,
    labels = label_number(big.mark = ",")  # or label_scientific()
  ) +
  annotation_logticks(sides = "l") +       # left-side log tick marks
  labs(
    title = "Sensitivity of Monitored Area vs Threshold",
    x = "Threshold (dB, peak-to-peak)",
    y = "Volume Monitored (km3)",
    color = "Location",
    shape = "Glider depth (m)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor.y = element_blank())  # optional: hide minor gridlines

print(p)

p <- ggplot(results_bellhop,
            aes(x = Threshold, y = AreaKm3,
                color = Location, 
                group = interaction(Location, Depth_m))) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = Depth_m), size = 2) +
  annotation_logticks(sides = "l") +       # left-side log tick marks
  labs(
    title = "Sensitivity of Monitored Area vs Threshold",
    x = "Threshold (dB, peak-to-peak)",
    y = "Volume Monitored (km3)",
    color = "Location",
    shape = "Glider depth (m)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor.y = element_blank())  # optional: hide minor gridlines

print(p)



# ----- Optional: export summarized table for inspection
# Wide-by-depth counts at a few example thresholds (e.g., 120, 140, 160 dB)
example_thr <- c(120, 140, 160)
summary_tbl <- results_bellhop %>%
  filter(Threshold %in% example_thr) %>%
  arrange(Location, Depth_m, Threshold) %>%
  select(Location, Depth_m, Threshold, CountAbove) %>%
  tidyr::pivot_wider(names_from = Threshold, values_from = CountAbove,
                     names_prefix = "thr_")

print(summary_tbl)




#%% Do THE SAME FOR THE SPERHICAL SPREADING


# ==============================
# Sensitivity vs Threshold (R) Spherical Spreading
# ==============================

library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(ggplot2)

# Folder with CSVs
data_dir <- "C:/Users/kaity/Documents/SpaciousData/CSVs"

# Thresholds to evaluate (x-axis)
threshold_seq <- seq(100, 175, by = 5)   # adjust as needed

# List all PeakToPeak CSVs
files <- list.files(data_dir, pattern = "^PeakToPeak_Spherical.*\\.csv$", full.names = TRUE)

stopifnot(length(files) > 0)

# Helper to parse metadata from filename
parse_meta <- function(path) {
  fname <- basename(path)
  loc_id <- str_match(fname, "dive_(\\d+)")[, 2]
  depth  <- str_match(fname, "GliderDepth_(\\d+)m")[, 2]
  tibble(Location = loc_id, Depth_m = as.numeric(depth))
}

# Read a CSV to a numeric matrix (no headers assumed)
read_grid <- function(path) {
  # If your CSVs sometimes include row/col names, change col_names accordingly
  as.matrix(read_csv(path, col_names = FALSE, show_col_types = FALSE))
}

# Count cells > threshold
count_above <- function(mat, thr) {
  sum(mat > thr, na.rm = TRUE)  # strictly greater-than per your spec
}

# Build results: one row per (file, threshold)
results_spherical <- lapply(files, function(f) {
  meta <- parse_meta(f)
  grid <- read_grid(f)
  
  
  # filter the depths between 300 and 1500
  depths = 100*(1:ncol(grid))
  grid[,which(depths>1500)] =NaN
  grid[,which(depths<300)] =NaN
  
  
  # Ensure numeric
  storage.mode(grid) <- "numeric"
  
  tibble(
    Threshold = threshold_seq,
    CountAbove = vapply(threshold_seq, function(t) count_above(grid, t), numeric(1))
  ) %>%
    bind_cols(meta)
}) %>% bind_rows()

# Clean up factor ordering (optional)
results_spherical <- results_spherical %>%
  mutate(
    Location = factor(Location, levels = sort(unique(Location))),
    Depth_m  = factor(Depth_m, levels = sort(unique(Depth_m)))
  )

results_spherical$AreaKm3 = results_spherical$CountAbove*0.152*(.1) # Approximate surface area *100m 

# ---- Plot: x = threshold, y = counts
# Lines colored by Location; point shapes by Depth; lines grouped by (Location, Depth)
p <- ggplot(results_spherical,
            aes(x = Threshold, y = AreaKm3,
                color = Location, group = interaction(Location, Depth_m))) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = Depth_m), size = 2) +
  scale_y_log10()+
  labs(
    title = "Sensitivity of Monitored Area vs Threshold",
    x = "Threshold (dB, peak-to-peak)",
    y = "Volume Montiored (km^3)",
    color = "Location",
    shape = "Glider depth (m)"
  ) +
  theme_minimal(base_size = 12)

print(p)


p <- ggplot(subset(results_spherical, Location =='86'),
            aes(x = Threshold, y = CountAbove,
                color = Location, group = interaction(Location, Depth_m))) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = Depth_m), size = 2) +
  scale_y_log10()+
  labs(
    title = "Sensitivity of Monitored Area vs Threshold",
    x = "Threshold (dB, peak-to-peak)",
    y = "Count of cells > threshold",
    color = "Location",
    shape = "Glider depth (m)"
  ) +
  theme_minimal(base_size = 12)

print(p)


library(scales)

# Data range (positive only for log scale)
yr <- subset(results_spherical)$CountAbove
yr <- yr[is.finite(yr) & yr > 0]
emin <- floor(log10(min(yr)))
emax <- ceiling(log10(max(yr)))

major_breaks <- 10^(emin:emax)
minor_breaks <- as.numeric(outer(2:9, 10^(emin:emax), `*`))

p <- ggplot(results_spherical,
            aes(x = Threshold, y = AreaKm3,
                color = Location, group = interaction(Location, Depth_m))) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = Depth_m), size = 2) +
  scale_y_log10(
    breaks = major_breaks,
    minor_breaks = minor_breaks,
    labels = label_number(big.mark = ",")  # or label_scientific()
  ) +
  annotation_logticks(sides = "l") +       # left-side log tick marks
  labs(
    title = "Sensitivity of Monitored Area vs Threshold",
    x = "Threshold (dB, peak-to-peak)",
    y = "Volume Monitored (km3)",
    color = "Location",
    shape = "Glider depth (m)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor.y = element_blank())  # optional: hide minor gridlines

print(p)

p <- ggplot(results_spherical,
            aes(x = Threshold, y = AreaKm3,
                color = Location, 
                group = interaction(Location, Depth_m))) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = Depth_m), size = 2) +
  annotation_logticks(sides = "l") +       # left-side log tick marks
  labs(
    title = "Sensitivity of Monitored Area vs Threshold",
    x = "Threshold (dB, peak-to-peak)",
    y = "Volume Monitored (km3)",
    color = "Location",
    shape = "Glider depth (m)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor.y = element_blank())  # optional: hide minor gridlines

print(p)



# ----- Optional: export summarized table for inspection
# Wide-by-depth counts at a few example thresholds (e.g., 120, 140, 160 dB)
example_thr <- c(120, 140, 160)
summary_tbl <- results_spherical %>%
  filter(Threshold %in% example_thr) %>%
  arrange(Location, Depth_m, Threshold) %>%
  select(Location, Depth_m, Threshold, CountAbove) %>%
  tidyr::pivot_wider(names_from = Threshold, values_from = CountAbove,
                     names_prefix = "thr_")

print(summary_tbl)

library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(purrr)
library(tibble)

# ---- Helpers ----
read_grid <- function(path) as.matrix(read_csv(path,
                                               col_names = FALSE, 
                                               show_col_types = FALSE))

parse_meta <- function(path) {
  fname <- basename(path)
  tibble(
    File     = fname,
    Model    = str_match(fname, "^PeakToPeak_([A-Za-z]+)")[,2],
    Location = str_match(fname, "dive_(\\d+)")[,2],
    Depth_m  = as.numeric(str_match(fname, "GliderDepth_(\\d+)m")[,2]),
    # Optional: parse band "…_1_10khz.csv" -> kHz → Hz
    f_low_hz  = as.numeric(str_match(fname, "_(\\d+)_([0-9]+)khz\\.csv$")[,2]) * 1000,
    f_high_hz = as.numeric(str_match(fname, "_(\\d+)_([0-9]+)khz\\.csv$")[,3]) * 1000
  )
}

# Convert peak-to-peak dB (re 1 µPa p-p) to RMS dB (re 1 µPa rms)
# For a sine wave, crest_db = 20*log10(2*sqrt(2)) ≈ 9.03 dB
pp_to_rms_db <- function(pp_db, crest_db = 9.03) pp_db - crest_db

# ---- Main function ----
snr_volume_from_dir <- function(
    data_dir,
    noise_csv_path,
    snr_thresholds = c(20),    # SNR thresholds in dB
    crest_db      = 9.03,      # p-p → RMS conversion (dB). Change if your pulse crest differs.
    cell_area_km2 = 0.152,     # surface area per grid cell (km^2)
    slice_thick_km = 0.1       # depth slice thickness (km)
) {
  stopifnot(dir.exists(data_dir), file.exists(noise_csv_path))
  
  # --- Load noise (band-integrated RMS dB) ---
  NL <- read.csv(noise_csv_path, check.names = FALSE)
  
  # Try common time column names; keep as POSIXct if present, else use row index
  time_col <- intersect(names(NL), c("time", "Time", "DateTime", "datetime", "timestamp"))
  time_vec <- if (length(time_col) > 0) as.POSIXct(NL[[time_col[1]]], tz = "UTC") else seq_len(nrow(NL))
  if (!("L_rms_db" %in% names(NL))) stop("Noise CSV must contain column 'L_rms_db' (band-integrated RMS dB).")
  nl_rms_db <- as.numeric(NL$L_rms_db)
  
  # File list
  files <- list.files(data_dir, pattern = "^PeakToPeak_(Bellhop|Spherical).*\\.csv$", full.names = TRUE)
  if (length(files) == 0) stop("No PeakToPeak_* CSVs found in ", data_dir)
  
  cell_vol_km3 <- cell_area_km2 * slice_thick_km
  
  # Process each file → results tibble
  results <- map_dfr(files, function(f) {
    meta <- parse_meta(f)
    
    # Read grid (dB peak-to-peak), convert to RMS dB once
    grid_pp <- read_grid(f)
    storage.mode(grid_pp) <- "numeric"
    grid_rms_db <- pp_to_rms_db(grid_pp, crest_db = crest_db)
    
    # For speed, flatten grid to a vector once (same for all timesteps)
    grid_vals <- as.numeric(grid_rms_db)
    
    # For each timestep, SNR > thr  <=>  grid_rms_db > nl_rms_db[t] + thr
    # We’ll compute counts for every threshold and timestep.
    counts_by_thr <- lapply(snr_thresholds, function(thr) {
      # Required RL_rms at each timestep
      req_rms <- nl_rms_db + thr  # vector length = nrow(NL)
      # Count cells > requirement for each timestep (loop over time; grid vectorized)
      cnt <- vapply(req_rms, function(req) sum(grid_vals > req, na.rm = TRUE), numeric(1))
      tibble(SNR_Thr = thr, CountAbove = cnt)
    }) %>% bind_rows()
    
    # Repeat/attach the time vector
    nT <- length(nl_rms_db)
    out <- counts_by_thr %>%
      group_by(SNR_Thr) %>%
      mutate(time = time_vec) %>%
      ungroup() %>%
      mutate(
        AreaKm3 = CountAbove * cell_vol_km3,
        Model    = meta$Model,
        Location = meta$Location,
        Depth_m  = meta$Depth_m,
        File     = meta$File
      ) %>%
      relocate(Model, Location, Depth_m, SNR_Thr, time, CountAbove, AreaKm3, File)
    
    # Trim the depth from 400-1500 m
    #out = out[out$Depth_m<1500 & out$Depth_m>400,]= NaN
    
    # Optional: keep band info in output (if parsed)
    if (!is.na(meta$f_low_hz) && !is.na(meta$f_high_hz)) {
      out <- out %>% mutate(f_low_hz = meta$f_low_hz, f_high_hz = meta$f_high_hz)
    }
    out
  })
  
  results
}


# ==============================
# Combine results for plotting
# ==============================

results_spherical$Model = as.factor('Spherical')
results_bellhop$Model = as.factor('Bellhop')

results_all = rbind(results_spherical, results_bellhop)



theme(panel.grid.minor.y = element_blank())  # optional: hide minor gridlines
library(dplyr)
results_all <- results_all %>%
  mutate(
    Threshold = as.numeric(Threshold)  # ensure numeric
  )

p <- ggplot(
  results_all,
  aes(x = Threshold,
      y = AreaKm3,
      color = Model,
      group = interaction(Model, Depth_m))   # <-- no Threshold here
) +
  facet_grid(~ Location) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = factor(Depth_m)), size = 2) +
  scale_shape_discrete(name = "Glider depth (m)") +
  scale_color_brewer(palette = "Set2", name = "Model") +
  scale_y_log10() +                            # if you want log scale
  annotation_logticks(sides = "l") +
  labs(
    title = "Sensitivity of Monitored Area vs Threshold",
    x = "Threshold (dB, peak-to-peak)",
    y = "Volume Monitored (km³)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor.y = element_blank())

print(p)


p <- ggplot(
  results_all,
  aes(x = Threshold,
      y = AreaKm3,
      color = Model)   # <-- no Threshold here
) +
  facet_grid(~ Location) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = factor(Depth_m)), size = 2) +
  scale_shape_discrete(name = "Glider depth (m)") +
  scale_color_brewer(palette = "Set2", name = "Model") +
  labs(
    title = "Sensitivity of Monitored Area vs Threshold",
    x = "Threshold (dB, peak-to-peak)",
    y = "Volume Monitored (km³)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor.y = element_blank())

print(p)
# Distribution of volume monitored
# Full SNR run

data_dir <- "C:/Users/kaity/Documents/SpaciousData/CSVs"
noise_csv_path <- "C:/Users/kaity/Downloads/MHI_679_NoisE1_20khz.csv"
# 
# # Example: SNR thresholds at 10 and 20 dB, sine crest (9.03 dB), same cell geometry you used
# snr_results <- snr_volume_from_dir(
#   data_dir,
#   noise_csv_path,
#   snr_thresholds = c(20),
#   crest_db = 9.03,
#   cell_area_km2 = 0.152,
#   slice_thick_km = 0.1
# )
# 
# 
# 
# # Quick look
# print(head(snr_results))
# 
# 
# library(dplyr)
# library(ggplot2)
# 
# 
# library(scales)
# 
# # "167" "42"  "86" 
# 
# 
# library(dplyr)
# library(ggplot2)
# library(scales)
# 
# df <- subset(snr_results, Depth_m ==500, is.finite(AreaKm3))
# 
# ggplot(snr_results, 
#        aes(x = AreaKm3, colour = Model, 
#            fill = Model, group = interaction(Location, Depth_m, Model))) +
#   facet_wrap(SNR_Thr ~ Depth_m+Location, 
#              labeller = label_both, 
#              scales = "free_x") +
#   geom_density(alpha = 0.35, adjust = 1, na.rm = TRUE, nrow=2) +
#   scale_x_continuous(labels = label_number(big.mark = ",")) +
#   labs(
#     title = "Volume Monitored",
#     x = "Volume Monitored (km³)",
#     y = "Density (per Model × Location)",
#     colour = "Model", fill = "Model"
#   ) +
#   theme_minimal(base_size = 12) +
#   theme(panel.grid.minor = element_blank())
# 
# library(ggplot2)
# library(scales)
# library(dplyr)
# 
# 
# ggplot(
#   df,
#   aes(x = AreaKm3, colour = Model, fill = Model, group = interaction(Location, Model))
# ) +
#   facet_grid(SNR_Thr ~ Location, labeller = label_both, scales = "free_x") +
#   geom_density(alpha = 0.35, adjust = 1, na.rm = TRUE) +
#   scale_x_continuous(labels = label_number(big.mark = ",")) +
#   labs(
#     title = "Per-Model, Per-Location Density of Volume Monitored",
#     x = "Volume Monitored (km³)",
#     y = "Density (per Model × Location)"
#   ) +
#   theme_minimal(base_size = 12) +
#   theme(panel.grid.minor = element_blank())
# 
# 
# df$Datetime <- as.POSIXct(df$time, format = "%Y-%m-%d %H:%M:%S %Z", tz = "UTC")
# 
# 
# ggplot(
#   df,
#   aes(x = Datetime, y = AreaKm3, color = Model, group = Model)) +
#   geom_line(linewidth = 0.7, na.rm = TRUE) +   # draw the lines
#   facet_wrap(~ Location, scales = "free_y", nrow = 3) +  # panels per location
#   scale_y_continuous(labels = comma) +
#   labs(
#     title = "Count of cells above SNR threshold over time",
#     x = NULL, y = "Count of cells > threshold", color = "Model") +
#   theme_minimal(base_size = 12) +
#   theme(panel.grid.minor = element_blank())

