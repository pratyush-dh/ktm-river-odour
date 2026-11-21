#!/usr/bin/env Rscript
# Kathmandu Valley joyplot (GLO-30 ridgelines) with the river-odour exposure overlaid.
#
# Overlay: every ~50 m along the OSM rivers, a dot sits on the ridgeline of the transect
# it falls in (so nearer ridges occlude it correctly). Dot colour = number of OSM buildings
# within `band_step` m of that point (from kathmandu_odour_map.py -> out/river_points.csv).
# No odour distance is assumed or drawn.
#
# Run kathmandu_odour_map.py first, then (from this folder):  Rscript kathmandu_joyplot_odour.R
# Install: install.packages(c("terra", "ggplot2", "scales"))

local({
  p <- system.file("proj", package = "terra")
  if (nzchar(p)) Sys.setenv(PROJ_LIB = p, PROJ_DATA = p)
})
suppressPackageStartupMessages({ library(terra); library(ggplot2); library(scales) })

# ---- parameters -------------------------------------------------------------
bbox         <- c(xmin = 85.15, ymin = 27.55, xmax = 85.55, ymax = 27.85)
dem_file     <- "data/kathmandu_glo30.tif"
points_csv   <- "out/river_points.csv"
out_png      <- "kathmandu_joyplot_odour.png"
n_lines      <- 150
n_points     <- 700
offset       <- 1
exaggeration <- 12
bg           <- "#000000"
line_width   <- 0.4
dot_size     <- 0.9
tint <- data.frame(
  elev = c(-Inf, 1250, 1400, 1700, 2100, Inf),
  col  = c("#9aa0aa", "#a6c95a", "#a7c85a", "#4caf50", "#23553f", "#013220")
)
# river dots: low -> high nearby building count
dot_cols     <- c("#22d3ee", "#facc15", "#f97316", "#ef4444", "#ffffff")
title_text   <- "KATHMANDU VALLEY"
subtitle_txt <- "Rivers over terrain, coloured by OSM buildings within 100 m"

# ---- 1. DEM -----------------------------------------------------------------
dem_url <- function(lat, lon) {
  ns <- if (lat >= 0) "N" else "S"; ew <- if (lon >= 0) "E" else "W"
  tile <- sprintf("Copernicus_DSM_COG_10_%s%02d_00_%s%03d_00_DEM", ns, abs(lat), ew, abs(lon))
  sprintf("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/%s/%s.tif", tile, tile)
}
get_dem <- function(bbox, dem_file) {
  if (file.exists(dem_file)) return(rast(dem_file))
  lats <- floor(bbox["ymin"]):floor(bbox["ymax"])
  lons <- floor(bbox["xmin"]):floor(bbox["xmax"])
  tiles <- lapply(
    as.vector(outer(lats, lons, Vectorize(function(a, b) paste(a, b)))),
    function(s) {
      ll <- as.integer(strsplit(s, " ")[[1]])
      tryCatch(rast(dem_url(ll[1], ll[2])),
               error = function(e) stop("Could not open DEM tile (lat ", ll[1], ", lon ", ll[2], "): ",
                                        conditionMessage(e), call. = FALSE))
    })
  e <- ext(bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"])
  r <- if (length(tiles) == 1) crop(tiles[[1]], e) else crop(do.call(mosaic, unname(tiles)), e)
  dir.create(dirname(dem_file), showWarnings = FALSE, recursive = TRUE)
  writeRaster(r, dem_file, overwrite = TRUE, gdal = c("COMPRESS=DEFLATE", "TILED=YES"))
  r
}
dem <- get_dem(bbox, dem_file)
ev <- as.vector(ext(dem))
if (!(isTRUE(is.lonlat(dem)) && ev[1] >= -180 && ev[2] <= 180 && ev[3] >= -90 && ev[4] <= 90))
  stop("Expected a lon/lat (EPSG:4326) DEM")

# ---- 2. transects -----------------------------------------------------------
m <- as.matrix(dem, wide = TRUE)
m[!is.finite(m)] <- min(m, na.rm = TRUE)
h <- nrow(m); w <- ncol(m)
edges <- as.integer(round(seq(0, h, length.out = n_lines + 1)))
bands <- t(vapply(seq_len(n_lines), function(i) {
  rows <- (edges[i] + 1):max(edges[i + 1], edges[i] + 1)
  colMeans(m[rows, , drop = FALSE])
}, numeric(w)))
xi <- seq(1, w, length.out = n_points)
z  <- t(apply(bands, 1, function(r) approx(seq_len(w), r, xout = xi)$y))

e <- ext(dem)
km_lon <- 111.320 * cos(((e$ymin + e$ymax) / 2) * pi / 180)
x_km   <- seq(0, (e$xmax - e$xmin) * km_lon, length.out = n_points)
zmin <- min(z); zmax <- max(z)
height <- (z - zmin) / (zmax - zmin) * exaggeration * offset

# ---- 2b. river points -> (band, x, y) ---------------------------------------
if (!file.exists(points_csv)) stop("Missing ", points_csv, ": run kathmandu_odour_map.py first")
rp <- read.csv(points_csv)
rp <- rp[rp$lon >= e$xmin & rp$lon <= e$xmax & rp$lat >= e$ymin & rp$lat <= e$ymax, ]
row  <- (e$ymax - rp$lat) / (e$ymax - e$ymin) * h
rp$band <- pmin(pmax(findInterval(row, edges, rightmost.closed = TRUE), 1L), n_lines)
rp$x <- (rp$lon - e$xmin) * km_lon
rp <- rp[is.finite(rp$n_buildings), ]
nb_max <- max(rp$n_buildings, 1)

# ---- 3. plot ----------------------------------------------------------------
p <- ggplot()
for (i in seq_len(n_lines)) {                    # back (north) to front (south)
  base <- (n_lines - i) * offset
  d <- data.frame(x = x_km, y = base + height[i, ], z = z[i, ], base = base)
  p <- p +
    geom_ribbon(data = d, aes(x = x, ymin = base - 0.5 * offset, ymax = y), fill = bg, colour = NA) +
    geom_path(data = d, aes(x = x, y = y, colour = z), linewidth = line_width,
              lineend = "round", linejoin = "round")
  pi <- rp[rp$band == i, ]
  if (nrow(pi)) {
    pi$y <- base + approx(x_km, height[i, ], xout = pi$x, rule = 2)$y
    p <- p + geom_point(data = pi, aes(x = x, y = y, fill = n_buildings),
                        shape = 21, size = dot_size, stroke = 0.15, colour = bg)
  }
}
y_top <- (n_lines - 1) * offset + offset * (exaggeration + 1)
anch_v <- rescale(pmin(pmax(tint$elev, zmin), zmax), from = c(zmin, zmax))
keep   <- !duplicated(anch_v)

p <- p +
  scale_colour_gradientn(
    colours = tint$col[keep], values = anch_v[keep], limits = c(zmin, zmax), name = "Elevation (m)",
    guide = guide_colourbar(order = 1, direction = "horizontal", title.position = "left", title.vjust = 0.8,
                            barwidth = unit(3, "in"), barheight = unit(0.09, "in"), ticks = FALSE)) +
  scale_fill_gradientn(
    colours = dot_cols, limits = c(0, nb_max), oob = squish, name = "Buildings within 100 m of river",
    guide = guide_colourbar(order = 2, direction = "horizontal", title.position = "left", title.vjust = 0.8,
                            barwidth = unit(3, "in"), barheight = unit(0.09, "in"), ticks = FALSE)) +
  scale_x_continuous(expand = c(0.0005, 0)) +
  scale_y_continuous(limits = c(-offset, y_top), expand = c(0, 0)) +
  labs(title = title_text, subtitle = subtitle_txt) +
  theme_void() +
  theme(
    plot.background  = element_rect(fill = bg, colour = NA),
    panel.background = element_rect(fill = bg, colour = NA),
    plot.title       = element_text(colour = "white", size = 26, face = "plain", margin = margin(b = 4)),
    plot.subtitle    = element_text(colour = "#9aa0aa", size = 11, margin = margin(b = 14)),
    plot.title.position = "plot",
    legend.position = "bottom", legend.justification = "right", legend.box = "vertical",
    legend.title = element_text(colour = "#9aa0aa", size = 9),
    legend.text  = element_text(colour = "#9aa0aa", size = 9),
    legend.margin = margin(t = 6),
    plot.margin = margin(24, 28, 18, 28)
  )

ggsave(out_png, p, width = 12, height = 10, dpi = 200, bg = bg)
message("Saved ", normalizePath(out_png))
