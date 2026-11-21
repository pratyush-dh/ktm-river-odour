#!/usr/bin/env Rscript
# Georeferenced joyplot basemap (dark + light) for the odour web map.
#
# Unlike the poster version, this is drawn so that it can sit under a web map:
#   * x = longitude, y = Web-Mercator northing  -> the PNG maps 1:1 onto its lat/lon bounds.
#   * Transect i is placed at its true latitude. Peaks rise ABOVE it, but heights are
#     measured from the valley-floor elevation (median DEM height at the river points), so
#     the flat valley floor, where the rivers and buildings are, sits at its true latitude
#     instead of being shifted north by the exaggeration.
#   * No rivers or exposure drawn here: the web map draws those live on top.
# Writes out/basemap_dark.png, out/basemap_light.png and out/basemap_meta.js (bounds).
#
# Run from this folder:  Rscript kathmandu_joyplot_basemap.R
local({
  p <- system.file("proj", package = "terra")
  if (nzchar(p)) Sys.setenv(PROJ_LIB = p, PROJ_DATA = p)
})
suppressPackageStartupMessages({ library(terra); library(ggplot2); library(scales) })

bbox         <- c(xmin = 85.15, ymin = 27.42, xmax = 85.55, ymax = 27.85)   # reaches the lower Bagmati
dem_file     <- "data/kathmandu_glo30_basemap.tif"
points_csv   <- "out/river_points.csv"
out_dir      <- "out"
n_lines      <- 150
n_points     <- 1200
exaggeration <- 12          # peak height above the valley floor, in transect spacings
pad_north    <- 0.02        # degrees of headroom above the bbox for the northern peaks
width_px     <- 4800
dpi          <- 300
line_width   <- 0.28

themes <- list(
  dark = list(bg = "#0a0c11",
              tint = data.frame(elev = c(-Inf, 1250, 1400, 1700, 2100, Inf),
                                col = c("#7d838d", "#8fb04c", "#90b04c", "#3f9244", "#1f4b37", "#0f3a28"))),
  light = list(bg = "#f3f1ec",
               tint = data.frame(elev = c(-Inf, 1250, 1400, 1700, 2100, Inf),
                                 col = c("#5d6674", "#8a9f35", "#7a9232", "#4c8a3a", "#2a6548", "#143f2c")))
)

merc <- function(lat) log(tan(pi / 4 + lat * pi / 360))   # degrees -> Mercator northing (radians)

dem_url <- function(lat, lon) {
  tile <- sprintf("Copernicus_DSM_COG_10_N%02d_00_E%03d_00_DEM", lat, lon)
  sprintf("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/%s/%s.tif", tile, tile)
}
if (!file.exists(dem_file)) {
  lats <- floor(bbox["ymin"]):floor(bbox["ymax"]); lons <- floor(bbox["xmin"]):floor(bbox["xmax"])
  tiles <- unlist(lapply(lats, function(a) lapply(lons, function(b) rast(dem_url(a, b)))))
  e0 <- ext(bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"])
  r <- if (length(tiles) == 1) crop(tiles[[1]], e0) else crop(do.call(mosaic, unname(tiles)), e0)
  dir.create(dirname(dem_file), showWarnings = FALSE, recursive = TRUE)
  writeRaster(r, dem_file, overwrite = TRUE, gdal = c("COMPRESS=DEFLATE", "TILED=YES"))
}
dem <- rast(dem_file)
m <- as.matrix(dem, wide = TRUE); m[!is.finite(m)] <- min(m, na.rm = TRUE)
h <- nrow(m); w <- ncol(m)
edges <- as.integer(round(seq(0, h, length.out = n_lines + 1)))
bands <- t(vapply(seq_len(n_lines), function(i) {
  rows <- (edges[i] + 1):max(edges[i + 1], edges[i] + 1); colMeans(m[rows, , drop = FALSE])
}, numeric(w)))
xi <- seq(1, w, length.out = n_points)
z  <- t(apply(bands, 1, function(r) approx(seq_len(w), r, xout = xi)$y))
e  <- ext(dem)
lon <- seq(e$xmin, e$xmax, length.out = n_points)
zmin <- min(z); zmax <- max(z)

zfloor <- 1300
if (file.exists(points_csv)) {
  rp <- read.csv(points_csv)
  v <- terra::extract(dem, cbind(rp$lon, rp$lat))[, 1]
  if (any(is.finite(v))) zfloor <- median(v, na.rm = TRUE)
}
message("valley-floor elevation used as reference: ", round(zfloor), " m")

dlat   <- (e$ymax - e$ymin) / n_lines
dh_deg <- exaggeration * dlat / (zmax - zmin)             # degrees of latitude per metre of relief
lat_i  <- e$ymax - (seq_len(n_lines) - 0.5) * dlat        # true latitude of each transect centre
north  <- e$ymax + pad_north

x_span <- (e$xmax - e$xmin) * pi / 180
y_span <- merc(north) - merc(e$ymin)
height_in <- (width_px / dpi) * y_span / x_span

render <- function(name, th) {
  tint <- th$tint
  anch_v <- rescale(pmin(pmax(tint$elev, zmin), zmax), from = c(zmin, zmax)); keep <- !duplicated(anch_v)
  p <- ggplot()
  for (i in seq_len(n_lines)) {
    yl <- lat_i[i] + (z[i, ] - zfloor) * dh_deg
    d  <- data.frame(x = lon, y = merc(yl), z = z[i, ], base = merc(lat_i[i] - 0.5 * dlat))
    p <- p +
      geom_ribbon(data = d, aes(x = x, ymin = base, ymax = y), fill = th$bg, colour = NA) +
      geom_path(data = d, aes(x = x, y = y, colour = z), linewidth = line_width,
                lineend = "round", linejoin = "round")
  }
  p <- p +
    scale_colour_gradientn(colours = tint$col[keep], values = anch_v[keep], limits = c(zmin, zmax),
                           guide = "none") +
    scale_x_continuous(limits = c(e$xmin, e$xmax), expand = c(0, 0)) +
    scale_y_continuous(limits = c(merc(e$ymin), merc(north)), expand = c(0, 0)) +
    theme_void() +
    theme(plot.background = element_rect(fill = th$bg, colour = NA),
          panel.background = element_rect(fill = th$bg, colour = NA),
          plot.margin = margin(0, 0, 0, 0), legend.position = "none")
  f <- file.path(out_dir, paste0("basemap_", name, ".png"))
  ggsave(f, p, width = width_px / dpi, height = height_in, dpi = dpi, bg = th$bg, limitsize = FALSE)
  message("Saved ", normalizePath(f))
}
dir.create(out_dir, showWarnings = FALSE)
for (nm in names(themes)) render(nm, themes[[nm]])
writeLines(sprintf("window.KTM_BASE={south:%.6f,west:%.6f,north:%.6f,east:%.6f,floor_m:%.0f};",
                   e$ymin, e$xmin, north, e$xmax, zfloor), file.path(out_dir, "basemap_meta.js"))
