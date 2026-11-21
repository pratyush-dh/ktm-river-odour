"""Valley outline and elevation contours from a DEM (no assumed rim height).

The outline is the drainage basin of the valley floor: a marker watershed on the
DEM, flooded upward from (a) river points sitting on the valley floor and
(b) everything that is not the valley (raster edge cells and river points that
are clearly downstream, i.e. the gorge). The line where the floods meet is the
ridge crest / outlet sill, so no elevation threshold is chosen.

Writes data_valley.js  ->  window.KTM_VALLEY = {outline, contours, floor_m, ...}
Run:  python odour_valley.py --dem data/kathmandu_glo30_basemap.tif --points out/river_points.csv --out out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _round(c, nd=5):
    return [[round(x, nd), round(y, nd)] for x, y in c]


def build(dem_path, points_csv, out_dir, interval=100, index_every=500, decim=3, sigma=1.2):
    import rasterio
    from rasterio import features
    from scipy import ndimage as ndi
    from shapely.geometry import MultiLineString, mapping, shape
    from shapely.ops import unary_union
    from skimage.measure import find_contours
    from skimage.segmentation import watershed

    with rasterio.open(dem_path) as s:
        z = s.read(1).astype(float)
        t = s.transform
        nod = s.nodata
    if nod is not None:
        z[z == nod] = np.nan
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    z = z[: z.shape[0] // decim * decim, : z.shape[1] // decim * decim]
    z = z.reshape(z.shape[0] // decim, decim, z.shape[1] // decim, decim).mean(axis=(1, 3))
    from affine import Affine
    t = t * Affine.scale(decim, decim)
    zs = ndi.gaussian_filter(z, sigma)

    pts = pd.read_csv(points_csv)
    rows, cols = rasterio.transform.rowcol(t, pts.lon.to_numpy(), pts.lat.to_numpy())
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = (rows >= 0) & (rows < z.shape[0]) & (cols >= 0) & (cols < z.shape[1])
    rows, cols = rows[ok], cols[ok]
    ez = zs[rows, cols]
    floor = float(np.median(ez))
    # valley-floor markers: river points near the median river elevation.
    # downstream (gorge) markers: river points clearly below the floor.
    on_floor = np.abs(ez - floor) <= 40
    down = ez < floor - 40
    if on_floor.sum() < 5:
        raise SystemExit("too few river points on the valley floor to seed the basin")

    mk = np.zeros(z.shape, np.int32)
    mk[0, :] = mk[-1, :] = mk[:, 0] = mk[:, -1] = 2
    mk[rows[down], cols[down]] = 2
    mk[rows[on_floor], cols[on_floor]] = 1
    lab = watershed(zs, markers=mk)
    basin = lab == 1
    basin = ndi.binary_closing(basin, iterations=2)
    basin = ndi.binary_fill_holes(basin)
    comp, n = ndi.label(basin)
    keep = np.bincount(comp.ravel())[1:].argmax() + 1
    basin = comp == keep

    polys = [shape(g) for g, v in features.shapes(basin.astype(np.uint8), mask=basin, transform=t) if v == 1]
    poly = unary_union(polys).buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    poly = poly.simplify(0.0006)
    # smooth the staircase a little without moving the line more than ~60 m
    poly = poly.buffer(0.0004).buffer(-0.0004).simplify(0.0004)
    cos = np.cos(np.radians(poly.centroid.y))
    area_km2 = poly.area * 111.32 * 111.32 * cos
    inside = ndi.binary_dilation(basin, iterations=1)
    zin = zs[inside]
    rim_hi = float(np.percentile(zin, 99.5))
    outline = {"type": "Feature", "properties": {"name": "Valley (drainage basin of the valley floor)",
                                                 "area_km2": round(area_km2, 1), "floor_m": round(floor), "rim_m": round(rim_hi)},
               "geometry": mapping(poly)}

    # contours inside the basin only
    lv0 = int(np.ceil(floor / interval) * interval)
    levels = np.arange(lv0, int(zin.max()) + 1, interval)
    zc = np.where(ndi.binary_dilation(basin, iterations=3), zs, np.nan)
    feats = []
    clip = poly.buffer(0.0005)
    for lv in levels:
        segs = []
        for c in find_contours(np.nan_to_num(zc, nan=-1e4), lv):
            if len(c) < 6:
                continue
            xy = [t * (cc[1] + 0.5, cc[0] + 0.5) for cc in c]
            segs.append(xy)
        if not segs:
            continue
        geom = MultiLineString(segs).intersection(clip).simplify(0.00025)
        if geom.is_empty:
            continue
        lines = [geom] if geom.geom_type == "LineString" else [g for g in getattr(geom, "geoms", []) if g.geom_type == "LineString"]
        lines = [g for g in lines if g.length > 0.002]
        if not lines:
            continue
        feats.append({"type": "Feature", "properties": {"elev": int(lv), "index": int(lv % index_every == 0)},
                      "geometry": {"type": "MultiLineString", "coordinates": [_round(g.coords) for g in lines]}})
    data = {"outline": outline, "contours": {"type": "FeatureCollection", "features": feats},
            "floor_m": round(floor), "interval": interval, "area_km2": round(area_km2, 1)}
    out = Path(out_dir) / "data_valley.js"
    out.write_text("window.KTM_VALLEY=" + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"valley basin {area_km2:.0f} km2, floor {floor:.0f} m, rim top {rim_hi:.0f} m, "
          f"{len(feats)} contour levels, {out.stat().st_size/1e6:.2f} MB -> {out}")
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=int, default=100)
    a = ap.parse_args()
    build(a.dem, a.points, a.out, a.interval)
