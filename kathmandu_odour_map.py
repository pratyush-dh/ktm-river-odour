#!/usr/bin/env python
"""Kathmandu Valley river-odour exposure map: evidence-driven, no assumed buffer.

Subcommands
-----------
templates   write empty CSV templates for field observations and water-quality stations
exposure    distance-to-river exposure curve + map (population vs distance, measured
            water quality on each reach). No odour distance is chosen.
calibrate   fit P(odour detected | distance, night, monsoon, downwind) from field
            observations; adds data-derived odour-reach contours with 95% CIs to the map.

Requires: geopandas>=1.0, shapely>=2, pyproj, rasterio, scipy, pandas, numpy, requests
    pip install geopandas rasterio scipy requests

Examples
--------
    python kathmandu_odour_map.py templates --out data/
    python kathmandu_odour_map.py exposure  --out out/ --pop npl_ppp_2020.tif --stations data/stations.csv
    python kathmandu_odour_map.py calibrate --out out/ --pop npl_ppp_2020.tif \
        --stations data/stations.csv --obs data/observations.csv --cluster session

What is and is not assumed
--------------------------
* CRS: distances in EPSG:32645 (UTM 45N, metres); web layers EPSG:4326.
* Distance to river = distance to the nearest OSM line, evaluated at raster cell centres
  using lines densified to <=5 m vertices (error <= ~2.5 m). Cells further than
  --max-dist are ignored. --max-dist is a display/compute extent, not an odour claim.
* Population counts are only resolved down to the raster cell size (reported in
  the output as resolution_m); curve values below that are flagged unresolved.
* Water quality is shown only where measured (stations.csv) and attached to the nearest
  station on the SAME river; the distance to that station is stored. Nothing is
  interpolated, no pollution classes are imposed.
* Odour-reach distances exist only in `calibrate`, from your observations. It refuses to
  run on too little data. Observation windows (night hours, monsoon months) are
  sampling labels; the model estimates their effect.
* OSM may omit culverted/covered reaches. The `tunnel` and `waterway` fields are kept.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon

import odour_core as oc
import odour_hotspots as oh
import odour_web as ow

LOG = logging.getLogger("odour")
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter"]
VALLEY_BBOX = (27.58, 85.20, 27.83, 85.55)  # S, W, N, E (with margin)
CRS_GEO, CRS_PROJ = "EPSG:4326", "EPSG:32645"
NAME_REGEX = "Bagmati|Bishnumati|Vishnumati|Dhobi|Tukucha|Tukuche|Hanumante|Manohara|Nakkhu|Nakhu|Balkhu"

STATION_COLS = ["station", "river", "lat", "lon", "date", "do_mgl", "bod_mgl", "orp_mv", "source"]
OBS_COLS = ["lat", "lon", "datetime", "detected", "intensity_0_5", "wind_dir_from_deg", "session", "observer"]


# ----------------------------------------------------------------------------- data
class Tiler:
    """Tiled, cached, rate-limit-friendly Overpass client.

    The bbox is cut into small tiles; each tile is its own request, cached on disk
    (re-runs and interrupted runs resume for free). Failed tiles are retried with
    exponential backoff (Retry-After honoured), rotated across mirrors, and finally
    split into four sub-tiles. Ways crossing tile edges come back from both tiles
    and are de-duplicated by OSM id.
    """

    def __init__(self, cache: Path, pause: float = 1.0, timeout: int = 90, retries: int = 4, min_tile_deg: float = 0.005):
        self.cache, self.pause, self.timeout, self.retries, self.min_tile = cache, pause, timeout, retries, min_tile_deg
        self.cache.mkdir(parents=True, exist_ok=True)
        self.headers = {"User-Agent": "ktm-odour-map/0.1", "Accept": "application/json"}
        self.n_req = self.n_cached = 0

    def _post(self, query: str) -> list:
        last = None
        for k in range(self.retries):
            url = OVERPASS_URLS[k % len(OVERPASS_URLS)]
            try:
                r = requests.post(url, data={"data": query}, headers=self.headers, timeout=self.timeout + 30)
                if r.status_code in (429, 502, 503, 504):
                    wait = float(r.headers.get("Retry-After", 0) or 0) or 5 * 2 ** k
                    last = f"HTTP {r.status_code}"
                    LOG.warning("Overpass %s on %s; waiting %.0fs", last, url, wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                js = r.json()
                if "runtime error" in str(js.get("remark", "")):
                    last = js["remark"]  # server-side timeout/memory: tile too big
                    break
                self.n_req += 1
                time.sleep(self.pause)
                return js.get("elements", [])
            except (requests.RequestException, ValueError) as exc:
                last = str(exc)
                time.sleep(2 ** k)
        raise RuntimeError(last or "overpass failed")

    def tile(self, kind: str, body: str, bbox: tuple) -> list:
        """`body` contains {bbox} placeholder, e.g. 'way["building"]({bbox});out geom;'"""
        s, w, n, e = bbox
        f = self.cache / f"{kind}_{s:.4f}_{w:.4f}_{n:.4f}_{e:.4f}.json"
        if f.exists():
            self.n_cached += 1
            return json.loads(f.read_text())
        q = f"[out:json][timeout:{self.timeout}];" + body.format(bbox=f"{s},{w},{n},{e}")
        try:
            els = self._post(q)
        except RuntimeError as exc:
            if max(n - s, e - w) / 2 < self.min_tile:
                raise RuntimeError(f"tile {bbox} failed at minimum size: {exc}") from exc
            LOG.warning("tile %s failed (%s); splitting", bbox, exc)
            ms, mw = (s + n) / 2, (w + e) / 2
            els = []
            for sub in ((s, w, ms, mw), (s, mw, ms, e), (ms, w, n, mw), (ms, mw, n, e)):
                els += self.tile(kind, body, sub)
        f.write_text(json.dumps(els))
        return els

    def run(self, kind: str, body: str, bbox: tuple, tile_deg: float, keep=None) -> dict:
        """Fetch every tile of bbox (optionally only those where keep(tile_bbox) is True); dedupe by id."""
        s, w, n, e = bbox
        ss = np.arange(s, n, tile_deg)
        ws = np.arange(w, e, tile_deg)
        tiles = [(a, b, min(a + tile_deg, n), min(b + tile_deg, e)) for a in ss for b in ws]
        if keep:
            tiles = [t for t in tiles if keep(t)]
        out: dict = {}
        for i, t in enumerate(tiles, 1):
            for el in self.tile(kind, body, t):
                out[el["id"]] = el
            if i % 10 == 0 or i == len(tiles):
                LOG.info("%s: tile %d/%d, %d features (requests %d, cached %d)",
                         kind, i, len(tiles), len(out), self.n_req, self.n_cached)
        return out


def fetch_waterways(tiler: Tiler, bbox=VALLEY_BBOX, tile_deg: float = 0.1) -> gpd.GeoDataFrame:
    body = f'way["waterway"~"river|stream|canal|drain"]["name"~"{NAME_REGEX}",i]({{bbox}});out geom tags;'
    rows = []
    for el in tiler.run("waterways", body, bbox, tile_deg).values():
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(pts) >= 2:
            t = el.get("tags", {})
            rows.append({"osm_id": el["id"], "name": t.get("name", ""), "waterway": t.get("waterway", ""),
                         "tunnel": t.get("tunnel", ""), "geometry": LineString(pts)})
    if not rows:
        raise RuntimeError("Overpass returned no waterways; use --waterways with your own line file.")
    return gpd.GeoDataFrame(rows, crs=CRS_GEO)


def fetch_buildings(tiler: Tiler, lines_utm: gpd.GeoDataFrame, reach_m: float, tile_deg: float = 0.03) -> gpd.GeoDataFrame:
    """OSM building footprints (closed ways) from only those tiles that touch the river corridor."""
    corridor = gpd.GeoSeries([shapely.union_all(lines_utm.geometry.to_numpy()).buffer(reach_m + 100)],
                             crs=CRS_PROJ).to_crs(CRS_GEO).iloc[0]
    w, s, e, n = corridor.bounds
    keep = lambda t: corridor.intersects(shapely.box(t[1], t[0], t[3], t[2]))
    body = 'way["building"]({bbox});out geom tags;'
    rows = []
    for el in tiler.run("buildings", body, (s, w, n, e), tile_deg, keep).values():
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            rows.append({"osm_id": el["id"], "building": el.get("tags", {}).get("building", "yes"),
                         "geometry": Polygon(pts)})
    if not rows:
        raise RuntimeError("No OSM building footprints found along the rivers.")
    g = gpd.GeoDataFrame(rows, crs=CRS_GEO).to_crs(CRS_PROJ)
    g = g[g.geometry.is_valid & ~g.geometry.is_empty].copy()
    g["area_m2"] = g.geometry.area
    LOG.info("%d building footprints (relation/multipolygon buildings are not included)", len(g))
    return g.reset_index(drop=True)


ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
                "living_street", "service", "pedestrian"]


def fetch_roads(tiler: Tiler, lines_utm: gpd.GeoDataFrame, reach_m: float, tile_deg: float = 0.03) -> gpd.GeoDataFrame:
    """OSM drivable/pedestrian-street ways (footpaths/steps/tracks excluded) from tiles touching the river corridor."""
    corridor = gpd.GeoSeries([shapely.union_all(lines_utm.geometry.to_numpy()).buffer(reach_m + 100)],
                             crs=CRS_PROJ).to_crs(CRS_GEO).iloc[0]
    w, s, e, n = corridor.bounds
    keep = lambda t: corridor.intersects(shapely.box(t[1], t[0], t[3], t[2]))
    rx = "^(" + "|".join(ROAD_CLASSES) + ")(_link)?$"
    body = f'way["highway"~"{rx}"]({{bbox}});out geom tags;'
    rows = []
    for el in tiler.run("roads", body, (s, w, n, e), tile_deg, keep).values():
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(pts) >= 2:
            rows.append({"osm_id": el["id"], "highway": el.get("tags", {}).get("highway", ""),
                         "geometry": LineString(pts)})
    if not rows:
        raise RuntimeError("No OSM roads found along the rivers.")
    g = gpd.GeoDataFrame(rows, crs=CRS_GEO).to_crs(CRS_PROJ)
    LOG.info("%d OSM road ways", len(g))
    return g.reset_index(drop=True)


def road_pieces(roads: gpd.GeoDataFrame, lines_utm: gpd.GeoDataFrame, max_d: float, step: float = 25.0,
                lcode: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Cut roads into <=step m pieces; each piece gets the distance (m) from its midpoint to the nearest river line."""
    tree = shapely.STRtree(lines_utm.geometry.to_numpy())
    idx, dmin = tree.query_nearest(roads.geometry.to_numpy(), return_distance=True, all_matches=False)
    near = np.zeros(len(roads), bool)
    near[idx[0][dmin <= max_d + step]] = True
    roads = roads[near]
    cls_code = {c: i for i, c in enumerate(ROAD_CLASSES)}
    X0, Y0, X1, Y1, C = [], [], [], [], []
    for geom, hw in zip(roads.geometry.to_numpy(), roads.highway.to_numpy()):
        n = max(1, int(np.ceil(geom.length / step)))
        xy = shapely.get_coordinates(shapely.line_interpolate_point(geom, np.linspace(0, 1, n + 1), normalized=True))
        X0.append(xy[:-1, 0]); Y0.append(xy[:-1, 1]); X1.append(xy[1:, 0]); Y1.append(xy[1:, 1])
        C.append(np.full(n, cls_code.get(hw.replace("_link", ""), 0)))
    df = pd.DataFrame({"x0": np.concatenate(X0), "y0": np.concatenate(Y0), "x1": np.concatenate(X1),
                       "y1": np.concatenate(Y1), "cls": np.concatenate(C)})
    mid = shapely.points((df.x0 + df.x1) / 2, (df.y0 + df.y1) / 2)
    i2, d = tree.query_nearest(mid, return_distance=True, all_matches=False)
    dist = np.full(len(df), np.nan)
    dist[i2[0]] = d
    df["dist_m"] = dist
    rc = np.zeros(len(df), int)
    if lcode is not None:
        rc[i2[0]] = lcode[i2[1]]          # river this piece is nearest to
    df["river_code"] = rc
    df["len_m"] = np.hypot(df.x1 - df.x0, df.y1 - df.y0)
    df = df[df.dist_m <= max_d].reset_index(drop=True)
    LOG.info("%d road pieces (%.0f km) within %.0f m of a river", len(df), df.len_m.sum() / 1000, max_d)
    return df


def building_distances(bld: gpd.GeoDataFrame, lines_utm: gpd.GeoDataFrame, max_d: float,
                       lcode: Optional[np.ndarray] = None):
    """Exact footprint-to-river-line distance (m, polygon edge to nearest line) for buildings within max_d.
    Also records which river (code) is nearest."""
    tree = shapely.STRtree(lines_utm.geometry.to_numpy())
    idx, d = tree.query_nearest(bld.geometry.to_numpy(), return_distance=True, all_matches=False)
    dist = np.full(len(bld), np.nan)
    dist[idx[0]] = d
    rc = np.zeros(len(bld), int)
    if lcode is not None:
        rc[idx[0]] = lcode[idx[1]]
    bld = bld.assign(dist_m=dist, river_code=rc)
    return bld[bld.dist_m <= max_d].copy()


def load_waterways(path: Optional[Path], aoi: Optional[Path], tiler: Optional[Tiler] = None) -> gpd.GeoDataFrame:
    if path:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"{path} has no CRS")
        gdf = gdf.to_crs(CRS_GEO)
        if "name" not in gdf.columns:
            gdf["name"] = ""
    else:
        gdf = fetch_waterways(tiler)
    if aoi:
        a = gpd.read_file(aoi)
        if a.crs is None:
            raise ValueError("AOI has no CRS")
        gdf = gpd.clip(gdf, a.to_crs(gdf.crs))
        if gdf.empty:
            raise RuntimeError("No river lines inside the AOI")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["river_key"] = gdf["name"].map(oc.canonical_river)
    return gdf.reset_index(drop=True)


def densify(lines_utm: gpd.GeoSeries, step: float = 5.0, codes: Optional[np.ndarray] = None):
    """Vertices every <= `step` m along all lines, as an (N,2) array in metres.
    With `codes` (one per line) also returns the code of the line each vertex came from."""
    out, oc_ = [], []
    for k, geom in enumerate(lines_utm.geometry):
        for part in getattr(geom, "geoms", [geom]):
            n = max(2, int(np.ceil(part.length / step)) + 1)
            pts = shapely.line_interpolate_point(part, np.linspace(0, part.length, n))
            xy = shapely.get_coordinates(pts)
            out.append(xy)
            if codes is not None:
                oc_.append(np.full(len(xy), codes[k]))
    xy = np.vstack(out)
    return xy if codes is None else (xy, np.concatenate(oc_))


def attach_stations(lines_utm: gpd.GeoDataFrame, st_csv: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Attach measured values from the nearest same-river station (no interpolation)."""
    st = pd.read_csv(st_csv)
    miss = {"station", "river", "lat", "lon"} - set(st.columns)
    if miss:
        raise ValueError(f"stations csv missing columns: {sorted(miss)}")
    for c in STATION_COLS:
        if c not in st.columns:
            st[c] = np.nan
    stg = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=CRS_GEO).to_crs(CRS_PROJ)
    stg["river_key"] = stg["river"].map(oc.canonical_river)
    for c in ["station_name", "do_mgl", "bod_mgl", "orp_mv", "station_date", "station_source", "dist_to_station_m"]:
        lines_utm[c] = np.nan if c not in ("station_name", "station_date", "station_source") else ""
    for i, row in lines_utm.iterrows():
        cand = stg[stg.river_key == row.river_key]
        if cand.empty:
            continue
        d = cand.geometry.distance(row.geometry)
        j = d.idxmin()
        s = stg.loc[j]
        lines_utm.loc[i, ["station_name", "do_mgl", "bod_mgl", "orp_mv", "station_date", "station_source",
                          "dist_to_station_m"]] = [s.station, s.do_mgl, s.bod_mgl, s.orp_mv, str(s.date),
                                                   str(s.source), float(d[j])]
    return lines_utm, stg


# ----------------------------------------------------------------------------- raster
def pixel_distances(raster: Path, tree: cKDTree, max_d: float, vcode: Optional[np.ndarray] = None):
    """Distance-to-river and population for each raster cell centre within max_d.

    Returns (dist_m, pop, resolution_m, cell_xy_utm, nearest_river_code). Reads only the window around the rivers.
    """
    import pyproj
    import rasterio
    from rasterio.transform import xy
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(raster) as src:
        if src.crs is None:
            raise ValueError("population raster has no CRS")
        pts = tree.data
        tf_in = pyproj.Transformer.from_crs(CRS_PROJ, src.crs, always_xy=True)
        x0, y0 = pts.min(axis=0) - max_d
        x1, y1 = pts.max(axis=0) + max_d
        bx, by = tf_in.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
        win = from_bounds(min(bx), min(by), max(bx), max(by), src.transform).round_offsets().round_lengths()
        win = rasterio.windows.intersection(win, rasterio.windows.Window(0, 0, src.width, src.height))
        arr = src.read(1, window=win, masked=True)
        tfm = src.window_transform(win)
        rows, cols = np.indices(arr.shape)
        xs, ys = xy(tfm, rows.ravel(), cols.ravel(), offset="center")
        to_utm = pyproj.Transformer.from_crs(src.crs, CRS_PROJ, always_xy=True)
        X, Y = to_utm.transform(np.asarray(xs), np.asarray(ys))
        xy2 = np.column_stack([X, Y])
        # cell size in metres from two neighbouring centres
        res = float(np.hypot(*(xy2[1] - xy2[0]))) if len(xy2) > 1 else float("nan")
        pop = arr.filled(np.nan).ravel().astype(float)
    d, vi = tree.query(xy2, k=1, distance_upper_bound=max_d)
    keep = np.isfinite(d) & np.isfinite(pop)
    LOG.info("%d raster cells within %.0f m of a river (cell ~%.0f m)", keep.sum(), max_d, res)
    code = vcode[vi[keep]] if vcode is not None else np.zeros(int(keep.sum()), int)
    return d[keep], pop[keep], res, xy2[keep], code


# ----------------------------------------------------------------------------- polygons
def bands(lines_utm: gpd.GeoDataFrame, step: float, max_d: float) -> gpd.GeoDataFrame:
    merged = shapely.union_all(lines_utm.geometry.to_numpy())
    edges = np.arange(0, max_d + step, step)
    out, prev = [], None
    for r in edges[1:]:
        full = merged.buffer(float(r), quad_segs=8)
        geom = full if prev is None else full.difference(prev)
        out.append({"inner_m": float(r - step), "outer_m": float(r), "geometry": geom})
        prev = full
    return gpd.GeoDataFrame(out, crs=CRS_PROJ)


def contour(merged, dist_m: float):
    """Filled buffer polygon at dist_m around the merged river geometry (nan/inf/0 -> None)."""
    if not np.isfinite(dist_m) or dist_m <= 0:
        return None
    return merged.buffer(float(dist_m), quad_segs=8)


# ----------------------------------------------------------------------------- observations
def observation_table(obs_csv: Path, lines_utm: gpd.GeoDataFrame, tree: cKDTree, args) -> pd.DataFrame:
    obs = pd.read_csv(obs_csv)
    miss = {"lat", "lon", "datetime", "detected"} - set(obs.columns)
    if miss:
        raise ValueError(f"observations csv missing columns: {sorted(miss)}")
    g = gpd.GeoDataFrame(obs, geometry=gpd.points_from_xy(obs.lon, obs.lat), crs=CRS_GEO).to_crs(CRS_PROJ)
    xy = np.column_stack([g.geometry.x, g.geometry.y])
    d, i = tree.query(xy, k=1)
    obs["dist_m"] = d
    obs = oc.add_time_covariates(obs, args.night_start, args.night_end, args.monsoon_months)
    covars = ["night", "monsoon"]
    if "wind_dir_from_deg" in obs.columns and obs.wind_dir_from_deg.notna().all():
        # downwind = wind carries air from the river toward the observer (<90 deg difference)
        vec = xy - tree.data[i]  # river -> observer
        bearing_to_obs = (np.degrees(np.arctan2(vec[:, 0], vec[:, 1])) + 360) % 360
        wind_to = (obs.wind_dir_from_deg.to_numpy(float) + 180) % 360
        diff = np.abs((bearing_to_obs - wind_to + 180) % 360 - 180)
        obs["downwind"] = (diff < 90).astype(int)
        covars.append("downwind")
    # drop covariates with no variation instead of failing silently later
    covars = [c for c in covars if obs[c].nunique() > 1]
    obs.attrs["covars"] = covars
    return obs


# ----------------------------------------------------------------------------- web export
def to_geojson(gdf: gpd.GeoDataFrame, cols: Sequence[str], simplify_m: float = 0.0) -> dict:
    g = gdf.copy()
    if simplify_m:
        g["geometry"] = g.geometry.simplify(simplify_m)
    g = g.to_crs(CRS_GEO)[[*cols, "geometry"]]
    return json.loads(g.to_json(na="null"))


# ----------------------------------------------------------------------------- commands
def cmd_templates(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=STATION_COLS).to_csv(a.out / "stations.csv", index=False)
    pd.DataFrame(columns=OBS_COLS).to_csv(a.out / "observations.csv", index=False)
    print(f"Wrote templates to {a.out}. Fill ONLY measured values; leave unknown cells empty.")
    return 0


def cmd_map(a, calibrate: bool) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    tiler = Tiler(a.cache, pause=a.pause)
    lines = load_waterways(a.waterways, a.aoi, tiler)
    lines_utm = lines.to_crs(CRS_PROJ)
    stations_geo = gpd.GeoDataFrame(columns=["station", "geometry"], geometry="geometry", crs=CRS_GEO)
    if a.stations:
        lines_utm, st_utm = attach_stations(lines_utm, a.stations)
        stations_geo = st_utm.to_crs(CRS_GEO)[["station", "river", "do_mgl", "geometry"]]
    # one code per river (spelling variants merged) so every feature knows which river it is nearest to
    rkeys = sorted(k for k in lines_utm.river_key.unique() if k)
    kcode = {k: i for i, k in enumerate(rkeys)}
    lcode = lines_utm.river_key.map(lambda k: kcode.get(k, 0)).to_numpy()
    lines_utm["rcode"] = lcode
    rnames = {k: (lines_utm.loc[lines_utm.river_key == k, "name"].replace("", np.nan).dropna().mode().tolist() or [k.title()])[0]
              for k in rkeys}
    rlen_km = (lines_utm.assign(_l=lines_utm.geometry.length).groupby("river_key")._l.sum() / 1000).to_dict()
    verts, vcode = densify(lines_utm, codes=lcode)
    tree = cKDTree(verts)
    pxy = np.zeros((0, 2)); pcode = np.zeros(0, int); pd_ = np.zeros(0); ppop = np.zeros(0)

    # exposure curve (exact at cell level; no threshold)
    grid = np.arange(0, a.max_dist + 1, a.curve_step, dtype=float)
    res = float("nan")
    if a.pop:
        d, pop, res, pxy, pcode = pixel_distances(a.pop, tree, a.max_dist, vcode)
        pd_, ppop = d, pop
        curve = oc.cumulative_curve(d, pop, grid)
        curve["resolved"] = curve.dist_m >= res
    else:
        LOG.warning("No --pop raster: map shows distance bands only, no population numbers.")
        curve = pd.DataFrame({"dist_m": grid, "cum_pop": np.nan, "resolved": False})
    # OSM building footprints: exact footprint-edge distance, counted per building
    bld = None
    if a.buildings:
        bld = building_distances(fetch_buildings(tiler, lines_utm, a.max_dist, a.tile_deg), lines_utm, a.max_dist, lcode)
        cb = oc.cumulative_curve(bld.dist_m.to_numpy(), np.ones(len(bld)), grid)
        ca = oc.cumulative_curve(bld.dist_m.to_numpy(), bld.area_m2.to_numpy(), grid)
        curve["cum_buildings"], curve["cum_footprint_m2"] = cb.cum_pop.to_numpy(), ca.cum_pop.to_numpy()
    else:
        curve["cum_buildings"], curve["cum_footprint_m2"] = np.nan, np.nan
    roads_df = None
    if a.roads:
        roads_df = road_pieces(fetch_roads(tiler, lines_utm, a.max_dist, a.tile_deg), lines_utm, a.max_dist, lcode=lcode)
        curve["cum_road_m"] = oc.cumulative_curve(roads_df.dist_m.to_numpy(), roads_df.len_m.to_numpy(), grid).cum_pop.to_numpy()
    else:
        curve["cum_road_m"] = np.nan
    curve.to_csv(a.out / "exposure_curve.csv", index=False)

    bnd = bands(lines_utm, a.band_step, a.max_dist)
    contours, reach, reach_notes = [], [], []
    merged = shapely.union_all(lines_utm.geometry.to_numpy())
    if calibrate:
        obs = observation_table(a.obs, lines_utm, tree, a)
        covars = obs.attrs["covars"]
        fit, tab = oc.calibrate(obs, covars=covars, n_boot=a.n_boot, cluster_col=a.cluster)
        tab.to_csv(a.out / "odour_reach_fit.csv", index=False)
        (a.out / "odour_model.json").write_text(json.dumps(
            {"coefficients": dict(zip(fit.names, map(float, fit.beta))), "auc": fit.auc, "n": fit.n,
             "n_detected": fit.n_pos, "covariates": covars, "cluster": a.cluster}, indent=2))
        print(tab.to_string(index=False))
        fin = lambda v: float(v) if np.isfinite(v) else None
        reach = [{"scenario": r.scenario, "p": float(r.p_detect), "dist_m": fin(r.dist_m),
                  "lo": fin(r.ci_lo_m), "hi": fin(r.ci_hi_m)} for r in tab.itertuples()]
        palette = ["#7f1d1d", "#d9480f", "#7c3aed", "#0f766e", "#1d4ed8", "#a16207", "#be185d", "#374151"]

        def emit(tb, geom, label, k0):
            for k, (sc, grp) in enumerate(tb.groupby("scenario", sort=False)):
                for _, r in grp.iterrows():
                    feats = []
                    for kind, dist in (("est", r.dist_m), ("lo", r.ci_lo_m), ("hi", r.ci_hi_m)):
                        poly = contour(geom, dist)
                        if poly is not None:
                            feats.append({"kind": kind, "geometry": poly})
                    if feats:
                        cg = gpd.GeoDataFrame(feats, crs=CRS_PROJ)
                        contours.append({"name": f"{label}P(detect)={r.p_detect:g} [{sc}] (95% CI dashed)",
                                         "color": palette[(k0 + k) % len(palette)],
                                         "geojson": to_geojson(cg, ["kind"], simplify_m=5)})

        emit(tab, merged, "", 0)
        # rivers are not alike: fit each river on its own reports, where there are enough of them
        if getattr(a, "by_river", False):
            if "river" not in obs.columns:
                raise ValueError("--by-river needs a 'river' column in the observations CSV (the web reports include it)")
            obs["river_key"] = obs["river"].fillna("").astype(str).map(oc.canonical_river)
            for j, (rk, sub) in enumerate(obs[obs.river_key != ""].groupby("river_key"), start=1):
                cv = [c for c in covars if sub[c].nunique() > 1]
                try:
                    _, t2 = oc.calibrate(sub.reset_index(drop=True), covars=cv, n_boot=a.n_boot, cluster_col=a.cluster)
                except ValueError as exc:
                    reach_notes.append(f"{rnames.get(rk, rk)}: not enough reports ({len(sub)}; "
                                       f"need at least 60 with 15 smelt and 15 not smelt)")
                    LOG.info("river %s skipped: %s", rk, exc)
                    continue
                t2["scenario"] = np.where(t2.scenario == "all", f"river={rk}", f"river={rk}," + t2.scenario)
                t2.to_csv(a.out / f"odour_reach_fit_{rk}.csv", index=False)
                print(t2.to_string(index=False))
                reach += [{"scenario": r.scenario, "p": float(r.p_detect), "dist_m": fin(r.dist_m),
                           "lo": fin(r.ci_lo_m), "hi": fin(r.ci_hi_m)} for r in t2.itertuples()]
                mr = shapely.union_all(lines_utm[lines_utm.river_key == rk].geometry.to_numpy())
                emit(t2, mr, f"{rnames.get(rk, rk)}: ", j)

    lines_out = lines_utm.drop(columns=[c for c in ("river_key",) if c in lines_utm.columns])
    keep = [c for c in ["name", "rcode", "waterway", "tunnel", "station_name", "do_mgl", "bod_mgl", "orp_mv",
                        "station_date", "station_source", "dist_to_station_m"] if c in lines_out.columns]
    # exposure zones: filled buffers around the rivers every `zone_step` m (simplified more with distance)
    zones = []
    for r in np.arange(a.zone_step, a.max_dist + 1, a.zone_step):
        zg = gpd.GeoDataFrame([{"d": float(r), "geometry": merged.buffer(float(r), quad_segs=8)}], crs=CRS_PROJ)
        zones.append({"d": float(r), "geojson": to_geojson(zg, ["d"], simplify_m=max(5.0, 0.04 * float(r)))})
    import pyproj
    tf = pyproj.Transformer.from_crs(CRS_PROJ, CRS_GEO, always_xy=True)
    to_ll = lambda x, y: tf.transform(np.asarray(x), np.asarray(y))
    # per-river exposure and hotspots of exposed buildings / people (see odour_hotspots.py for the statistics)
    hs, rivers_info = None, []
    if bld is not None and len(bld):
        cen = bld.geometry.centroid
        bxy = np.column_stack([cen.x, cen.y])
        rd = roads_df.dist_m.to_numpy() if roads_df is not None else np.zeros(0)
        rl = roads_df.len_m.to_numpy() if roads_df is not None else np.zeros(0)
        rc = roads_df.river_code.to_numpy() if roads_df is not None else np.zeros(0, int)
        rivers_info = oh.by_river(bld.dist_m.to_numpy(), bld.river_code.to_numpy(), bld.area_m2.to_numpy(), pd_, ppop, pcode,
                                  rd, rl, rc, rkeys, rnames, rlen_km, a.max_dist)
        hs = oh.hotspots(bxy, bld.dist_m.to_numpy(), bld.river_code.to_numpy(), pxy, pd_, ppop, len(rkeys), to_ll, a.max_dist)
        LOG.info("hotspots: %d candidate cells (200 m) of %d with buildings", len(hs["cells"]), hs["n_domain"])
    data = {
        "hotspots": hs, "rivers": rivers_info, "reach_notes": reach_notes,
        "max_dist": a.max_dist, "zone_step": a.zone_step, "resolution_m": 0 if np.isnan(res) else res,
        "has_pop": bool(a.pop), "has_bld": bool(a.buildings), "has_roads": bool(a.roads),
        "curve": [[float(r.dist_m)] + [0.0 if np.isnan(v) else float(v) for v in
                                       (r.cum_pop, r.cum_buildings, r.cum_footprint_m2, r.cum_road_m)]
                  for r in curve.itertuples()],
        "curve_step": a.curve_step, "zones": zones,
        "lines": to_geojson(lines_out, keep),
        "stations": to_geojson(stations_geo, [c for c in ["station"] if c in stations_geo.columns]),
        "contours": contours, "reach": reach,
    }
    (a.out / "data_meta.js").write_text("window.KTM=" + json.dumps(data) + ";", encoding="utf-8")
    if bld is not None:
        ow.export_buildings(a.out / "data_buildings.js", bld, to_ll, ow.LON0, ow.LAT0)
    if roads_df is not None:
        ow.export_roads(a.out / "data_roads.js", roads_df, to_ll, ow.LON0, ow.LAT0)
    ow.copy_page(a.out)
    gpkg = a.out / "odour_layers.gpkg"
    lines_utm.to_file(gpkg, layer="waterways", driver="GPKG")
    if bld is not None:
        bld.to_file(gpkg, layer="buildings_in_reach", driver="GPKG")
    # river sample points (every 50 m) with nearby-building counts, for the R joyplot overlay
    rp = densify(lines_utm, step=50.0)
    ll = gpd.GeoSeries(gpd.points_from_xy(rp[:, 0], rp[:, 1]), crs=CRS_PROJ).to_crs(CRS_GEO)
    if bld is not None:
        cen = bld.geometry.centroid
        nb = cKDTree(np.column_stack([cen.x, cen.y])).query_ball_point(rp, r=a.band_step, return_length=True)
    else:
        nb = np.full(len(rp), np.nan)
    pd.DataFrame({"lon": ll.x, "lat": ll.y, "n_buildings": nb}).to_csv(a.out / "river_points.csv", index=False)
    bnd.to_file(gpkg, layer="distance_bands", driver="GPKG")
    LOG.info("Wrote %s", a.out)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("templates"); t.add_argument("--out", type=Path, default=Path("data"))
    for name in ("exposure", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=Path("out"))
        p.add_argument("--waterways", type=Path, help="line file instead of Overpass")
        p.add_argument("--aoi", type=Path, help="polygon to clip rivers to")
        p.add_argument("--pop", type=Path, help="population raster, people per cell (e.g. WorldPop)")
        p.add_argument("--stations", type=Path, help="measured water-quality stations CSV")
        p.add_argument("--max-dist", type=float, default=1000.0, help="compute/display extent, m (not an odour claim)")
        p.add_argument("--band-step", type=float, default=100.0, help="map band width, m")
        p.add_argument("--curve-step", type=float, default=25.0)
        p.add_argument("--zone-step", type=float, default=50.0, help="exposure-zone slider step, m")
        p.add_argument("--no-buildings", dest="buildings", action="store_false", help="skip OSM building footprints")
        p.add_argument("--no-roads", dest="roads", action="store_false", help="skip OSM roads")
        p.add_argument("--tile-deg", type=float, default=0.03, help="Overpass tile size for buildings, degrees (~3.3 km)")
        p.add_argument("--cache", type=Path, default=Path("overpass_cache"), help="per-tile Overpass cache (resumable)")
        p.add_argument("--pause", type=float, default=1.0, help="seconds to wait between Overpass requests")
        if name == "calibrate":
            p.add_argument("--obs", type=Path, required=True)
            p.add_argument("--by-river", action="store_true",
                           help="also fit each river separately (needs a 'river' column; rivers with too few reports are listed, not guessed)")
            p.add_argument("--cluster", help="column grouping correlated observations (e.g. session)")
            p.add_argument("--n-boot", type=int, default=1000)
            p.add_argument("--night-start", type=int, default=19)
            p.add_argument("--night-end", type=int, default=6)
            p.add_argument("--monsoon-months", type=int, nargs="+", default=[6, 7, 8, 9])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if a.cmd == "templates":
            return cmd_templates(a)
        return cmd_map(a, calibrate=(a.cmd == "calibrate"))
    except (requests.RequestException, FileNotFoundError, ValueError, RuntimeError) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
