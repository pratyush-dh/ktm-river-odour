"""Web-asset export for the odour map: compact binary packing of footprints/roads and JS data files.

Why binary: ~250k building polygons as GeoJSON is ~60 MB; packed as quantised deltas it is ~8 MB, loads
via a plain <script> tag (works from file://, no server) and is drawn on a canvas by the page.

Layout of one blob (little-endian), n groups, m delta vertices:
  fx int32[n], fy int32[n]   first vertex, (lon-lon0)*1e6 / (lat-lat0)*1e6
  nv uint16[n]               vertices in group (first included)
  dist uint16[n]             distance to nearest river line, m
  dx int16[m], dy int16[m]   deltas of the remaining vertices in 1e-6 deg
  cls uint8[n]               optional class code
Groups are sorted by first-vertex latitude so the page can binary-search the visible band.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

import numpy as np

Q = 1e6
LON0, LAT0 = 85.0, 27.0


def pack_groups(lon, lat, counts, dist, cls: Optional[np.ndarray], lon0: float, lat0: float) -> tuple[dict, bytes]:
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    counts = np.asarray(counts, np.int64)
    n = len(counts)
    starts = np.cumsum(counts) - counts
    qx = np.round((lon - lon0) * Q).astype(np.int64)
    qy = np.round((lat - lat0) * Q).astype(np.int64)
    order = np.argsort(qy[starts], kind="stable")
    c2 = counts[order]
    s2 = np.cumsum(c2) - c2
    vidx = np.repeat(starts[order] - s2, c2) + np.arange(int(c2.sum()))
    qx, qy = qx[vidx], qy[vidx]
    first = np.zeros(len(qx), bool)
    first[s2] = True
    ddx, ddy = np.diff(qx, prepend=qx[0]), np.diff(qy, prepend=qy[0])
    dx, dy = ddx[~first], ddy[~first]
    if np.abs(dx).max(initial=0) > 32767 or np.abs(dy).max(initial=0) > 32767:
        raise ValueError("vertex delta exceeds int16; shorten the groups")
    parts = [qx[first].astype("<i4"), qy[first].astype("<i4"), c2.astype("<u2"),
             np.clip(np.round(np.asarray(dist, float)[order]), 0, 65535).astype("<u2"),
             dx.astype("<i2"), dy.astype("<i2")]
    if cls is not None:
        parts.append(np.asarray(cls)[order].astype("<u1"))
    blob = b"".join(p.tobytes() for p in parts)
    return {"n": int(n), "m": int(len(dx)), "lon0": lon0, "lat0": lat0, "has_cls": cls is not None}, blob


def write_packed(path: Path, key: str, header: dict, blob: bytes) -> None:
    header = dict(header, b64=base64.b64encode(blob).decode("ascii"))
    path.write_text(f"window.KTM_DATA=window.KTM_DATA||{{}};window.KTM_DATA.{key}={json.dumps(header)};", encoding="utf-8")


def export_buildings(path: Path, bld, to_ll, lon0: float, lat0: float) -> None:
    """bld: GeoDataFrame in UTM with dist_m. to_ll: callable(x, y) -> lon, lat arrays."""
    import shapely
    geoms = shapely.simplify(bld.geometry.to_numpy(), 0.4)
    xy, idx = shapely.get_coordinates(shapely.get_exterior_ring(geoms), return_index=True)
    counts = np.bincount(idx, minlength=len(geoms))
    ends = np.cumsum(counts) - 1                      # closing vertex duplicates the first: drop it
    keep = np.ones(len(xy), bool)
    keep[ends] = False
    counts = counts - 1
    ok = counts >= 3
    gid = np.repeat(np.arange(len(geoms)), counts + 1)[keep]
    xy = xy[keep]
    sel = ok[gid]
    lon, lat = to_ll(xy[sel, 0], xy[sel, 1])
    cls = bld.river_code.to_numpy()[ok] if "river_code" in bld.columns else None   # nearest river, for the page's river focus
    h, blob = pack_groups(lon, lat, counts[ok], bld.dist_m.to_numpy()[ok], cls, lon0, lat0)
    write_packed(path, "bld", h, blob)


def export_roads(path: Path, pieces, to_ll, lon0: float, lat0: float) -> None:
    n = len(pieces)
    xs = np.empty(2 * n); ys = np.empty(2 * n)
    xs[0::2], xs[1::2] = pieces.x0.to_numpy(), pieces.x1.to_numpy()
    ys[0::2], ys[1::2] = pieces.y0.to_numpy(), pieces.y1.to_numpy()
    lon, lat = to_ll(xs, ys)
    cls = pieces.river_code.to_numpy() if "river_code" in pieces.columns else pieces.cls.to_numpy()
    h, blob = pack_groups(lon, lat, np.full(n, 2), pieces.dist_m.to_numpy(), cls, lon0, lat0)
    write_packed(path, "road", h, blob)


def copy_page(out: Path) -> None:
    """Copy the static page and vendored Leaflet next to the data files (offline, no CDN)."""
    import shutil
    here = Path(__file__).parent
    shutil.copy(here / "web" / "odour_map.html", out / "odour_map.html")
    if not (out / "config.js").exists():  # never overwrite a report_url the user has set
        shutil.copy(here / "web" / "config.js", out / "config.js")
    shutil.copytree(here / "vendor", out / "vendor", dirs_exist_ok=True)
