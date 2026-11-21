"""Major roads (motorway..tertiary) across the whole valley, for the map's reference layer.

The exposure map only packs roads inside the river corridor. This adds the valley-wide
road skeleton so the dashboard can show where the main roads run. Same tiled, cached,
resumable Overpass client as the main script.

Writes data_mroads.js -> window.KTM_MROADS = GeoJSON (props: c = class index, n = name/ref)
Run:  python odour_roads_major.py --out out          (needs out/data_valley.js; run odour_valley.py first)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from shapely.geometry import LineString, shape, mapping

import kathmandu_odour_map as km

MAJOR = ["motorway", "trunk", "primary", "secondary", "tertiary"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--cache", type=Path, default=Path("overpass_cache"))
    ap.add_argument("--tile-deg", type=float, default=0.1)
    ap.add_argument("--pause", type=float, default=1.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    txt = (a.out / "data_valley.js").read_text(encoding="utf-8")
    v = json.loads(txt[txt.index("=") + 1 : txt.rindex(";")])
    poly = shape(v["outline"]["geometry"]).buffer(0.01)  # ~1 km margin: roads leaving the valley are cut
    w, s, e, n = poly.bounds

    tiler = km.Tiler(a.cache, pause=a.pause)
    rx = "^(" + "|".join(MAJOR) + ")(_link)?$"
    body = f'way["highway"~"{rx}"]({{bbox}});out geom tags;'
    els = tiler.run("majorroads", body, (s, w, n, e), a.tile_deg)
    feats = []
    for el in els.values():
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(pts) < 2:
            continue
        t = el.get("tags", {})
        hw = t.get("highway", "")
        base = hw.replace("_link", "")
        if base not in MAJOR:
            continue
        g = LineString(pts).intersection(poly)
        if g.is_empty:
            continue
        g = g.simplify(0.00002)  # ~2 m
        parts = [g] if g.geom_type == "LineString" else [x for x in getattr(g, "geoms", []) if x.geom_type == "LineString"]
        for p in parts:
            feats.append({"type": "Feature",
                          "properties": {"c": MAJOR.index(base), "l": int(hw.endswith("_link")), "n": t.get("name") or t.get("ref") or ""},
                          "geometry": {"type": "LineString", "coordinates": [[round(x, 5), round(y, 5)] for x, y in p.coords]}})
    fc = {"type": "FeatureCollection", "features": feats}
    out = a.out / "data_mroads.js"
    out.write_text("window.KTM_MROADS=" + json.dumps(fc, separators=(",", ":")) + ";\n", encoding="utf-8")
    cnt = {c: sum(1 for f in feats if f["properties"]["c"] == i) for i, c in enumerate(MAJOR)}
    print(f"{len(feats)} major-road pieces {cnt}, {out.stat().st_size/1e6:.2f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
