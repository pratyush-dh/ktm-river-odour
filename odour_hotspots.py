"""Hotspots of exposed buildings and people, and the per-river breakdown.

Nothing here assumes how far an odour travels or that rivers smell alike. "Exposed" means
"within T metres of the nearest mapped river"; the page lets the user choose T, and this
module precomputes the answer for every T from 50 m to max_dist so the page can switch instantly.

Hotspot statistic: Getis-Ord Gi* on a square grid (default 200 m), queen contiguity including the
cell itself, binary weights, on the count of exposed buildings per cell. Gi* is a z-score: how
unusual a neighbourhood's total is compared with the other cells that have an exposed building at
that same distance. The page uses z >= 2.58 ("clustered") and z >= 3.29 ("strong hotspot"). With
over a thousand cells tested and strong spatial autocorrelation these are screening thresholds,
not proof of anything about smell, so the page says "concentrated", not "polluted".
Only cells reaching z >= zmin at some distance are exported.

Households are not available as open data here. OSM building count is the proxy (an apartment
block holds many households, a shed none). WorldPop people are attached per cell, with the
inflation caveat the page already shows.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def gi_star(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Getis-Ord Gi* z-scores on a 2-D grid; cells outside `mask` are ignored and get z = 0."""
    m = mask.astype(float)
    xv = np.where(mask, x, 0.0)
    n = m.sum()
    if n < 3:
        return np.zeros_like(xv)
    mean = xv.sum() / n
    s = np.sqrt(max((xv ** 2).sum() / n - mean ** 2, 0.0))
    k = np.ones((3, 3))
    wx = ndi.convolve(xv, k, mode="constant")   # sum of w_ij * x_j over in-domain neighbours + self
    w = ndi.convolve(m, k, mode="constant")     # sum of w_ij (binary), also = sum of w_ij^2
    with np.errstate(invalid="ignore", divide="ignore"):
        den = s * np.sqrt((n * w - w ** 2) / (n - 1))
        z = np.where(den > 0, (wx - mean * w) / den, 0.0)
    return np.where(mask, z, 0.0)


def t_grid(max_dist: float, step: float = 50.0) -> np.ndarray:
    return np.arange(step, max_dist + 1e-9, step)


def hotspots(b_xy, b_d, b_code, p_xy, p_d, p_pop, n_rivers, to_ll, max_dist=1000.0, cell=200.0,
             step=50.0, zmin=2.58) -> dict:
    """b_*: buildings (centroid xy in metres, distance to nearest river, nearest-river code).
    p_*: raster cells (xy, distance, people). Returns a JSON-able dict for the page."""
    T = t_grid(max_dist, step)
    allxy = np.vstack([b_xy, p_xy]) if len(p_xy) else b_xy
    x0, y0 = allxy.min(axis=0) - cell
    bix = np.floor((b_xy[:, 0] - x0) / cell).astype(int)
    biy = np.floor((b_xy[:, 1] - y0) / cell).astype(int)
    if len(p_xy):
        pix = np.floor((p_xy[:, 0] - x0) / cell).astype(int)
        piy = np.floor((p_xy[:, 1] - y0) / cell).astype(int)
    nx = int(max(bix.max(), pix.max() if len(p_xy) else 0)) + 2
    ny = int(max(biy.max(), piy.max() if len(p_xy) else 0)) + 2
    shape = (nx, ny)
    domain = np.zeros(shape, bool)
    domain[bix, biy] = True                      # cells with at least one building within max_dist (for reporting)
    B, P, Z = [], [], []
    for t in T:
        sel = b_d <= t
        b = np.zeros(shape)
        np.add.at(b, (bix[sel], biy[sel]), 1.0)
        p = np.zeros(shape)
        if len(p_xy):
            sp = p_d <= t
            np.add.at(p, (pix[sp], piy[sp]), p_pop[sp])
        # compare each exposed cell with the other EXPOSED cells at this T (cells with no exposed building are
        # not part of the comparison), otherwise everything near a river is "significant" next to the empty hills
        B.append(b); P.append(p); Z.append(gi_star(b, b > 0))
    B, P, Z = np.stack(B), np.stack(P), np.stack(Z)
    cand = domain & (Z.max(axis=0) >= zmin)
    ii, jj = np.nonzero(cand)
    # which river dominates each candidate cell (buildings within 500 m)
    dom = np.zeros(len(ii), int)
    if len(ii):
        cnt = np.zeros(shape + (n_rivers,))
        sel = b_d <= min(500.0, max_dist)
        np.add.at(cnt, (bix[sel], biy[sel], b_code[sel]), 1.0)
        dom = cnt[ii, jj].argmax(axis=1)
    cx, cy = x0 + (ii + 0.5) * cell, y0 + (jj + 0.5) * cell
    lon, lat = to_ll(cx, cy)
    cells = []
    for k in range(len(ii)):
        cells.append([int(ii[k]), int(jj[k]), round(float(lon[k]), 5), round(float(lat[k]), 5), int(dom[k]),
                      B[:, ii[k], jj[k]].astype(int).tolist(),
                      np.round(P[:, ii[k], jj[k]]).astype(int).tolist(),
                      np.clip(np.round(Z[:, ii[k], jj[k]] * 10), -999, 999).astype(int).tolist()])
    return {"cell_m": cell, "T": T.astype(int).tolist(), "zmin": zmin, "n_domain": int(domain.sum()), "cells": cells}


def by_river(b_d, b_code, b_area, p_d, p_pop, p_code, r_d, r_len, r_code, keys, names, lengths_km, max_dist=1000.0, step=50.0):
    """Cumulative exposure (buildings, footprint m2, people, road m) for each river at every T."""
    T = t_grid(max_dist, step)
    out = []
    for c, key in enumerate(keys):
        sb, sp, sr = b_code == c, (p_code == c) if len(p_d) else np.zeros(0, bool), (r_code == c) if len(r_d) else np.zeros(0, bool)
        bd, ba = b_d[sb], b_area[sb]
        o = np.argsort(bd)
        cum_a = np.concatenate([[0.0], np.cumsum(ba[o])])
        ib = np.searchsorted(bd[o], T, side="right")
        row = {"key": key, "name": names[key], "len_km": round(float(lengths_km[key]), 2),
               "b": ib.astype(int).tolist(), "fp": np.round(cum_a[ib]).astype(int).tolist()}
        if len(p_d):
            pd_, pp_ = p_d[sp], p_pop[sp]
            o = np.argsort(pd_)
            cp = np.concatenate([[0.0], np.cumsum(pp_[o])])
            row["p"] = np.round(cp[np.searchsorted(pd_[o], T, side="right")]).astype(int).tolist()
        if len(r_d):
            rd_, rl_ = r_d[sr], r_len[sr]
            o = np.argsort(rd_)
            cr = np.concatenate([[0.0], np.cumsum(rl_[o])])
            row["r"] = np.round(cr[np.searchsorted(rd_[o], T, side="right")]).astype(int).tolist()
        out.append(row)
    return out
