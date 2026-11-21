"""Pure numpy/pandas core for the Kathmandu river-odour map (no GIS imports).

Two jobs:
  1. cumulative_curve(): exact distance-to-river exposure curve from per-pixel
     distances and population. No distance threshold is chosen.
  2. calibrate(): fit P(odour detected | distance, covariates) from field
     observations, with cluster/stratified bootstrap CIs. Any "odour reach"
     distance comes from this fit, never from an assumed number.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
_RIVER_KEYS = {
    "bagmati": r"bagmati",
    "bishnumati": r"bishnumati|vishnumati",
    "dhobi": r"dhobi",
    "tukucha": r"tukucha|tukuche",
    "hanumante": r"hanumante",
    "manohara": r"manohara",
    "nakkhu": r"nakkhu|nakhu",
    "balkhu": r"balkhu",
}


def canonical_river(name: str) -> str:
    """Map spelling variants (Vishnumati/Bishnumati ...) to one key; else lower-case name."""
    s = (name or "").lower()
    for key, pat in _RIVER_KEYS.items():
        if re.search(pat, s):
            return key
    return s.strip()


# --------------------------------------------------------------------------
# Exposure curve (assumption-free)
# --------------------------------------------------------------------------
def cumulative_curve(dist_m: np.ndarray, pop: np.ndarray, grid_m: np.ndarray) -> pd.DataFrame:
    """Cumulative population within each grid distance.

    dist_m, pop: 1-D arrays, one entry per raster cell (distance from cell
    centre to nearest river line, metres; population count in the cell).
    Cells with non-finite distance or population are dropped, negatives -> 0.
    """
    dist_m = np.asarray(dist_m, float)
    pop = np.asarray(pop, float)
    if dist_m.shape != pop.shape:
        raise ValueError("dist_m and pop must have the same shape")
    ok = np.isfinite(dist_m) & np.isfinite(pop)
    d, p = dist_m[ok], np.clip(pop[ok], 0, None)
    order = np.argsort(d, kind="stable")
    d, cum = d[order], np.cumsum(p[order])
    grid = np.asarray(grid_m, float)
    idx = np.searchsorted(d, grid, side="right")  # cells with d <= grid
    cum_at = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)
    return pd.DataFrame({"dist_m": grid, "cum_pop": cum_at})


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
@dataclass
class Fit:
    beta: np.ndarray
    names: list[str]
    auc: float
    n: int
    n_pos: int


def _design(df: pd.DataFrame, covars: Sequence[str]) -> np.ndarray:
    cols = [np.ones(len(df)), np.log1p(df["dist_m"].to_numpy(float))]
    cols += [df[c].to_numpy(float) for c in covars]
    return np.column_stack(cols)


def _fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """Penalised logistic regression (tiny ridge on slopes guards perfect separation)."""
    def nll(b: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ b
        loss = np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * np.sum(b[1:] ** 2)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y)
        grad[1:] += l2 * b[1:]
        return loss, grad

    res = minimize(nll, np.zeros(X.shape[1]), jac=True, method="BFGS")
    if not res.success and not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"logistic fit failed: {res.message}")
    return res.x


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def dist_at_prob(beta: np.ndarray, p: float, covar_values: Sequence[float]) -> float:
    """Distance (m) where P(detect)=p for given covariate values.

    Returns 0.0 if p is not reached even at the river bank, nan if the fitted
    distance slope is not negative (no decay detected).
    """
    b0, b1, bc = beta[0], beta[1], beta[2:]
    if b1 >= 0:
        return float("nan")
    target = np.log(p / (1 - p))
    log_d = (target - b0 - float(np.dot(bc, covar_values))) / b1
    if log_d > 14:  # > ~1,000 km: effectively unbounded
        return float("inf")
    return float(max(np.expm1(log_d), 0.0))


def calibrate(
    obs: pd.DataFrame,
    covars: Sequence[str] = (),
    probs: Sequence[float] = (0.5, 0.1),
    n_boot: int = 1000,
    cluster_col: Optional[str] = None,
    min_n: int = 60,
    min_class: int = 15,
    seed: int = 42,
) -> tuple[Fit, pd.DataFrame]:
    """Fit detection vs distance and return distance-at-probability with 95% CIs.

    obs columns: dist_m (to nearest river line), detected (0/1), plus covars.
    cluster_col (e.g. transect/session id): resample whole clusters so that
    spatially/temporally correlated observations are not treated as independent.
    Scenarios = every 0/1 combination of the binary covariates in `covars`.
    """
    need = {"dist_m", "detected", *covars}
    miss = need - set(obs.columns)
    if miss:
        raise ValueError(f"missing columns: {sorted(miss)}")
    df = obs.dropna(subset=list(need)).reset_index(drop=True)
    y = df["detected"].to_numpy(float)
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("detected must be 0/1")
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if len(df) < min_n or n_pos < min_class or n_neg < min_class:
        raise ValueError(
            f"Too few observations to calibrate: n={len(df)}, detected={n_pos}, "
            f"not detected={n_neg} (need n>={min_n} and >={min_class} of each). "
            "Collect more field data instead of assuming a distance."
        )
    for c in covars:
        if not set(df[c].unique()) <= {0, 1, 0.0, 1.0}:
            raise ValueError(f"covariate {c} must be binary 0/1")
        if df[c].nunique() < 2:
            raise ValueError(f"covariate {c} has no variation in the data")

    X = _design(df, covars)
    beta = _fit_logit(X, y)
    fit = Fit(beta, ["intercept", "log1p(dist_m)", *covars], _auc(y, X @ beta), len(df), n_pos)

    scenarios = [dict(zip(covars, v)) for v in itertools.product([0, 1], repeat=len(covars))] or [{}]
    rng = np.random.default_rng(seed)
    boots: list[np.ndarray] = []
    if cluster_col and cluster_col in df.columns:
        groups = [g.index.to_numpy() for _, g in df.groupby(cluster_col)]
    else:
        groups = None
    pos_idx, neg_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        if groups is not None:
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in pick])
        else:  # stratified so both classes always present
            idx = np.concatenate([rng.choice(pos_idx, len(pos_idx)), rng.choice(neg_idx, len(neg_idx))])
        yb = y[idx]
        if yb.sum() < 2 or (len(yb) - yb.sum()) < 2:
            continue
        try:
            boots.append(_fit_logit(X[idx], yb))
        except RuntimeError:
            continue

    rows = []
    for sc in scenarios:
        vals = [sc[c] for c in covars]
        for p in probs:
            est = dist_at_prob(beta, p, vals)
            bd = np.array([dist_at_prob(b, p, vals) for b in boots]) if boots else np.array([])
            bd = bd[~np.isnan(bd)]  # inf kept: an unbounded upper CI is a real finding
            lo, hi = (np.percentile(bd, [2.5, 97.5], method="nearest") if len(bd) >= 50 else (np.nan, np.nan))
            rows.append(
                {
                    "scenario": ",".join(f"{k}={v}" for k, v in sc.items()) or "all",
                    "p_detect": p,
                    "dist_m": est,
                    "ci_lo_m": lo,
                    "ci_hi_m": hi,
                    "n_boot_ok": int(len(bd)),
                }
            )
    return fit, pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Observation covariates from timestamps (sampling-design labels, not physics)
# --------------------------------------------------------------------------
def add_time_covariates(
    df: pd.DataFrame, night_start: int = 19, night_end: int = 6, monsoon_months: Sequence[int] = (6, 7, 8, 9)
) -> pd.DataFrame:
    """Add binary night / monsoon columns from `datetime` (local time).

    The hour/month windows are labels YOU choose for stratifying the sample;
    the fitted model estimates their effect, they are not assumed to matter.
    """
    out = df.copy()
    dt = pd.to_datetime(out["datetime"])
    h = dt.dt.hour
    out["night"] = ((h >= night_start) | (h < night_end)).astype(int)
    out["monsoon"] = dt.dt.month.isin(list(monsoon_months)).astype(int)
    return out
