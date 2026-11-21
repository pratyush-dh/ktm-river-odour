"""Run: python test_odour_core.py  (numpy/pandas/scipy only)"""
import numpy as np
import pandas as pd

import odour_core as oc


def test_canonical():
    assert oc.canonical_river("Vishnumati River") == "bishnumati"
    assert oc.canonical_river("Nakhu Khola") == "nakkhu"
    assert oc.canonical_river("Bagmati") == "bagmati"
    assert oc.canonical_river("") == ""


def test_curve():
    d = np.array([10, 60, 60, 400, np.nan, 900.0])
    p = np.array([5, 10, 10, 20, 99, 1.0])
    c = oc.cumulative_curve(d, p, np.array([0, 10, 50, 60, 399, 400, 1000.0]))
    assert c.cum_pop.tolist() == [0, 5, 5, 25, 25, 45, 46]
    try:
        oc.cumulative_curve(d, p[:-1], np.array([1.0]))
        raise AssertionError("shape mismatch not caught")
    except ValueError:
        pass


def synth(n=600, seed=1, b=(5.0, -1.0, 0.8)):
    rng = np.random.default_rng(seed)
    d = rng.uniform(0, 1500, n)
    night = rng.integers(0, 2, n)
    z = b[0] + b[1] * np.log1p(d) + b[2] * night
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"dist_m": d, "night": night, "detected": y, "cl": rng.integers(0, 30, n)})


def test_recovery():
    df = synth()
    fit, tab = oc.calibrate(df, covars=["night"], n_boot=300, cluster_col="cl")
    assert abs(fit.beta[1] - (-1.0)) < 0.15, fit.beta
    # truth: p=0.5 at log1p(d)=5 (day) -> 147 m ; night 5.8 -> 329 m
    day = tab[(tab.scenario == "night=0") & (tab.p_detect == 0.5)].iloc[0]
    night = tab[(tab.scenario == "night=1") & (tab.p_detect == 0.5)].iloc[0]
    assert day.ci_lo_m < 147 < day.ci_hi_m, day
    assert night.ci_lo_m < 329 < night.ci_hi_m, night
    assert night.dist_m > day.dist_m
    assert fit.auc > 0.7


def test_refuses_small_or_flat():
    df = synth(n=30)
    try:
        oc.calibrate(df, covars=["night"])
        raise AssertionError("should refuse small n")
    except ValueError as e:
        assert "Too few" in str(e)
    df = synth()
    df["detected"] = 0
    try:
        oc.calibrate(df, covars=["night"])
        raise AssertionError("should refuse single class")
    except ValueError:
        pass


def test_no_decay():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"dist_m": rng.uniform(0, 1000, 400), "detected": rng.integers(0, 2, 400)})
    fit, tab = oc.calibrate(df, n_boot=200)
    # no real relationship: either nan (slope >= 0) or very wide CI; must not claim a tight reach
    r = tab[tab.p_detect == 0.5].iloc[0]
    assert np.isnan(r.dist_m) or (r.ci_hi_m - r.ci_lo_m) > 300 or np.isnan(r.ci_lo_m), r


def test_time_covariates():
    df = pd.DataFrame({"datetime": ["2026-07-01 22:00", "2026-12-01 12:00", "2026-12-02 05:30"]})
    o = oc.add_time_covariates(df)
    assert o.night.tolist() == [1, 0, 1] and o.monsoon.tolist() == [1, 0, 0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
