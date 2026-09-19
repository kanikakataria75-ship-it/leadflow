"""Early Sector Rotation blueprint, §5.1 + §5.2 + §3/§9 only, as a feature test.

Scope is deliberately narrow. The blueprint proposes nine phases and a
standalone system; this builds the minimum needed to test its **core claim**
against the AES signal population that already exists, which is a week of work
rather than nine phases.

Built:  §5.1 price/RS layer, §5.2 breadth layer, §3 lifecycle states via the
        §9 classifier, §28 lead-time.
Skipped and why:
  §5.3 delivery/participation -- REJECTED #1: six formulations across two
        universes, no edge in any, and as a filter it turned +0.715% into
        -0.380% while cutting the sample 83%.
  §5.4 derivatives -- no clean free historical OI series for this universe.
  §5.5 volatility, §11-18 -- out of scope until the core claim survives.

**Sector indices are built from the AES universe's own constituents**, equal
weighted, rather than from NSE's published sector indices. Three reasons: the
breadth layer needs constituents anyway so the index and the breadth measure
stay consistent; the AES universe is mid/smallcap, so published (large-cap
dominated) sector indices would describe a different population from the one
the signals are drawn from; and it avoids introducing a second data source with
its own corporate-action history.

**The survivorship limitation is unchanged and applies here too.** These are
today's index constituents, so a sector's early-period breadth is computed over
the subset that survived. STATE.md §1 quantifies it: 53% of the universe existed
in 2010, 59% in 2015, 80% in 2021.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MIN_CONSTITUENTS = 8
STATES = ["Dormant", "Accumulating", "Emerging", "Leader", "Crowded", "Fading"]


# --------------------------------------------------------------------------- #
# Sector indices and breadth
# --------------------------------------------------------------------------- #
def build_sector_panel(
    frames: dict[str, pd.DataFrame],
    sector_of: dict[str, str],
    min_constituents: int = MIN_CONSTITUENTS,
) -> dict[str, pd.DataFrame]:
    """One frame per sector: equal-weight index plus breadth, all causal.

    The index is the mean of constituents' closes rebased to their own first
    observation, so a name joining later cannot create a jump: it enters the
    average at 1.0 on its first bar.
    """
    out: dict[str, pd.DataFrame] = {}
    by_sector: dict[str, list[str]] = {}
    for sym, sec in sector_of.items():
        if sym in frames and isinstance(sec, str):
            by_sector.setdefault(sec, []).append(sym)

    for sec, members in sorted(by_sector.items()):
        if len(members) < min_constituents:
            continue
        norm, a20, a50 = [], [], []
        for m in members:
            c = frames[m]["close"]
            norm.append(c / c.iloc[0])
            a20.append((c > c.rolling(20).mean()).astype(float))
            a50.append((c > c.rolling(50).mean()).astype(float))
        idx = pd.concat(norm, axis=1).mean(axis=1, skipna=True)
        df = pd.DataFrame({
            "index": idx,
            "pct_above_20dma": pd.concat(a20, axis=1).mean(axis=1, skipna=True) * 100,
            "pct_above_50dma": pd.concat(a50, axis=1).mean(axis=1, skipna=True) * 100,
            "n_constituents": pd.concat(norm, axis=1).notna().sum(axis=1),
        })
        out[sec] = df.dropna(subset=["index"])
    return out


def _slope(s: pd.Series, window: int) -> pd.Series:
    """Per-bar change over ``window``, in the series' own units."""
    return (s - s.shift(window)) / window


def sector_features(
    panel: dict[str, pd.DataFrame],
    bench_close: pd.Series,
    rs_short: int = 20,
    rs_long: int = 60,
) -> dict[str, pd.DataFrame]:
    """§5.1 relative strength and §5.2 breadth, each as level, slope and acceleration.

    Every column at date T uses only data available by the close of T. Forward
    returns are attached elsewhere and always start after T.
    """
    out = {}
    for sec, df in panel.items():
        b = bench_close.reindex(df.index).ffill()
        f = pd.DataFrame(index=df.index)

        # --- §5.1 price / relative strength ---
        f["ret_20"] = df["index"] / df["index"].shift(20) - 1.0
        f["ret_60"] = df["index"] / df["index"].shift(60) - 1.0
        f["excess_20"] = f["ret_20"] - (b / b.shift(20) - 1.0)
        f["excess_60"] = f["ret_60"] - (b / b.shift(60) - 1.0)
        rs = df["index"] / b
        f["rs"] = rs
        f["rs_slope"] = _slope(rs, rs_short) / rs.shift(rs_short) * 100    # %/bar
        f["rs_slope_long"] = _slope(rs, rs_long) / rs.shift(rs_long) * 100
        f["rs_accel"] = f["rs_slope"] - f["rs_slope"].shift(rs_short)

        # --- §5.2 breadth ---
        for col, tag in (("pct_above_20dma", "b20"), ("pct_above_50dma", "b50")):
            f[tag] = df[col]
            f[f"{tag}_slope"] = _slope(df[col], rs_short)
            f[f"{tag}_accel"] = f[f"{tag}_slope"] - f[f"{tag}_slope"].shift(rs_short)
        f["n_constituents"] = df["n_constituents"]
        out[sec] = f
    return out


# --------------------------------------------------------------------------- #
# Cross-sectional percentile ranks
# --------------------------------------------------------------------------- #
RANK_COLS = [
    "excess_20", "excess_60", "rs_slope", "rs_accel",
    "b20", "b50", "b20_slope", "b20_accel", "b50_slope", "b50_accel",
]


def rank_panel(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long frame of cross-sectional percentile ranks, one row per sector-day.

    Ranked **across sectors within each day**, per the blueprint's §8 note that
    percentile scoring is more robust than raw values when sectors differ in
    volatility. A same-day cross-sectional rank uses no future information.
    """
    rows = []
    for sec, f in features.items():
        g = f.copy()
        g["sector"] = sec
        g["date"] = g.index
        rows.append(g)
    long = pd.concat(rows, ignore_index=True)
    for c in RANK_COLS:
        long[f"r_{c}"] = long.groupby("date")[c].rank(pct=True) * 100
    return long


# --------------------------------------------------------------------------- #
# §9 state classifier -- fixed rules, equal weights, nothing fitted
# --------------------------------------------------------------------------- #
def classify(long: pd.DataFrame, hi: float = 60.0, lo: float = 40.0) -> pd.DataFrame:
    """Assign one §3 lifecycle state per sector-day.

    The rules are the blueprint's §9 text turned into percentile comparisons,
    with two fixed cut points (60/40) applied to every term. They are not tuned:
    no threshold here was chosen by looking at an outcome, and the composite
    scores that feed them are equal-weighted averages of ranks, per §8's
    "start with equal weights or simple rank aggregation".

    Order matters because the states are not mutually exclusive as written;
    the sequence below is the lifecycle order from §3, so a sector that
    qualifies as both Crowded and Leader is called Crowded (the later state).
    """
    d = long
    momentum = d[["r_excess_20", "r_excess_60"]].mean(axis=1)
    rs_improving = d[["r_rs_slope", "r_rs_accel"]].mean(axis=1)
    breadth_level = d[["r_b20", "r_b50"]].mean(axis=1)
    breadth_improving = d[["r_b20_slope", "r_b20_accel",
                           "r_b50_slope", "r_b50_accel"]].mean(axis=1)
    accel = d[["r_rs_accel", "r_b20_accel", "r_b50_accel"]].mean(axis=1)

    out = pd.Series("Dormant", index=d.index, dtype=object)
    # Fading: deteriorating internals on both axes
    out[(breadth_improving < lo) & (d["r_rs_accel"] < lo)] = "Fading"
    # Accumulating: low/moderate momentum, but RS and breadth turning up
    out[(momentum < hi) & (rs_improving > hi) & (breadth_improving > hi)] = "Accumulating"
    # Emerging: strong acceleration + broadening participation + improving RS
    out[(accel > hi) & (breadth_improving > hi) & (rs_improving > hi)
        & (momentum < 80.0)] = "Emerging"
    # Leader: high momentum + high breadth + established outperformance
    out[(momentum > hi) & (breadth_level > hi) & (d["r_excess_60"] > hi)] = "Leader"
    # Crowded: high momentum but acceleration slowing and/or breadth narrowing
    out[(momentum > hi) & ((accel < lo) | (breadth_improving < lo))] = "Crowded"

    res = d.copy()
    res["state"] = out.to_numpy()
    res["momentum_pct"] = momentum.to_numpy()
    res["rs_improving_pct"] = rs_improving.to_numpy()
    res["breadth_level_pct"] = breadth_level.to_numpy()
    res["breadth_improving_pct"] = breadth_improving.to_numpy()
    res["accel_pct"] = accel.to_numpy()
    # §7 Silent Rotation: internals improving while headline momentum is ordinary
    res["silent_rotation"] = (
        (res["accel_pct"] > 70.0)
        & (res["breadth_improving_pct"] > 70.0)
        & (res["rs_improving_pct"] > 60.0)
        & (res["momentum_pct"] < 60.0)
    )
    return res


# --------------------------------------------------------------------------- #
# §28 lead time
# --------------------------------------------------------------------------- #
def lead_time(
    states: pd.DataFrame,
    horizon: int = 60,
    top_decile: float = 90.0,
) -> dict[str, Any]:
    """How far ahead of "obvious momentum" does Emerging fire?

    Obvious momentum is the blueprint's own definition: the sector's 20-day
    excess return entering the **top decile cross-sectionally**. For each
    Emerging onset where momentum is not already obvious, the lead time is the
    number of sessions until it becomes obvious within ``horizon``; onsets that
    never get there inside the horizon are counted as false positives rather
    than dropped, because dropping them would measure only the successes.
    """
    leads, fp, total = [], 0, 0
    for sec, g in states.sort_values("date").groupby("sector"):
        g = g.reset_index(drop=True)
        obvious = (g["r_excess_20"] >= top_decile).to_numpy()
        st = g["state"].to_numpy()
        onset = np.r_[False, (st[1:] == "Emerging") & (st[:-1] != "Emerging")]
        for i in np.flatnonzero(onset):
            if obvious[i]:
                continue          # already obvious: not an early signal
            total += 1
            fut = obvious[i + 1 : i + 1 + horizon]
            hit = np.flatnonzero(fut)
            if hit.size:
                leads.append(int(hit[0] + 1))
            else:
                fp += 1
    a = np.array(leads, dtype=float)
    return {
        "n_onsets": total,
        "n_reached_obvious": len(leads),
        "false_positive_pct": round(fp / total * 100, 1) if total else None,
        "mean_lead_days": round(float(a.mean()), 1) if a.size else None,
        "median_lead_days": round(float(np.median(a)), 1) if a.size else None,
        "p25_lead_days": round(float(np.percentile(a, 25)), 1) if a.size else None,
        "p75_lead_days": round(float(np.percentile(a, 75)), 1) if a.size else None,
        "horizon": horizon,
    }


__all__ = [
    "MIN_CONSTITUENTS", "RANK_COLS", "STATES", "build_sector_panel",
    "classify", "lead_time", "rank_panel", "sector_features",
]
