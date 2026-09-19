"""The backtest results section: three views, each strategy-only beside total-book.

Two honesty requirements are structural here rather than cosmetic, because the
UI renders whatever this module emits:

1. **Provenance.** Every window carries ``provenance`` -- ``"discovery"`` or
   ``"validated"``. As of Phase 16 nothing in this project is out-of-sample
   validated: 2015-2021 is the original discovery window and 2022-2026 was
   spent as discovery by the runner study. Both are labelled honestly rather
   than one being dressed up as a test.

2. **Concentration.** Every headline return is emitted with its
   ``ex_best_period`` twin and the best period's share of the total. In both
   windows a single year is 66-70% of the summed return; a headline shown
   alone is a lie by omission, so the pair travels together.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .metrics import compounded_return_on_dates, drawdown_episodes, summarise
from .model import simulate
from .params import BookParams

SMALLCAP_PEAK = pd.Timestamp("2024-09-24")   # §7.4's Nifty Smallcap 250 peak


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #
def concentration(returns: list[float]) -> dict[str, Any]:
    """Headline mean beside the mean with the best period removed."""
    r = np.array([x for x in returns if np.isfinite(x)], dtype=float)
    if len(r) < 2:
        return {}
    total = float(r.sum())
    best = float(r.max())
    return {
        "mean": round(float(r.mean()), 2),
        "median": round(float(np.median(r)), 2),
        "ex_best_period": round(float(r[r != best].mean()), 2),
        "ex_top2": round(float(np.sort(r)[:-2].mean()), 2) if len(r) > 2 else None,
        "best_period_share_pct": round(best / total * 100, 1) if abs(total) > 1e-9 else None,
        "positive": f"{int((r > 0).sum())}/{len(r)}",
    }


# --------------------------------------------------------------------------- #
# Market phase classification
# --------------------------------------------------------------------------- #
def annual_breadth(frames: dict[str, pd.DataFrame], ma: int = 200) -> pd.Series:
    """Median share of the universe above its own ``ma``-day average, per year.

    A participation measure, not an index level: a year where the index rises on
    a handful of names scores low. Used only to *group* years for reporting --
    §4a and REJECTED #10 established breadth is not tradeable as a gate, and
    nothing here treats it as one.
    """
    cols = {}
    for sym, df in frames.items():
        c = df["close"]
        cols[sym] = (c > c.rolling(ma).mean()).astype(float)
    panel = pd.DataFrame(cols)
    daily = panel.mean(axis=1, skipna=True) * 100
    return daily.groupby(daily.index.year).median()


def phase_label(year: int, breadth: pd.Series, cut: float) -> str:
    b = breadth.get(year)
    if b is None or not np.isfinite(b):
        return "unknown"
    return "broad" if b >= cut else "narrow"


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def _slice_metrics(
    curve: pd.DataFrame,
    strat_eq: pd.Series,
    bench: pd.Series,
    trades: pd.DataFrame | None,
    idx: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Strategy-only and total-book metrics over one date slice.

    ``idx`` need not be contiguous (e.g. "all narrow-breadth years" skips the
    broad years in between). Most of ``performance()`` -- volatility, Sharpe,
    drawdown episodes -- is computed correctly on a row-filtered slice because
    it only ever looks at consecutive *rows* of the filtered series, which are
    consecutive *trading days within the group* by construction. The one
    exception is total/CAGR-style return, which naively computed as
    ``eq.iloc[-1] / eq.iloc[0] - 1`` silently re-includes every day excluded
    from the middle of a non-contiguous set. That figure is overridden here
    with the correctly compounded version.
    """
    c = curve.loc[idx]
    s = strat_eq.loc[idx]
    tr = None
    if trades is not None and not trades.empty and "entry_date" in trades:
        ed = pd.to_datetime(trades["entry_date"])
        tr = trades[(ed >= idx[0]) & (ed <= idx[-1])]

    strat_m = summarise(s, tr, bench, (c["positions"] / c["strategy"]).clip(0, 5))
    book_m = summarise(c["book"], tr, bench, c["deployment"])
    strat_dr = strat_eq.pct_change()
    book_dr = curve["book"].pct_change()
    strat_m["total_return_pct"] = compounded_return_on_dates(strat_dr, idx)
    book_m["total_return_pct"] = compounded_return_on_dates(book_dr, idx)
    return {
        "strategy": strat_m,
        "book": book_m,
        "start": str(idx[0].date()),
        "end": str(idx[-1].date()),
        "days": len(idx),
    }


def year_view(curve, strat_eq, bench, trades) -> list[dict[str, Any]]:
    out = []
    for year, g in curve.groupby(curve.index.year):
        if len(g) < 20:
            continue
        m = _slice_metrics(curve, strat_eq, bench, trades, g.index)
        m["period"] = str(year)
        m["sleeve_returns"] = {
            "strategy_pct": round(float((1 + g["strat_ret"]).prod() - 1) * 100, 2),
            "arb_pct": round(float((1 + g["arb_ret"]).prod() - 1) * 100, 2),
            "gold_pct": round(float((1 + g["gold_ret"]).prod() - 1) * 100, 2),
        }
        m["mean_deployment_pct"] = round(float(g["deployment"].mean() * 100), 2)
        out.append(m)
    return out


def fold_view(curve, strat_eq, bench, trades) -> dict[str, Any]:
    """Annual walk-forward folds, with each fold's share of the summed return."""
    folds = year_view(curve, strat_eq, bench, trades)
    s_rets = [f["strategy"].get("total_return_pct", 0.0) for f in folds]
    b_rets = [f["book"].get("total_return_pct", 0.0) for f in folds]
    s_tot, b_tot = sum(s_rets), sum(b_rets)
    for f, sr, br in zip(folds, s_rets, b_rets):
        f["strategy_share_pct"] = round(sr / s_tot * 100, 1) if abs(s_tot) > 1e-9 else None
        f["book_share_pct"] = round(br / b_tot * 100, 1) if abs(b_tot) > 1e-9 else None
    return {
        "folds": folds,
        "strategy_concentration": concentration(s_rets),
        "book_concentration": concentration(b_rets),
    }


def phase_view(curve, strat_eq, bench, trades, breadth: pd.Series) -> list[dict[str, Any]]:
    """Grouped by market regime rather than by the calendar."""
    years = sorted({int(y) for y in curve.index.year})
    vals = [breadth.get(y) for y in years if np.isfinite(breadth.get(y, np.nan))]
    cut = float(np.median(vals)) if vals else 50.0

    groups: dict[str, list[pd.Timestamp]] = {}
    for y in years:
        lab = phase_label(y, breadth, cut)
        mask = curve.index.year == y
        groups.setdefault(f"{lab}-breadth years", []).extend(curve.index[mask])

    # The 2024-25 smallcap peak-and-correction split, when the window covers it
    if curve.index.min() <= SMALLCAP_PEAK <= curve.index.max():
        pre = curve.index[(curve.index >= pd.Timestamp("2024-01-01")) & (curve.index < SMALLCAP_PEAK)]
        post = curve.index[(curve.index >= SMALLCAP_PEAK) & (curve.index <= pd.Timestamp("2025-12-31"))]
        if len(pre) > 20:
            groups["2024 pre-smallcap-peak"] = list(pre)
        if len(post) > 20:
            groups["post-peak correction"] = list(post)

    out = []
    for name, dates in groups.items():
        if len(dates) < 20:
            continue
        idx = pd.DatetimeIndex(sorted(set(dates)))
        m = _slice_metrics(curve, strat_eq, bench, trades, idx)
        m["period"] = name
        m["breadth_cut"] = round(cut, 1)
        out.append(m)
    return out


# --------------------------------------------------------------------------- #
# Allocation sweep
# --------------------------------------------------------------------------- #
def allocation_sweep(
    strat_eq: pd.Series,
    strat_pos: pd.Series,
    gold: pd.Series,
    base: BookParams | None = None,
    weights: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.70, 1.00),
) -> list[dict[str, Any]]:
    """Strategy weight vs return and drawdown, arbitrage/gold kept in 5:2 ratio.

    Requested so the allocation can be chosen from the tradeoff rather than
    assumed. The remainder splits 5:2 because that is the brief's own 50:20.
    """
    base = base or BookParams()
    out = []
    for w in weights:
        rest = 1.0 - w
        p = replace(base, w_strategy=w, w_arbitrage=rest * 5 / 7, w_gold=rest * 2 / 7)
        r = simulate(strat_eq, strat_pos, gold, p)
        m = summarise(r.curve["book"], None, None, r.curve["deployment"])
        eps = drawdown_episodes(r.curve["book"])
        out.append({
            "w_strategy": round(w, 2),
            "w_arbitrage": round(p.w_arbitrage, 3),
            "w_gold": round(p.w_gold, 3),
            "cagr_pct": round(m.get("cagr_pct", 0.0), 2),
            "volatility_pct": round(m.get("volatility_pct", 0.0), 2),
            "sharpe": round(m.get("sharpe", 0.0), 2),
            "max_drawdown_pct": round(m.get("max_drawdown_pct", 0.0), 2),
            "avg_drawdown_pct": round(float(eps["depth_pct"].mean()) if not eps.empty else 0.0, 2),
            "mean_deployment_pct": round(r.diagnostics["mean_deployment_pct"], 2),
            "borrow_day_pct": round(r.diagnostics["borrow_day_pct"], 1),
        })
    return out


__all__ = [
    "SMALLCAP_PEAK", "allocation_sweep", "annual_breadth", "concentration",
    "fold_view", "phase_view", "year_view",
]
