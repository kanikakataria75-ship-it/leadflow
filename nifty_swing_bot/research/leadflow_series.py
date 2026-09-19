"""Persist the time series the terminal renders, under the frozen configuration.

``leadflow_frozen_results.json`` carries summary statistics only. Eight of the
things the Backtest and Book pages draw are series, not scalars, and none of
them were stored: the strategy and book equity curves, the benchmark overlay,
the drawdown curve, rolling 12-month return, the daily-return distribution, the
sleeve allocation over time, deployment, the rebalance dates, and the
trade-level R distribution.

This module regenerates the frozen run and writes those series out, then
**reconciles** the regenerated summary against the stored one. A silent
divergence between the numbers a page draws and the numbers the research file
reports is exactly the failure this project has already been bitten by twice
(the quarterly-rebalance book, the orphaned ladder grid), so the reconciliation
is part of the artefact rather than a thing done once by hand.

Only the frozen ladder is persisted. The old-ladder arm stays in the research
file as a comparison and is not something the terminal should ever render as if
it were live.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from ..aes.locked import assert_live_config, assert_locked
from ..aes.portfolio import AESPortfolioBacktester
from ..aes.scoring import attach_score
from ..aes_scanner import live_config as LC
from ..book.gold import load_gold
from ..book.metrics import drawdown_series, positions_from_rows, summarise
from ..book.model import correlation_by_year, simulate
from ..config import get_config
from .aes_calibration import load_universe
from .gamma_study import EQUAL_WEIGHTS, WINDOWS, signals_for_window
from .geometry_sweep import _truncate

#: Metrics checked against the stored research file. A mismatch beyond this
#: tolerance means the page and the paper would disagree, and that is a stop.
RECONCILE_KEYS = (
    "total_return_pct", "cagr_pct", "sharpe", "sortino", "calmar",
    "max_drawdown_pct", "volatility_pct", "win_rate_pct", "avg_r_multiple",
)
TOL = 1e-6


def _round(xs, dp: int = 4) -> list[float]:
    return [None if x is None or (isinstance(x, float) and np.isnan(x))
            else round(float(x), dp) for x in xs]


def build_window(frames, index_df, gold, start: str, end: str) -> dict[str, Any]:
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    raw = signals_for_window(frames, index_df, t0, t1, None)
    scored = [attach_score(s, EQUAL_WEIGHTS) for s in raw]

    ft = _truncate(frames, t1)
    icut = int(index_df.index.searchsorted(t1, side="right")) + 40
    idx = index_df.iloc[:icut]

    res = AESPortfolioBacktester(params=LC.PORTFOLIO).run(
        ft, idx, scored, min_bucket="wait_and_watch"
    )
    eq = res.equity_curve
    strat_eq = eq["equity"].astype(float)
    trades = res.trades_frame

    bk = simulate(strat_eq, eq["positions_value"].astype(float), gold, LC.BOOK)
    curve = bk.curve
    bench = idx["close"].astype(float).reindex(curve.index).ffill()

    # --- series ---------------------------------------------------------- #
    strat_norm = strat_eq / float(strat_eq.iloc[0]) * 100.0
    book_norm = curve["book"] / float(curve["book"].iloc[0]) * 100.0
    bench_norm = bench / float(bench.dropna().iloc[0]) * 100.0
    dd_strat = drawdown_series(strat_eq) * 100.0
    dd_book = drawdown_series(curve["book"]) * 100.0

    daily = strat_eq.pct_change().dropna() * 100.0
    # rolling 12-month (252 trading days) return, in percent
    roll = (strat_eq / strat_eq.shift(252) - 1.0) * 100.0

    rebal_dates = [str(d.date()) for d in curve.index[curve["rebalanced"].astype(bool)]] \
        if "rebalanced" in curve else []

    pos = positions_from_rows(trades) if not trades.empty else pd.DataFrame()
    r_mult = (pos["r_multiple"].astype(float).tolist()
              if not pos.empty and "r_multiple" in pos else [])

    # deployment as a fraction OF THE SLEEVE, which is the 30% target's own
    # basis -- the curve stores it as a fraction of the book.
    deploy_book = curve["deployment"].astype(float) * 100.0

    out: dict[str, Any] = {
        "start": start, "end": end,
        "dates": [str(d.date()) for d in curve.index],
        "strategy_index": _round(strat_norm.reindex(curve.index).ffill(), 3),
        "book_index": _round(book_norm, 3),
        "benchmark_index": _round(bench_norm, 3),
        "drawdown_strategy_pct": _round(dd_strat.reindex(curve.index).ffill(), 3),
        "drawdown_book_pct": _round(dd_book, 3),
        "sleeve_strategy": _round(curve["strategy"], 2),
        "sleeve_arb": _round(curve["arb"], 2),
        "sleeve_gold": _round(curve["gold"], 2),
        "book_value": _round(curve["book"], 2),
        "deployment_pct_of_book": _round(deploy_book, 3),
        "rebalance_dates": rebal_dates,
        "rolling_12m_pct": {
            "dates": [str(d.date()) for d in roll.dropna().index],
            "values": _round(roll.dropna(), 3),
        },
        "daily_returns_pct": _round(daily, 4),
        "r_multiples": _round(r_mult, 4),
        "trades": (trades.to_dict("records") if not trades.empty else []),
        "gold_correlation_by_year": correlation_by_year(bk, gold).round(4).to_dict("records"),
        "n_positions": int(trades.groupby(["symbol", "entry_date"]).ngroups)
                       if not trades.empty else 0,
        "n_signals": len(scored),
    }

    # --- the summary, recomputed, for reconciliation ---------------------- #
    out["_summary"] = {
        "strategy": summarise(strat_eq, trades, bench,
                              eq["exposure"] if "exposure" in eq else None),
        "book": summarise(curve["book"], trades, bench, curve["deployment"]),
    }
    return out


def reconcile(built: dict[str, Any], stored_path: str) -> dict[str, Any]:
    """Compare the regenerated summary against the stored research file."""
    with open(stored_path, encoding="utf-8") as fh:
        stored = json.load(fh)
    report: dict[str, Any] = {}
    for wname, w in built.items():
        s = stored["windows"][wname]["ladders"]["variant_c_12_20"]
        rows = []
        for side in ("strategy", "book"):
            for k in RECONCILE_KEYS:
                a = w["_summary"][side].get(k)
                b = s[side].get(k)
                if a is None or b is None:
                    continue
                ok = abs(float(a) - float(b)) <= TOL
                rows.append({"side": side, "metric": k, "series": float(a),
                             "research_file": float(b), "match": ok})
        report[wname] = {
            "checked": len(rows),
            "mismatches": [r for r in rows if not r["match"]],
            "all_match": all(r["match"] for r in rows),
            "rows": rows,
        }
    return report


def main() -> dict[str, Any]:
    assert_live_config()
    assert_locked(score=EQUAL_WEIGHTS)
    cfg = get_config()
    ohlcv, results = str(cfg.paths.ohlcv_dir), str(cfg.paths.results_dir)
    frames = load_universe(ohlcv)
    index_df = pd.read_parquet(os.path.join(ohlcv, "CRSLDX.parquet"))
    gold, _ = load_gold(os.path.join(ohlcv, "GOLDBEES.parquet"))

    built: dict[str, Any] = {}
    for wname, (start, end, _folds) in WINDOWS.items():
        print(f"[series] {wname}", flush=True)
        built[wname] = build_window(frames, index_df, gold, start, end)
        print(f"   {len(built[wname]['dates'])} days, "
              f"{built[wname]['n_positions']} positions", flush=True)

    rep = reconcile(built, os.path.join(results, "leadflow_frozen_results.json"))
    for wname, r in rep.items():
        print(f"[reconcile] {wname}: {r['checked']} metrics, "
              f"{'ALL MATCH' if r['all_match'] else str(len(r['mismatches'])) + ' MISMATCH'}",
              flush=True)

    payload = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "frozen_on": str(LC.FROZEN_ON),
        "config": LC.describe(),
        "ladder": "variant_c_12_20",
        "reconciliation": rep,
        "windows": {k: {kk: vv for kk, vv in v.items() if kk != "_summary"}
                    for k, v in built.items()},
        "summary": {k: v["_summary"] for k, v in built.items()},
    }
    path = os.path.join(results, "leadflow_series.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=str, separators=(",", ":"))
    payload["_path"] = path
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.2f} MB)", flush=True)
    return payload


__all__ = ["RECONCILE_KEYS", "build_window", "main", "reconcile"]
