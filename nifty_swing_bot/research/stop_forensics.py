"""Phase 24 diagnostic: what happened AFTER a stop exit. Measurement only.

No adoption question, no baseline comparison, no config change. This module
describes the forward behaviour of names the engine stopped out of, and stops
there.

Two facts about the population, established before any number is produced
------------------------------------------------------------------------
1. **The frozen config produces zero ``sma10_close_stop`` exits.** The SMA(10)
   close stop is the *spec* stop. It was superseded by ATR(14) x 2.5 in Phase
   15 (PROVEN #1), and ``live_config.PORTFOLIO`` sets ``atr_stop_mult=2.5``,
   so the SMA branch in ``portfolio.py`` is unreachable under the live
   configuration. Its stop exits are ``atr_trail_stop`` and
   ``big_box_bottom_stop``. A diagnostic of the SMA(10) stop is therefore a
   post-mortem on a stop the system no longer uses.

2. **Stored history covers one window, not two.**
   ``results/aes_portfolio_trades.csv`` holds 187 ``sma10_close_stop`` exits,
   but 177 of them are in 2015-2021 and **10** are in 2022-2026. "Both windows
   separately" is not answerable from what is on disk.

So the exit population is regenerated here for both windows, under an
explicitly declared configuration, and the 2015-2021 result is cross-checked
against the stored file. Producing exit events requires the simulator; that is
not a strategy run and nothing is compared against a baseline.

Both stops are harvested: ``spec`` (the SMA(10) stop, which is what was asked
for) and ``frozen`` (the ATR trail, which is the stop actually in use). The
second is an addendum, not a substitution.

The marginality split
---------------------
"Closed just below the stop" vs "closed meaningfully below" is defined in units
of the name's own ATR(14) on the exit bar, because a fixed percentage means
different things for a 2%-ATR name and an 8%-ATR name:

    depth_atr = (stop_level - exit_close) / ATR14(exit_bar)

    MARGINAL          0 < depth_atr <= 0.5
    CLEAN BREAKDOWN   depth_atr > 0.5

The raw percentage below the stop is reported alongside so the cut can be
sanity-checked against something unnormalised.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ..aes.locked import assert_live_config, assert_locked
from ..aes.params import PortfolioParams
from ..aes.portfolio import AESPortfolioBacktester
from ..aes.scoring import attach_score
from ..aes_scanner import live_config as LC
from ..config import get_config
from ..strategy.indicators import atr
from .aes_calibration import load_universe
from .gamma_study import EQUAL_WEIGHTS, WINDOWS, signals_for_window
from .geometry_sweep import _truncate

HORIZONS = (5, 10, 15, 20, 25, 30)
MAX_WINDOW = 40
THRESHOLDS = (10.0, 20.0, 50.0, 100.0)
MARGINAL_ATR = 0.5

#: The two stop regimes harvested. ``spec`` is the SMA(10) close stop, the
#: subject of the request; ``frozen`` is the live ATR trail.
REGIMES: dict[str, tuple[PortfolioParams, tuple[str, ...]]] = {
    "spec_sma10": (PortfolioParams(), ("sma10_close_stop",)),
    "frozen_atr": (LC.PORTFOLIO, ("atr_trail_stop",)),
}


def harvest_exits(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    scored: list,
    end: str,
    pp: PortfolioParams,
    reasons: tuple[str, ...],
) -> pd.DataFrame:
    """Every exit matching ``reasons``, with the stop level recomputed.

    The engine's trade record does not carry the stop level or the ATR at the
    exit bar, and the marginality split needs both. They are recomputed here
    from the same price frames the engine read, using the same definitions the
    exit rule uses, rather than being inferred from the exit price.
    """
    t1 = pd.Timestamp(end)
    ft = _truncate(frames, t1)
    icut = int(index_df.index.searchsorted(t1, side="right")) + 40
    res = AESPortfolioBacktester(params=pp).run(
        ft, index_df.iloc[:icut], scored, min_bucket="wait_and_watch"
    )
    tf = res.trades_frame
    if tf.empty:
        return pd.DataFrame()
    tf = tf[tf["exit_reason"].isin(reasons)].copy()
    if tf.empty:
        return tf

    rows = []
    for r in tf.itertuples():
        df = frames.get(r.symbol)
        if df is None:
            continue
        ed = pd.Timestamp(r.exit_date)
        if ed not in df.index:
            continue
        i = int(df.index.get_loc(ed))
        close = df["close"]
        a = atr(df["high"], df["low"], close, pp.atr_stop_period)
        atr_now = float(a.iloc[i]) if np.isfinite(a.iloc[i]) else np.nan

        if pp.atr_stop_mult is None:
            sma = close.rolling(pp.sma_stop_period).mean()
            level = float(sma.iloc[i]) * (1 - pp.sma_stop_buffer_pct)
        else:
            # the trail level implied by the exit: the engine stopped because
            # close fell below peak_close - mult*ATR, and peak_close is the
            # highest close from entry to exit.
            e0 = df.index.get_loc(pd.Timestamp(r.entry_date))
            peak = float(close.iloc[int(e0): i + 1].max())
            level = peak - pp.atr_stop_mult * atr_now if np.isfinite(atr_now) else np.nan

        exit_px = float(r.exit_price)
        depth_atr = (level - exit_px) / atr_now if (np.isfinite(atr_now) and atr_now > 0
                                                    and np.isfinite(level)) else np.nan
        row: dict[str, Any] = {
            "symbol": r.symbol, "exit_date": ed, "exit_price": exit_px,
            "bucket": r.bucket, "score": float(r.score),
            "stop_level": level, "atr14": atr_now,
            "depth_atr": depth_atr,
            "depth_pct": (level - exit_px) / exit_px * 100 if exit_px > 0 else np.nan,
        }
        # forward returns from the exit price
        fwd = close.to_numpy(dtype=float)
        for h in HORIZONS:
            j = i + h
            row[f"fwd_{h}"] = (fwd[j] / exit_px - 1.0) * 100 if j < len(fwd) else np.nan
        seg = fwd[i + 1: i + 1 + MAX_WINDOW]
        if seg.size:
            row["max_fwd_40"] = (seg.max() / exit_px - 1.0) * 100
            row["argmax_bar"] = int(seg.argmax()) + 1
        else:
            row["max_fwd_40"] = np.nan
            row["argmax_bar"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def describe(sub: pd.DataFrame) -> dict[str, Any]:
    """The requested statistics for one cut. ``n`` travels with every figure."""
    out: dict[str, Any] = {"n": int(len(sub))}
    if sub.empty:
        return out
    for h in HORIZONS:
        col = sub[f"fwd_{h}"].dropna()
        d: dict[str, Any] = {"n": int(len(col))}
        if len(col):
            d["mean_pct"] = round(float(col.mean()), 2)
            d["median_pct"] = round(float(col.median()), 2)
            for t in THRESHOLDS:
                d[f"pct_over_{int(t)}"] = round(float((col > t).mean() * 100), 1)
        out[f"h{h}"] = d
    am = sub["argmax_bar"].dropna()
    mx = sub["max_fwd_40"].dropna()
    out["max_within_40"] = {
        "n": int(len(mx)),
        "median_max_pct": round(float(mx.median()), 2) if len(mx) else None,
        "mean_max_pct": round(float(mx.mean()), 2) if len(mx) else None,
        "median_argmax_bar": float(am.median()) if len(am) else None,
        "mean_argmax_bar": round(float(am.mean()), 1) if len(am) else None,
        "argmax_q25": float(am.quantile(0.25)) if len(am) else None,
        "argmax_q75": float(am.quantile(0.75)) if len(am) else None,
    }
    out["median_depth_atr"] = round(float(sub["depth_atr"].median()), 3)
    out["median_depth_pct"] = round(float(sub["depth_pct"].median()), 2)
    return out


def splits(ex: pd.DataFrame) -> dict[str, Any]:
    """The two requested splits, each reported with its own sample sizes."""
    if ex.empty:
        return {"all": {"n": 0}}
    hi = ex[ex["bucket"] == "high_conviction"]
    rest = ex[ex["bucket"] != "high_conviction"]
    marg = ex[(ex["depth_atr"] > 0) & (ex["depth_atr"] <= MARGINAL_ATR)]
    clean = ex[ex["depth_atr"] > MARGINAL_ATR]
    odd = ex[~((ex["depth_atr"] > 0) & np.isfinite(ex["depth_atr"]))]
    return {
        "all": describe(ex),
        "by_score": {
            "high_conviction": describe(hi),
            "wait_and_watch": describe(rest),
        },
        "by_marginality": {
            f"marginal_le_{MARGINAL_ATR}atr": describe(marg),
            f"clean_breakdown_gt_{MARGINAL_ATR}atr": describe(clean),
            "unclassified": {"n": int(len(odd))},
        },
    }


def main(out_name: str = "stop_forensics.json") -> dict[str, Any]:
    assert_live_config()
    assert_locked(score=EQUAL_WEIGHTS)
    cfg = get_config()
    ohlcv, results = str(cfg.paths.ohlcv_dir), str(cfg.paths.results_dir)
    frames = load_universe(ohlcv)
    index_df = pd.read_parquet(os.path.join(ohlcv, "CRSLDX.parquet"))

    res: dict[str, Any] = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "note": (
            "Diagnostic only. The frozen config emits no sma10_close_stop exits; "
            "the SMA(10) stop is the spec stop, superseded by ATR 2.5x (PROVEN #1). "
            "Exit populations are regenerated under declared configs."
        ),
        "marginal_rule": f"0 < (stop - close)/ATR14 <= {MARGINAL_ATR}",
        "windows": {},
    }

    for wname, (start, end, _folds) in WINDOWS.items():
        raw = signals_for_window(frames, index_df, pd.Timestamp(start), pd.Timestamp(end), None)
        scored = [attach_score(s, EQUAL_WEIGHTS) for s in raw]
        print(f"[{wname}] {len(scored)} signals", flush=True)
        res["windows"][wname] = {}
        for rname, (pp, reasons) in REGIMES.items():
            if pp.atr_stop_mult is None:
                assert_locked(
                    portfolio=pp,
                    deviations={"atr_stop_mult", "max_hold_bars"},
                    reason="Phase 24 diagnostic: harvesting the spec SMA(10) stop population",
                )
            ex = harvest_exits(frames, index_df, scored, end, pp, reasons)
            print(f"   {rname}: {len(ex)} exits", flush=True)
            res["windows"][wname][rname] = splits(ex)
            if not ex.empty:
                ex.to_csv(os.path.join(results, f"stop_exits_{wname}_{rname}.csv"), index=False)

    path = os.path.join(results, out_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    res["_path"] = path
    return res


__all__ = ["HORIZONS", "MARGINAL_ATR", "REGIMES", "describe", "harvest_exits", "main", "splits"]
