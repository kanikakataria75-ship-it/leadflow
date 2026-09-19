"""Phase 23b: re-entry restricted to high-conviction positions stopped within 3 bars.

Why this population and not every stop-out
------------------------------------------
Phase 23 tested re-entry against **all** stop exits. One config cleared its bar
and then failed attribution: the re-entry positions' own P&L was 5.3% and 3.7%
of the equity gain, the rest being slot reshuffling, and a score-jitter null
reproduced two thirds of the 2022-2026 gain with no new positions at all.

Phase 24b then found the pooled population contains two nearly disjoint groups.
Among 23 high-conviction SMA10 stop-outs:

    stopped <= 3 bars   n=11   ran >= +20% in 40 bars: 7 (64%)   median max 27.87%
    stopped  > 3 bars   n=12   ran >= +20% in 40 bars: 3 (25%)   median max  8.36%

and zero of the fast group matched the compression pattern while all eight
compressed trades were slow grinds. Pooling those two is what Phase 23 did.
This runs the narrow population on its own.

THE PRE-REGISTERED BAR -- attribution first, not performance first
-------------------------------------------------------------------
Stated before the run, and deliberately stricter than Phase 23's:

    1. net P&L of the re-entered trades themselves must be > 0, AND
    2. that P&L must be >= 50% of the total equity gain over baseline, AND
    3. the gain must exceed the score-jitter null's 95th percentile.

"The config beats baseline" is explicitly NOT sufficient, because that is
exactly what passed last time and meant nothing. A config that improves equity
while its own re-entries contribute a minority of the gain is recorded as a
reshuffling artefact, not a re-entry edge.

The null
--------
Identical to Phase 24's: scores are jittered +/-1%, which perturbs candidate
ordering and adds no positions. It is an imperfect analogue -- it cannot
reproduce the deployment effect of adding trades -- and that limitation is
stated in the output rather than left for the reader to infer.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ..aes.locked import assert_live_config, assert_locked
from ..aes.portfolio import AESPortfolioBacktester
from ..aes.reentry import ReentryParams
from ..aes.scoring import attach_score
from ..aes_scanner import live_config as LC
from ..book.gold import load_gold
from ..book.metrics import positions_from_rows, summarise
from ..book.model import simulate
from ..book.results import concentration
from ..config import get_config
from .aes_calibration import load_universe
from .gamma_study import EQUAL_WEIGHTS, WINDOWS, signals_for_window
from .geometry_sweep import _truncate
from .reentry_study import WATCH_BARS, VOL_MULTS, CAPS, _position_edge, _reentry_outcomes

NULL_SEEDS = 20
JITTER = 0.01
MAJORITY = 0.50


def configs() -> dict[str, ReentryParams]:
    out: dict[str, ReentryParams] = {"baseline": ReentryParams(enabled=False)}
    for n in WATCH_BARS:
        for v in VOL_MULTS:
            for c in CAPS:
                rp = ReentryParams(
                    enabled=True, watch_bars=n, vol_mult=v, max_reentries=c,
                    require_bucket="high_conviction", max_bars_held=3,
                )
                out[rp.label()] = rp
    return out


def _attribute(base_eq: pd.Series, res, trades: pd.DataFrame) -> dict[str, Any]:
    """Split the equity gain into re-entry P&L and everything else.

    This is the number the bar is set on, so it is computed for every config
    rather than only for the ones that look good.
    """
    eq = res.equity_curve["equity"].astype(float)
    delta = float(eq.iloc[-1] - base_eq.iloc[-1])
    if not res.reentries or trades is None or trades.empty:
        return {"equity_delta": round(delta, 2), "reentry_pnl": 0.0,
                "reentry_share_pct": None, "n_reentries": len(res.reentries)}
    pos = positions_from_rows(trades)
    keys = {(r["symbol"], pd.Timestamp(r["entry_date"]).strftime("%Y-%m-%d"))
            for r in res.reentries}
    pe = pos.copy()
    pe["_k"] = list(zip(pe["symbol"], pd.to_datetime(pe["entry_date"]).dt.strftime("%Y-%m-%d")))
    hit = pe[pe["_k"].isin(keys)]
    pnl = float(hit["net_pnl"].sum()) if not hit.empty else 0.0
    return {
        "equity_delta": round(delta, 2),
        "reentry_pnl": round(pnl, 2),
        "reentry_share_pct": round(pnl / delta * 100, 1) if abs(delta) > 1e-9 else None,
        "n_reentries": len(res.reentries),
        "n_matched_positions": int(len(hit)),
    }


def null_distribution(frames, index_df, scored, ft, idx, base_eq) -> dict[str, Any]:
    """Equity delta under ordering perturbation only -- no positions added."""
    deltas = []
    for seed in range(NULL_SEEDS):
        rng = random.Random(seed)
        # ``Signal`` is a frozen, slotted dataclass, so plain attribute
        # assignment raises and a bare try/except would silently leave every
        # seed identical to the baseline -- a null that always returns zero and
        # looks like a very tight distribution instead of a broken one. Use
        # dataclasses.replace, and assert the jitter actually bit.
        jit = [replace(s, score=s.score * (1 + rng.uniform(-JITTER, JITTER)))
               for s in scored]
        if seed == 0 and all(a.score == b.score for a, b in zip(jit, scored)):
            raise RuntimeError(
                "score jitter had no effect -- the null would be vacuously zero"
            )
        rr = AESPortfolioBacktester(params=LC.PORTFOLIO).run(
            ft, idx, jit, min_bucket="wait_and_watch"
        )
        deltas.append(float(rr.equity_curve["equity"].astype(float).iloc[-1] - base_eq.iloc[-1]))
    d = np.array(deltas, float)
    return {
        "seeds": NULL_SEEDS,
        "mean": round(float(d.mean()), 2),
        "sd": round(float(d.std(ddof=1)), 2),
        "min": round(float(d.min()), 2),
        "max": round(float(d.max()), 2),
        "p95": round(float(np.percentile(d, 95)), 2),
        "caveat": ("ordering perturbation only; cannot reproduce the deployment "
                   "effect of adding positions, so it is a floor on the null, "
                   "not a complete one"),
    }


def run_config(frames, index_df, gold, scored, start, end, folds, rp, base_eq, ft, idx):
    t1 = pd.Timestamp(end)
    years = max(1e-9, (t1 - pd.Timestamp(start)).days / 365.25)

    fold_rows = []
    for year in folds:
        y0, y1 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        sub = [s for s in scored if y0 <= s.entry_date <= y1]
        if not sub:
            continue
        fty = _truncate(frames, y1)
        ic = int(index_df.index.searchsorted(y1, side="right")) + 40
        r = AESPortfolioBacktester(params=LC.PORTFOLIO, reentry=rp).run(
            fty, index_df.iloc[:ic], sub, min_bucket="wait_and_watch"
        )
        tf = r.trades_frame
        fold_rows.append({
            "fold": year,
            "ret": float(r.stats.get("total_return_pct", 0.0)),
            "dd": float(r.stats.get("max_drawdown_pct", 0.0)),
            "positions": int(tf.groupby(["symbol", "entry_date"]).ngroups) if not tf.empty else 0,
            "reentries": len(r.reentries),
        })

    out: dict[str, Any] = {"label": rp.label(), "signals": len(scored),
                           "signals_per_year": round(len(scored) / years, 1)}
    if fold_rows:
        rr = np.array([x["ret"] for x in fold_rows], float)
        dd = np.array([x["dd"] for x in fold_rows], float)
        out["fold_summary"] = {
            "ann_ret": round(float(rr.mean()), 2),
            "median_fold": round(float(np.median(rr)), 2),
            "mean_dd": round(float(dd.mean()), 2),
            "worst_dd": round(float(dd.max()), 2),
            "pos_folds": f"{int((rr > 0).sum())}/{len(rr)}",
            "positions_per_year": round(float(np.mean([x['positions'] for x in fold_rows])), 1),
            "reentries_per_year": round(float(np.mean([x['reentries'] for x in fold_rows])), 1),
            **concentration(list(rr)),
        }
        out["folds"] = fold_rows

    res = AESPortfolioBacktester(params=LC.PORTFOLIO, reentry=rp).run(
        ft, idx, scored, min_bucket="wait_and_watch"
    )
    eq, trades = res.equity_curve, res.trades_frame
    if len(eq) > 3:
        strat_eq = eq["equity"].astype(float)
        bk = simulate(strat_eq, eq["positions_value"].astype(float), gold, LC.BOOK)
        bench = idx["close"].astype(float)
        out["strategy"] = summarise(strat_eq, trades, bench,
                                    eq["exposure"] if "exposure" in eq else None)
        out["book"] = summarise(bk.curve["book"], trades, bench, bk.curve["deployment"])
        out["position_edge"] = _position_edge(trades)
        out["reentry_outcomes"] = _reentry_outcomes(res.reentries, trades)
        out["attribution"] = _attribute(base_eq, res, trades)
        out["n_reentries_total"] = len(res.reentries)
    return out


def main(out_name: str = "reentry_narrow_results.json") -> dict[str, Any]:
    assert_live_config()
    assert_locked(score=EQUAL_WEIGHTS)
    cfg = get_config()
    ohlcv, results = str(cfg.paths.ohlcv_dir), str(cfg.paths.results_dir)
    frames = load_universe(ohlcv)
    index_df = pd.read_parquet(os.path.join(ohlcv, "CRSLDX.parquet"))
    gold, _ = load_gold(os.path.join(ohlcv, "GOLDBEES.parquet"))

    cfgs = configs()
    res: dict[str, Any] = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "population": "high_conviction at entry AND stopped within 3 bars",
        "bar_for_adoption": (
            "reentry_pnl > 0 AND reentry_share_pct >= 50 AND equity_delta > null p95"
        ),
        "windows": {},
    }

    for wname, (start, end, folds) in WINDOWS.items():
        print(f"[signals] {wname}", flush=True)
        raw = signals_for_window(frames, index_df, pd.Timestamp(start), pd.Timestamp(end), None)
        scored = [attach_score(s, EQUAL_WEIGHTS) for s in raw]
        t1 = pd.Timestamp(end)
        ft = _truncate(frames, t1)
        ic = int(index_df.index.searchsorted(t1, side="right")) + 40
        idx = index_df.iloc[:ic]
        base = AESPortfolioBacktester(params=LC.PORTFOLIO).run(
            ft, idx, scored, min_bucket="wait_and_watch")
        base_eq = base.equity_curve["equity"].astype(float)
        print(f"          {len(scored)} signals; running null ({NULL_SEEDS} seeds)", flush=True)
        null = null_distribution(frames, index_df, scored, ft, idx, base_eq)
        print(f"          null: mean={null['mean']:,.0f} sd={null['sd']:,.0f} p95={null['p95']:,.0f}", flush=True)

        res["windows"][wname] = {"null": null, "configs": {}}
        for name, rp in cfgs.items():
            c = run_config(frames, index_df, gold, scored, start, end, folds, rp,
                           base_eq, ft, idx)
            res["windows"][wname]["configs"][name] = c
            a = c.get("attribution", {})
            print(f"  {name:32} re={c.get('n_reentries_total',0):3} "
                  f"delta={a.get('equity_delta',0):>12,.0f} "
                  f"repnl={a.get('reentry_pnl',0):>11,.0f} "
                  f"share={a.get('reentry_share_pct')}", flush=True)

    path = os.path.join(results, out_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    res["_path"] = path
    return res


__all__ = ["MAJORITY", "NULL_SEEDS", "configs", "main", "null_distribution", "run_config"]
