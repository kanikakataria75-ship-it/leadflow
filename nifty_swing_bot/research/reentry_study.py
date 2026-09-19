"""Phase 23: re-entry after a stop-out, swept.

Pre-registered bar (set before the run, checked by ``reentry_tables.verdict``)
-----------------------------------------------------------------------------
    improve per-trade edge AND ex-best-fold, in BOTH windows, with drawdown
    not materially worse (mean drawdown no more than 2pp worse than baseline).

One measurement note that changes what "per-trade edge" can mean here
---------------------------------------------------------------------
Re-entry is a **portfolio-level** mechanism. It creates no new signals: the
admission gate, the box stage, the scorer and the signal population are
untouched, and ``signals`` and ``signals/year`` are therefore **identical in
every cell by construction**. The signal-level ``edge`` / ``edge_vs_base`` that
``geometry_sweep.per_trade_edge`` computes is a property of that population, so
it is invariant too, and quoting it across cells would be meaningless -- it
would look stable because nothing could move it.

The quantity that does move, and the one reported here as the per-trade edge,
is the **realised per-position return**: mean net return per POSITION (ladder
rows collapsed, per ``positions_from_rows``), with its own t-statistic. Both
are reported, with the invariant one labelled as such, rather than silently
substituting one for the other.

Why this is not simply the sixth signal-count attempt, restated as a risk
------------------------------------------------------------------------
It adds trades, and the last five things that added trades all failed on
ex-best-fold. It differs in that it adds no new *names*: every re-entry is a
name the pipeline already admitted, scored and chose to hold. That is a reason
to measure it separately, not a reason to expect a different answer, and the
bar above is the same shape as GAMMA's for exactly that reason.
"""

from __future__ import annotations

import json
import os
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

#: N x volume x cap, as briefed. ``None`` volume is "no volume condition".
WATCH_BARS = (5, 10, 15, 20)
VOL_MULTS: tuple[float | None, ...] = (1.5, 2.0, None)
CAPS = (1, 2)


def configs() -> dict[str, ReentryParams]:
    out: dict[str, ReentryParams] = {"baseline": ReentryParams(enabled=False)}
    for n in WATCH_BARS:
        for v in VOL_MULTS:
            for c in CAPS:
                rp = ReentryParams(enabled=True, watch_bars=n, vol_mult=v, max_reentries=c)
                out[rp.label()] = rp
    return out


def _position_edge(trades: pd.DataFrame) -> dict[str, Any]:
    """Mean net return per POSITION, with a t-statistic.

    Position level, not trade-row level: a profit ladder splits one position
    into 2-4 rows and every ladder row is profitable by construction, so a
    row-level mean is inflated and its t is computed on a sample that is not
    independent.
    """
    if trades is None or trades.empty:
        return {"position_edge_pct": float("nan"), "position_edge_t": float("nan"), "n": 0}
    pos = positions_from_rows(trades)
    r = pos["ret_pct"].astype(float).dropna()
    if len(r) < 3:
        return {"position_edge_pct": float("nan"), "position_edge_t": float("nan"), "n": len(r)}
    sd = float(r.std(ddof=1))
    return {
        "position_edge_pct": float(r.mean()),
        "position_edge_t": float(r.mean() / (sd / np.sqrt(len(r)))) if sd > 0 else float("nan"),
        "n": int(len(r)),
    }


def _reentry_outcomes(reentries: list[dict], trades: pd.DataFrame) -> dict[str, Any]:
    """How the re-entries themselves did, separately from the population.

    "A small number of trades with a high hit rate" and "churn" are different
    outcomes and are distinguished here rather than averaged together.
    """
    if not reentries:
        return {"n": 0, "win_rate_pct": None, "mean_ret_pct": None,
                "total_net_pnl": None, "share_of_positions_pct": 0.0}
    pos = positions_from_rows(trades) if trades is not None and not trades.empty else pd.DataFrame()
    if pos.empty:
        return {"n": len(reentries), "win_rate_pct": None, "mean_ret_pct": None,
                "total_net_pnl": None, "share_of_positions_pct": None}

    keys = {(r["symbol"], pd.Timestamp(r["entry_date"]).strftime("%Y-%m-%d")) for r in reentries}
    pe = pos.copy()
    pe["_k"] = list(zip(pe["symbol"], pd.to_datetime(pe["entry_date"]).dt.strftime("%Y-%m-%d")))
    hit = pe[pe["_k"].isin(keys)]
    if hit.empty:
        return {"n": len(reentries), "win_rate_pct": None, "mean_ret_pct": None,
                "total_net_pnl": None, "share_of_positions_pct": 0.0}
    r = hit["ret_pct"].astype(float)
    return {
        "n": int(len(hit)),
        "win_rate_pct": round(float((hit["net_pnl"] > 0).mean() * 100), 2),
        "mean_ret_pct": round(float(r.mean()), 3),
        "median_ret_pct": round(float(r.median()), 3),
        "total_net_pnl": round(float(hit["net_pnl"].sum()), 2),
        "share_of_positions_pct": round(len(hit) / len(pos) * 100, 2),
    }


def run_config(
    frames, index_df, gold, scored, start: str, end: str, folds: list[int],
    rp: ReentryParams,
) -> dict[str, Any]:
    t1 = pd.Timestamp(end)
    years = max(1e-9, (t1 - pd.Timestamp(start)).days / 365.25)

    # --- per-fold, on the REJECTED #8/#12/#18/#23 footing -------------------- #
    fold_rows = []
    for year in folds:
        y0, y1 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        sub = [s for s in scored if y0 <= s.entry_date <= y1]
        if not sub:
            continue
        ft = _truncate(frames, y1)
        icut = int(index_df.index.searchsorted(y1, side="right")) + 40
        res = AESPortfolioBacktester(params=LC.PORTFOLIO, reentry=rp).run(
            ft, index_df.iloc[:icut], sub, min_bucket="wait_and_watch"
        )
        tf = res.trades_frame
        fold_rows.append({
            "fold": year,
            "ret": float(res.stats.get("total_return_pct", 0.0)),
            "dd": float(res.stats.get("max_drawdown_pct", 0.0)),
            "positions": int(tf.groupby(["symbol", "entry_date"]).ngroups) if not tf.empty else 0,
            "reentries": len(res.reentries),
        })

    out: dict[str, Any] = {"label": rp.label(), "signals": len(scored),
                           "signals_per_year": round(len(scored) / years, 1)}
    if fold_rows:
        r = np.array([x["ret"] for x in fold_rows], float)
        d = np.array([x["dd"] for x in fold_rows], float)
        out["folds"] = fold_rows
        out["fold_summary"] = {
            "ann_ret": round(float(r.mean()), 2),
            "median_fold": round(float(np.median(r)), 2),
            "mean_dd": round(float(d.mean()), 2),
            "worst_dd": round(float(d.max()), 2),
            "pos_folds": f"{int((r > 0).sum())}/{len(r)}",
            "positions_per_year": round(float(np.mean([x["positions"] for x in fold_rows])), 1),
            "reentries_per_year": round(float(np.mean([x["reentries"] for x in fold_rows])), 1),
            **concentration(list(r)),
        }

    # --- continuous run: risk metrics, book, re-entry outcomes -------------- #
    ft = _truncate(frames, t1)
    icut = int(index_df.index.searchsorted(t1, side="right")) + 40
    idx = index_df.iloc[:icut]
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
        out["n_reentries_total"] = len(res.reentries)
    return out


def main(out_name: str = "reentry_results.json") -> dict[str, Any]:
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
        "frozen_on": str(LC.FROZEN_ON),
        "config": LC.describe(),
        "bar_for_adoption": (
            "improve position edge AND ex-best-fold in BOTH windows, "
            "mean drawdown no more than 2pp worse"
        ),
        "note_signals_invariant": (
            "re-entry is portfolio-level: the signal population, and therefore "
            "signals/year and the signal-level edge, are identical in every cell"
        ),
        "windows": {},
    }

    for wname, (start, end, folds) in WINDOWS.items():
        print(f"[signals] {wname}", flush=True)
        raw = signals_for_window(frames, index_df, pd.Timestamp(start), pd.Timestamp(end), None)
        scored = [attach_score(s, EQUAL_WEIGHTS) for s in raw]
        print(f"          {len(scored)} signals (same for every cell)", flush=True)
        res["windows"][wname] = {}
        for name, rp in cfgs.items():
            res["windows"][wname][name] = run_config(
                frames, index_df, gold, scored, start, end, folds, rp
            )
            s = res["windows"][wname][name]
            fs = s.get("fold_summary", {})
            print(f"  {name:26} reent={s.get('n_reentries_total',0):4} "
                  f"ann={fs.get('ann_ret')} exbest={fs.get('ex_best_period')}", flush=True)

    path = os.path.join(results, out_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    res["_path"] = path
    return res


__all__ = ["CAPS", "VOL_MULTS", "WATCH_BARS", "configs", "main", "run_config"]
