"""Phase 22: GAMMA -- alternative daily structures at the box stage.

The prior, stated before the result rather than after it
--------------------------------------------------------
This is the **fifth** attempt to raise signal count. The previous four:

    REJECTED #8   watchlist memory 90 -> 750    4.5x signals, return fell
    REJECTED #12  weak re-admission            sig/yr 120 -> 259, edge 2.39 -> 1.53
    REJECTED #18  range floor 15% -> 5%        +59 sig/yr, zero coverage gain,
                                               fold return 4.78 -> 2.79
    REJECTED #14  geometry loosening           run in Phase 15, nothing adopted

All four raised count and lowered fold return. If GAMMA does the same it is
**REJECTED #23** and gets written up as such. The bar for it being anything
else is explicit and set here, before the numbers exist:

    it must raise signals/year AND hold per-trade edge (``edge_vs_base``)
    AND hold ex-best-fold return, in BOTH windows.

Raising the headline while ex-best-fold falls is the failure mode every one of
the four had, and it is exactly what a structure that mostly adds noise looks
like when one lucky fold carries it.

Measurement footing
-------------------
Fold statistics use per-fold portfolio runs on annual folds with the Phase 9
truncation convention -- the same footing REJECTED #8/#12/#18 were measured on,
so "GAMMA follows the pattern" is a like-for-like claim rather than a rhyme.
Risk-adjusted metrics and the book need a continuous daily equity series, so a
second, continuous run per arm supplies those. Both are reported.

Everything downstream of the structure is the frozen live configuration, and
``assert_live_config()`` runs before anything is computed.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from ..aes.boxes import scan_box_history
from ..aes.gamma import GammaParams, detect_gamma_structure
from ..aes.locked import assert_live_config, assert_locked
from ..aes.params import BoxParams, ScoreParams
from ..aes.portfolio import AESPortfolioBacktester
from ..aes.scoring import attach_score
from ..aes.screener import screen_universe
from ..aes.watchlist import run_watchlist_for_symbol
from ..aes_scanner import live_config as LC
from ..book.gold import load_gold
from ..book.metrics import summarise
from ..book.model import simulate
from ..book.results import concentration
from ..config import get_config
from .aes_calibration import _finish_signal_frame, load_universe, universe_base_rate
from .geometry_sweep import (
    _truncate,
    coverage,
    load_real_trades,
    per_trade_edge,
)

WINDOWS: dict[str, tuple[str, str, list[int]]] = {
    "2015-2021": ("2015-01-01", "2021-12-31", list(range(2015, 2022))),
    "2022-2026": ("2022-01-01", "2026-09-18", list(range(2022, 2027))),
}

#: baseline first, each arm alone, then combined -- the order the brief asks for.
ARMS: dict[str, GammaParams | None] = {
    "baseline": None,
    "a_hhhl_2": GammaParams(arms=("hhhl",), hhhl_min_swings=2),
    "a_hhhl_3": GammaParams(arms=("hhhl",), hhhl_min_swings=3),
    "a_hhhl_4": GammaParams(arms=("hhhl",), hhhl_min_swings=4),
    "b_rising_ma": GammaParams(arms=("rising_ma",)),
    "c_pullback": GammaParams(arms=("pullback",)),
    "combined": GammaParams(arms=("hhhl", "rising_ma", "pullback"), hhhl_min_swings=3),
}

EQUAL_WEIGHTS = ScoreParams(
    w_fast_resolution=0.25, w_absorbed=0.25, w_rs_capture=0.25, w_prior_cycles=0.25
)


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def signals_for_window(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    gamma_params: GammaParams | None,
) -> list:
    """``slot_sweep.signals_for_window`` with the GAMMA fallback threaded through.

    Admission, box params and watchlist params are the frozen ones and are not
    parameters of this function: the brief is explicit that only the box stage
    changes, and making the rest un-passable is the cheapest way to guarantee
    that.
    """
    adm = screen_universe(frames, LC.SCREENER)
    adm = adm[(adm["date"] >= start) & (adm["date"] <= end)]
    out = []
    for sym in sorted(adm["symbol"].unique()):
        df = frames[sym]
        sa = adm[adm["symbol"] == sym]
        last = df.index.get_indexer([sa["date"].max()])[0]
        hist = scan_box_history(df, LC.BOX, end_idx=min(len(df) - 1, last + 100), step=3)
        out.extend(
            run_watchlist_for_symbol(
                df, index_df, sa["date"], symbol=sym,
                box_params=LC.BOX, watch_params=LC.WATCHLIST,
                box_history=hist, gamma_params=gamma_params,
            )
        )
    return out


def _gamma_share(scored: list) -> dict[str, Any]:
    """How much of the signal population each arm actually contributed."""
    kinds: dict[str, int] = {}
    for s in scored:
        k = str(s.nested.big.kind)
        k = k.split(":", 1)[1] if k.startswith("gamma:") else "box"
        kinds[k] = kinds.get(k, 0) + 1
    total = max(1, len(scored))
    return {
        "by_structure": kinds,
        "gamma_pct": round(
            sum(v for k, v in kinds.items() if k != "box") / total * 100, 1
        ),
    }


# --------------------------------------------------------------------------- #
# Real-trade recovery
# --------------------------------------------------------------------------- #
def real_trade_recovery(
    real: list[dict[str, Any]], gamma_params: GammaParams | None
) -> dict[str, Any]:
    """How many of the 39 testable real trades each arm sees at the entry bar.

    The number that matters is ``recovered``: trades where the **box detector
    finds nothing** and the GAMMA structure does. Trades the box already sees
    are not a gain -- they are already in the baseline -- so they are counted
    separately rather than folded into a flattering total.
    """
    bp = BoxParams()
    box_hits = set()
    for t in real:
        from ..aes.boxes import detect_nested_box

        if detect_nested_box(t["df"], t["idx"], bp, symbol=t["ticker"]) is not None:
            box_hits.add(t["stock"])

    if gamma_params is None:
        return {
            "testable": len(real), "box_at_entry": len(box_hits),
            "gamma_at_entry": 0, "recovered": 0, "recovered_names": [],
            "union": len(box_hits),
        }

    gamma_hits, recovered = set(), []
    for t in real:
        nb = detect_gamma_structure(
            t["df"], t["idx"], gamma_params, symbol=t["ticker"], box_params=bp
        )
        if nb is None:
            continue
        gamma_hits.add(t["stock"])
        if t["stock"] not in box_hits:
            recovered.append(t["stock"])
    return {
        "testable": len(real),
        "box_at_entry": len(box_hits),
        "gamma_at_entry": len(gamma_hits),
        "recovered": len(recovered),
        "recovered_names": sorted(recovered),
        "union": len(box_hits | gamma_hits),
    }


# --------------------------------------------------------------------------- #
# One arm, one window
# --------------------------------------------------------------------------- #
def run_arm(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    gold: pd.Series,
    start: str,
    end: str,
    folds: list[int],
    gamma_params: GammaParams | None,
    base_rate: dict[int, pd.Series],
) -> dict[str, Any]:
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    assert_locked(score=EQUAL_WEIGHTS)

    raw = signals_for_window(frames, index_df, t0, t1, gamma_params)
    scored = [attach_score(s, EQUAL_WEIGHTS) for s in raw]

    years = max(1e-9, (t1 - t0).days / 365.25)
    out: dict[str, Any] = {
        "signals": len(scored),
        "signals_per_year": round(len(scored) / years, 1),
        "structures": _gamma_share(scored),
    }

    # --- per-trade edge, on the admission_variants footing ------------------ #
    frame = _finish_signal_frame(raw, frames, index_df, LC.SCREENER)
    out["edge"] = (
        per_trade_edge(frame, base_rate) if not frame.empty
        else {"edge": float("nan"), "edge_vs_base": float("nan")}
    )

    # --- per-fold runs: the REJECTED #8/#12/#18 footing ---------------------- #
    fold_rows = []
    for year in folds:
        y0, y1 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        sub = [s for s in scored if y0 <= s.entry_date <= y1]
        if not sub:
            continue
        ft = _truncate(frames, y1)
        icut = int(index_df.index.searchsorted(y1, side="right")) + 40
        res = AESPortfolioBacktester(params=LC.PORTFOLIO).run(
            ft, index_df.iloc[:icut], sub, min_bucket="wait_and_watch"
        )
        st, tf = res.stats, res.trades_frame
        top1 = top3 = None
        if not tf.empty:
            by = tf.groupby("symbol")["net_pnl"].sum().sort_values(ascending=False)
            tot = float(by.sum())
            if abs(tot) > 1e-9:
                top1 = float(by.iloc[0] / tot * 100)
                top3 = float(by.iloc[:3].sum() / tot * 100)
        fold_rows.append({
            "fold": year,
            "signals": len(sub),
            "positions": int(tf.groupby(["symbol", "entry_date"]).ngroups) if not tf.empty else 0,
            "ret": float(st.get("total_return_pct", 0.0)),
            "dd": float(st.get("max_drawdown_pct", 0.0)),
            "top1": top1,
            "top3": top3,
        })

    if fold_rows:
        r = np.array([x["ret"] for x in fold_rows], float)
        d = np.array([x["dd"] for x in fold_rows], float)
        def _m(k):
            v = [x[k] for x in fold_rows if x[k] is not None]
            return round(float(np.mean(v)), 1) if v else None
        out["folds"] = fold_rows
        out["fold_summary"] = {
            "ann_ret": round(float(r.mean()), 2),
            "median_fold": round(float(np.median(r)), 2),
            "mean_dd": round(float(d.mean()), 2),
            "worst_dd": round(float(d.max()), 2),
            "pos_folds": f"{int((r > 0).sum())}/{len(r)}",
            "positions_per_year": round(
                float(np.mean([x["positions"] for x in fold_rows])), 1),
            "top1": _m("top1"),
            "top3": _m("top3"),
            **concentration(list(r)),
        }

    # --- continuous run: risk metrics and the book -------------------------- #
    ft = _truncate(frames, t1)
    icut = int(index_df.index.searchsorted(t1, side="right")) + 40
    idx = index_df.iloc[:icut]
    res = AESPortfolioBacktester(params=LC.PORTFOLIO).run(
        ft, idx, scored, min_bucket="wait_and_watch"
    )
    eq = res.equity_curve
    trades = res.trades_frame
    if len(eq) > 3:
        strat_eq = eq["equity"].astype(float)
        bk = simulate(strat_eq, eq["positions_value"].astype(float), gold, LC.BOOK)
        bench = idx["close"].astype(float)
        out["strategy"] = summarise(
            strat_eq, trades, bench, eq["exposure"] if "exposure" in eq else None
        )
        out["book"] = summarise(
            bk.curve["book"], trades, bench, bk.curve["deployment"]
        )
        out["book_diagnostics"] = bk.diagnostics
        out["n_positions"] = (
            int(trades.groupby(["symbol", "entry_date"]).ngroups) if not trades.empty else 0
        )
    return out


def main(out_name: str = "gamma_results.json") -> dict[str, Any]:
    assert_live_config()
    cfg = get_config()
    ohlcv = str(cfg.paths.ohlcv_dir)
    results = str(cfg.paths.results_dir)

    frames = load_universe(ohlcv)
    index_df = pd.read_parquet(os.path.join(ohlcv, "CRSLDX.parquet"))
    gold, _ = load_gold(os.path.join(ohlcv, "GOLDBEES.parquet"))
    base_rate = universe_base_rate(frames, index_df)
    real = load_real_trades(
        os.path.join(str(cfg.paths.cache_dir), "recs"),
        os.path.join(results, "recs_mapping.csv"),
    )

    res: dict[str, Any] = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "frozen_on": str(LC.FROZEN_ON),
        "config": LC.describe(),
        "prior": "5th attempt to raise signal count; previous four all lowered return",
        "bar_for_adoption": (
            "raise signals/yr AND hold edge_vs_base AND hold ex-best-fold, in BOTH windows"
        ),
        "real_trade_recovery": {},
        "windows": {},
    }

    for arm, gp in ARMS.items():
        res["real_trade_recovery"][arm] = real_trade_recovery(real, gp)
        print(f"[recovery] {arm}: {res['real_trade_recovery'][arm]}", flush=True)

    for wname, (start, end, folds) in WINDOWS.items():
        res["windows"][wname] = {}
        for arm, gp in ARMS.items():
            print(f"[run] {wname} / {arm}", flush=True)
            res["windows"][wname][arm] = run_arm(
                frames, index_df, gold, start, end, folds, gp, base_rate
            )
            s = res["windows"][wname][arm]
            print(f"      signals={s['signals']} "
                  f"ann_ret={s.get('fold_summary', {}).get('ann_ret')} "
                  f"ex_best={s.get('fold_summary', {}).get('ex_best_period')}", flush=True)

    path = os.path.join(results, out_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    res["_path"] = path
    return res


__all__ = ["ARMS", "WINDOWS", "main", "real_trade_recovery", "run_arm", "signals_for_window"]
