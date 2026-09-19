"""Phase 18: slots x signal-count x risk, with proper risk sizing.

The open question after §17. Per-trade edge is positive and stable (+0.48% on
2015-2021, +0.17% to +1.04% on 2022-2026), and PROVEN 4 says the binding
constraint is capital deployment, not holding period: mean exposure is ~6%, and
reaching 10%/yr on the measured edge needs roughly 48%.

§17's portfolio run made the gap concrete in the wrong way -- +272% at ~33% per
position, against a locked config deploying 6-9% on the same signals. That
difference is deployment, but §17 sized positions flat, which is not a policy
anyone would run: it ignores where the stop sits. This sweep uses
``AESPortfolioBacktester`` throughout -- risk-based sizing, ATR(14) x 2.5
trailing stop, the §6.2 ladder, the §8 regime rules -- and asks what the
measured edge actually supports.

**Sizing convention.** ``target_concurrent_positions`` scales with the slot
count at the locked config's own ratio (2.5 / 3 = 0.833), so base notional per
position is 1/S of equity rather than a fixed 40%. Holding it at 2.5 while
raising slots to 15 would let one position take 40% of equity, which is not the
policy being tested and would make the notional cap, not the risk budget, the
operative constraint (REJECTED #13 measured that cap never binding).

Two windows, reported separately and never pooled, because they disagree:
2015-2021 (seven folds) and **2022-2026 (five folds, and DISCOVERY as of
Phase 16 -- see STATE.md §5)**.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ..aes.boxes import scan_box_history
from ..aes.params import BoxParams, PortfolioParams, ScreenerParams, WatchlistParams
from ..aes.portfolio import AESPortfolioBacktester
from ..aes.scoring import attach_score
from ..aes.watchlist import run_watchlist_for_symbol
from .geometry_sweep import BUCKET_ORDER, EQUAL_WEIGHTS, LOCKED_PORTFOLIO, _truncate

#: Locked config's slots-to-sizing-target ratio, preserved as slots scale.
TARGET_RATIO = 2.5 / 3.0


def signals_for_window(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    box_params: BoxParams | None = None,
    screener_params: ScreenerParams | None = None,
    watch_params: WatchlistParams | None = None,
) -> list:
    """``discovery_signals_raw`` for an arbitrary window.

    The calibration helper hard-codes ``AES_DISCOVERY``; this is the same walk
    with the window as a parameter, so 2015-2021 and 2022-2026 go through
    identical code.
    """
    from ..aes.screener import screen_universe

    SP = screener_params or ScreenerParams()
    BP = box_params or BoxParams()
    WP = watch_params or WatchlistParams()

    adm = screen_universe(frames, SP)
    adm = adm[(adm["date"] >= start) & (adm["date"] <= end)]
    out = []
    for sym in sorted(adm["symbol"].unique()):
        df = frames[sym]
        sa = adm[adm["symbol"] == sym]
        last = df.index.get_indexer([sa["date"].max()])[0]
        hist = scan_box_history(df, BP, end_idx=min(len(df) - 1, last + 100), step=3)
        out.extend(
            run_watchlist_for_symbol(
                df, index_df, sa["date"], symbol=sym,
                box_params=BP, watch_params=WP, box_history=hist,
            )
        )
    return out


def portfolio_params(slots: int, risk_pct: float) -> PortfolioParams:
    return replace(
        LOCKED_PORTFOLIO,
        max_open_positions=slots,
        target_concurrent_positions=max(1.0, slots * TARGET_RATIO),
        risk_per_trade_pct=risk_pct,
    )


def run_cell(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    scored: list,
    years: list[int],
    slots: int,
    risk_pct: float,
) -> dict[str, Any]:
    """One (slots, risk) cell over the given annual folds."""
    pp = portfolio_params(slots, risk_pct)
    rows = []
    for year in years:
        y0, y1 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        sub = [s for s in scored if y0 <= s.entry_date <= y1]
        if not sub:
            continue
        ft = _truncate(frames, y1)
        icut = int(index_df.index.searchsorted(y1, side="right")) + 40
        res = AESPortfolioBacktester(params=pp).run(
            ft, index_df.iloc[:icut], sub, min_bucket="wait_and_watch"
        )
        st, tf = res.stats, res.trades_frame
        dep = (
            float(res.equity_curve["exposure"].mean() * 100)
            if "exposure" in res.equity_curve else 0.0
        )
        top1 = top3 = None
        if not tf.empty:
            by = tf.groupby("symbol")["net_pnl"].sum().sort_values(ascending=False)
            tot = float(by.sum())
            if abs(tot) > 1e-9:
                top1 = float(by.iloc[0] / tot * 100)
                top3 = float(by.iloc[:3].sum() / tot * 100)
        rows.append({
            "fold": year,
            "n_signals": len(sub),
            "positions": int(tf.groupby(["symbol", "entry_date"]).ngroups) if not tf.empty else 0,
            "ret": float(st.get("total_return_pct", 0.0)),
            "alpha": float(st.get("alpha_annual_pct", 0.0)),
            "beta": float(st.get("beta", 0.0)),
            "dd": float(st.get("max_drawdown_pct", 0.0)),
            "deploy": dep,
            "top1": top1,
            "top3": top3,
        })
    if not rows:
        return {}
    r = np.array([x["ret"] for x in rows], dtype=float)
    d = np.array([x["dd"] for x in rows], dtype=float)
    def _m(k):
        v = [x[k] for x in rows if x[k] is not None]
        return float(np.mean(v)) if v else None
    return {
        "slots": slots,
        "risk_pct": risk_pct,
        "ann_ret": round(float(r.mean()), 2),
        "compounded": round(float((np.prod(1 + r / 100) ** (1 / len(r)) - 1) * 100), 2),
        "alpha": round(_m("alpha") or 0.0, 2),
        "beta": round(_m("beta") or 0.0, 3),
        "mean_dd": round(float(d.mean()), 2),
        "worst_dd": round(float(d.max()), 2),
        "deploy": round(_m("deploy") or 0.0, 2),
        "top1": round(_m("top1"), 1) if _m("top1") is not None else None,
        "top3": round(_m("top3"), 1) if _m("top3") is not None else None,
        "pos_folds": f"{int((r > 0).sum())}/{len(r)}",
        "ret_dd": round(float(r.mean() / d.mean()), 2) if d.mean() > 0 else None,
        "folds": rows,
    }


def score_pool(signals: list) -> list:
    return [attach_score(s, EQUAL_WEIGHTS) for s in signals]


def traded_count(scored: list) -> int:
    return sum(1 for s in scored if BUCKET_ORDER.get(s.bucket, 0) >= 1)


__all__ = [
    "TARGET_RATIO", "portfolio_params", "run_cell", "score_pool",
    "signals_for_window", "traded_count",
]
