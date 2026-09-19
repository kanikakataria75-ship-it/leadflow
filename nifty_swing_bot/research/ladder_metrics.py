"""Phase 21: the risk-adjusted metric grid, under the FROZEN live configuration.

Why this module exists rather than a read of ``book_results_final.json``
-----------------------------------------------------------------------
That file already carries an old-ladder/variant-(c) grid with the full metric
set, generated 2026-09-17 06:01 -- the freeze date. It cannot be used, for a
reason visible only in one diagnostic field: it reports **28** rebalances over
2015-2021 and **19** over 2022-2026. Annual rebalancing on those spans is 7
and 5. Those are quarterly counts, so the book in that file was simulated on
``BookParams()``'s default ``rebalance="quarterly"``, not on the frozen
``annual``.

``aes/locked.py`` did not catch it because it guarded ``ScoreParams`` and
``PortfolioParams`` and nothing else. It now guards ``BookParams`` too, and
this module calls ``assert_live_config()`` before computing anything.

The strategy-sleeve half of that file is probably sound -- sleeve equity does
not depend on how the book rebalances around it -- but the file records no
parameters at all, so "probably" is the strongest available statement about it.
Everything is recomputed here from the frozen config, and this run is
self-describing: the parameters travel in the output.

What the comparison isolates
----------------------------
Both ladders run at the frozen sizing (6 slots, 3% risk, ATR(14) x 2.5 trail,
25-bar cap) and differ **only** in the tranche schedule, so the gap between the
two columns is the ladder and nothing else:

    old        +5% / 50%, +8% / 20%, +20% / 20%    (the spec ladder)
    variant c  +12% / 50%, +20% / 20%, remainder trails

Variant (c) is a **forward-test candidate**, not an established improvement.
Its best paired t across folds is 1.98. Nothing here changes that.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from typing import Any

import pandas as pd

from ..aes.locked import assert_live_config, assert_locked
from ..aes.portfolio import AESPortfolioBacktester
from ..aes.scoring import attach_score
from ..aes_scanner import live_config as LC
from ..book.gold import load_gold
from ..book.metrics import summarise
from ..book.model import correlation_by_year, simulate
from ..book.results import concentration
from ..config import get_config
from .geometry_sweep import _truncate
from .slot_sweep import signals_for_window

#: Both windows are DISCOVERY as of Phase 16. Neither is out-of-sample.
WINDOWS: dict[str, tuple[str, str, str]] = {
    "2015-2021": ("2015-01-01", "2021-12-31", "discovery"),
    "2022-2026": ("2022-01-01", "2026-09-18", "discovery"),
}

#: Fields worth recording per run, so a result can never again be orphaned
#: from the configuration that produced it.
_PARAM_KEYS = (
    "max_open_positions", "target_concurrent_positions", "risk_per_trade_pct",
    "atr_stop_mult", "atr_stop_period", "max_hold_bars",
    "size_mult_wait_and_watch", "size_mult_high_conviction",
    "ladder1_trigger_pct", "ladder1_fraction",
    "ladder2_trigger_pct", "ladder2_fraction",
    "ladder3_trigger_pct", "ladder3_fraction",
)


def ladders() -> dict[str, Any]:
    """The frozen portfolio config, and the same config with the spec ladder."""
    frozen = LC.PORTFOLIO
    old = replace(
        frozen,
        ladder1_trigger_pct=0.05, ladder1_fraction=0.50,
        ladder2_trigger_pct=0.08, ladder2_fraction=0.20,
        ladder3_trigger_pct=0.20, ladder3_fraction=0.20,
    )
    return {"old_ladder_5_8_20": old, "variant_c_12_20": frozen}


def _equal_weights():
    from ..aes.params import ScoreParams

    return ScoreParams(
        w_fast_resolution=0.25, w_absorbed=0.25,
        w_rs_capture=0.25, w_prior_cycles=0.25,
    )


def _annual_returns(curve: pd.DataFrame, col: str, ret_col: str | None = None):
    """Per-calendar-year return in percent, with the year it belongs to.

    Returns ``(years, values)``. The years travel with the values because a
    fold list indexed by position silently mislabels every fold as soon as a
    window does not start in January.
    """
    groups = [(int(y), g) for y, g in curve.groupby(curve.index.year)]

    # The window run-off (40 bars past the window end, so a 25-bar-capped
    # position can close naturally) spills a two-month stub into the next
    # calendar year. Left alone it stands as a fold of its own and is counted
    # by every concentration statistic as if it were a year. Its P&L belongs
    # to the window, so it is merged into the preceding fold rather than
    # dropped -- dropping it would lose real closing P&L from the fold table
    # while leaving it in total return.
    MIN_FOLD_DAYS = 60
    if len(groups) > 1 and len(groups[-1][1]) < MIN_FOLD_DAYS:
        y_prev, g_prev = groups[-2]
        groups[-2] = (y_prev, pd.concat([g_prev, groups[-1][1]]))
        groups.pop()

    years, out = [], []
    for year, g in groups:
        if len(g) < 20:
            continue
        years.append(year)
        if ret_col is not None:
            out.append(float((1 + g[ret_col]).prod() - 1) * 100)
        else:
            out.append(float(g[col].iloc[-1] / g[col].iloc[0] - 1) * 100)
    return years, out


def run_window(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    gold: pd.Series,
    start: str,
    end: str,
    provenance: str,
) -> dict[str, Any]:
    """Signals once, then each ladder through the sleeve and the whole book."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    sw = _equal_weights()
    assert_locked(score=sw)

    raw = signals_for_window(
        frames, index_df, t0, t1,
        box_params=LC.BOX, screener_params=LC.SCREENER, watch_params=LC.WATCHLIST,
    )
    scored = [attach_score(s, sw) for s in raw]

    # Phase 9's isolation convention. Without it the 2015-2021 run keeps
    # walking to the end of the data: the strategy sits flat from 2022 (no
    # signals) while arbitrage and gold keep accruing, which drags the sleeve's
    # CAGR below the risk-free rate, turns Sharpe negative, and hands the book
    # five free positive folds it did not earn.
    frames = _truncate(frames, t1)
    icut = int(index_df.index.searchsorted(t1, side="right")) + 40
    index_df = index_df.iloc[:icut]
    bench = index_df["close"].astype(float)

    out: dict[str, Any] = {
        "provenance": provenance,
        "start": start,
        "end": end,
        "signals": len(scored),
        "ladders": {},
    }

    for name, pp in ladders().items():
        res = AESPortfolioBacktester(params=pp).run(
            frames, index_df, scored, min_bucket="wait_and_watch"
        )
        eq = res.equity_curve
        strat_eq = eq["equity"].astype(float)
        strat_pos = eq["positions_value"].astype(float)
        trades = res.trades_frame

        bk = simulate(strat_eq, strat_pos, gold, LC.BOOK)
        curve = bk.curve

        strat_m = summarise(
            strat_eq, trades, bench,
            eq["exposure"] if "exposure" in eq else None,
        )
        book_m = summarise(curve["book"], trades, bench, curve["deployment"])

        s_years, s_folds = _annual_returns(curve, "strategy", ret_col="strat_ret")
        b_years, b_folds = _annual_returns(curve, "book")

        out["ladders"][name] = {
            "params": {k: asdict(pp)[k] for k in _PARAM_KEYS},
            "n_positions": (
                int(trades.groupby(["symbol", "entry_date"]).ngroups)
                if not trades.empty else 0
            ),
            "n_traded_rows": int(len(trades)),
            "strategy": strat_m,
            "book": book_m,
            "strategy_concentration": concentration(s_folds),
            "book_concentration": concentration(b_folds),
            "fold_years": s_years,
            "strategy_folds_pct": [round(x, 2) for x in s_folds],
            "book_fold_years": b_years,
            "book_folds_pct": [round(x, 2) for x in b_folds],
            "diagnostics": bk.diagnostics,
            "gold_correlation_by_year": (
                correlation_by_year(bk, gold).round(4).to_dict("records")
            ),
            "curve_span": [str(curve.index[0].date()), str(curve.index[-1].date())],
        }
    return out


def main() -> dict[str, Any]:
    assert_live_config()          # refuse to compute on a drifted config

    cfg = get_config()
    from .aes_calibration import load_universe

    ohlcv = str(cfg.paths.ohlcv_dir)
    frames = load_universe(ohlcv)
    index_df = pd.read_parquet(os.path.join(ohlcv, "CRSLDX.parquet"))
    gold, gold_info = load_gold(os.path.join(ohlcv, "GOLDBEES.parquet"))

    res: dict[str, Any] = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "frozen_on": str(LC.FROZEN_ON),
        "config": LC.describe(),
        "gold_info": gold_info,
        "windows": {},
    }
    for label, (start, end, prov) in WINDOWS.items():
        res["windows"][label] = run_window(
            frames, index_df, gold, start, end, prov
        )

    out = os.path.join(str(cfg.paths.results_dir), "leadflow_frozen_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    res["_path"] = out
    return res


__all__ = ["WINDOWS", "ladders", "main", "run_window"]
