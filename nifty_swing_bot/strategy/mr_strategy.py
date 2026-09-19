"""Pullback Reversion (PBR) — the strategy the evidence actually supports.

The DAB rule set was measured to be *anti*-predictive: its entries underperformed
a random pick from the same universe by ~0.59pp over 10 bars, because every one
of its five rules selects for short-term extension, and extension in the NSE
mid/small-cap tier mean-reverts.

This module inverts that. It buys **weakness inside strength**:

* **Trend filter** — close above its 50-day average. The stock must still be in
  an uptrend; this is what separates a pullback from a falling knife.
* **Dip trigger** — RSI(2) below a threshold, i.e. two days of concentrated
  selling. A fast oscillator is essential: RSI(14) needs so much sustained
  decline to reach oversold that it drags price under the 50-day average, so
  "oversold *and* in an uptrend" is nearly an empty set on a slow RSI.
* **Optional pullback depth** — a minimum 3-day decline, so the setup is a real
  flush rather than a drift.

**Exits are reversion-based, not clock-based.** A mean-reversion trade is over
when the thing reverts, so the position closes when price recovers above a short
moving average or the oscillator normalises. The max-hold clock remains as a
backstop, not as the primary exit. Reusing DAB's exit stack here would be a
category error.

Measured edge over the universe base rate at a 10-bar horizon, on a
date-separated train/holdout split:

===========================  ==============  ==============
setup                        train           holdout
===========================  ==============  ==============
RSI(2) < 10 above 50d MA     +0.715% (t=6.5)  +0.345% (t=3.5)
Pullback + RSI(2) < 15       +0.714% (t=7.3)  +0.355% (t=4.0)
DAB (for comparison)         -0.997%          -0.697%
===========================  ==============  ==============

Holdout edges run roughly half of train, which is the normal shrinkage; the sign
and significance survive. That is a shortlist entry, not a validated system --
the portfolio backtest is what decides whether it survives costs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import AppConfig, MRParams, RiskParams, get_config
from . import indicators as ind
from .dab_strategy import prepare_input

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume",
    "sma_trend", "sma_exit", "rsi_fast", "ret_pullback",
    "cond_trend", "cond_dip", "cond_depth", "cond_liquidity",
    "atr_stop", "rank_score", "signal", "exit_signal",
)


def compute_features(
    ohlcv: pd.DataFrame,
    delivery: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | pd.Series | None = None,
    *,
    params: MRParams | None = None,
    risk: RiskParams | None = None,
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    """Compute Pullback Reversion entry and exit signals for one symbol.

    Args:
        ohlcv: Daily bars with ``open/high/low/close/volume``.
        delivery: Accepted and joined for display, but **not** used by any rule.
            The delivery thesis was tested in six formulations and none showed
            predictive value in this universe; keeping the column lets the UI
            still chart it without pretending it drives the signal.
        benchmark: Unused. Accepted so this module is drop-in compatible with the
            backtester's calling convention.
        params: Strategy parameters. Defaults to config.
        risk: Risk parameters, for the ATR stop column.
        cfg: Config override.

    Returns:
        A frame carrying :data:`FEATURE_COLUMNS`, including a boolean ``signal``
        for entries and ``exit_signal`` for reversion exits.
    """
    cfg = cfg or get_config()
    params = params or cfg.mr
    risk = risk or cfg.risk

    frame = prepare_input(ohlcv, delivery)
    if frame.empty:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))

    close, high, low = frame["close"], frame["high"], frame["low"]

    # -- Trend filter ------------------------------------------------------- #
    frame["sma_trend"] = ind.sma(close, params.trend_ma)
    cond_trend = close > frame["sma_trend"]
    if params.require_long_trend:
        long_ma = ind.sma(close, params.long_trend_ma)
        cond_trend &= close > long_ma
    frame["cond_trend"] = cond_trend.fillna(False)

    # -- Dip trigger --------------------------------------------------------- #
    frame["rsi_fast"] = ind.rsi(close, params.rsi_period)
    frame["cond_dip"] = (frame["rsi_fast"] < params.rsi_entry).fillna(False)

    # -- Pullback depth ------------------------------------------------------ #
    frame["ret_pullback"] = ind.pct_return(close, params.pullback_lookback)
    if params.min_pullback_pct > 0:
        frame["cond_depth"] = (frame["ret_pullback"] <= -params.min_pullback_pct).fillna(False)
    else:
        frame["cond_depth"] = True

    # -- Liquidity sanity ---------------------------------------------------- #
    # A dip on near-zero volume is usually a stale or illiquid print rather than
    # real selling, and the fill assumptions would not hold.
    vol_baseline = ind.rolling_baseline(frame["volume"].astype(float), 20)
    frame["cond_liquidity"] = (
        ind.safe_ratio(frame["volume"].astype(float), vol_baseline) > params.min_volume_ratio
    ).fillna(False)

    frame["atr_stop"] = ind.atr(high, low, close, risk.atr_stop_period)

    frame["signal"] = (
        frame["cond_trend"] & frame["cond_dip"] & frame["cond_depth"] & frame["cond_liquidity"]
    )

    # Conviction ranking for days when signals outnumber available slots, which
    # is most days. Deeper oversold = stronger snap-back expectation, so the
    # score inverts RSI. Without this the backtester falls back to sorting by
    # symbol name and the results measure nothing.
    frame["rank_score"] = 100.0 - frame["rsi_fast"]

    # -- Reversion exit (off by default) ------------------------------------- #
    # It is tempting to exit the moment price reverts -- the premise has played
    # out, so why hold? Measurement says otherwise, emphatically. The reversion
    # exit fires ~2.7 bars in, banking a small gain while the position carried
    # the full 2.5x ATR stop the whole time. Win rate goes UP (47.7% -> 55.7%)
    # and the strategy goes from +7.95% to -41.50%, because average win/average
    # loss collapses from 1.16 to 0.57.
    #
    # The entry edge was measured over 10 bars; exiting at bar 2.7 collects a
    # fraction of it and pays the whole round-trip cost. So the default is to
    # hold for `risk.max_hold_days` and let the clock do the work.
    frame["sma_exit"] = ind.sma(close, params.exit_ma)
    if params.use_reversion_exit:
        frame["exit_signal"] = (
            (close > frame["sma_exit"]) | (frame["rsi_fast"] > params.rsi_exit)
        ).fillna(False)
    else:
        frame["exit_signal"] = False

    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame


__all__ = ["FEATURE_COLUMNS", "compute_features"]
