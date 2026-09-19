"""Multi-timeframe breakout classification (AES spec section 3.1).

"Analysis is done on daily and weekly. Monthly is a one-time overview pass.
Breakout significance scales with timeframe: Daily breakout -> good; Weekly
breakout -> strong; Monthly breakout -> strongest." This module answers, for a
given daily bar, whether that bar is also a fresh high on the weekly and
monthly charts.

**Why this cannot be a plain resample-then-check.** ``df.resample("W").agg(...)``
over the *whole* history builds a weekly bar for the current week using every
daily bar that week eventually contains -- including days after the evaluation
date. A trader looking at their weekly chart on a Tuesday does not get to see
Thursday and Friday's candle yet, but a naive resample would hand it to them.
Every function here truncates to ``df.iloc[: i + 1]`` *before* resampling, so
the in-progress week or month is built only from bars that have actually
happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .params import TimeframeParams


@dataclass(frozen=True, slots=True)
class BreakoutClassification:
    """Which timeframes confirm a fresh high, as of one evaluation bar."""

    daily: bool
    weekly: bool
    monthly: bool
    daily_high: float
    weekly_high: float | None
    monthly_high: float | None

    @property
    def score(self) -> int:
        """Ordinal strength: 0 none, 1 daily, 2 weekly, 3 monthly.

        Not a claim about relative weight -- section 4's scorer decides that.
        This is only a compact way to say "the highest timeframe confirming".
        Monthly confirming implies checking weekly/daily separately still
        matters, since a name can make a fresh monthly high while pulling back
        within the week; callers wanting the full picture should read the
        three booleans, not just the ordinal.
        """
        if self.monthly:
            return 3
        if self.weekly:
            return 2
        if self.daily:
            return 1
        return 0


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV, keeping the final (possibly in-progress) bar as-is.

    The caller is responsible for having already truncated ``df`` to the
    evaluation bar -- this function has no bar index to enforce that itself,
    only the frame it is given.
    """
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(how="all")


def _fresh_high(closes: pd.Series, lookback: int, margin: float) -> tuple[bool, float]:
    """Is the final bar's close a fresh high over the ``lookback`` bars before it?

    Uses closing prices on every timeframe -- not the intraday high -- so that
    daily, weekly and monthly are measuring the same thing and a single wick
    cannot manufacture a breakout on its own. This mirrors the closing-basis
    convention the spec itself uses for stops (section 6.1).

    Returns:
        ``(is_breakout, prior_high)``. ``prior_high`` excludes the final bar,
        so it is always a meaningful comparison level even when there is not
        yet a full ``lookback`` bars of history (it just uses what exists).
    """
    if len(closes) < 2:
        return False, float("nan")
    history = closes.iloc[-(lookback + 1) : -1]
    if history.empty:
        return False, float("nan")
    prior_high = float(history.max())
    current = float(closes.iloc[-1])
    return current >= prior_high * (1.0 + margin), prior_high


def classify_breakout(
    df: pd.DataFrame, i: int, params: TimeframeParams | None = None
) -> BreakoutClassification:
    """Daily/weekly/monthly fresh-high confirmation as of bar ``i``.

    Args:
        df: Daily OHLCV frame, ascending date index.
        i: Evaluation bar. Only bars ``[0 .. i]`` are read.
        params: Thresholds.

    Returns:
        Which timeframes are making a fresh high, and the prior high on each.
    """
    params = params or TimeframeParams()
    truncated = df.iloc[: i + 1]

    daily_is, daily_prior = _fresh_high(truncated["close"], params.daily_lookback, params.breakout_margin_pct)

    weekly = _resample_ohlc(truncated, params.weekly_rule)
    weekly_is, weekly_prior = _fresh_high(weekly["close"], params.weekly_lookback, params.breakout_margin_pct)

    monthly = _resample_ohlc(truncated, params.monthly_rule)
    monthly_is, monthly_prior = _fresh_high(monthly["close"], params.monthly_lookback, params.breakout_margin_pct)

    return BreakoutClassification(
        daily=daily_is,
        weekly=weekly_is,
        monthly=monthly_is,
        daily_high=daily_prior,
        weekly_high=weekly_prior if not np.isnan(weekly_prior) else None,
        monthly_high=monthly_prior if not np.isnan(monthly_prior) else None,
    )


__all__ = ["BreakoutClassification", "classify_breakout"]
