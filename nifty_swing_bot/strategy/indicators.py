"""Vectorised technical indicators used by the DAB rule set.

Everything here is pure: given a frame in, a Series/frame out, no I/O and no
config. That keeps the rules unit-testable against synthetic data with known
answers.

**Look-ahead discipline.** Several helpers take an ``exclude_today`` flag. A
"spike versus baseline" comparison is only meaningful if the baseline does not
already contain the spike, so rolling baselines default to the *prior* N bars.
Indicators that describe the current bar's state (ATR, range position) do
include it. No function in this module ever reads a future bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range.

    ``max(high - low, |high - prev_close|, |low - prev_close|)``. The first bar
    has no previous close, so it falls back to the plain high-low range.
    """
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = ranges.max(axis=1)
    tr.iloc[0] = (high.iloc[0] - low.iloc[0]) if len(high) else np.nan
    return tr.rename("true_range")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    *,
    method: str = "wilder",
) -> pd.Series:
    """Average True Range.

    Args:
        high, low, close: Price series sharing an index.
        period: Averaging period.
        method: ``"wilder"`` for Wilder's RMA smoothing (the standard, and what
            charting packages show), or ``"sma"`` for a simple moving average.

    Returns:
        ATR series named ``atr_{period}``.
    """
    tr = true_range(high, low, close)
    if method == "sma":
        out = tr.rolling(period, min_periods=period).mean()
    else:
        # Wilder's smoothing is an EWM with alpha = 1/period. min_periods keeps
        # the early, under-seeded values as NaN rather than emitting a number
        # computed from two bars.
        out = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return out.rename(f"atr_{period}")


def rolling_baseline(
    series: pd.Series,
    window: int,
    *,
    exclude_today: bool = True,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling mean used as a "normal level" baseline.

    Args:
        series: Input series.
        window: Lookback length.
        exclude_today: Shift the window back one bar so the current value is not
            part of its own baseline. This is what makes a spike ratio honest.
        min_periods: Defaults to ``window`` so partial windows stay NaN.

    Returns:
        The rolling mean, aligned to the input index.
    """
    base = series.shift(1) if exclude_today else series
    return base.rolling(window, min_periods=min_periods or window).mean()


def rolling_high(
    series: pd.Series,
    window: int,
    *,
    exclude_today: bool = True,
    min_periods: int | None = None,
) -> pd.Series:
    """Highest value over the lookback window.

    ``exclude_today=True`` gives the prior N-bar high, which is the only
    definition under which "close breaks above the 20-day high" can ever be
    true -- including today's own high would make the test self-defeating.
    """
    base = series.shift(1) if exclude_today else series
    return base.rolling(window, min_periods=min_periods or window).max()


def rolling_low(
    series: pd.Series,
    window: int,
    *,
    exclude_today: bool = True,
    min_periods: int | None = None,
) -> pd.Series:
    """Lowest value over the lookback window."""
    base = series.shift(1) if exclude_today else series
    return base.rolling(window, min_periods=min_periods or window).min()


def close_range_position(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Where the close sits within the bar's high-low range.

    Returns 0.0 at the low, 1.0 at the high. A doji bar where high == low has no
    meaningful range; it is reported as 1.0 because an unchanged bar closed at
    its (single) price, i.e. at its high, and treating it as 0.0 would silently
    veto otherwise valid signals on thin days.
    """
    span = high - low
    pos = (close - low) / span.replace(0, np.nan)
    return pos.fillna(1.0).clip(0.0, 1.0).rename("range_position")


def pct_return(series: pd.Series, periods: int) -> pd.Series:
    """Simple N-period return as a fraction (0.05 == +5%)."""
    return (series / series.shift(periods) - 1.0).rename(f"ret_{periods}")


def swing_low(low: pd.Series, lookback: int = 3) -> pd.Series:
    """Most recent confirmed swing low.

    A bar is a swing low when its low is the minimum of the window centred on
    it. Confirmation needs ``lookback`` bars on the right, so the result is
    shifted forward by that many bars -- a swing low is only *known* to be one
    after those bars have printed. Values are forward-filled so every bar
    carries the latest confirmed swing low.
    """
    window = 2 * lookback + 1
    is_pivot = low == low.rolling(window, center=True, min_periods=window).min()
    # Shift by `lookback` so we only use the pivot once its right side exists.
    confirmed = low.where(is_pivot).shift(lookback)
    return confirmed.ffill().rename("swing_low")


def rolling_mean_volume(volume: pd.Series, window: int, *, exclude_today: bool = True) -> pd.Series:
    """Baseline volume, excluding today by default."""
    return rolling_baseline(volume, window, exclude_today=exclude_today).rename(
        f"vol_avg_{window}"
    )


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise ratio with zero/NaN denominators mapped to NaN.

    Used for the spike ratios, where a zero baseline (a stock that did not trade
    for the whole lookback) must not become an infinite spike.
    """
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window, min_periods=window).mean().rename(f"sma_{window}")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Uses the same RMA smoothing as :func:`atr`, so values match what charting
    packages display rather than the simple-average variant.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # A window with no losses is maximally overbought, not undefined.
    return out.where(avg_loss != 0, 100.0).rename(f"rsi_{period}")


__all__ = [
    "atr",
    "close_range_position",
    "pct_return",
    "rolling_baseline",
    "rolling_high",
    "rolling_low",
    "rolling_mean_volume",
    "rsi",
    "safe_ratio",
    "sma",
    "swing_low",
    "true_range",
]
