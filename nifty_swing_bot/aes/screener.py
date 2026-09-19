"""The daily screener (AES spec section 1.1) and hard deletes (section 1.2).

This is emphatically **not** the entry signal. It only decides who gets onto the
watchlist. Section 0 is blunt about why that matters: "Most backtests of 'buy
the breakout' fail because they buy on screener day." The screener finds a
stock that has already moved 20% and sits near its 52-week high -- which is
precisely the moment of maximum short-term stretch. Entry comes days or weeks
later, after a box has formed and resolved.

Everything here is vectorised over a single symbol's frame and returns a
boolean Series aligned to it, so a scan over the universe is one pass per
symbol rather than one pass per symbol per day.

**Known gap.** Section 1.1 also requires market cap above 500 Cr. There is no
market-cap history in this project's cache, and back-filling point-in-time
market cap is a data problem, not a modelling one. The turnover floor is a
partial proxy. This is recorded rather than silently dropped, because it means
the reconstructed universe is slightly wider than the one actually traded.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def screen(df: pd.DataFrame, params) -> pd.Series:
    """Bars on which this stock qualifies for watchlist admission.

    Args:
        df: OHLCV frame with an ascending date index.
        params: A ``ScreenerParams``.

    Returns:
        Boolean Series aligned to ``df.index``. Every condition reads the
        current bar or earlier, never a later one.
    """
    close = df["close"]
    volume = df["volume"]

    moved = close >= (1.0 + params.move_pct) * close.shift(params.move_lookback)

    # "At or near the 52-week high", inclusive of today: the high being made
    # today is the whole point of the condition.
    high_52w = close.rolling(params.high_lookback, min_periods=params.high_lookback).max()
    near_high = close >= high_52w * (1.0 - params.high_proximity)

    priced = close > params.min_price
    turnover = close * volume.rolling(params.turnover_lookback).mean()
    liquid = turnover >= params.min_turnover_inr

    qualifies = moved & near_high & priced & liquid
    return qualifies.fillna(False).astype(bool)


def circuit_hits(df: pd.DataFrame, params) -> pd.Series:
    """Rolling count of suspected circuit days (section 1.2).

    Exchange circuit data is not in the cache, so a circuit day is inferred
    from its signature: the bar has essentially no range and closes at a large
    move. An upper circuit opens, locks and never trades lower, so high, low
    and close coincide.

    This is an approximation and is flagged as one. The threshold ``N`` it feeds
    is a calibration target in Phase 3, which is the honest way to handle a
    proxy: sweep it rather than trust it.

    Returns:
        Integer Series: circuit-like bars in the trailing window, inclusive of
        the current bar.
    """
    move = df["close"].pct_change()
    span = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    locked = span <= params.circuit_max_range
    big = move.abs() >= params.circuit_min_move
    hits = (locked & big).astype(int)
    return hits.rolling(params.circuit_lookback, min_periods=1).sum().fillna(0).astype(int)


def hard_delete_mask(df: pd.DataFrame, params) -> pd.Series:
    """Bars on which section 1.2 says the name leaves the watchlist for good.

    Three reasons: the price band, sustained illiquidity below the screener
    floor, and being a frequent circuit mover.
    """
    close = df["close"]
    out_of_band = (close < params.delete_below_price) | (close > params.delete_above_price)

    turnover = close * df["volume"].rolling(params.turnover_lookback).mean()
    thin = turnover < params.min_turnover_inr
    sustained_thin = (
        thin.rolling(params.illiquid_sustained_bars, min_periods=1).min().fillna(0).astype(bool)
    )

    circuits = circuit_hits(df, params) > params.max_circuit_hits
    return (out_of_band | sustained_thin | circuits).fillna(False).astype(bool)


def screen_universe(
    frames: dict[str, pd.DataFrame], params
) -> pd.DataFrame:
    """Run the screener across a universe.

    Args:
        frames: ``{symbol: ohlcv frame}``.
        params: A ``ScreenerParams``.

    Returns:
        Long frame of ``(symbol, date, close, turnover)`` admission events,
        sorted by date.
    """
    rows: list[pd.DataFrame] = []
    for symbol, df in frames.items():
        if len(df) < params.high_lookback + params.move_lookback:
            continue
        hits = screen(df, params)
        if not hits.any():
            continue
        dates = df.index[hits]
        rows.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "close": df.loc[dates, "close"].to_numpy(),
                    "turnover": (
                        df["close"] * df["volume"].rolling(params.turnover_lookback).mean()
                    ).loc[dates].to_numpy(),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "close", "turnover"])
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


__all__ = ["circuit_hits", "hard_delete_mask", "screen", "screen_universe"]
