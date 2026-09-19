"""Drawdown-conditional relative strength (AES spec section 3.4).

The spec is explicit that this is **not** "stock return minus index return"
over some window -- that is ordinary relative strength, and it rewards a stock
that simply went up more, which says nothing about how it behaves under
pressure. Instead:

    "During periods where Nifty 500 declined X%, measure the stock's decline:
    Fell less than the index -> best (strong hands holding). Fell roughly
    equal -> good. Fell more -> significantly negative, downgrade heavily."

This isolates the index's own down days over a trailing window and measures
what the stock did on exactly those days -- nothing else. A stock that hugs
the index when the tape is calm but barely moves when the index sells off
scores well here; a stock with an equally strong-looking chart that simply
beta-chases the index down on bad days does not, and the two are
indistinguishable under plain relative strength.

The result is a **downside-capture ratio**: the stock's cumulative return on
the index's down days, divided by the index's own cumulative return on those
same days. Both sums are negative, so a ratio below 1 means the stock fell
*less* than the index on the days that mattered.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .params import RelativeStrengthParams


@dataclass(frozen=True, slots=True)
class DrawdownRelativeStrength:
    """Section 3.4's measurement, as of one evaluation bar."""

    index_down_days: int
    index_decline_pct: float | None     # sum of index returns on its down days
    stock_decline_pct: float | None     # sum of the SAME days' stock returns
    downside_capture: float | None      # stock_decline / index_decline
    classification: str                 # "resilient" | "inline" | "fragile" | "insufficient_data"


def drawdown_relative_strength(
    stock: pd.DataFrame,
    index: pd.DataFrame,
    i: int,
    params: RelativeStrengthParams | None = None,
) -> DrawdownRelativeStrength:
    """Downside capture of ``stock`` against ``index`` over the trailing window.

    Args:
        stock: The stock's daily OHLCV frame, ascending date index.
        index: The benchmark's daily OHLCV frame (e.g. Nifty 500 / CRSLDX),
            same convention.
        i: Evaluation bar into ``stock``. Only bars ``[0 .. i]`` of ``stock``
            are read; ``index`` is aligned to those same dates, so a future
            index bar can never enter the calculation even if ``index``
            itself runs further.
        params: Thresholds.

    Returns:
        The decomposition and a classification band.
    """
    params = params or RelativeStrengthParams()
    stock_window = stock["close"].iloc[max(0, i - params.lookback_bars) : i + 1]
    if stock_window.size < params.min_down_days + 1:
        return DrawdownRelativeStrength(0, None, None, None, "insufficient_data")

    # Align by date, not by position: the two frames are not guaranteed to
    # share a calendar (a stock can have a trading holiday the index does
    # not, or a shorter listing history). Truncate the index to this bar's
    # own date *first* -- so a future index bar can never enter the
    # calculation even if ``index`` runs further than ``stock`` -- then
    # reindex onto the stock's own dates; any date missing from the index
    # becomes NaN and is dropped by the ``valid`` mask below.
    as_of = stock.index[i]
    idx_close = index["close"].loc[:as_of].reindex(stock_window.index)

    stock_ret = stock_window.pct_change()
    index_ret = idx_close.pct_change()

    valid = stock_ret.notna() & index_ret.notna()
    stock_ret, index_ret = stock_ret[valid], index_ret[valid]

    down_mask = index_ret <= params.index_down_threshold
    n_down = int(down_mask.sum())
    if n_down < params.min_down_days:
        return DrawdownRelativeStrength(n_down, None, None, None, "insufficient_data")

    index_decline = float(index_ret[down_mask].sum())
    stock_decline = float(stock_ret[down_mask].sum())

    if index_decline == 0:
        return DrawdownRelativeStrength(n_down, index_decline, stock_decline, None, "insufficient_data")

    capture = stock_decline / index_decline
    if capture <= params.capture_resilient_max:
        label = "resilient"
    elif capture >= params.capture_fragile_min:
        label = "fragile"
    else:
        label = "inline"

    return DrawdownRelativeStrength(n_down, index_decline, stock_decline, float(capture), label)


__all__ = ["DrawdownRelativeStrength", "drawdown_relative_strength"]
