"""Market-behaviour hypotheses for short-swing trading in Indian mid/small caps.

Each hypothesis names a *phenomenon* and a reason it should exist, then commits
to a measurable signature. None of them is an indicator level; several are
sequences or cross-sectional comparisons, because a level is a state and most of
the exploitable behaviour here is a *transition*.

The structural facts about this market that the hypotheses lean on:

* **Thin, episodic liquidity.** A smallcap order book can be empty enough that a
  modest order moves price several percent. Such a move carries no information.
* **Retail-heavy, leverage-heavy participation.** Margin funding and pledged
  promoter holdings create forced sellers who trade on a deadline rather than on
  a view -- an inventory shock, not an information shock.
* **Strong thematic/sector co-movement.** Real news about a business usually
  moves its peers too. A move that leaves peers untouched is more likely flow.
* **Circuit bands.** Price discovery is interrupted rather than continuous,
  which delays and then concentrates adjustment.

The common thread: **separate information from flow.** Information should be
followed; flow should be faded. Every hypothesis below is an attempt to
measure which of the two just happened.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy import indicators as ind

logger = logging.getLogger(__name__)

#: A hypothesis maps a per-symbol feature frame to a boolean event mask.
EventFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One testable market-behaviour hypothesis."""

    name: str
    phenomenon: str
    why: str
    horizon: str
    fails_when: str
    fn: EventFn
    direction: str = "long"


def _mask(series: pd.Series, frame: pd.DataFrame) -> pd.Series:
    """Coerce to a clean boolean mask aligned to the frame."""
    return series.reindex(frame.index).fillna(False).astype(bool)


# --------------------------------------------------------------------------- #
# Shared feature construction
# --------------------------------------------------------------------------- #
def build_features(
    ohlcv: pd.DataFrame,
    *,
    sector_return: pd.Series | None = None,
    market_return: pd.Series | None = None,
    sector_gap: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute the feature set every hypothesis draws from.

    All rolling statistics use only bars up to and including *t*; baselines that
    describe a "normal level" exclude *t* itself so an event is not measured
    against a baseline that already contains it.

    Args:
        ohlcv: Daily bars for one symbol.
        sector_return: Peer-group median return series, aligned by date. Already
            computed cross-sectionally from bars up to *t*.
        market_return: Benchmark return series.
        sector_gap: Peer-group median overnight gap.

    Returns:
        Feature frame indexed like ``ohlcv``.
    """
    f = ohlcv.copy()
    f.index = pd.DatetimeIndex(f.index).tz_localize(None).normalize()
    f = f[~f.index.duplicated(keep="last")].sort_index()

    o, h, low, c, v = f["open"], f["high"], f["low"], f["close"], f["volume"].astype(float)

    # -- participation ------------------------------------------------------ #
    f["vol_baseline"] = ind.rolling_baseline(v, 20)
    f["vol_ratio"] = ind.safe_ratio(v, f["vol_baseline"])
    f["turnover"] = c * v
    f["turnover_baseline"] = ind.rolling_baseline(f["turnover"], 20)

    # -- returns ------------------------------------------------------------ #
    for k in (1, 2, 3, 5, 10):
        f[f"ret_{k}"] = ind.pct_return(c, k)
    f["gap"] = o / c.shift(1) - 1.0

    # -- volatility --------------------------------------------------------- #
    f["atr_14"] = ind.atr(h, low, c, 14)
    f["atr_pct"] = f["atr_14"] / c
    f["atr_5"] = ind.atr(h, low, c, 5)
    f["atr_ratio"] = ind.safe_ratio(f["atr_5"], ind.atr(h, low, c, 20))
    # Move measured in units of the stock's own recent volatility, so a 5% move
    # in a quiet name and a 5% move in a wild one are not treated alike.
    f["ret_3_atr"] = f["ret_3"] / f["atr_pct"].replace(0, np.nan)
    f["ret_1_atr"] = f["ret_1"] / f["atr_pct"].replace(0, np.nan)

    # -- structure ---------------------------------------------------------- #
    f["range_position"] = ind.close_range_position(h, low, c)
    f["low_20"] = ind.rolling_low(low, 20)
    f["high_20"] = ind.rolling_high(h, 20)
    f["sma_50"] = ind.sma(c, 50)
    f["sma_200"] = ind.sma(c, 200)
    f["above_50"] = (c > f["sma_50"]).fillna(False)
    f["above_200"] = (c > f["sma_200"]).fillna(False)

    # Consecutive lower closes ending at t.
    down = (c.diff() < 0).astype(int)
    streak = down.groupby((down != down.shift()).cumsum()).cumsum()
    f["down_streak"] = streak.where(down == 1, 0)

    # -- relative behaviour -------------------------------------------------- #
    if sector_return is not None:
        aligned = sector_return.reindex(f.index)
        f["sector_ret_3"] = aligned
        f["rel_sector_3"] = f["ret_3"] - aligned
    else:
        f["sector_ret_3"] = np.nan
        f["rel_sector_3"] = np.nan

    if market_return is not None:
        aligned = market_return.reindex(f.index)
        f["market_ret_3"] = aligned
        f["rel_market_3"] = f["ret_3"] - aligned
    else:
        f["market_ret_3"] = np.nan
        f["rel_market_3"] = np.nan

    if sector_gap is not None:
        f["sector_gap"] = sector_gap.reindex(f.index)
        f["rel_gap"] = f["gap"] - f["sector_gap"]
    else:
        f["sector_gap"] = np.nan
        f["rel_gap"] = np.nan

    return f


def build_sector_aggregates(
    prices: Mapping[str, pd.DataFrame],
    industry_of: Mapping[str, str],
    *,
    lookback: int = 3,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], pd.Series]:
    """Cross-sectional peer aggregates, computed without look-ahead.

    For each industry and each date, the median k-day return and median overnight
    gap across that industry's members. Only bars up to and including *t* enter
    the calculation, so a value at *t* is knowable at *t*.

    Note the one unavoidable compromise: industry membership comes from today's
    constituent list, so a stock's peer group is assigned with information that
    was not available historically. The effect is second-order (industry
    classification is stable) but it is a real deviation from point-in-time
    purity and is recorded as such.

    Returns:
        ``(sector_returns, sector_gaps, market_return)`` where the first two are
        keyed by industry name.
    """
    returns: dict[str, pd.DataFrame] = {}
    gaps: dict[str, pd.DataFrame] = {}

    for symbol, frame in prices.items():
        if frame is None or frame.empty or len(frame) < lookback + 2:
            continue
        industry = industry_of.get(symbol)
        if not industry:
            continue
        idx = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        close = pd.Series(frame["close"].to_numpy(), index=idx)
        open_ = pd.Series(frame["open"].to_numpy(), index=idx)
        returns.setdefault(industry, pd.DataFrame())[symbol] = close / close.shift(lookback) - 1.0
        gaps.setdefault(industry, pd.DataFrame())[symbol] = open_ / close.shift(1) - 1.0

    sector_returns = {
        industry: frame.median(axis=1, skipna=True).sort_index()
        for industry, frame in returns.items()
    }
    sector_gaps = {
        industry: frame.median(axis=1, skipna=True).sort_index()
        for industry, frame in gaps.items()
    }

    all_returns = pd.concat(returns.values(), axis=1) if returns else pd.DataFrame()
    market_return = (
        all_returns.median(axis=1, skipna=True).sort_index()
        if not all_returns.empty else pd.Series(dtype=float)
    )
    return sector_returns, sector_gaps, market_return


# --------------------------------------------------------------------------- #
# H1 - Idiosyncratic dislocation: flow shock, not information
# --------------------------------------------------------------------------- #
def h1_idiosyncratic_drop(f: pd.DataFrame) -> pd.Series:
    """Stock falls hard while its sector does not.

    A business-specific shock large enough to move a stock 2+ ATRs usually moves
    its peers as well, because the news is about the industry. When peers are
    flat, the move is more likely one participant liquidating than the market
    repricing the asset.
    """
    idiosyncratic = f["rel_sector_3"] < -0.05     # 5pp worse than peers over 3 days
    sector_calm = f["sector_ret_3"].abs() < 0.02  # sector itself went nowhere
    real_move = f["ret_3_atr"] < -1.5             # in the stock's own vol units
    return _mask(idiosyncratic & sector_calm & real_move, f)


# --------------------------------------------------------------------------- #
# H2 - Trapped sellers: the failed breakdown
# --------------------------------------------------------------------------- #
def h2_failed_breakdown(f: pd.DataFrame) -> pd.Series:
    """Price breaks a 20-day low intraday, then closes back above it.

    Breaking a visible low triggers resting stops and attracts momentum sellers.
    If price closes back above the level the same day, every one of those
    participants is offside and their covering supplies the demand for a short,
    sharp squeeze.
    """
    broke = f["low"] < f["low_20"]
    reclaimed = f["close"] > f["low_20"]
    conviction = f["range_position"] > 0.5        # closed in the upper half
    participation = f["vol_ratio"] > 1.0          # someone actually traded it
    return _mask(broke & reclaimed & conviction & participation, f)


# --------------------------------------------------------------------------- #
# H3 - Absorption: a decline that stops going down on heavy volume
# --------------------------------------------------------------------------- #
def h3_absorption(f: pd.DataFrame) -> pd.Series:
    """A multi-day slide meets a high-volume bar that closes strong.

    Distribution ends when a buyer with size steps in. The signature is not the
    decline itself but the bar where heavy volume no longer produces a lower
    close -- supply is being absorbed rather than overwhelming demand.
    """
    slide = f["down_streak"].shift(1) >= 2        # was already falling
    heavy = f["vol_ratio"] > 1.5
    strong_close = f["range_position"] > 0.6
    return _mask(slide & heavy & strong_close, f)


# --------------------------------------------------------------------------- #
# H4 - Liquidity vacuum: a big move with nobody behind it
# --------------------------------------------------------------------------- #
def h4_liquidity_vacuum(f: pd.DataFrame) -> pd.Series:
    """A large decline on BELOW-average volume.

    This inverts the usual reading of volume. Conventional analysis treats
    volume as confirmation; here its absence is the signal. In a thin order book
    a modest sell order walks the price down several percent simply because
    there are no bids, not because anyone repriced the business. Once ordinary
    two-way flow resumes, the price recovers. The behaviour should be specific
    to illiquid markets -- which is exactly the tier this universe covers.
    """
    big_drop = f["ret_3_atr"] < -1.5
    no_participation = f["vol_ratio"] < 0.9       # fewer shares than usual traded
    return _mask(big_drop & no_participation, f)


# --------------------------------------------------------------------------- #
# H5 - Overnight flow shock: gap down that peers did not share
# --------------------------------------------------------------------------- #
def h5_orphan_gap(f: pd.DataFrame) -> pd.Series:
    """Stock gaps down while its sector opens flat, then closes off its low.

    Overnight moves are where flow imbalance concentrates: an order left to
    execute at the open, a margin call settled before the bell. If the sector did
    not gap, the cause is specific to the seller rather than to the business.
    """
    gapped = f["gap"] < -0.02
    orphan = f["rel_gap"] < -0.015                # peers did not gap with it
    recovered = f["range_position"] > 0.5         # intraday demand showed up
    return _mask(gapped & orphan & recovered, f)


# --------------------------------------------------------------------------- #
# Controls: generic setups that must be BEATEN, not merely matched
# --------------------------------------------------------------------------- #
def c1_oversold_rsi2(f: pd.DataFrame) -> pd.Series:
    """Plain RSI(2) oversold — the standard retail mean-reversion trigger."""
    return _mask(ind.rsi(f["close"], 2) < 10.0, f)


def c2_simple_drop(f: pd.DataFrame) -> pd.Series:
    """Any 3-day drop of more than 5%, with no conditioning at all."""
    return _mask(f["ret_3"] < -0.05, f)


def c3_breakout(f: pd.DataFrame) -> pd.Series:
    """20-day breakout — the momentum control."""
    return _mask(f["close"] > f["high_20"], f)


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        "H1_idiosyncratic_drop",
        "Stock falls 5pp more than its sector over 3 days while the sector is flat",
        "Real business news moves peers too; a solo move is inventory, not information",
        "3-8 bars",
        "When the idiosyncratic move IS information: earnings miss, fraud, promoter exit",
        h1_idiosyncratic_drop,
    ),
    Hypothesis(
        "H2_failed_breakdown",
        "Price breaks the 20-day low intraday then closes back above it",
        "Stops and momentum sellers are triggered, then trapped; covering forces price up",
        "2-5 bars",
        "When the breakdown is genuine and simply resumes the next day",
        h2_failed_breakdown,
    ),
    Hypothesis(
        "H3_absorption",
        "A 2+ day slide meets a heavy-volume bar that closes in its upper third",
        "Heavy volume that fails to make a lower close means supply is being absorbed",
        "3-7 bars",
        "In a genuine liquidation, heavy volume and strong closes alternate for weeks",
        h3_absorption,
    ),
    Hypothesis(
        "H4_liquidity_vacuum",
        "A 1.5-ATR decline on BELOW-average volume",
        "In a thin book price moves without information; normal flow restores it",
        "2-6 bars",
        "In a genuine repricing, low volume means nobody wants it at any price",
        h4_liquidity_vacuum,
    ),
    Hypothesis(
        "H5_orphan_gap",
        "Gaps down 2%+ while the sector does not, then closes off the low",
        "Overnight imbalance is where forced flow concentrates",
        "1-4 bars",
        "When the gap reflects stock-specific news released overnight",
        h5_orphan_gap,
    ),
    Hypothesis(
        "C1_oversold_rsi2", "RSI(2) < 10", "Control: the generic retail trigger",
        "n/a", "n/a", c1_oversold_rsi2,
    ),
    Hypothesis(
        "C2_simple_drop", "3-day return < -5%", "Control: unconditioned weakness",
        "n/a", "n/a", c2_simple_drop,
    ),
    Hypothesis(
        "C3_breakout", "Close > 20-day high", "Control: momentum",
        "n/a", "n/a", c3_breakout,
    ),
)


__all__ = [
    "HYPOTHESES",
    "EventFn",
    "Hypothesis",
    "build_features",
    "build_sector_aggregates",
]
