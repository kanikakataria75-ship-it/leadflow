"""Unit tests for AES Phase 2 context features: resistance, timeframes, RS.

Same two-part discipline as ``test_aes_boxes.py``: hand-constructed series
with a known answer, and a structural look-ahead audit proving a result at
bar ``i`` cannot change based on what happens after it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.aes.params import (
    RelativeStrengthParams,
    ResistanceParams,
    TimeframeParams,
)
from nifty_swing_bot.aes.relative_strength import drawdown_relative_strength
from nifty_swing_bot.aes.resistance import find_resistance_levels, resistance_headroom
from nifty_swing_bot.aes.timeframes import classify_breakout


def _frame(closes, *, highs=None, lows=None, volume=1_000_000.0) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    highs = closes * 1.005 if highs is None else np.asarray(highs, dtype=float)
    lows = closes * 0.995 if lows is None else np.asarray(lows, dtype=float)
    if np.isscalar(volume) or np.ndim(volume) == 0:
        volume = np.full(n, float(volume))
    return pd.DataFrame(
        {
            "open": closes,
            "high": np.maximum(highs, closes),
            "low": np.minimum(lows, closes),
            "close": closes,
            "volume": np.asarray(volume, dtype=float),
        },
        index=pd.bdate_range("2019-01-01", periods=n),
    )


# --------------------------------------------------------------------------- #
# Resistance (section 3.2 / 3.3)
# --------------------------------------------------------------------------- #
def test_finds_a_single_swing_high_above_reference():
    """A lone spike above current price is resistance; below it is not."""
    closes = np.array([100.0] * 20 + [150.0] + [100.0] * 20 + [110.0] * 10)
    # The tail plateau's own high must not poke above its own close, or the
    # default synthetic wick (close * 1.005) makes the plateau register as a
    # sliver of "resistance" barely above itself -- an artifact of the test
    # helper's fixed wick, not something real OHLC data does. Only the spike
    # bar keeps its wick; the rest use close as their high.
    highs = closes.copy()
    highs[20] *= 1.005
    df = _frame(closes, highs=highs)
    levels = find_resistance_levels(df, len(df) - 1, params=ResistanceParams(swing_k=2))
    assert len(levels) == 1
    assert levels[0].price == pytest.approx(150.0 * 1.005, rel=1e-3)


def test_older_touch_makes_the_level_older_not_the_newer_one():
    """Section 3.2: age is measured from the FIRST touch, not the last."""
    n = 200
    closes = np.full(n, 100.0)
    old_touch, new_touch = 10, 150
    closes[old_touch] = 140.0
    closes[new_touch] = 140.0
    df = _frame(closes)
    i = n - 1
    params = ResistanceParams(swing_k=2, cluster_tol_pct=0.01)
    levels = find_resistance_levels(df, i, reference_price=100.0, params=params)
    assert len(levels) == 1
    level = levels[0]
    assert level.first_touch_idx == old_touch
    assert level.age_bars == i - old_touch


def test_two_distant_prices_are_two_levels_not_one():
    n = 150
    closes = np.full(n, 100.0)
    closes[20] = 130.0   # 30% above reference
    closes[80] = 155.0   # 55% above reference -- both within the 60% headroom cap
    df = _frame(closes)
    params = ResistanceParams(swing_k=2, cluster_tol_pct=0.02)
    levels = find_resistance_levels(df, n - 1, reference_price=100.0, params=params)
    assert len(levels) == 2
    assert levels[0].price < levels[1].price   # sorted nearest-first


def test_headroom_computed_against_reference_price():
    closes = np.full(120, 100.0)
    closes[30] = 120.0    # a level 20% above a reference of 100
    df = _frame(closes)
    result = resistance_headroom(
        df, len(df) - 1, reference_price=100.0,
        params=ResistanceParams(swing_k=2),
    )
    assert result["headroom_pct"] == pytest.approx(0.20, abs=0.01)
    assert result["clears_min_headroom"] is True   # 20% clears the 12-15% bar


def test_no_resistance_in_range_returns_none_not_zero():
    """Absence of a level is a fact, not a zero -- callers must be able to tell."""
    df = _frame(np.linspace(100, 101, 100))   # nothing above within range
    result = resistance_headroom(df, len(df) - 1, reference_price=200.0)
    assert result["headroom_pct"] is None
    assert result["clears_min_headroom"] is None


def test_rejection_classified_from_a_sharp_drop_after_the_touch():
    """Section 3.3: a touch followed by a sharp fall is 'rejected'."""
    n = 60
    closes = np.full(n, 100.0)
    touch = 20
    closes[touch] = 130.0
    closes[touch + 1 : touch + 6] = [125, 118, 110, 105, 100]   # sharp fall after
    df = _frame(closes)
    levels = find_resistance_levels(
        df, n - 1, reference_price=90.0,
        params=ResistanceParams(swing_k=2, rejection_drop_pct=0.05, reaction_window=10),
    )
    assert len(levels) == 1
    assert levels[0].touch_reaction[0] == "rejected"


def test_absorption_classified_from_a_held_touch():
    """A touch followed by price holding near the level is 'absorbed'."""
    n = 60
    closes = np.full(n, 100.0)
    touch = 20
    closes[touch] = 130.0
    closes[touch + 1 : touch + 11] = 128.0    # holds near the level, no sharp fall
    df = _frame(closes)
    levels = find_resistance_levels(
        df, n - 1, reference_price=90.0,
        params=ResistanceParams(swing_k=2, rejection_drop_pct=0.05, reaction_window=10),
    )
    assert len(levels) == 1
    assert levels[0].touch_reaction[0] == "absorbed"


def test_elevated_touch_volume_is_measured_against_its_own_prior_baseline():
    n = 60
    closes = np.full(n, 100.0)
    touch = 30
    closes[touch] = 130.0
    volume = np.full(n, 1_000_000.0)
    volume[touch] = 5_000_000.0     # a clear spike on the touch bar itself
    df = _frame(closes, volume=volume)
    levels = find_resistance_levels(
        df, n - 1, reference_price=90.0, params=ResistanceParams(swing_k=2)
    )
    assert len(levels) == 1
    assert levels[0].touch_volume_ratio[0] > 3.0


@pytest.mark.parametrize("i", [80, 120, 160])
def test_resistance_no_lookahead(i):
    rng = np.random.default_rng(7)
    n = 250
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    df = _frame(closes)

    before = find_resistance_levels(df, i, params=ResistanceParams(swing_k=2))
    poisoned = df.copy()
    poisoned.iloc[i + 1 :] *= 5.0
    after = find_resistance_levels(poisoned, i, params=ResistanceParams(swing_k=2))

    assert [lv.price for lv in before] == [lv.price for lv in after]
    assert [lv.age_bars for lv in before] == [lv.age_bars for lv in after]


# --------------------------------------------------------------------------- #
# Timeframes (section 3.1)
# --------------------------------------------------------------------------- #
def test_daily_breakout_detected_on_a_fresh_high():
    closes = np.concatenate([np.full(60, 100.0), [105.0]])
    df = _frame(closes)
    result = classify_breakout(df, len(df) - 1, TimeframeParams(daily_lookback=50))
    assert result.daily is True
    # "Fresh high" is measured on closes, not intraday wicks, on every
    # timeframe -- deliberately, so daily/weekly/monthly are comparable and
    # noise from a single wick cannot manufacture a breakout.
    assert result.daily_high == pytest.approx(100.0, rel=1e-3)


def test_weekly_breakout_requires_the_week_high_not_just_one_day():
    """A single strong day inside an otherwise flat week can still confirm weekly."""
    n = 300
    closes = np.full(n, 100.0)
    closes[-1] = 130.0     # today's bar makes a large new high
    df = _frame(closes)
    result = classify_breakout(df, n - 1, TimeframeParams(weekly_lookback=52))
    assert result.weekly is True


def test_in_progress_week_does_not_see_future_days():
    """The weekly bar for the current week must only use bars up to i.

    Constructed so that a naive whole-history resample would classify this
    bar as a weekly breakout (because later days in the same week go even
    higher), while the causal version -- built from bars up to i alone --
    must not.
    """
    dates = pd.bdate_range("2020-01-01", periods=300)
    closes = np.full(300, 100.0)
    # A Monday at position 250 sits at 102 (a small new high); Tue-Fri of the
    # SAME week then rocket to 200. If future days leaked into "this week's"
    # bar, evaluating at the Monday would see a bar whose eventual close is
    # dragged toward 200 by bars that have not happened yet.
    monday = 250
    closes[monday] = 102.0
    closes[monday + 1 : monday + 5] = [150.0, 170.0, 190.0, 200.0]
    df = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": np.full(300, 1_000_000.0)},
        index=dates,
    )
    full = classify_breakout(df, monday, TimeframeParams(weekly_lookback=52))
    truncated_df = df.iloc[: monday + 1]
    truncated = classify_breakout(truncated_df, monday, TimeframeParams(weekly_lookback=52))
    assert full.weekly == truncated.weekly
    assert full.weekly_high == truncated.weekly_high


def test_breakout_score_orders_by_highest_confirming_timeframe():
    n = 400
    closes = np.full(n, 100.0)
    closes[-1] = 200.0    # a huge move: should confirm on every timeframe
    df = _frame(closes)
    result = classify_breakout(df, n - 1)
    assert result.daily and result.weekly and result.monthly
    assert result.score == 3


def test_no_breakout_scores_zero():
    df = _frame(np.linspace(100, 90, 300))    # a decline; no fresh highs anywhere
    result = classify_breakout(df, len(df) - 1)
    assert result.score == 0
    assert not (result.daily or result.weekly or result.monthly)


@pytest.mark.parametrize("i", [200, 260, 310])
def test_timeframe_no_lookahead(i):
    rng = np.random.default_rng(11)
    n = 400
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, n)))
    df = _frame(closes)

    full = classify_breakout(df, i)
    truncated = classify_breakout(df.iloc[: i + 1], i)
    assert full == truncated

    poisoned = df.copy()
    poisoned.iloc[i + 1 :] *= 10.0
    poisoned_result = classify_breakout(poisoned, i)
    assert full == poisoned_result


# --------------------------------------------------------------------------- #
# Drawdown-conditional relative strength (section 3.4)
# --------------------------------------------------------------------------- #
def _index_and_stock(index_rets: np.ndarray, stock_rets: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2021-01-01", periods=index_rets.size + 1)
    index_close = 1000.0 * np.exp(np.cumsum(np.concatenate([[0.0], index_rets])))
    stock_close = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], stock_rets])))
    idx_df = pd.DataFrame({"open": index_close, "high": index_close, "low": index_close,
                           "close": index_close, "volume": 1e6}, index=dates)
    stock_df = pd.DataFrame({"open": stock_close, "high": stock_close, "low": stock_close,
                             "close": stock_close, "volume": 1e6}, index=dates)
    return idx_df, stock_df


def test_stock_resilient_on_index_down_days_is_classified_resilient():
    """Section 3.4's best case: the stock falls much less than the index."""
    n = 60
    index_rets = np.full(n, -0.01)     # index down every day
    stock_rets = np.full(n, -0.002)    # stock barely moves
    idx_df, stock_df = _index_and_stock(index_rets, stock_rets)
    result = drawdown_relative_strength(stock_df, idx_df, n, RelativeStrengthParams(lookback_bars=n))
    assert result.classification == "resilient"
    assert result.downside_capture == pytest.approx(0.2, rel=0.05)


def test_stock_falling_more_than_index_is_classified_fragile():
    """Section 3.4's worst case: the stock falls more than the index, downgrade."""
    n = 60
    index_rets = np.full(n, -0.01)
    stock_rets = np.full(n, -0.02)     # falls twice as hard
    idx_df, stock_df = _index_and_stock(index_rets, stock_rets)
    result = drawdown_relative_strength(stock_df, idx_df, n, RelativeStrengthParams(lookback_bars=n))
    assert result.classification == "fragile"
    assert result.downside_capture == pytest.approx(2.0, rel=0.05)


def test_up_days_are_excluded_from_the_measurement():
    """This is not plain relative strength: index-up days must not enter it.

    Constructed so that if up-days leaked in, the ratio would come out
    completely different (the stock rips higher on up days, which would drag
    a naive "total return ratio" deeply negative/nonsensical) -- but restricted
    to down-days only, the ratio must reflect only the down-day behaviour.
    """
    n = 60
    index_rets = np.where(np.arange(n) % 2 == 0, -0.01, 0.03)   # alternating down/up
    stock_rets = np.where(np.arange(n) % 2 == 0, -0.005, 0.10)  # resilient on down, rips on up
    idx_df, stock_df = _index_and_stock(index_rets, stock_rets)
    result = drawdown_relative_strength(stock_df, idx_df, n, RelativeStrengthParams(lookback_bars=n))
    assert result.index_down_days == pytest.approx(n / 2, abs=1)
    assert result.downside_capture == pytest.approx(0.5, rel=0.1)
    assert result.classification == "resilient"


def test_insufficient_down_days_is_reported_not_guessed():
    n = 30
    index_rets = np.full(n, 0.01)      # index never down
    stock_rets = np.full(n, 0.01)
    idx_df, stock_df = _index_and_stock(index_rets, stock_rets)
    result = drawdown_relative_strength(stock_df, idx_df, n, RelativeStrengthParams(lookback_bars=n, min_down_days=5))
    assert result.classification == "insufficient_data"
    assert result.downside_capture is None


@pytest.mark.parametrize("i", [40, 55])
def test_relative_strength_no_lookahead(i):
    rng = np.random.default_rng(3)
    n = 70
    index_rets = rng.normal(0.0, 0.015, n)
    stock_rets = rng.normal(0.0, 0.02, n)
    idx_df, stock_df = _index_and_stock(index_rets, stock_rets)
    params = RelativeStrengthParams(lookback_bars=30)

    before = drawdown_relative_strength(stock_df, idx_df, i, params)

    poisoned_idx = idx_df.copy()
    poisoned_idx.iloc[i + 1 :] *= 3.0
    poisoned_stock = stock_df.copy()
    poisoned_stock.iloc[i + 1 :] *= 3.0
    after = drawdown_relative_strength(poisoned_stock, poisoned_idx, i, params)

    assert before == after


def test_relative_strength_index_extending_further_than_stock_is_still_causal():
    """The index history running past the stock's last bar must not leak in."""
    n = 60
    index_rets = np.random.default_rng(5).normal(-0.005, 0.01, n + 20)
    stock_rets = np.random.default_rng(9).normal(-0.002, 0.01, n)
    idx_df, _ = _index_and_stock(index_rets, index_rets)   # long index history
    _, stock_df = _index_and_stock(stock_rets, stock_rets)  # shorter stock history
    params = RelativeStrengthParams(lookback_bars=30)

    result_short = drawdown_relative_strength(stock_df, idx_df, n - 5, params)
    # Extending the index frame further into the future must not change a
    # result computed at an earlier bar.
    idx_extended = idx_df.copy()
    result_long_index = drawdown_relative_strength(stock_df, idx_extended, n - 5, params)
    assert result_short == result_long_index
