"""Unit tests for AES nested box detection.

Two kinds of test live here.

**Hand-constructed answers.** Synthetic series are built with a box whose top,
bottom, duration and internal structure are known by construction, and the
detector is required to recover them. If a future change to the scoring breaks
the geometry, these fail with a readable diff rather than a shifted statistic.

**A structural look-ahead audit.** ``test_no_lookahead_*`` do not check a
threshold, they check an invariant: a detection at bar ``i`` must be identical
whether the frame ends at ``i`` or continues for years, and must not change when
the bars after ``i`` are overwritten with garbage. This is the same audit the
earlier DAB/MR studies carried, kept because it is the one bug class that
silently manufactures edge.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.aes.boxes import (
    _trim_isolated_extreme,
    detect_big_box,
    detect_nested_box,
    detect_small_box,
    scan_box_history,
    swing_low_indices,
)
from nifty_swing_bot.aes.params import BoxParams


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _frame(closes, *, highs=None, lows=None, volume=1_000_000.0) -> pd.DataFrame:
    """OHLCV frame from a close path, with optional explicit highs/lows."""
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    highs = closes * 1.005 if highs is None else np.asarray(highs, dtype=float)
    lows = closes * 0.995 if lows is None else np.asarray(lows, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": np.maximum(highs, closes),
            "low": np.minimum(lows, closes),
            "close": closes,
            "volume": np.full(n, float(volume)),
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )


def _leg_then_box(
    *,
    lead: int = 150,
    leg_from: float = 100.0,
    leg_to: float = 130.0,
    leg_bars: int = 20,
    box_lows: list[float],
    box_highs: list[float],
) -> pd.DataFrame:
    """Flat lead-in, an expansion leg, then a box with explicit bar extremes.

    The expansion leg's final bar is the intended box top, so the detector's
    anchor (the running-maximum high) should land exactly there.
    """
    lead_path = np.full(lead, leg_from)
    leg = np.linspace(leg_from, leg_to, leg_bars + 1)[1:]
    box_lows = np.asarray(box_lows, dtype=float)
    box_highs = np.asarray(box_highs, dtype=float)
    box_closes = (box_lows + box_highs) / 2.0

    closes = np.concatenate([lead_path, leg, box_closes])
    highs = np.concatenate([lead_path, leg, box_highs])
    lows = np.concatenate([lead_path, leg, box_lows])
    return _frame(closes, highs=highs, lows=lows)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_swing_low_indices_finds_hand_marked_lows():
    lows = np.array([10, 9, 8, 9, 10, 11, 9.5, 9.0, 9.5, 10, 11], dtype=float)
    #                          ^idx 2                 ^idx 7
    assert swing_low_indices(lows, k=2).tolist() == [2, 7]


def test_swing_low_needs_k_bars_on_each_side():
    """The last k bars can never be confirmed as swing lows."""
    lows = np.array([10, 9, 8, 7, 6], dtype=float)  # falling; final bar lowest
    assert swing_low_indices(lows, k=2).size == 0


def test_swing_low_window_too_short():
    assert swing_low_indices(np.array([1.0, 2.0]), k=2).size == 0


# --------------------------------------------------------------------------- #
# Big box geometry
# --------------------------------------------------------------------------- #
def test_big_box_recovers_known_top_bottom_and_duration():
    """A 130-top / 112-bottom box over 20 bars must be read back exactly."""
    lows = [112.0, 113.0, 114.0, 113.5, 115.0] * 4
    highs = [128.0, 127.0, 129.0, 128.5, 127.5] * 4
    df = _leg_then_box(box_lows=lows, box_highs=highs)
    box = detect_big_box(df, len(df) - 1)

    assert box is not None
    assert box.top == pytest.approx(130.0)      # the expansion leg's peak
    assert box.bottom == pytest.approx(112.0)   # lowest low inside the box
    assert box.bars == 21                       # peak bar + 20 box bars
    assert box.range_pct == pytest.approx(18.0 / 112.0, rel=1e-6)
    assert box.start_idx == len(df) - 21
    assert box.end_idx == len(df) - 1


def test_big_box_counts_higher_lows():
    """Ascending troughs, each a genuine local minimum with k bars either side."""
    # Troughs at box positions 2, 5, 8, 11 -> 112, 113, 114, 115.
    lows = [120.0, 120.0, 112.0, 120.0, 120.0, 113.0,
            120.0, 120.0, 114.0, 120.0, 120.0, 115.0, 120.0, 120.0]
    highs = [128.0] * len(lows)
    df = _leg_then_box(box_lows=lows, box_highs=highs)
    box = detect_big_box(df, len(df) - 1)

    assert box is not None
    assert box.swing_lows >= 4
    assert box.higher_lows == 3      # spec section 2.4 wants at least 2
    assert box.lows_tilt > 0         # visual upward tilt


def test_swing_low_plateau_convention():
    """A flat bottom yields one swing low at its first bar, not one per bar.

    The rule is strict on the left and non-strict on the right, which is what
    makes a plateau resolve to a single point. Pinned because it changes the
    higher-low count on the flat-bottomed boxes that are common in practice.
    """
    lows = np.array([10.0, 8.0, 8.0, 8.0, 10.0])
    assert swing_low_indices(lows, k=1).tolist() == [1]


def test_big_box_measures_time_above_midpoint():
    """Section 2.4's "% of closes above box midpoint", recomputed independently.

    The assertion is against the box the detector actually returned rather than
    against a hard-coded level, because which anchor wins is the detector's
    decision; what must be right is the arithmetic on top of it.
    """
    highs = [129.0] * 20
    lows = [112.0, 122.0, 122.0, 122.0, 113.0] + [122.0] * 15   # floor tested twice
    df = _leg_then_box(box_lows=lows, box_highs=highs)
    box = detect_big_box(df, len(df) - 1)

    assert box is not None
    window = df["close"].iloc[box.start_idx : box.end_idx + 1]
    assert box.mid == pytest.approx((box.top + box.bottom) / 2.0)
    assert box.pct_closes_above_mid == pytest.approx(float((window > box.mid).mean()))
    assert box.pct_closes_above_mid > 0.5   # this construction sits high


def test_big_box_rejects_range_outside_the_band():
    """15-25% is the spec's band; 7% and 45% are not boxes."""
    too_tight = _leg_then_box(box_lows=[126.0] * 20, box_highs=[129.0] * 20)
    assert detect_big_box(too_tight, len(too_tight) - 1) is None

    too_wide = _leg_then_box(box_lows=[85.0] * 20, box_highs=[129.0] * 20)
    assert detect_big_box(too_wide, len(too_wide) - 1) is None


def test_big_box_refuses_a_breakout_bar():
    """A window whose high is today is an expansion, not an accumulation."""
    lows = [112.0] * 20
    highs = [128.0] * 19 + [140.0]     # new high on the final bar
    df = _leg_then_box(box_lows=lows, box_highs=highs)
    assert detect_big_box(df, len(df) - 1) is None


def test_trim_isolated_extreme_drops_a_lone_wick():
    """One low far below everything else: excluded, the next value is the edge."""
    values = np.array([120.0, 121.0, 119.0, 122.0, 104.0])   # 104 is alone
    edge = _trim_isolated_extreme(values, "min", gap_pct=0.05, max_trim=3)
    assert edge == pytest.approx(119.0)


def test_trim_isolated_extreme_keeps_a_cluster():
    """Two lows close together: the rule says include -- the far one wins."""
    values = np.array([120.0, 121.0, 122.0, 104.0, 105.5])   # 104/105.5 cluster, ~1.4% apart
    edge = _trim_isolated_extreme(values, "min", gap_pct=0.05, max_trim=3)
    assert edge == pytest.approx(104.0)   # the far edge of the cluster, not 119-ish


def test_trim_isolated_extreme_respects_the_trim_ceiling():
    """A run of isolated points longer than max_trim stops being trimmed."""
    # Each point is >5% from its neighbour, so left alone this would trim all
    # the way to 100. With max_trim=2, only 70 and 80 may be discarded, and
    # the third-lowest (90) becomes the edge even though it is still isolated.
    values = np.array([100.0, 90.0, 80.0, 70.0])
    edge = _trim_isolated_extreme(values, "min", gap_pct=0.05, max_trim=2)
    assert edge == pytest.approx(90.0)


def test_big_box_floor_is_not_set_by_a_single_wick():
    """A lone spike low must not become the box bottom.

    Regression test for a bug visual validation caught: a stock consolidating
    in a 6% range printed one bar with a long lower tail, and the detector drew
    an 18.8% box whose floor no other bar had been near -- which then inflated
    every health metric, because dragging the midpoint down put every close
    above it.
    """
    # Twelve bars holding 120-128, then one bar tailing down to 104.
    lows = [120.0] * 12 + [104.0]
    highs = [128.0] * 13
    df = _leg_then_box(box_lows=lows, box_highs=highs)
    box = detect_big_box(df, len(df) - 1)

    if box is not None:
        assert box.bottom > 110.0, "the spike low was taken as the box floor"
        assert box.bottom_touches >= 2


def test_big_box_accepts_a_floor_touched_once_if_it_is_not_a_spike():
    """A single-visit floor is fine; only an *isolated* one is rejected.

    Guards the over-correction: an earlier fix required two separated visits to
    the floor, which forced price to have traded back down to it and so pushed
    the nested small box into the lower half by construction. Upper-zone boxes
    -- the §2.1 positive -- disappeared entirely. How well the floor is held is
    a scoring question (§4), not a detection gate.
    """
    # Floor at 112 touched once, but 113 sits right beside it: not a spike.
    lows = [112.0, 113.0] + [122.0] * 17
    highs = [129.0] * 19
    df = _leg_then_box(box_lows=lows, box_highs=highs)
    box = detect_big_box(df, len(df) - 1)
    assert box is not None
    assert box.bottom == pytest.approx(112.0)


def test_upper_zone_boxes_remain_reachable():
    """The spec's best setup must be producible by the detector at all.

    Not a threshold test -- a smoke alarm. A detection rule that makes
    "small box in the upper half of the big box" impossible has broken §2.1,
    and that failure is invisible in aggregate statistics.
    """
    # Deep pullback, then a tight 5.3% coil high in the big box, held long
    # enough (22 bars) that every admissible small-box window sits entirely
    # inside the coil and cannot reach back down into the pullback.
    lows = [112.0, 113.0, 114.0, 115.0] + [122.0] * 22
    highs = [129.0, 128.0, 118.0, 119.0] + [128.5] * 22
    df = _leg_then_box(box_lows=lows, box_highs=highs)
    nested = detect_nested_box(df, len(df) - 1, symbol="TEST")
    assert nested is not None
    assert nested.small is not None
    assert nested.small_zone == "upper", (
        f"position {nested.small_position:.2f} -> {nested.small_zone}"
    )


def test_big_box_rejects_a_clean_uptrend():
    """No stall means no box: the running maximum is always the last bar."""
    df = _frame(np.linspace(100, 200, 300))
    assert detect_big_box(df, len(df) - 1) is None


def test_big_box_needs_minimum_duration():
    """A box three bars old is below the section 2.3 floor."""
    lows = [112.0, 113.0, 114.0]
    highs = [128.0, 127.0, 129.0]
    df = _leg_then_box(box_lows=lows, box_highs=highs)
    assert detect_big_box(df, len(df) - 1) is None


# --------------------------------------------------------------------------- #
# Small box nesting
# --------------------------------------------------------------------------- #
def test_small_box_nests_in_upper_half_and_is_scored_upper():
    """A tight range sitting high inside the big box is the section 2.1 positive."""
    # Big box 112..130. Recent 12 bars coil between 124 and 129 (about 4%),
    # earlier bars establish the 112 bottom.
    lows = [112.0, 120.0, 120.0, 113.0, 120.0, 122.0] + [124.0] * 12
    highs = [120.0, 121.0, 122.0, 123.0, 125.0, 127.0] + [129.0] * 12
    df = _leg_then_box(box_lows=lows, box_highs=highs)

    big = detect_big_box(df, len(df) - 1)
    assert big is not None
    small = detect_small_box(df, len(df) - 1, big)
    assert small is not None
    assert small.top <= big.top + 1e-9
    assert small.bottom >= big.bottom - 1e-9

    position = (small.mid - big.bottom) / big.height
    assert position > 0.65, f"expected an upper-zone small box, got {position:.2f}"


def test_small_box_near_big_box_bottom_is_the_red_flag_case():
    """Section 2.1: a small box near the big-box bottom is a downgrade."""
    lows = [112.0] * 14 + [112.5] * 6
    highs = [130.0, 128.0] + [118.0] * 18
    df = _leg_then_box(box_lows=lows, box_highs=highs)

    nested = detect_nested_box(df, len(df) - 1, symbol="TEST")
    assert nested is not None
    assert nested.small is not None
    assert nested.small_zone == "bottom", nested.small_position


def test_small_box_range_band_and_nesting_hold_for_every_detection():
    """Whatever comes back must be 5-10% tall and inside the big box.

    Asserted as an invariant over many real detections rather than on one
    fixture. A single hand-built "too tight" series does not prove the band is
    enforced, because the small box sweeps window lengths: a 22-bar window over
    the same bars can easily land in the band even when the last five do not.
    """
    df = _real_frame()
    params = BoxParams()
    checked = 0
    for i in range(300, len(df), 3):
        big = detect_big_box(df, i, params)
        if big is None:
            continue
        small = detect_small_box(df, i, big, params)
        if small is None:
            continue
        checked += 1
        assert params.small_range_min <= small.range_pct <= params.small_range_max
        slack = params.small_wick_tolerance * big.height
        assert small.top <= big.top + slack
        assert small.bottom >= big.bottom - slack
        assert params.small_min_bars <= small.bars <= params.small_max_bars
        assert small.bars <= big.bars
    assert checked > 30, f"only {checked} small boxes seen; test is not exercising much"


def test_small_box_allows_wicks_outside_but_measures_containment():
    """Section 2.1: wicks may exceed the small box, closing bodies mostly not.

    The spike sits mid-box on purpose. Put on the final bar it would be a new
    high two bars old, which the detector correctly refuses to call a box at
    all -- a separate behaviour, covered by ``test_big_box_refuses_a_breakout_bar``.
    """
    lows = [112.0, 120.0, 120.0, 113.0] + [120.0] * 16
    highs = [128.0, 127.0, 126.0, 126.0] + [124.0] * 6 + [127.5] + [124.0] * 9
    df = _leg_then_box(box_lows=lows, box_highs=highs)
    big = detect_big_box(df, len(df) - 1)
    assert big is not None
    small = detect_small_box(df, len(df) - 1, big)
    assert small is not None
    # Quantile edges mean the spike can sit outside the small box while the
    # closing bodies stay in -- containment is a real measurement, not a 1.0.
    assert 0.0 < small.close_containment <= 1.0
    assert small.top < big.top


# --------------------------------------------------------------------------- #
# Context and history
# --------------------------------------------------------------------------- #
def test_nested_box_measures_the_expansion_leg_that_built_the_top():
    """Section 2.2: the box forms after an expansion; record how big it was."""
    df = _leg_then_box(
        leg_from=100.0, leg_to=130.0,
        box_lows=[112.0] * 20, box_highs=[128.0] * 20,
    )
    nested = detect_nested_box(df, len(df) - 1, symbol="TEST")
    assert nested is not None
    # Leg ran 100 -> 130, so the rise into the box top is about 30%.
    assert nested.prior_leg_pct == pytest.approx(0.30, abs=0.01)


def test_nested_box_detects_volume_dry_up():
    """Section 2.5: volume should fall off relative to the expansion leg."""
    df = _leg_then_box(box_lows=[112.0] * 20, box_highs=[128.0] * 20)
    df = df.copy()
    df.iloc[:-20, df.columns.get_loc("volume")] = 5_000_000.0
    df.iloc[-20:, df.columns.get_loc("volume")] = 1_000_000.0

    nested = detect_nested_box(df, len(df) - 1, symbol="TEST")
    assert nested is not None
    assert nested.vol_dryup_ratio < 1.0


def test_per_stock_norm_ignores_boxes_that_have_not_finished():
    """Section 2.3's norm may only use episodes that ended before this box."""
    df = _leg_then_box(box_lows=[112.0] * 20, box_highs=[128.0] * 20)
    i = len(df) - 1
    big = detect_big_box(df, i)
    assert big is not None

    history = scan_box_history(df, step=3)
    nested = detect_nested_box(df, i, symbol="TEST", history=history)
    assert nested is not None
    # Every box counted toward the norm must have closed before this one opened.
    assert all(b.end_idx < big.start_idx for b in history[: nested.prior_boxes])


def test_min_history_gate():
    df = _frame(np.linspace(100, 130, 60))
    assert detect_nested_box(df, 59) is None


# --------------------------------------------------------------------------- #
# Look-ahead audit (structural)
# --------------------------------------------------------------------------- #
def _real_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260912)
    n = 900
    steps = rng.normal(0.0006, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    noise = np.abs(rng.normal(0, 0.012, n))
    return _frame(close, highs=close * (1 + noise), lows=close * (1 - noise))


@pytest.mark.parametrize("i", [300, 450, 601, 750, 899])
def test_no_lookahead_truncation_invariance(i):
    """A detection at bar i must not change when later bars are removed."""
    df = _real_frame()
    full = detect_nested_box(df, i, symbol="T")
    truncated = detect_nested_box(df.iloc[: i + 1], i, symbol="T")
    assert (full is None) == (truncated is None)
    if full is not None:
        assert full.as_dict() == truncated.as_dict()


@pytest.mark.parametrize("i", [300, 450, 601, 750])
def test_no_lookahead_future_mutation_invariance(i):
    """Overwriting every bar after i with garbage must change nothing at i."""
    df = _real_frame()
    before = detect_nested_box(df, i, symbol="T")

    poisoned = df.copy()
    poisoned.iloc[i + 1 :] *= 7.5
    poisoned.iloc[i + 1 :, poisoned.columns.get_loc("volume")] = 9.9e9
    after = detect_nested_box(poisoned, i, symbol="T")

    assert (before is None) == (after is None)
    if before is not None:
        assert before.as_dict() == after.as_dict()


def test_no_lookahead_in_prior_cycle_count():
    """A past box's upward resolution may only count bars at or before i."""
    df = _real_frame()
    i = 600
    history = scan_box_history(df.iloc[: i + 1], step=3)

    poisoned = df.copy()
    poisoned.iloc[i + 1 :] *= 7.5     # every future bar a huge breakout
    a = detect_nested_box(df, i, symbol="T", history=history)
    b = detect_nested_box(poisoned, i, symbol="T", history=history)

    assert (a is None) == (b is None)
    if a is not None:
        assert a.prior_cycles == b.prior_cycles


def test_scan_box_history_is_causal():
    """Episodes found up to bar i must not shift when later data arrives."""
    df = _real_frame()
    i = 700
    truncated = scan_box_history(df.iloc[: i + 1], step=5)
    full = scan_box_history(df, step=5, end_idx=i)
    assert [(b.start_idx, b.end_idx) for b in truncated] == [
        (b.start_idx, b.end_idx) for b in full
    ]


# --------------------------------------------------------------------------- #
# Params
# --------------------------------------------------------------------------- #
def test_box_params_reject_inverted_bands():
    with pytest.raises(ValueError):
        dataclasses.replace(BoxParams(), big_range_min=0.30, big_range_max=0.25)
    with pytest.raises(ValueError):
        dataclasses.replace(BoxParams(), zone_bottom_max=0.8, zone_upper_min=0.2)


def test_quality_is_not_a_proxy_for_duration():
    """The grade must describe shape, not length.

    Count-based structure metrics made quality correlate 0.60 with bar count,
    which meant the detector was choosing the longest candidate rather than the
    best-formed one. Rates fixed that; this test keeps it fixed.
    """
    df = _real_frame()
    boxes = [b for b in (detect_big_box(df, i) for i in range(300, 900, 3)) if b]
    assert len(boxes) > 40
    corr = np.corrcoef([b.quality for b in boxes], [float(b.bars) for b in boxes])[0, 1]
    assert abs(corr) < 0.25, f"quality still tracks duration (corr={corr:.2f})"
