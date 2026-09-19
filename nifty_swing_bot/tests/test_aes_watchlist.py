"""Unit tests for the persistent watchlist state machine and all three entry modes.

Every fixture here was built by first probing the box detector for its
actual anchor bar and operative top (printed, then hard-coded as a comment),
because the anchor search finds the *earliest* valid box, which is rarely
where a hand-written series "conceptually" ends. Guessing the timing instead
of measuring it produced several false failures while writing this file --
staleness firing before an intended breakout because the box was detected 11
bars earlier than assumed, and a spurious box-bottom re-entry on the box's own
ordinary internal oscillation -- both left as regression tests below rather
than silently fixed away.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.aes.boxes import detect_nested_box
from nifty_swing_bot.aes.params import BoxParams, WatchlistParams
from nifty_swing_bot.aes.watchlist import run_watchlist_for_symbol


def _index_df(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    idx_close = 1000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    return pd.DataFrame(
        {"open": idx_close, "high": idx_close * 1.005, "low": idx_close * 0.995,
         "close": idx_close, "volume": 1e8},
        index=dates,
    )


def _lead_leg_box(box_tile=(112.0, 113.0, 114.0, 113.5, 115.0), reps: int = 4):
    """The one recurring base shape: flat(150) -> leg 100->130(20) -> box(20).

    Anchor-detected as a valid box starting at bar 169 (the leg's own peak)
    ending at bar 178 -- a 10-bar box, top 131.3, bottom 110.88 -- *before*
    the full 20-bar tile finishes, which is the detector picking the earliest
    valid window rather than "the whole box array". Bars 179-189 are simply
    more of the same oscillation and never change that.
    """
    lead = np.full(150, 100.0)
    leg = np.linspace(100.0, 130.0, 21)[1:]
    box = np.tile(np.asarray(box_tile), reps)
    return np.concatenate([lead, leg, box])


def _frame(closes, opens=None, highs=None, lows=None, volume=1_000_000.0) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    highs = closes * 1.01 if highs is None else np.asarray(highs, dtype=float)
    lows = closes * 0.99 if lows is None else np.asarray(lows, dtype=float)
    if np.ndim(volume) == 0:
        volume = np.full(n, float(volume))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.asarray(volume, dtype=float)},
        index=pd.bdate_range("2018-01-01", periods=n),
    )


BP = BoxParams()
WP = WatchlistParams()


# --------------------------------------------------------------------------- #
# Mode 1: breakout momentum
# --------------------------------------------------------------------------- #
def test_mode1_fires_on_strong_volume_breakout():
    base = _lead_leg_box()
    breakout = np.full(30, 145.0)          # clears the 131.3 top comfortably
    closes = np.concatenate([base, breakout])
    n = len(closes)
    volume = np.full(n, 1_000_000.0)
    volume[190] = 5_000_000.0              # the breakout bar itself: clear spike
    df = _frame(closes, volume=volume)

    sigs = run_watchlist_for_symbol(df, _index_df(n), [df.index[145]], symbol="T", box_params=BP, watch_params=WP)
    assert len(sigs) == 1
    assert sigs[0].entry_mode == 1
    assert sigs[0].breakout_date == df.index[190]
    assert sigs[0].entry_date == df.index[191]


def test_mode1_requires_a_fresh_cross_not_merely_being_above():
    """Sitting above the top on later bars must not each re-fire mode 1."""
    base = _lead_leg_box()
    breakout = np.full(30, 145.0)
    closes = np.concatenate([base, breakout])
    n = len(closes)
    volume = np.full(n, 1_000_000.0)
    volume[190:195] = 5_000_000.0           # elevated volume for several bars, not just one
    df = _frame(closes, volume=volume)

    sigs = run_watchlist_for_symbol(df, _index_df(n), [df.index[145]], symbol="T", box_params=BP, watch_params=WP)
    assert len(sigs) == 1                   # only the first crossing counts


# --------------------------------------------------------------------------- #
# Mode 2: breakout + retest
# --------------------------------------------------------------------------- #
def _mode2_frame():
    """A weak-volume breakout at bar 190 (top 131.3), retest resolving at bar 192."""
    base = _lead_leg_box()
    box_obj = detect_nested_box(_frame(base), 189, BP, symbol="T")
    top = box_obj.small.top if box_obj.small else box_obj.big.top   # 131.3
    weak_close, retest_close = top * 1.01, top * 1.02
    closes = np.concatenate([base, [weak_close, retest_close, retest_close + 1, retest_close + 2]])
    opens = np.concatenate([base, [weak_close - 0.3, top * 0.995, retest_close, retest_close + 1]])
    lows = np.concatenate([base * 0.99, [weak_close - 0.5, top * 1.005, retest_close - 0.3, retest_close + 1]])
    highs = np.concatenate([base * 1.01, [weak_close + 0.3, retest_close + 0.2, retest_close + 1, retest_close + 3]])
    return _frame(closes, opens=opens, highs=highs, lows=lows), top


def test_mode2_fires_on_a_held_retest_with_a_confirming_candle():
    df, top = _mode2_frame()
    n = len(df)
    sigs = run_watchlist_for_symbol(df, _index_df(n), [df.index[145]], symbol="T", box_params=BP, watch_params=WP)
    assert len(sigs) == 1
    assert sigs[0].entry_mode == 2
    assert sigs[0].breakout_volume_ratio < WP.mode1_vol_threshold   # confirms it was routed as "weak"


def test_mode2_abandons_a_failed_retest():
    """A close well below the broken level, right after the weak breakout, kills the attempt."""
    base = _lead_leg_box()
    box_obj = detect_nested_box(_frame(base), 189, BP, symbol="T")
    top = box_obj.small.top if box_obj.small else box_obj.big.top
    weak_close = top * 1.01
    fail_close = top * 0.90     # the very next bar breaks hard back down
    df = _frame(np.concatenate([base, [weak_close, fail_close]]))
    sigs = run_watchlist_for_symbol(df, _index_df(len(df)), [df.index[145]], symbol="T", box_params=BP, watch_params=WP)
    assert len(sigs) == 0


def test_mode2_abandons_an_expired_retest():
    """No qualifying retest within the window: the attempt is dropped, not entered."""
    base = _lead_leg_box()
    box_obj = detect_nested_box(_frame(base), 189, BP, symbol="T")
    top = box_obj.small.top if box_obj.small else box_obj.big.top
    weak_close = top * 1.01
    # Weak breakout, then drift sideways ABOVE the level (never dips back to
    # retest it) for longer than retest_window.
    drift = np.full(WP.retest_window + 5, weak_close)
    closes = np.concatenate([base, [weak_close], drift])
    df = _frame(closes)
    sigs = run_watchlist_for_symbol(df, _index_df(len(df)), [df.index[145]], symbol="T", box_params=BP, watch_params=WP)
    assert len(sigs) == 0


# --------------------------------------------------------------------------- #
# Mode 3: box-bottom entry
# --------------------------------------------------------------------------- #
#: A decline into the box floor, a flat base, then a small uptick -- built so
#: SMA(10) is still catching up (lagging below price) right as price sits
#: within the bottom zone. ``box_stale_bars`` is widened for this fixture
#: alone: the default 22-bar budget from first detection (bar 178) to a clean
#: SMA/zone crossing left no room to build a realistic-looking approach
#: without the transition happening in a single, too-abrupt bar. Staleness
#: itself is exercised by its own test, at the default value, below.
_MODE3_WP = dataclasses.replace(WatchlistParams(), box_stale_bars=60)


def _mode3_tail():
    decline = np.linspace(115.0, 113.0, 8)
    flat = np.full(12, 113.0)
    uptick = np.array([113.2, 113.4, 113.5])
    return np.concatenate([decline, flat, uptick])


def test_mode3_fires_near_the_box_bottom_when_respecting_sma10():
    base = _lead_leg_box()
    closes = np.concatenate([base, _mode3_tail()])
    df = _frame(closes, highs=closes * 1.005, lows=closes * 0.995)
    sigs = run_watchlist_for_symbol(df, _index_df(len(df)), [df.index[145]], symbol="T", box_params=BP, watch_params=_MODE3_WP)
    assert len(sigs) == 1
    assert sigs[0].entry_mode == 3


def test_mode3_does_not_refire_on_further_oscillation_after_entry():
    """The bug this file's docstring describes: box noise re-triggering itself.

    Without the "must anchor on a genuinely later box" guard, resolving one
    mode-3 entry immediately re-detects the *same* structure on the very next
    down-tick and fires again -- and again -- on ordinary internal wiggle.
    Extending the series with more oscillation after the entry must still
    yield exactly one signal, not one per touch of the bottom.
    """
    base = _lead_leg_box()
    tail = np.concatenate([_mode3_tail(), np.tile([113.0, 112.8, 113.0, 113.3, 113.6], 6)])
    closes = np.concatenate([base, tail])
    df = _frame(closes, highs=closes * 1.005, lows=closes * 0.995)
    sigs = run_watchlist_for_symbol(df, _index_df(len(df)), [df.index[145]], symbol="T", box_params=BP, watch_params=_MODE3_WP)
    assert len(sigs) == 1


# --------------------------------------------------------------------------- #
# The watchlist itself: persistence, invalidation, staleness
# --------------------------------------------------------------------------- #
def test_invalidation_then_a_fresh_box_across_two_admissions():
    """The core improvement over a per-admission search: an invalidated first
    box followed by a wholly new one is found without a dedicated search
    window being reopened for it -- the watchlist that leg2's own admission
    opens simply keeps running until it finds box2, rather than needing box2
    to already be anticipated.

    Two admissions, not one: leg2's ~150% run would realistically re-trigger
    the screener on its own, and the eligibility clock is deliberately tied
    to admission recency (see the module docstring) -- a single admission
    from months before a box ever forms should not grant unlimited patience.
    """
    lead = np.full(150, 100.0)
    leg1 = np.linspace(100.0, 130.0, 21)[1:]
    box1 = np.tile([112.0, 113.0, 114.0, 113.5, 115.0], 4)
    invalidate = np.full(3, 70.0)          # closes well below box1's floor (110.88)
    recover = np.full(20, 70.0)
    leg2 = np.linspace(70.0, 200.0, 21)[1:]
    box2 = np.tile([173.0, 183.0, 186.0, 181.0, 190.0], 4)   # ~18% range off the leg2 peak
    breakout2 = np.full(30, 260.0)
    closes = np.concatenate([lead, leg1, box1, invalidate, recover, leg2, box2, breakout2])
    n = len(closes)
    volume = np.full(n, 1_000_000.0)
    breakout2_start = 150 + 20 + 20 + 3 + 20 + 20 + 20
    volume[breakout2_start] = 6_000_000.0
    df = _frame(closes, volume=volume)

    leg2_start = 150 + 20 + 20 + 3 + 20
    admissions = [df.index[145], df.index[leg2_start + 10]]   # original + one part-way through leg2
    sigs = run_watchlist_for_symbol(df, _index_df(n), admissions, symbol="T", box_params=BP, watch_params=WP)
    assert len(sigs) == 1
    assert sigs[0].entry_mode == 1
    assert sigs[0].box_date > df.index[150 + 20 + 20]   # the second box, not the first
    assert sigs[0].nested.big.bottom > 150.0            # box2's floor, not box1's (~110.88)
    assert sigs[0].admission_date == df.index[leg2_start + 10]   # attributed to the RELEVANT admission


def test_a_single_ancient_admission_does_not_grant_unlimited_patience():
    """Without the second admission, the same series must find nothing: the
    gap from admission to box2 exceeds max_watch_without_box, and only one
    admission exists to measure that clock against.
    """
    lead = np.full(150, 100.0)
    leg1 = np.linspace(100.0, 130.0, 21)[1:]
    box1 = np.tile([112.0, 113.0, 114.0, 113.5, 115.0], 4)
    invalidate = np.full(3, 70.0)
    recover = np.full(20, 70.0)
    leg2 = np.linspace(70.0, 200.0, 21)[1:]
    box2 = np.tile([173.0, 183.0, 186.0, 181.0, 190.0], 4)
    breakout2 = np.full(30, 260.0)
    closes = np.concatenate([lead, leg1, box1, invalidate, recover, leg2, box2, breakout2])
    n = len(closes)
    volume = np.full(n, 1_000_000.0)
    volume[150 + 20 + 20 + 3 + 20 + 20 + 20] = 6_000_000.0
    df = _frame(closes, volume=volume)

    sigs = run_watchlist_for_symbol(df, _index_df(n), [df.index[145]], symbol="T", box_params=BP, watch_params=WP)
    assert len(sigs) == 0


def test_box_invalidates_on_a_close_below_the_floor():
    """No entry at all if the only box on offer breaks down and nothing replaces it."""
    lead = np.full(150, 100.0)
    leg = np.linspace(100.0, 130.0, 21)[1:]
    box = np.tile([112.0, 113.0, 114.0, 113.5, 115.0], 4)
    breakdown = np.full(30, 90.0)          # well below the 110.88 floor; no recovery
    closes = np.concatenate([lead, leg, box, breakdown])
    df = _frame(closes)
    sigs = run_watchlist_for_symbol(df, _index_df(len(df)), [df.index[145]], symbol="T", box_params=BP, watch_params=WP)
    assert len(sigs) == 0


def test_staleness_with_no_touch_produces_no_signal():
    """A box that just sits mid-range -- neither breaking out nor dipping to
    the bottom -- for longer than box_stale_bars must produce nothing."""
    base = _lead_leg_box()
    stale = np.full(40, 122.0)             # mid-range: not near 131.3, not near 110.88
    closes = np.concatenate([base, stale])
    df = _frame(closes)
    sigs = run_watchlist_for_symbol(df, _index_df(len(df)), [df.index[145]], symbol="T", box_params=BP, watch_params=WP)
    assert len(sigs) == 0


def test_no_box_ever_forming_does_not_error():
    flat = np.full(300, 100.0)
    df = _frame(flat)
    sigs = run_watchlist_for_symbol(df, _index_df(len(df)), [df.index[130]], symbol="T", box_params=BP, watch_params=WP)
    assert sigs == []


def test_repeated_admissions_do_not_multiply_signals():
    """Many consecutive admissions of the same move must not each search anew."""
    base = _lead_leg_box()
    breakout = np.full(30, 145.0)
    closes = np.concatenate([base, breakout])
    n = len(closes)
    volume = np.full(n, 1_000_000.0)
    volume[190] = 5_000_000.0
    df = _frame(closes, volume=volume)
    admissions = [df.index[i] for i in range(140, 150)]   # ten consecutive admissions

    sigs = run_watchlist_for_symbol(df, _index_df(n), admissions, symbol="T", box_params=BP, watch_params=WP)
    assert len(sigs) == 1


# --------------------------------------------------------------------------- #
# Look-ahead audit
# --------------------------------------------------------------------------- #
def _real_frame(n: int = 500, seed: int = 20260912) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    noise = np.abs(rng.normal(0, 0.012, n))
    return _frame(close, highs=close * (1 + noise), lows=close * (1 - noise))


def test_no_lookahead_truncation_invariance():
    """Signals found up to bar i must not change once later bars are visible."""
    df = _real_frame()
    idx_df = _index_df(len(df))
    admissions = [df.index[130], df.index[250], df.index[380]]
    cut = 400

    full = run_watchlist_for_symbol(df, idx_df, admissions, symbol="T", box_params=BP, watch_params=WP)
    full_before_cut = [s for s in full if s.breakout_date <= df.index[cut]]

    truncated = run_watchlist_for_symbol(
        df.iloc[: cut + 1], idx_df.iloc[: cut + 1],
        [d for d in admissions if d <= df.index[cut]], symbol="T", box_params=BP, watch_params=WP,
    )
    assert [s.breakout_date for s in full_before_cut] == [s.breakout_date for s in truncated]
    for a, b in zip(full_before_cut, truncated):
        assert a.entry_price == pytest.approx(b.entry_price)
        assert a.entry_mode == b.entry_mode


def test_no_lookahead_future_mutation_invariance():
    """Overwriting every bar after i with garbage must not change a signal found by i."""
    df = _real_frame()
    idx_df = _index_df(len(df))
    admissions = [df.index[130], df.index[250]]
    cut = 400

    before = run_watchlist_for_symbol(df, idx_df, admissions, symbol="T", box_params=BP, watch_params=WP)
    before_by_cut = [s for s in before if s.breakout_date <= df.index[cut]]

    poisoned = df.copy()
    poisoned.iloc[cut + 1 :] *= 6.0
    after = run_watchlist_for_symbol(poisoned, idx_df, admissions, symbol="T", box_params=BP, watch_params=WP)
    after_by_cut = [s for s in after if s.breakout_date <= df.index[cut]]

    assert [s.breakout_date for s in before_by_cut] == [s.breakout_date for s in after_by_cut]


# --------------------------------------------------------------------------- #
# Carry-forward: a resolved box held as an entry candidate (Phase 8)
# --------------------------------------------------------------------------- #
def test_carry_forward_allows_a_breakout_after_the_box_goes_stale():
    """A thrust arriving after box_stale_bars fires only when carry is on.

    The box is detected at bar 169 and goes stale 22 bars later; the breakout
    here lands well past that, which is precisely the case Phase 7 measured
    as the detector's largest miss category.
    """
    base = _lead_leg_box()
    drift = np.full(30, 122.0)              # mid-range, outlives box_stale_bars
    thrust = np.full(20, 145.0)             # clears the 131.3 operative top
    closes = np.concatenate([base, drift, thrust])
    # The thrust needs above-average volume or it routes to mode 2 and waits
    # for a retest that a flat series never delivers -- measured, not assumed.
    vol = np.full(closes.size, 1_000_000.0)
    vol[base.size + drift.size:] = 3_000_000.0
    df = _frame(closes, volume=vol)

    off = run_watchlist_for_symbol(
        df, _index_df(len(df)), [df.index[145]], symbol="T",
        box_params=BP, watch_params=dataclasses.replace(WP, box_carry_bars=0))
    on = run_watchlist_for_symbol(
        df, _index_df(len(df)), [df.index[145]], symbol="T",
        box_params=BP, watch_params=dataclasses.replace(WP, box_carry_bars=45))

    assert len(off) == 0
    assert len(on) == 1
    assert on[0].entry_mode in (1, 2)


def test_carry_window_expires():
    """Carry is bounded: a thrust beyond the carry window still fires nothing."""
    base = _lead_leg_box()
    drift = np.full(60, 122.0)
    thrust = np.full(20, 145.0)
    closes = np.concatenate([base, drift, thrust])
    vol = np.full(closes.size, 1_000_000.0)
    vol[base.size + drift.size:] = 3_000_000.0
    df = _frame(closes, volume=vol)
    sigs = run_watchlist_for_symbol(
        df, _index_df(len(df)), [df.index[145]], symbol="T",
        box_params=BP, watch_params=dataclasses.replace(WP, box_carry_bars=10))
    assert sigs == []


def test_carried_box_still_dies_on_a_close_below_the_floor():
    """Carry extends the entry window, not the setup's validity."""
    base = _lead_leg_box()
    drift = np.full(30, 122.0)
    breakdown = np.full(5, 100.0)           # below the 110.88 big-box bottom
    thrust = np.full(20, 145.0)
    closes = np.concatenate([base, drift, breakdown, thrust])
    vol = np.full(closes.size, 1_000_000.0)
    vol[base.size + drift.size + breakdown.size:] = 3_000_000.0
    df = _frame(closes, volume=vol)
    trace: list = []
    sigs = run_watchlist_for_symbol(
        df, _index_df(len(df)), [df.index[145]], symbol="T",
        box_params=BP, watch_params=dataclasses.replace(WP, box_carry_bars=45),
        trace=trace)

    assert any(e["kind"] == "box_invalidated" for e in trace)
    # A genuinely *new* box may still form on the wreckage and fire on its own
    # merits -- what must not happen is the carried box surviving the breach,
    # so nothing may be anchored on the original structure (big.start_idx 169).
    assert all(s.nested.big.start_idx != 169 for s in sigs)


def test_carry_does_not_enable_a_box_bottom_entry():
    """Mode 3 into an already-stale structure is a different trade, not this one."""
    base = _lead_leg_box()
    drift = np.full(30, 122.0)
    to_floor = np.full(20, 112.0)           # within 2% of the 110.88 floor
    df = _frame(np.concatenate([base, drift, to_floor]))
    sigs = run_watchlist_for_symbol(
        df, _index_df(len(df)), [df.index[145]], symbol="T",
        box_params=BP, watch_params=dataclasses.replace(WP, box_carry_bars=45))
    assert all(s.entry_mode != 3 for s in sigs)


def test_trace_hook_is_off_by_default_and_records_when_passed():
    base = _lead_leg_box()
    df = _frame(np.concatenate([base, np.full(30, 122.0)]))
    tr: list = []
    run_watchlist_for_symbol(df, _index_df(len(df)), [df.index[145]], symbol="T",
                             box_params=BP, watch_params=WP, trace=tr)
    assert tr and {"kind", "idx", "date"} <= set(tr[0])
    assert any(e["kind"] == "box_armed" for e in tr)
