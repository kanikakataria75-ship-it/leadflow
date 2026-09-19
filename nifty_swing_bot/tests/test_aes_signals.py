"""Unit tests for AES signal assembly: admission -> box -> breakout -> forward return."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.aes.params import BoxParams
from nifty_swing_bot.aes.signals import (
    SignalParams,
    build_signal,
    build_signals_for_symbol,
    episode_starts,
    find_breakout,
)


def _index_and_stock(n: int = 500, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    idx_close = 1000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    idx_df = pd.DataFrame(
        {"open": idx_close, "high": idx_close * 1.005, "low": idx_close * 0.995,
         "close": idx_close, "volume": 1e8}, index=dates,
    )
    return idx_df, dates


def _breakout_frame(n: int = 400) -> pd.DataFrame:
    """A flat lead-in, an expansion leg, a box, a flat gap, then a clean breakout."""
    dates = pd.bdate_range("2018-01-01", periods=n)
    lead = np.full(150, 100.0)
    leg = np.linspace(100.0, 130.0, 21)[1:]                  # 20 bars: 150-169
    box = np.tile([112.0, 113.0, 114.0, 113.5, 115.0], 4)    # 20 bars: 170-189
    post_box_flat = np.full(20, 122.0)                       # 20 bars: 190-209
    breakout_start = 150 + 20 + 20 + 20                      # bar 210
    breakout = np.full(n - breakout_start, 145.0)
    closes = np.concatenate([lead, leg, box, post_box_flat, breakout])[:n]
    volume = np.full(n, 1_000_000.0)
    volume[breakout_start] = 5_000_000.0    # the volume spike on the ACTUAL breakout bar
    df = pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "volume": volume}, index=dates,
    )
    return df


def test_build_signal_finds_box_then_breakout_then_forward_return():
    df = _breakout_frame()
    idx_df, dates = _index_and_stock(n=len(df))
    admission_idx = 145   # a few bars before the box starts forming
    signal = build_signal(
        df, idx_df, admission_idx, symbol="TEST", admission_date=df.index[admission_idx],
        signal_params=SignalParams(max_box_wait=40, max_breakout_wait=40),
    )
    assert signal is not None
    assert signal.box_date > df.index[admission_idx]       # the screener/entry gap
    assert signal.breakout_date > signal.box_date
    assert signal.entry_date > signal.breakout_date
    assert signal.breakout_volume_ratio > 2.0               # the spike is real
    assert 20 in signal.forward_gross


def test_no_box_within_wait_window_returns_none():
    df = _breakout_frame()
    idx_df, _ = _index_and_stock(n=len(df))
    signal = build_signal(
        df, idx_df, 145, symbol="TEST", admission_date=df.index[145],
        signal_params=SignalParams(max_box_wait=3, max_breakout_wait=40),  # far too short
    )
    assert signal is None


def test_no_breakout_within_wait_window_returns_none():
    df = _breakout_frame()
    idx_df, _ = _index_and_stock(n=len(df))
    signal = build_signal(
        df, idx_df, 145, symbol="TEST", admission_date=df.index[145],
        signal_params=SignalParams(max_box_wait=40, max_breakout_wait=1),  # box found, no room to break out
    )
    assert signal is None


def test_breakout_uses_small_box_top_when_present():
    """The operative top for entry is the small box's, not the big box's, when nested."""
    # Exactly the geometry proven in test_aes_boxes.py's
    # test_upper_zone_boxes_remain_reachable: a deep pullback, then a tight
    # coil held 22 bars so every admissible small-box window sits inside it.
    lows = [112.0, 113.0, 114.0, 115.0] + [122.0] * 22
    highs = [129.0, 128.0, 118.0, 119.0] + [128.5] * 22
    box_closes = [(a + b) / 2 for a, b in zip(lows, highs)]
    lead = np.full(150, 100.0)
    leg = np.linspace(100.0, 130.0, 21)[1:]
    breakout = np.full(60, 145.0)   # clears both the small (128.5) and big (130) tops
    closes = np.concatenate([lead, leg, box_closes, breakout])
    highs_full = np.concatenate([lead, leg, highs, breakout])
    lows_full = np.concatenate([lead, leg, lows, breakout])
    dates = pd.bdate_range("2018-01-01", periods=len(closes))
    df = pd.DataFrame(
        {"open": closes, "high": highs_full, "low": lows_full, "close": closes,
         "volume": np.full(len(closes), 1_000_000.0)}, index=dates,
    )
    from nifty_swing_bot.aes.boxes import detect_nested_box
    box_idx = 150 + 20 + len(lows) - 1     # the last bar of the coil itself
    nested = detect_nested_box(df, box_idx, BoxParams())
    assert nested is not None and nested.small is not None
    assert nested.small_zone == "upper"

    found = find_breakout(df, nested, SignalParams(max_breakout_wait=30))
    assert found is not None
    breakout_idx, _, used_small = found
    assert used_small is True
    # The breakout must fire at the first bar clearing the SMALL box's lower
    # top (128.5), not wait for the bigger box top (130).
    assert df["close"].iloc[breakout_idx] > nested.small.top
    assert df["close"].iloc[breakout_idx - 1] <= nested.small.top


@pytest.mark.parametrize("i", [200, 250])
def test_signal_no_lookahead(i):
    """A signal built from a truncated frame must match one built from the full frame."""
    df = _breakout_frame(n=350)
    idx_df, _ = _index_and_stock(n=350)
    admission_idx = 145
    if i <= admission_idx:
        pytest.skip("evaluation index must be after admission")

    full = build_signal(df, idx_df, admission_idx, symbol="T", admission_date=df.index[admission_idx])
    truncated_df = df.iloc[: i + 1]
    truncated_idx = idx_df.iloc[: i + 1]
    if full is not None and full.entry_date > df.index[i]:
        pytest.skip("signal resolves after the truncation point; nothing to compare")
    truncated = build_signal(
        truncated_df, truncated_idx, admission_idx, symbol="T", admission_date=df.index[admission_idx]
    )
    if full is None:
        assert truncated is None or truncated.breakout_date > df.index[i]
    else:
        assert truncated is not None
        assert truncated.breakout_date == full.breakout_date
        assert truncated.entry_price == pytest.approx(full.entry_price)


# --------------------------------------------------------------------------- #
# Episode de-duplication (the repeated-admission bug)
# --------------------------------------------------------------------------- #
def test_episode_starts_collapses_consecutive_admissions():
    """A run of daily admissions from one multi-week move is one episode."""
    sp = SignalParams(max_box_wait=30, max_breakout_wait=40)   # horizon 70
    admissions = list(range(100, 115))    # 15 consecutive daily admissions
    starts = episode_starts(admissions, sp)
    assert starts == [100]


def test_episode_starts_keeps_admissions_past_the_horizon():
    """A second, later move is a genuinely new episode, not a duplicate."""
    sp = SignalParams(max_box_wait=30, max_breakout_wait=40)   # horizon 70
    admissions = [100, 101, 102, 200, 201]   # 200 is 100 bars after 100
    starts = episode_starts(admissions, sp)
    assert starts == [100, 200]


def test_episode_starts_boundary_is_inclusive_of_the_cooldown():
    """Exactly at the horizon is still covered by the prior search; one past it is new."""
    sp = SignalParams(max_box_wait=10, max_breakout_wait=10)    # horizon 20
    assert episode_starts([100, 120], sp) == [100]          # exactly at the edge: same episode
    assert episode_starts([100, 121], sp) == [100, 121]     # one bar past it: new


def test_build_signals_for_symbol_collapses_repeated_admissions_to_one_signal():
    """The exact bug found in the discovery-period data, reproduced and fixed.

    Real data showed the same box+breakout counted up to 6x because the
    screener admitted the name on every one of several consecutive days.
    """
    df = _breakout_frame()
    idx_df, _ = _index_and_stock(n=len(df))
    # Every one of these admission dates walks forward to the SAME box and
    # SAME breakout -- exactly the pattern observed in the real discovery set.
    admissions = pd.DataFrame({"date": [df.index[140], df.index[142], df.index[144],
                                         df.index[145], df.index[147]]})
    signals = build_signals_for_symbol(
        df, idx_df, admissions, symbol="TEST",
        signal_params=SignalParams(max_box_wait=40, max_breakout_wait=40),
    )
    assert len(signals) == 1


def test_build_signals_for_symbol_keeps_genuinely_separate_episodes():
    """Two distinct breakouts, far enough apart, must both be counted."""
    df = _breakout_frame(n=600)
    # A second cycle: another leg + box + breakout, well past the first one's
    # search horizon (which ends around bar 250: box_wait 30 + breakout_wait 40
    # from the box at ~178).
    second_start = 350
    leg2 = np.linspace(145.0, 175.0, 21)[1:]
    box2 = np.tile([155.0, 156.0, 157.0, 156.5, 158.0], 4)
    tail = np.full(len(df) - second_start - 40, 190.0)
    df.iloc[second_start:second_start + 20, df.columns.get_loc("close")] = leg2
    df.iloc[second_start + 20:second_start + 40, df.columns.get_loc("close")] = box2
    df.iloc[second_start + 40:, df.columns.get_loc("close")] = tail[: len(df) - second_start - 40]
    for col in ("open", "high", "low"):
        df[col] = df["close"] * (1.01 if col == "high" else (0.99 if col == "low" else 1.0))

    idx_df, _ = _index_and_stock(n=len(df))
    admissions = pd.DataFrame({"date": [df.index[145], df.index[147], df.index[second_start]]})
    signals = build_signals_for_symbol(
        df, idx_df, admissions, symbol="TEST",
        signal_params=SignalParams(max_box_wait=40, max_breakout_wait=40),
    )
    assert len(signals) == 2
    assert signals[0].breakout_date < signals[1].breakout_date
