"""Unit tests for the AES portfolio backtest: exits (§6), sizing (§7), regime (§8)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.aes.boxes import BoxGeometry, NestedBox
from nifty_swing_bot.aes.params import PortfolioParams
from nifty_swing_bot.aes.portfolio import AESPortfolioBacktester
from nifty_swing_bot.aes.relative_strength import DrawdownRelativeStrength
from nifty_swing_bot.aes.signals import Signal
from nifty_swing_bot.aes.timeframes import BreakoutClassification
from nifty_swing_bot.config import ExecutionParams

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
_ZERO_COST = ExecutionParams(
    cost_model="simple_bps", brokerage_bps=0.0, slippage_bps=0.0,
    stt_bps_buy=0.0, stt_bps_sell=0.0,
)


def _box(bottom: float, top: float) -> BoxGeometry:
    return BoxGeometry(
        kind="big", start_idx=0, end_idx=10,
        start_date=pd.Timestamp("2020-01-01"), end_date=pd.Timestamp("2020-01-15"),
        top=top, bottom=bottom, bars=11, range_pct=(top - bottom) / bottom,
        higher_lows=0, swing_lows=0, lows_tilt=0.0, pct_closes_above_mid=0.5,
        top_touches=1, bottom_touches=1, close_containment=1.0,
        drift_ratio=0.0, oscillation=0.0, bars_since_top=2, quality=0.5,
    )


def _signal(
    symbol: str, entry_date: pd.Timestamp, entry_price: float, *,
    box_bottom: float, score: float = 0.5, bucket: str = "high_conviction", entry_mode: int = 1,
) -> Signal:
    big = _box(box_bottom, entry_price * 1.3)
    nested = NestedBox(
        symbol=symbol, as_of=entry_date - pd.Timedelta(days=1), as_of_idx=0,
        big=big, small=None, small_position=None, small_zone="none",
        prior_leg_pct=0.2, vol_dryup_ratio=0.8, own_norm_bars=None,
        duration_vs_own_norm=None, prior_boxes=0, prior_cycles=0,
    )
    return Signal(
        symbol=symbol, admission_date=entry_date - pd.Timedelta(days=10),
        box_date=entry_date - pd.Timedelta(days=1), breakout_date=entry_date - pd.Timedelta(days=1),
        entry_date=entry_date, entry_price=entry_price, breakout_volume_ratio=2.5,
        used_small_top=False, nested=nested,
        resistance={"headroom_pct": None, "clears_min_headroom": None, "resistance_age_strength": None,
                    "resistance_touches": 0, "resistance_rejected_count": 0, "resistance_absorbed_count": 0},
        breakout_tf=BreakoutClassification(daily=True, weekly=False, monthly=False, daily_high=0, weekly_high=None, monthly_high=None),
        rel_strength=DrawdownRelativeStrength(20, -0.05, -0.03, 0.6, "resilient"),
        forward_gross={}, entry_mode=entry_mode, score=score, bucket=bucket,
    )


def _flat_frame(price: float, n: int = 60, start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    closes = np.full(n, price)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": np.full(n, 1_000_000.0)},
        index=dates,
    )


def _index_df(n: int = 60, start: str = "2020-01-01", flat: bool = True) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    if flat:
        close = np.full(n, 1000.0)
    else:
        close = 1000.0 * np.exp(np.cumsum(np.full(n, 0.001)))
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": np.full(n, 1e8)}, index=dates,
    )


def _apply_path(frame: pd.DataFrame, closes: dict, wicks: dict | None = None) -> pd.DataFrame:
    """Overwrite specific dates' OHLC on a copy of ``frame``."""
    frame = frame.copy()
    wicks = wicks or {}
    for date, close in closes.items():
        lo, hi = wicks.get(date, (close, close))
        frame.loc[date, ["open", "high", "low", "close"]] = [close, max(hi, close), min(lo, close), close]
    return frame


# --------------------------------------------------------------------------- #
# Closing-basis stop
# --------------------------------------------------------------------------- #
def test_sma10_stop_triggers_on_a_closing_breach():
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=40)
    # A sustained decline in closes drags SMA(10) down with price; force a
    # clean break by dropping close well under the (still-elevated) average.
    later = pd.bdate_range(entry_date, periods=15)[1:]
    drop = {d: 100.0 for d in later[:5]}
    drop.update({d: 70.0 for d in later[5:]})   # sharp break, well under SMA10
    df = _apply_path(df, drop)

    sig = _signal("T", entry_date, 100.0, box_bottom=50.0)   # box floor far below, won't trigger first
    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=1), execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(60), [sig])
    reasons = {t.exit_reason for t in result.trades}
    assert "sma10_close_stop" in reasons


def test_sma10_stop_does_not_trigger_on_an_intraday_wick():
    """A low that pierces SMA(10) intraday, closing back above it, must not exit."""
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=40)
    later = pd.bdate_range(entry_date, periods=10)[1:]
    wick_day = later[3]
    df = _apply_path(df, {wick_day: 100.0}, wicks={wick_day: (40.0, 105.0)})  # deep wick, closes flat

    sig = _signal("T", entry_date, 100.0, box_bottom=20.0)   # also below the wick's low
    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=1), execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(60), [sig])
    reasons = {t.exit_reason for t in result.trades}
    assert "sma10_close_stop" not in reasons
    assert "big_box_bottom_stop" not in reasons


def test_big_box_bottom_stop_triggers_while_sma10_has_not_yet_broken():
    """A window, found numerically rather than assumed, where a dip sits
    below a tight box floor while still *above* SMA(10) -- proving the
    box-bottom condition is independently load-bearing, not a branch
    SMA(10) always pre-empts.

    A flat entry price makes this hard to construct: SMA(10) including
    today's own close triggers whenever today is below the *prior* 9 bars'
    average, so a plain decline off a flat base trips both conditions on the
    same bar (see the module-level note this replaced). A mildly *rising*
    run-up before entry gives SMA(10) room to lag behind a subsequent small
    dip -- verified here by construction, not asserted on faith.
    """
    dates = pd.bdate_range("2020-01-01", periods=40)
    closes = np.full(40, 100.0)
    closes[:15] = np.linspace(90.0, 100.0, 15)   # rising run-up into entry
    closes[15:] = 98.0                            # a small, sustained dip
    df = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": np.full(40, 1e6)},
        index=dates,
    )
    entry_date = dates[14]
    sma10 = pd.Series(closes, index=dates).rolling(10).mean()
    assert closes[15] < 99.0 and closes[15] > sma10.iloc[15]   # the window this test relies on

    sig = _signal("T", entry_date, 100.0, box_bottom=99.0)
    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=1), execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(40), [sig])
    exits = [t for t in result.trades if t.exit_reason in ("sma10_close_stop", "big_box_bottom_stop")]
    assert exits and exits[0].exit_reason == "big_box_bottom_stop"
    assert exits[0].exit_date == dates[15]


# --------------------------------------------------------------------------- #
# Profit ladder
# --------------------------------------------------------------------------- #
def test_profit_ladder_sells_partial_stages_and_leaves_a_remainder():
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=60)
    later = pd.bdate_range(entry_date, periods=40)[1:]
    # Climb through each ladder trigger in turn, then hold flat -- staying
    # above every trigger and never re-touching SMA(10)/box bottom, so any
    # exits recorded must be ladder-driven, not stop-driven.
    path = {later[0]: 106.0, later[1]: 109.0, later[2]: 121.0}
    for d in later[3:]:
        path[d] = 121.0
    df = _apply_path(df, path)

    sig = _signal("T", entry_date, 100.0, box_bottom=10.0)
    params = PortfolioParams(max_open_positions=1, base_position_pct_of_equity=1.0, size_mult_high_conviction=1.0)
    bt = AESPortfolioBacktester(params=params, execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(60), [sig])

    ladder_trades = [t for t in result.trades if t.exit_reason.startswith("ladder")]
    assert {t.exit_reason for t in ladder_trades} == {"ladder1", "ladder2", "ladder3"}
    total_sold = sum(t.qty for t in ladder_trades)
    initial_qty = ladder_trades[0].qty / params.ladder1_fraction
    # 50% + 20% + 20% = 90% sold across the three stages; ~10% remains open
    # or exits later some other way.
    assert total_sold == pytest.approx(initial_qty * 0.90, rel=0.05)


def test_ladder_stage_does_not_refire_once_triggered():
    """Price oscillating back and forth across a trigger must not resell the same stage."""
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=30)
    later = pd.bdate_range(entry_date, periods=15)[1:]
    path = {}
    for i, d in enumerate(later):
        path[d] = 106.0 if i % 2 == 0 else 103.0   # bounces above and below ladder1's 5% trigger
    df = _apply_path(df, path)

    sig = _signal("T", entry_date, 100.0, box_bottom=10.0)
    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=1), execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(30), [sig])
    assert sum(1 for t in result.trades if t.exit_reason == "ladder1") == 1


# --------------------------------------------------------------------------- #
# Time stop
# --------------------------------------------------------------------------- #
def test_time_stop_fires_on_a_stagnant_modest_profit():
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=40)
    later = pd.bdate_range(entry_date, periods=20)[1:]
    df = _apply_path(df, {d: 102.0 for d in later})   # +2%: positive, below the first ladder trigger

    sig = _signal("T", entry_date, 100.0, box_bottom=10.0)
    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=1, time_stop_bars=15), execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(40), [sig])
    assert any(t.exit_reason == "time_stop" for t in result.trades)


def test_time_stop_does_not_override_a_working_trade():
    """A trade already past the first ladder trigger is not "stagnant"."""
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=40)
    later = pd.bdate_range(entry_date, periods=20)[1:]
    df = _apply_path(df, {d: 107.0 for d in later})   # +7%: past ladder1, held there

    sig = _signal("T", entry_date, 100.0, box_bottom=10.0)
    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=1, time_stop_bars=15), execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(40), [sig])
    assert not any(t.exit_reason == "time_stop" for t in result.trades)


def test_time_stop_does_not_fire_on_a_losing_trade():
    """A trade below entry is not "modest profit" -- the stop should have handled it, not the clock."""
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=40)
    later = pd.bdate_range(entry_date, periods=20)[1:]
    # Held just above the box floor, below entry, never crossing the stop.
    df = _apply_path(df, {d: 96.0 for d in later})

    sig = _signal("T", entry_date, 100.0, box_bottom=90.0)
    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=1, time_stop_bars=15), execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(40), [sig])
    assert not any(t.exit_reason == "time_stop" for t in result.trades)


# --------------------------------------------------------------------------- #
# Sizing, capacity, regime, risk appetite
# --------------------------------------------------------------------------- #
def test_reject_bucket_signals_are_never_traded():
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=30)
    sig = _signal("T", entry_date, 100.0, box_bottom=10.0, bucket="reject", score=0.1)
    bt = AESPortfolioBacktester(execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(30), [sig])
    assert result.trades == []


def test_high_conviction_sized_larger_than_wait_and_watch():
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    dfA = _flat_frame(100.0, n=30)
    dfB = _flat_frame(100.0, n=30)
    sig_hc = _signal("A", entry_date, 100.0, box_bottom=10.0, bucket="high_conviction", score=0.9)
    sig_ww = _signal("B", entry_date, 100.0, box_bottom=10.0, bucket="wait_and_watch", score=0.5)
    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=2), execp=_ZERO_COST)
    result = bt.run({"A": dfA, "B": dfB}, _index_df(30), [sig_hc, sig_ww])
    qty_a = next(t.qty for t in result.trades if t.symbol == "A" and t.entry_date == entry_date)
    qty_b = next(t.qty for t in result.trades if t.symbol == "B" and t.entry_date == entry_date)
    assert qty_a > qty_b


def test_capacity_cap_prioritises_by_score():
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    frames = {s: _flat_frame(100.0, n=30) for s in ["A", "B", "C"]}
    sigs = [
        _signal("A", entry_date, 100.0, box_bottom=10.0, score=0.9, bucket="high_conviction"),
        _signal("B", entry_date, 100.0, box_bottom=10.0, score=0.4, bucket="wait_and_watch"),
        _signal("C", entry_date, 100.0, box_bottom=10.0, score=0.6, bucket="high_conviction"),
    ]
    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=2), execp=_ZERO_COST)
    result = bt.run(frames, _index_df(30), sigs)
    entered = {t.symbol for t in result.trades if t.entry_date == entry_date}
    assert entered == {"A", "C"}   # the two highest scores; B rejected on capacity
    assert result.rejected["no_capacity"] >= 1


def test_weak_regime_restricts_to_high_conviction_only():
    entry_date = pd.bdate_range("2020-01-01", periods=80)[-1]
    frames = {s: _flat_frame(100.0, n=90, start="2020-01-01") for s in ["A", "B"]}
    sig_hc = _signal("A", entry_date, 100.0, box_bottom=10.0, bucket="high_conviction", score=0.9)
    sig_ww = _signal("B", entry_date, 100.0, box_bottom=10.0, bucket="wait_and_watch", score=0.5)

    # A benchmark that is clearly BELOW its own 50-day SMA on entry_date: a
    # sharp recent decline after a long flat/higher base.
    idx_dates = pd.bdate_range("2020-01-01", periods=90)
    idx_close = np.full(90, 1200.0)
    idx_close[60:] = np.linspace(1200.0, 900.0, 30)   # decline into entry_date
    index_df = pd.DataFrame(
        {"open": idx_close, "high": idx_close, "low": idx_close, "close": idx_close, "volume": np.full(90, 1e8)},
        index=idx_dates,
    )

    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=2), execp=_ZERO_COST)
    result = bt.run(frames, index_df, [sig_hc, sig_ww])
    entered = {t.symbol for t in result.trades if t.entry_date == entry_date}
    assert entered == {"A"}
    assert result.rejected["regime_or_appetite_restricted"] >= 1


def test_negative_trailing_pnl_lowers_risk_appetite_and_restricts_entries():
    """A string of realised losses should eventually gate out wait_and_watch entries."""
    params = PortfolioParams(max_open_positions=5, risk_appetite_restrict_below=0.95)
    bt = AESPortfolioBacktester(params=params, execp=_ZERO_COST)

    dates = pd.bdate_range("2020-01-01", periods=100)
    frames = {}
    sigs = []
    # Five quick losing round trips on separate symbols, each closing before
    # the next entry, to build up trailing realised losses.
    for k in range(5):
        entry = dates[10 + k * 5]
        exit_ = dates[10 + k * 5 + 2]
        sym = f"L{k}"
        df = _flat_frame(100.0, n=100)
        df = _apply_path(df, {exit_: 80.0, **{d: 80.0 for d in dates[10 + k * 5 + 2:]}})
        frames[sym] = df
        sigs.append(_signal(sym, entry, 100.0, box_bottom=10.0, bucket="high_conviction", score=0.9))

    late_entry = dates[45]
    frames["WW"] = _flat_frame(100.0, n=100)
    sigs.append(_signal("WW", late_entry, 100.0, box_bottom=10.0, bucket="wait_and_watch", score=0.5))
    frames["HC"] = _flat_frame(100.0, n=100)
    sigs.append(_signal("HC", late_entry, 100.0, box_bottom=10.0, bucket="high_conviction", score=0.9))

    result = bt.run(frames, _index_df(100), sigs)
    entered_late = {t.symbol for t in result.trades if t.entry_date == late_entry}
    assert "HC" in entered_late
    assert "WW" not in entered_late


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def test_stats_include_alpha_beta_and_time_in_market():
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=60)
    sig = _signal("T", entry_date, 100.0, box_bottom=10.0)
    bt = AESPortfolioBacktester(params=PortfolioParams(max_open_positions=1), execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(60), [sig])
    for key in ("alpha_annual_pct", "beta", "time_in_market_pct"):
        assert key in result.stats


def test_no_eligible_signals_returns_an_empty_result_without_error():
    bt = AESPortfolioBacktester(execp=_ZERO_COST)
    result = bt.run({}, _index_df(10), [])
    assert result.trades == []


# --------------------------------------------------------------------------- #
# Hard maximum hold (Phase 5.2) and a disabled ladder
# --------------------------------------------------------------------------- #
def test_max_hold_closes_a_winner_the_time_stop_would_let_run():
    """A strongly trending position is exactly what the time stop ignores."""
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=80)
    later = pd.bdate_range(entry_date, periods=45)[1:]
    # Grinds steadily higher, never closing below SMA(10), and stays well past
    # the time stop's profit ceiling -- so only the hard cap can close it.
    df = _apply_path(df, {d: 110.0 + i for i, d in enumerate(later)})

    sig = _signal("T", entry_date, 100.0, box_bottom=10.0)
    params = PortfolioParams(
        max_open_positions=1, max_hold_bars=25,
        ladder1_fraction=0.0, ladder2_fraction=0.0, ladder3_fraction=0.0,
    )
    bt = AESPortfolioBacktester(params=params, execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(80), [sig])

    assert any(t.exit_reason == "max_hold" for t in result.trades)
    assert max(t.bars_held for t in result.trades) == 25


def test_max_hold_is_inert_when_unset():
    """``None`` must reproduce the original section 6 behaviour exactly."""
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=80)
    later = pd.bdate_range(entry_date, periods=45)[1:]
    df = _apply_path(df, {d: 110.0 + i for i, d in enumerate(later)})

    sig = _signal("T", entry_date, 100.0, box_bottom=10.0)
    params = PortfolioParams(
        max_open_positions=1, max_hold_bars=None,
        ladder1_fraction=0.0, ladder2_fraction=0.0, ladder3_fraction=0.0,
    )
    bt = AESPortfolioBacktester(params=params, execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(80), [sig])

    assert not any(t.exit_reason == "max_hold" for t in result.trades)
    assert max(t.bars_held for t in result.trades) > 25


def test_zero_fraction_ladder_stage_sells_nothing():
    """A disabled stage must sell nothing -- not a single token share.

    ``sell_qty`` is ``max(1, round(qty * frac))``, so without an explicit
    guard a zero-weight stage would still dribble out one share per stage and
    a "no ladder" variant would not actually be one.
    """
    entry_date = pd.bdate_range("2020-01-01", periods=15)[-1]
    df = _flat_frame(100.0, n=40)
    later = pd.bdate_range(entry_date, periods=20)[1:]
    df = _apply_path(df, {d: 125.0 for d in later})   # clears all three triggers

    sig = _signal("T", entry_date, 100.0, box_bottom=10.0)
    params = PortfolioParams(
        max_open_positions=1,
        ladder1_fraction=0.0, ladder2_fraction=0.0, ladder3_fraction=0.0,
    )
    bt = AESPortfolioBacktester(params=params, execp=_ZERO_COST)
    result = bt.run({"T": df}, _index_df(40), [sig])

    assert not any(t.exit_reason.startswith("ladder") for t in result.trades)
