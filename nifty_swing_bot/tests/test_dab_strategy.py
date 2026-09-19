"""Unit tests for the DAB signal engine.

The strategy is tested against a synthetic dataset that is constructed so that
exactly one bar (index 50) satisfies all five entry conditions. Each test then
breaks exactly one condition and asserts the signal disappears, which pins down
every rule independently rather than testing them only in aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.config import AppConfig
from nifty_swing_bot.strategy import indicators as ind
from nifty_swing_bot.strategy.dab_strategy import (
    compute_features,
    compute_stop,
    extract_signals,
    position_size,
)

N_BARS = 60
SIGNAL_BAR = 50
#: Default bar carrying the delivery-accumulation spike. In the default
#: ``prior_window`` mode the spike must precede the breakout bar, so it sits
#: four sessions before it.
SPIKE_BAR = SIGNAL_BAR - 4


@pytest.fixture
def cfg() -> AppConfig:
    """Default config, without touching the user's config.yaml."""
    return AppConfig()


def build_synthetic(
    *,
    deliv_spike: float = 60.0,
    deliv_base: float = 30.0,
    volume_spike: float = 300_000.0,
    volume_base: float = 100_000.0,
    signal_close: float = 110.0,
    signal_low: float = 104.0,
    signal_high: float = 110.5,
    bench_trend: float = 0.0,
    ramp_top: float = 103.0,
    spike_bar: int = SPIKE_BAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build (ohlcv, delivery, benchmark) with one engineered signal at bar 50.

    Bars 0-44 are flat around 100, bars 45-49 ramp gently to ``ramp_top`` so the
    prior 20-bar high is well defined, and bar 50 is the engineered breakout.
    Bars 51-59 drift up so exit logic has somewhere to go.

    The delivery spike sits on ``spike_bar``, which defaults to four sessions
    *before* the breakout -- the accumulation-then-breakout sequence the default
    ``prior_window`` mode looks for.

    Args:
        deliv_spike: Delivery% on the spike bar.
        deliv_base: Delivery% on every other bar.
        volume_spike: Volume on the signal bar.
        volume_base: Volume on every other bar.
        signal_close/low/high: Signal-bar prices.
        bench_trend: Per-bar benchmark drift; raise it to break relative strength.
        ramp_top: Price the pre-signal ramp reaches, which sets the 20-bar high.
        spike_bar: Bar index carrying the delivery spike.
    """
    dates = pd.bdate_range("2024-01-01", periods=N_BARS, name="date")
    close = np.full(N_BARS, 100.0)
    ramp = np.linspace(100.0, ramp_top, 5)
    close[45:50] = ramp
    close[SIGNAL_BAR] = signal_close
    close[SIGNAL_BAR + 1 :] = np.linspace(signal_close, signal_close * 1.06, N_BARS - SIGNAL_BAR - 1)

    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.2
    high[SIGNAL_BAR] = signal_high
    low[SIGNAL_BAR] = signal_low
    open_[SIGNAL_BAR] = signal_low + 0.5

    volume = np.full(N_BARS, volume_base)
    volume[SIGNAL_BAR] = volume_spike

    ohlcv = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    deliv = np.full(N_BARS, deliv_base)
    deliv[spike_bar] = deliv_spike
    delivery = pd.DataFrame({"deliv_pct": deliv}, index=dates)

    bench_close = 1000.0 * (1.0 + bench_trend) ** np.arange(N_BARS)
    benchmark = pd.DataFrame(
        {
            "open": bench_close,
            "high": bench_close,
            "low": bench_close,
            "close": bench_close,
            "volume": np.full(N_BARS, 1e6),
        },
        index=dates,
    )
    return ohlcv, delivery, benchmark


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_engineered_bar_produces_exactly_one_signal(cfg: AppConfig) -> None:
    ohlcv, delivery, bench = build_synthetic()
    feats = compute_features(ohlcv, delivery, bench, cfg=cfg)

    fired = feats.index[feats["signal"]]
    assert len(fired) == 1, f"expected 1 signal, got {len(fired)}: {list(fired)}"
    assert fired[0] == ohlcv.index[SIGNAL_BAR]


def test_all_five_conditions_true_on_signal_bar(cfg: AppConfig) -> None:
    ohlcv, delivery, bench = build_synthetic()
    row = compute_features(ohlcv, delivery, bench, cfg=cfg).iloc[SIGNAL_BAR]

    assert row["cond_delivery"], "delivery spike should pass"
    assert row["cond_volume"], "volume surge should pass"
    assert row["cond_close"], "strong close should pass"
    assert row["cond_rs"], "relative strength should pass"
    assert row["cond_setup"], "breakout/VCP should pass"


def test_component_values_match_hand_computation(cfg: AppConfig) -> None:
    """The ratios the LLM layer explains must be arithmetically correct."""
    ohlcv, delivery, bench = build_synthetic()
    row = compute_features(ohlcv, delivery, bench, cfg=cfg).iloc[SIGNAL_BAR]

    # The accumulation spike happened four sessions before the breakout.
    assert row["deliv_spike_ratio"] == pytest.approx(2.0)
    assert row["deliv_spike_age"] == pytest.approx(4.0)
    assert row["vol_baseline"] == pytest.approx(100_000.0)
    assert row["vol_ratio"] == pytest.approx(3.0)
    # close 110 in a 104.0-110.5 range -> (110-104)/6.5
    assert row["range_position"] == pytest.approx(6.0 / 6.5)
    # 10-bar return: close[50]/close[40] - 1 = 110/100 - 1
    assert row["ret_stock"] == pytest.approx(0.10)
    assert row["ret_bench"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# One test per rule: break it, and the signal must vanish
# --------------------------------------------------------------------------- #
def test_rule1_no_signal_without_delivery_spike(cfg: AppConfig) -> None:
    # 1.4x is below the 1.5x threshold.
    ohlcv, delivery, bench = build_synthetic(deliv_spike=30.0 * 1.4)
    feats = compute_features(ohlcv, delivery, bench, cfg=cfg)
    assert not feats["cond_delivery"].iloc[SIGNAL_BAR]
    assert not feats["signal"].any()


def test_rule1_spike_outside_the_window_does_not_qualify(cfg: AppConfig) -> None:
    """Accumulation must be recent; a spike 20 sessions back is stale."""
    ohlcv, delivery, bench = build_synthetic(spike_bar=SIGNAL_BAR - 20)
    feats = compute_features(ohlcv, delivery, bench, cfg=cfg)
    assert not feats["cond_delivery"].iloc[SIGNAL_BAR]
    assert not feats["signal"].any()


def test_rule1_spike_on_the_breakout_bar_alone_does_not_qualify(cfg: AppConfig) -> None:
    """In prior_window mode the spike must PRECEDE the breakout, not coincide.

    This is the whole point of the reformulation: delivery% is suppressed by the
    very volume surge rule 2 demands, so a same-bar spike is the rare case, not
    the signal the strategy is looking for.
    """
    ohlcv, delivery, bench = build_synthetic(spike_bar=SIGNAL_BAR)
    feats = compute_features(ohlcv, delivery, bench, cfg=cfg)
    assert feats["is_deliv_spike"].iloc[SIGNAL_BAR], "the bar itself is a spike"
    assert not feats["cond_delivery"].iloc[SIGNAL_BAR], "but it must not qualify"
    assert not feats["signal"].any()


def test_same_day_mode_reproduces_the_original_specification(cfg: AppConfig) -> None:
    """The original same-bar rule is preserved and selectable."""
    same_day = cfg.model_copy(deep=True)
    same_day.dab.delivery_mode = "same_day"

    ohlcv, delivery, bench = build_synthetic(spike_bar=SIGNAL_BAR)
    feats = compute_features(ohlcv, delivery, bench, cfg=same_day)
    assert feats["cond_delivery"].iloc[SIGNAL_BAR]
    assert bool(feats["signal"].iloc[SIGNAL_BAR])

    # And the prior-window layout must NOT fire under same-day semantics.
    ohlcv2, delivery2, bench2 = build_synthetic(spike_bar=SPIKE_BAR)
    assert not compute_features(ohlcv2, delivery2, bench2, cfg=same_day)["signal"].any()


def test_rule1_missing_delivery_data_blocks_signal(cfg: AppConfig) -> None:
    """No delivery data must fail closed, never silently pass."""
    ohlcv, _delivery, bench = build_synthetic()
    feats = compute_features(ohlcv, None, bench, cfg=cfg)
    assert not feats["signal"].any()


def test_rule2_no_signal_without_volume_surge(cfg: AppConfig) -> None:
    # 1.9x is below the 2.0x threshold.
    ohlcv, delivery, bench = build_synthetic(volume_spike=100_000.0 * 1.9)
    feats = compute_features(ohlcv, delivery, bench, cfg=cfg)
    assert not feats["cond_volume"].iloc[SIGNAL_BAR]
    assert not feats["signal"].any()


def test_rule3_no_signal_on_weak_close(cfg: AppConfig) -> None:
    """Close near the low of the range fails the strong-close test."""
    ohlcv, delivery, bench = build_synthetic(
        signal_close=105.0, signal_low=104.0, signal_high=110.5
    )
    feats = compute_features(ohlcv, delivery, bench, cfg=cfg)
    row = feats.iloc[SIGNAL_BAR]
    assert row["range_position"] == pytest.approx(1.0 / 6.5)
    assert not row["cond_close"]
    assert not feats["signal"].any()


def test_rule4_no_signal_when_index_outruns_stock(cfg: AppConfig) -> None:
    """Benchmark up ~2%/bar over 10 bars beats the stock's 10% move."""
    ohlcv, delivery, bench = build_synthetic(bench_trend=0.02)
    feats = compute_features(ohlcv, delivery, bench, cfg=cfg)
    row = feats.iloc[SIGNAL_BAR]
    assert row["ret_bench"] > row["ret_stock"]
    assert not row["cond_rs"]
    assert not feats["signal"].any()


def test_rule5_no_signal_below_high_without_contraction(cfg: AppConfig) -> None:
    """Well below the 20-bar high, and not contracting, fails the setup rule."""
    # Ramp to 130 so the prior 20-bar high is far above the signal close of 110.
    ohlcv, delivery, bench = build_synthetic(ramp_top=130.0, signal_close=110.0,
                                             signal_low=104.0, signal_high=110.5)
    feats = compute_features(ohlcv, delivery, bench, cfg=cfg)
    row = feats.iloc[SIGNAL_BAR]
    assert not row["is_breakout"]
    assert row["dist_from_high"] > cfg.dab.breakout_proximity_pct
    assert not row["cond_setup"]
    assert not feats["signal"].any()


def test_rule5_near_high_plus_contraction_qualifies(cfg: AppConfig) -> None:
    """The VCP branch: not a breakout, but within 3% of the high and contracting."""
    dates = pd.bdate_range("2024-01-01", periods=N_BARS, name="date")
    close = np.full(N_BARS, 100.0)
    # A wide-range early section then a very tight one collapses ATR(5)/ATR(20).
    # The wide part must end before bar 30 so that the prior 20-bar window at
    # bar 50 (bars 30-49) contains only tight bars, putting the 20-bar high at
    # 100.15 rather than 104.
    high = close + np.concatenate([np.full(30, 4.0), np.full(N_BARS - 30, 0.15)])
    low = close - np.concatenate([np.full(30, 4.0), np.full(N_BARS - 30, 0.15)])
    # Prior 20-bar high at bar 50 comes from the tight section: 100.15.
    close[SIGNAL_BAR] = 100.0
    high[SIGNAL_BAR] = 100.10
    low[SIGNAL_BAR] = 99.95
    open_ = close - 0.02

    volume = np.full(N_BARS, 100_000.0)
    volume[SIGNAL_BAR] = 300_000.0
    ohlcv = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates
    )
    deliv = np.full(N_BARS, 30.0)
    deliv[SIGNAL_BAR] = 60.0
    delivery = pd.DataFrame({"deliv_pct": deliv}, index=dates)

    feats = compute_features(ohlcv, delivery, None, cfg=cfg)
    row = feats.iloc[SIGNAL_BAR]
    assert not row["is_breakout"], "this case must exercise the VCP branch, not breakout"
    assert row["is_near_high"]
    assert row["is_contraction"], f"atr_ratio={row['atr_ratio']}"
    assert row["cond_setup"]


# --------------------------------------------------------------------------- #
# Look-ahead discipline
# --------------------------------------------------------------------------- #
def test_signal_is_unaffected_by_future_bars(cfg: AppConfig) -> None:
    """Truncating everything after the signal bar must not change the signal."""
    ohlcv, delivery, bench = build_synthetic()
    full = compute_features(ohlcv, delivery, bench, cfg=cfg)

    cut = SIGNAL_BAR + 1
    truncated = compute_features(
        ohlcv.iloc[:cut], delivery.iloc[:cut], bench.iloc[:cut], cfg=cfg
    )
    assert bool(truncated["signal"].iloc[-1]) == bool(full["signal"].iloc[SIGNAL_BAR])
    for col in ("deliv_ratio", "vol_ratio", "range_position", "ret_stock", "atr_ratio"):
        assert truncated[col].iloc[-1] == pytest.approx(full[col].iloc[SIGNAL_BAR], nan_ok=True)


def test_baseline_excludes_the_signal_bar_itself(cfg: AppConfig) -> None:
    """With the flag off, the spike contaminates its own baseline and dilutes the ratio."""
    ohlcv, delivery, bench = build_synthetic()
    inclusive = cfg.model_copy(deep=True)
    inclusive.dab.baseline_excludes_today = False

    # Measured on the spike bar itself, which is where contamination shows up.
    excl_ratio = compute_features(ohlcv, delivery, bench, cfg=cfg)["deliv_ratio"].iloc[SPIKE_BAR]
    incl_ratio = compute_features(ohlcv, delivery, bench, cfg=inclusive)["deliv_ratio"].iloc[SPIKE_BAR]
    assert excl_ratio > incl_ratio


# --------------------------------------------------------------------------- #
# Entry, stop and sizing
# --------------------------------------------------------------------------- #
def test_entry_reference_is_next_bar_open(cfg: AppConfig) -> None:
    ohlcv, delivery, bench = build_synthetic()
    feats = compute_features(ohlcv, delivery, bench, cfg=cfg)
    sigs = extract_signals(feats, "TEST", cfg=cfg)

    assert len(sigs) == 1
    assert sigs[0].entry_ref == pytest.approx(float(ohlcv["open"].iloc[SIGNAL_BAR + 1]))


def test_last_bar_signal_falls_back_to_close(cfg: AppConfig) -> None:
    """The live scanner has no next open, so it must quote the close instead."""
    ohlcv, delivery, bench = build_synthetic()
    cut = SIGNAL_BAR + 1
    feats = compute_features(ohlcv.iloc[:cut], delivery.iloc[:cut], bench.iloc[:cut], cfg=cfg)
    sigs = extract_signals(feats, "TEST", cfg=cfg)

    assert len(sigs) == 1
    assert sigs[0].entry_ref == pytest.approx(float(ohlcv["close"].iloc[SIGNAL_BAR]))


def test_stop_is_the_lower_of_signal_low_and_atr_stop(cfg: AppConfig) -> None:
    # ATR branch is lower: 100 - 1.5*8 = 88 < 95
    assert compute_stop(95.0, 100.0, 8.0, risk=cfg.risk) == pytest.approx(88.0)
    # Signal-low branch is lower: 90 < 100 - 1.5*2 = 97
    assert compute_stop(90.0, 100.0, 2.0, risk=cfg.risk) == pytest.approx(90.0)


def test_stop_never_lands_at_or_above_entry(cfg: AppConfig) -> None:
    """Degenerate inputs must still produce a usable, strictly-below stop."""
    stop = compute_stop(signal_low=100.0, entry_price=100.0, atr_value=0.0, risk=cfg.risk)
    assert stop < 100.0


def test_position_size_risks_exactly_one_percent_when_cap_is_slack(cfg: AppConfig) -> None:
    """Full 1% risk is achieved only when the notional cap does not bind.

    The cap binds whenever the stop is closer than
    ``max_position_pct / risk_per_trade_pct`` in percentage terms -- with the
    10%/1% defaults, any stop tighter than 10% away. A 12% stop is therefore
    slack and the risk formula governs alone.
    """
    equity = 1_000_000.0
    # Stop 12 below entry -> 10,000 risk budget / 12 = 833 shares,
    # notional 83,300 which is inside the 100,000 cap.
    qty, rupee_risk = position_size(equity, entry_price=100.0, stop_price=88.0, risk=cfg.risk)
    assert qty == 833
    assert qty * 100.0 <= equity * cfg.risk.max_position_pct
    assert rupee_risk == pytest.approx(equity * cfg.risk.risk_per_trade_pct, rel=1e-3)


def test_tight_stop_is_capped_and_therefore_risks_less_than_one_percent(cfg: AppConfig) -> None:
    """The documented consequence of the notional cap taking precedence.

    A 5% stop wants 20% of equity to put 1% at risk. The cap cuts that to 10%,
    which halves the realised risk to 0.5%. This is the intended trade-off:
    bounded small-cap concentration is worth more than a uniform risk figure.
    """
    equity = 1_000_000.0
    qty, rupee_risk = position_size(equity, entry_price=100.0, stop_price=95.0, risk=cfg.risk)
    assert qty == 1_000                       # 100,000 notional cap / Rs 100
    assert rupee_risk == pytest.approx(5_000.0)  # 0.5% of equity, not 1%


def test_max_position_cap_overrides_risk_formula(cfg: AppConfig) -> None:
    """A very tight stop must not produce an oversized small-cap position."""
    equity = 1_000_000.0
    # Risk formula alone wants 10,000/0.1 = 100,000 shares = Rs 1cr notional.
    qty, _ = position_size(equity, entry_price=100.0, stop_price=99.9, risk=cfg.risk)
    notional = qty * 100.0
    assert notional <= equity * cfg.risk.max_position_pct + 1e-6
    assert qty == int(equity * cfg.risk.max_position_pct // 100.0)


def test_position_size_respects_liquidity_cap(cfg: AppConfig) -> None:
    qty, _ = position_size(
        1_000_000.0, entry_price=100.0, stop_price=95.0, risk=cfg.risk, max_notional=50_000.0
    )
    assert qty == 500


def test_position_size_zero_on_invalid_stop(cfg: AppConfig) -> None:
    assert position_size(1_000_000.0, 100.0, 100.0, risk=cfg.risk)[0] == 0
    assert position_size(1_000_000.0, 100.0, 105.0, risk=cfg.risk)[0] == 0


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def test_true_range_uses_previous_close() -> None:
    high = pd.Series([10.0, 12.0])
    low = pd.Series([9.0, 11.5])
    close = pd.Series([9.5, 11.8])
    tr = ind.true_range(high, low, close)
    assert tr.iloc[0] == pytest.approx(1.0)          # first bar: high - low
    assert tr.iloc[1] == pytest.approx(2.5)          # |12 - 9.5| beats 0.5


def test_atr_is_positive_and_seeded_with_nans() -> None:
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)))
    high, low = close + 1.0, close - 1.0
    out = ind.atr(high, low, close, 14)
    assert out.iloc[:13].isna().all(), "under-seeded bars must stay NaN"
    assert (out.iloc[14:] > 0).all()


def test_range_position_bounds_and_doji() -> None:
    high = pd.Series([10.0, 10.0, 10.0])
    low = pd.Series([8.0, 8.0, 10.0])
    close = pd.Series([10.0, 8.0, 10.0])
    pos = ind.close_range_position(high, low, close)
    assert pos.iloc[0] == pytest.approx(1.0)
    assert pos.iloc[1] == pytest.approx(0.0)
    assert pos.iloc[2] == pytest.approx(1.0)  # zero-range bar


def test_rolling_high_excludes_current_bar() -> None:
    series = pd.Series([1.0, 5.0, 2.0, 3.0])
    out = ind.rolling_high(series, 2, exclude_today=True)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(5.0)   # max of bars 0-1
    assert out.iloc[3] == pytest.approx(5.0)   # max of bars 1-2


def test_swing_low_is_confirmed_not_clairvoyant() -> None:
    """A pivot low must only appear after its right-hand bars have printed."""
    low = pd.Series([10.0, 9.0, 5.0, 9.5, 10.5, 11.0, 12.0, 13.0])
    out = ind.swing_low(low, lookback=2)
    pivot_pos = 2
    assert out.iloc[pivot_pos : pivot_pos + 2].isna().all(), "pivot leaked early"
    assert out.iloc[pivot_pos + 2] == pytest.approx(5.0)


def test_safe_ratio_handles_zero_baseline() -> None:
    out = ind.safe_ratio(pd.Series([5.0, 5.0]), pd.Series([0.0, 2.5]))
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(2.0)
