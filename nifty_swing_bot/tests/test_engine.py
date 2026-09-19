"""Mechanical tests for the portfolio backtester.

These use hand-built price paths where the correct fill price and exit reason
can be worked out on paper, so a regression in the fill logic fails loudly
rather than quietly changing the reported edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.backtest.engine import PortfolioBacktester
from nifty_swing_bot.backtest.metrics import max_drawdown, profit_factor, sharpe_ratio
from nifty_swing_bot.config import AppConfig


@pytest.fixture
def cfg() -> AppConfig:
    """Frictionless config so expected fills are exact; costs tested separately."""
    c = AppConfig()
    # The realistic india_delivery model charges flat brokerage, two-sided
    # STT and a depository fee, none of which can be zeroed to a clean
    # number. These tests assert exact fill prices and R multiples, so they
    # opt into the simple percentage model with every rate set to zero.
    c.execution.cost_model = "simple_bps"
    c.execution.slippage_bps = 0.0
    c.execution.brokerage_bps = 0.0
    c.execution.stt_bps_sell = 0.0
    c.execution.gap_through_extra_slippage_bps = 0.0
    c.risk.starting_capital = 1_000_000.0
    c.risk.max_open_positions = 5
    return c


def make_frame(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLCV frame from (date, o, h, l, c, v) tuples."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows], name="date")
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
        },
        index=idx,
    )


def features_with_signal_on(
    frame: pd.DataFrame, signal_date: str, *, atr_stop: float = 2.0
) -> pd.DataFrame:
    """A minimal feature frame that fires exactly one signal, bypassing the rules.

    The DAB rules are covered by ``test_dab_strategy``; here we want to isolate
    the *execution* machinery from signal generation.
    """
    feats = frame.copy()
    feats["signal"] = False
    feats.loc[pd.Timestamp(signal_date), "signal"] = True
    feats["atr_stop"] = atr_stop
    feats["deliv_ratio"] = 2.0
    feats["deliv_pct"] = 60.0
    feats["vol_ratio"] = 3.0
    feats["range_position"] = 0.9
    feats["rs_excess"] = 0.05
    feats["atr_ratio"] = 0.5
    feats["turnover_cr"] = 500.0  # slack liquidity cap
    return feats


def test_entry_fills_at_next_bar_open(cfg: AppConfig) -> None:
    frame = make_frame([
        ("2024-01-01", 100, 101, 99, 100, 1e6),
        ("2024-01-02", 100, 106, 99, 105, 3e6),   # signal bar
        ("2024-01-03", 106, 108, 105, 107, 1e6),  # entry at 106
        ("2024-01-04", 107, 109, 106, 108, 1e6),
    ])
    res = PortfolioBacktester(cfg).run(
        {"X": frame}, {}, None, features={"X": features_with_signal_on(frame, "2024-01-02")}
    )
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.entry_date == pd.Timestamp("2024-01-03")
    assert t.entry_price == pytest.approx(106.0)


def test_stop_fills_at_the_stop_when_price_trades_through_it(cfg: AppConfig) -> None:
    """Stop = min(signal low 99, entry 106 - 1.5*2 = 103) = 99."""
    frame = make_frame([
        ("2024-01-01", 100, 101, 99, 100, 1e6),
        ("2024-01-02", 100, 106, 99, 105, 3e6),
        ("2024-01-03", 106, 108, 105, 107, 1e6),
        ("2024-01-04", 107, 108, 98, 100, 1e6),   # dips to 98, through the 99 stop
    ])
    res = PortfolioBacktester(cfg).run(
        {"X": frame}, {}, None, features={"X": features_with_signal_on(frame, "2024-01-02")}
    )
    t = res.trades[0]
    assert t.initial_stop == pytest.approx(99.0)
    assert t.exit_price == pytest.approx(99.0)
    assert t.exit_reason == "stop"
    assert t.r_multiple == pytest.approx(-1.0, abs=0.01), "a clean stop must lose exactly 1R"


def test_gap_through_stop_fills_at_the_open_not_the_stop(cfg: AppConfig) -> None:
    """Modelling this as a clean stop fill is the classic way to flatter a backtest."""
    frame = make_frame([
        ("2024-01-01", 100, 101, 99, 100, 1e6),
        ("2024-01-02", 100, 106, 99, 105, 3e6),
        ("2024-01-03", 106, 108, 105, 107, 1e6),
        ("2024-01-04", 96, 97, 94, 95, 1e6),      # opens 96, below the 99 stop
    ])
    res = PortfolioBacktester(cfg).run(
        {"X": frame}, {}, None, features={"X": features_with_signal_on(frame, "2024-01-02")}
    )
    t = res.trades[0]
    assert t.exit_reason == "gap_through_stop"
    assert t.exit_price == pytest.approx(96.0)
    assert t.r_multiple < -1.0, "gapping through the stop must lose more than 1R"


def test_locked_lower_circuit_defers_the_exit_and_costs_more(cfg: AppConfig) -> None:
    """A circuit-locked bar has no bid, so the stop cannot be filled that day."""
    cfg.execution.circuit_band_pct = 0.10
    cfg.execution.locked_circuit_penalty_pct = 0.02
    frame = make_frame([
        ("2024-01-01", 100, 101, 99, 100, 1e6),
        ("2024-01-02", 100, 106, 99, 105, 3e6),
        ("2024-01-03", 106, 108, 105, 107, 1e6),   # entry 106, prev close 107
        # Opens at 96.3 == 107 * 0.90, and never trades away from it: locked.
        ("2024-01-04", 96.3, 96.3, 96.3, 96.3, 1e3),
        ("2024-01-05", 94, 95, 92, 93, 1e6),       # trapped exit happens here
    ])
    res = PortfolioBacktester(cfg).run(
        {"X": frame}, {}, None, features={"X": features_with_signal_on(frame, "2024-01-02")}
    )
    t = res.trades[0]
    assert t.exit_reason == "circuit_locked_stop"
    assert t.exit_date == pd.Timestamp("2024-01-05"), "exit must be deferred a bar"
    # Next open 94, minus the 2% locked-circuit penalty.
    assert t.exit_price == pytest.approx(94.0 * 0.98)


def test_max_hold_exit_after_configured_bars(cfg: AppConfig) -> None:
    cfg.risk.max_hold_days = 3
    cfg.risk.trail_activate_r = 99.0  # keep trailing out of the way
    rows = [("2024-01-01", 100, 101, 99, 100, 1e6), ("2024-01-02", 100, 106, 99, 105, 3e6)]
    price = 106.0
    for i in range(3, 12):
        rows.append((f"2024-01-{i:02d}", price, price + 1, price - 0.2, price + 0.5, 1e6))
        price += 0.5
    frame = make_frame(rows)
    res = PortfolioBacktester(cfg).run(
        {"X": frame}, {}, None, features={"X": features_with_signal_on(frame, "2024-01-02")}
    )
    t = res.trades[0]
    assert t.exit_reason == "max_hold"
    assert t.bars_held == 3


def test_trailing_stop_ratchets_up_and_never_down(cfg: AppConfig) -> None:
    """Once past 1R the stop follows swing lows upward, locking in gains."""
    cfg.risk.trail_activate_r = 1.0
    cfg.risk.trail_swing_lookback = 1
    cfg.risk.max_hold_days = 30
    rows = [("2024-01-01", 100, 101, 99, 100, 1e6), ("2024-01-02", 100, 106, 99, 105, 3e6)]
    # Rally past 1R, pull back to carve a genuine swing low at 116, resume, then
    # collapse. The pullback matters: a swing low is a *local minimum*, so a
    # monotonic rally would never create one (see the test below).
    path = [
        (106, 112, 105, 111),
        (111, 120, 110, 119),
        (119, 126, 118, 125),
        (125, 127, 116, 126),   # pullback low of 116 -- the pivot
        (126, 134, 122, 133),   # confirms the pivot; stop trails to 116
        (133, 134, 100, 101),   # collapse: must exit at 116, not the initial 99
    ]
    for i, (o, h, low, c) in enumerate(path, start=3):
        rows.append((f"2024-01-{i:02d}", o, h, low, c, 1e6))
    frame = make_frame(rows)
    res = PortfolioBacktester(cfg).run(
        {"X": frame}, {}, None, features={"X": features_with_signal_on(frame, "2024-01-02")}
    )
    t = res.trades[0]
    assert t.exit_reason == "trailing_stop"
    assert t.initial_stop == pytest.approx(99.0)
    assert t.exit_price == pytest.approx(116.0), "must fill at the trailed stop"
    assert t.r_multiple == pytest.approx((116.0 - 106.0) / 7.0, abs=0.01)


def test_monotonic_rally_leaves_the_stop_at_its_initial_level(cfg: AppConfig) -> None:
    """A documented property of trailing on swing lows, not a bug.

    A swing low is a local minimum. A rally that never pulls back contains none,
    so there is nothing to trail to and the stop stays at its initial level until
    the max-hold clock ends the trade. This is why ``max_hold_days`` is the real
    backstop for winners, and it is worth knowing before trusting the trail to
    protect open profit.
    """
    cfg.risk.trail_activate_r = 1.0
    cfg.risk.trail_swing_lookback = 1
    cfg.risk.max_hold_days = 30
    rows = [("2024-01-01", 100, 101, 99, 100, 1e6), ("2024-01-02", 100, 106, 99, 105, 3e6)]
    # Strictly increasing lows: 105, 110, 118, 121, 128 -- no local minimum.
    path = [
        (106, 112, 105, 111), (111, 120, 110, 119), (119, 126, 118, 125),
        (125, 130, 121, 129), (129, 134, 128, 133),
        (133, 134, 100, 101),
    ]
    for i, (o, h, low, c) in enumerate(path, start=3):
        rows.append((f"2024-01-{i:02d}", o, h, low, c, 1e6))
    frame = make_frame(rows)
    res = PortfolioBacktester(cfg).run(
        {"X": frame}, {}, None, features={"X": features_with_signal_on(frame, "2024-01-02")}
    )
    t = res.trades[0]
    # Low of 100 on the collapse bar never reaches the untouched 99 stop.
    assert t.initial_stop == pytest.approx(99.0)
    assert t.exit_reason == "end_of_data"


def test_max_open_positions_is_respected(cfg: AppConfig) -> None:
    cfg.risk.max_open_positions = 2
    cfg.risk.max_hold_days = 30
    cfg.risk.max_portfolio_heat_pct = 1.0
    prices, feats = {}, {}
    rows = [("2024-01-01", 100, 101, 99, 100, 1e6), ("2024-01-02", 100, 106, 99, 105, 3e6)]
    for i in range(3, 20):
        rows.append((f"2024-01-{i:02d}", 106, 107, 105.5, 106.5, 1e6))
    for sym in ("A", "B", "C", "D", "E"):
        frame = make_frame(rows)
        prices[sym] = frame
        feats[sym] = features_with_signal_on(frame, "2024-01-02")

    res = PortfolioBacktester(cfg).run(prices, {}, None, features=feats)
    assert res.equity_curve["open_positions"].max() <= 2
    assert res.rejected["no_capacity"] >= 3


def test_costs_reduce_net_pnl_below_gross() -> None:
    cfg = AppConfig()
    cfg.risk.starting_capital = 1_000_000.0
    frame = make_frame([
        ("2024-01-01", 100, 101, 99, 100, 1e6),
        ("2024-01-02", 100, 106, 99, 105, 3e6),
        ("2024-01-03", 106, 108, 105, 107, 1e6),
        ("2024-01-04", 107, 120, 106, 119, 1e6),
        ("2024-01-05", 119, 121, 100, 101, 1e6),
    ])
    res = PortfolioBacktester(cfg).run(
        {"X": frame}, {}, None, features={"X": features_with_signal_on(frame, "2024-01-02")}
    )
    t = res.trades[0]
    assert t.costs > 0
    assert t.net_pnl == pytest.approx(t.gross_pnl - t.costs)


def test_no_signals_produces_flat_equity(cfg: AppConfig) -> None:
    frame = make_frame([
        ("2024-01-01", 100, 101, 99, 100, 1e6),
        ("2024-01-02", 100, 106, 99, 105, 3e6),
        ("2024-01-03", 106, 108, 105, 107, 1e6),
    ])
    feats = frame.copy()
    feats["signal"] = False
    feats["atr_stop"] = 2.0
    res = PortfolioBacktester(cfg).run({"X": frame}, {}, None, features={"X": feats})
    assert not res.trades
    assert res.equity_curve["equity"].nunique() == 1
    assert res.equity_curve["equity"].iloc[-1] == pytest.approx(cfg.risk.starting_capital)


def test_equity_curve_matches_cash_plus_positions(cfg: AppConfig) -> None:
    """Accounting identity: nothing may leak between cash and holdings."""
    frame = make_frame([
        ("2024-01-01", 100, 101, 99, 100, 1e6),
        ("2024-01-02", 100, 106, 99, 105, 3e6),
        ("2024-01-03", 106, 108, 105, 107, 1e6),
        ("2024-01-04", 107, 112, 106, 111, 1e6),
        ("2024-01-05", 111, 113, 95, 96, 1e6),
    ])
    res = PortfolioBacktester(cfg).run(
        {"X": frame}, {}, None, features={"X": features_with_signal_on(frame, "2024-01-02")}
    )
    eq = res.equity_curve
    assert np.allclose(eq["equity"], eq["cash"] + eq["positions_value"])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_max_drawdown_on_a_known_path() -> None:
    equity = pd.Series(
        [100.0, 120.0, 90.0, 110.0], index=pd.bdate_range("2024-01-01", periods=4)
    )
    depth, peak, trough, _ = max_drawdown(equity)
    assert depth == pytest.approx(0.25)  # 120 -> 90
    assert peak == equity.index[1]
    assert trough == equity.index[2]


def test_profit_factor_arithmetic() -> None:
    assert profit_factor([10.0, 20.0, -10.0]) == pytest.approx(3.0)
    assert profit_factor([10.0, 5.0]) == float("inf")
    assert profit_factor([]) == 0.0


def test_sharpe_is_zero_for_a_constant_curve() -> None:
    returns = pd.Series([0.0] * 50)
    assert sharpe_ratio(returns, 0.065, 252) == 0.0
