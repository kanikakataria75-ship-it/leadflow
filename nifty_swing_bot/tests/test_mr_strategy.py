"""Tests for the Pullback Reversion strategy and the engine's exit-signal hook.

The mean-reversion strategy inverts DAB, so the tests check the inversion is real
(it fires on weakness, not strength) and that its reversion exit actually drives
the backtester rather than being ignored in favour of the max-hold clock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.backtest.engine import PortfolioBacktester
from nifty_swing_bot.config import AppConfig
from nifty_swing_bot.strategy import indicators as ind
from nifty_swing_bot.strategy import mr_strategy


@pytest.fixture
def cfg() -> AppConfig:
    c = AppConfig()
    # The realistic india_delivery model charges flat brokerage, two-sided
    # STT and a depository fee, none of which can be zeroed to a clean
    # number. These tests assert exact fill prices and R multiples, so they
    # opt into the simple percentage model with every rate set to zero.
    c.execution.cost_model = "simple_bps"
    c.execution.slippage_bps = 0.0
    c.execution.brokerage_bps = 0.0
    c.execution.stt_bps_sell = 0.0
    c.risk.trail_activate_r = 1e9      # exits are reversion-based, not trailed
    return c


def uptrend_with_dip(
    *, n: int = 120, dip_bar: int = 100, dip_pct: float = 0.06
) -> pd.DataFrame:
    """A steady uptrend with one sharp two-day flush, then recovery.

    Built so the dip bar sits comfortably above the 50-day average — the setup
    the strategy is meant to find — and so RSI(2) collapses on it.
    """
    idx = pd.bdate_range("2024-01-01", periods=n, name="date")
    close = 100.0 * (1.0 + 0.004) ** np.arange(n)   # ~0.4% per bar uptrend
    close[dip_bar - 1] *= 1.0 - dip_pct / 2
    close[dip_bar] *= 1.0 - dip_pct
    # Recovery after the dip.
    for i in range(dip_bar + 1, n):
        close[i] = close[i - 1] * 1.012

    frame = pd.DataFrame(
        {
            "open": close * 1.002,
            "high": close * 1.008,
            "low": close * 0.992,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )
    return frame


def test_fires_on_the_dip_not_the_strength(cfg: AppConfig) -> None:
    """The whole point: it must trigger on weakness inside an uptrend."""
    frame = uptrend_with_dip()
    feats = mr_strategy.compute_features(frame, None, cfg=cfg)

    fired = feats.index[feats["signal"]]
    assert len(fired) > 0, "no signal on an engineered dip"

    dip_date = frame.index[100]
    assert dip_date in fired, "the dip bar itself should qualify"
    # And the strategy must be above its trend filter there, not below it.
    assert bool(feats.loc[dip_date, "cond_trend"])
    assert feats.loc[dip_date, "rsi_fast"] < cfg.mr.rsi_entry


def test_does_not_fire_below_the_trend_filter(cfg: AppConfig) -> None:
    """A falling knife must be rejected: oversold is not enough on its own."""
    n = 120
    idx = pd.bdate_range("2024-01-01", periods=n, name="date")
    close = 100.0 * (1.0 - 0.006) ** np.arange(n)   # steady downtrend
    frame = pd.DataFrame(
        {"open": close * 1.002, "high": close * 1.008, "low": close * 0.99,
         "close": close, "volume": np.full(n, 1e6)},
        index=idx,
    )
    feats = mr_strategy.compute_features(frame, None, cfg=cfg)

    oversold = (feats["rsi_fast"] < cfg.mr.rsi_entry).sum()
    assert oversold > 0, "a downtrend should produce oversold bars"
    assert not feats["signal"].any(), "but none may become signals"


def test_rank_score_prefers_the_more_oversold(cfg: AppConfig) -> None:
    """Ranking drives most of what actually gets traded, so it must be sane."""
    frame = uptrend_with_dip()
    feats = mr_strategy.compute_features(frame, None, cfg=cfg)
    signals = feats[feats["signal"]]
    assert not signals.empty

    # rank_score is 100 - RSI, so a lower RSI must score higher.
    assert np.allclose(signals["rank_score"], 100.0 - signals["rsi_fast"])
    assert signals["rank_score"].min() > 100.0 - cfg.mr.rsi_entry


def test_reversion_exit_is_disabled_by_default(cfg: AppConfig) -> None:
    """The measured-harmful exit must not be on unless explicitly requested.

    Enabling it cut mean holding period from 9.1 bars to 2.7 and turned +7.95%
    into -41.50% on the same data, by collecting a fraction of a 10-bar edge
    while carrying the full stop. Off is the correct default.
    """
    frame = uptrend_with_dip()
    feats = mr_strategy.compute_features(frame, None, cfg=cfg)
    assert not feats["exit_signal"].any()
    assert cfg.mr.use_reversion_exit is False


def test_exit_signal_is_set_once_price_recovers_when_enabled(cfg: AppConfig) -> None:
    cfg.mr.use_reversion_exit = True
    frame = uptrend_with_dip()
    feats = mr_strategy.compute_features(frame, None, cfg=cfg)
    # After the recovery leg, price is above the short MA, so exits are flagged.
    assert bool(feats["exit_signal"].iloc[-1])
    # And on the dip bar it is not.
    assert not bool(feats["exit_signal"].iloc[100])


def test_delivery_data_does_not_change_the_signal(cfg: AppConfig) -> None:
    """Delivery is carried for display only; it must not influence any rule."""
    frame = uptrend_with_dip()
    deliv = pd.DataFrame(
        {"deliv_pct": np.random.default_rng(0).uniform(10, 90, len(frame))},
        index=frame.index,
    )
    without = mr_strategy.compute_features(frame, None, cfg=cfg)["signal"]
    with_deliv = mr_strategy.compute_features(frame, deliv, cfg=cfg)["signal"]
    assert without.equals(with_deliv)


# --------------------------------------------------------------------------- #
# Engine exit-signal hook
# --------------------------------------------------------------------------- #
def make_frame(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows], name="date")
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
         "volume": [r[5] for r in rows]},
        index=idx,
    )


def test_engine_honours_exit_signal_before_max_hold(cfg: AppConfig) -> None:
    """A reversion exit must close the trade rather than waiting for the clock."""
    cfg.risk.max_hold_days = 30
    frame = make_frame([
        ("2024-01-01", 100, 101, 99, 100, 1e6),
        ("2024-01-02", 100, 101, 99, 100, 1e6),   # signal bar
        ("2024-01-03", 100, 103, 99, 102, 1e6),   # entry at 100
        ("2024-01-04", 102, 108, 101, 107, 1e6),  # exit signal fires here
        ("2024-01-05", 107, 109, 106, 108, 1e6),
    ])
    feats = frame.copy()
    feats["signal"] = [False, True, False, False, False]
    feats["exit_signal"] = [False, False, False, True, False]
    feats["atr_stop"] = 2.0
    feats["rank_score"] = 50.0

    res = PortfolioBacktester(cfg).run({"X": frame}, {}, None, features={"X": feats})
    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.exit_reason == "signal_exit"
    assert trade.exit_date == pd.Timestamp("2024-01-04")
    assert trade.exit_price == pytest.approx(107.0)


def test_stop_still_takes_precedence_over_exit_signal(cfg: AppConfig) -> None:
    """The protective stop must not be pre-empted by a same-bar exit signal."""
    cfg.risk.max_hold_days = 30
    cfg.risk.atr_stop_mult = 1.5
    frame = make_frame([
        ("2024-01-01", 100, 101, 99, 100, 1e6),
        ("2024-01-02", 100, 101, 97, 100, 1e6),   # signal bar, low 97
        ("2024-01-03", 100, 103, 99, 102, 1e6),   # entry at 100, stop 97
        ("2024-01-04", 100, 104, 95, 103, 1e6),   # breaches 97 AND flags exit
    ])
    feats = frame.copy()
    feats["signal"] = [False, True, False, False]
    feats["exit_signal"] = [False, False, False, True]
    feats["atr_stop"] = 2.0
    feats["rank_score"] = 50.0

    res = PortfolioBacktester(cfg).run({"X": frame}, {}, None, features={"X": feats})
    trade = res.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(97.0)


def test_features_without_exit_signal_still_work(cfg: AppConfig) -> None:
    """The hook is optional: DAB frames carry no exit_signal column."""
    cfg.risk.max_hold_days = 2
    frame = make_frame([
        ("2024-01-01", 100, 101, 99, 100, 1e6),
        ("2024-01-02", 100, 101, 99, 100, 1e6),
        ("2024-01-03", 100, 103, 99, 102, 1e6),
        ("2024-01-04", 102, 104, 101, 103, 1e6),
        ("2024-01-05", 103, 105, 102, 104, 1e6),
        ("2024-01-06", 104, 106, 103, 105, 1e6),
    ])
    feats = frame.copy()
    feats["signal"] = [False, True, False, False, False, False]
    feats["atr_stop"] = 2.0
    res = PortfolioBacktester(cfg).run({"X": frame}, {}, None, features={"X": feats})
    assert res.trades[0].exit_reason == "max_hold"


def test_rsi2_is_fast_enough_to_reach_oversold_in_an_uptrend() -> None:
    """The premise behind choosing RSI(2) over RSI(14).

    On the same engineered dip, the fast oscillator reaches oversold while the
    slow one does not — which is exactly why the RSI(14) variant produced almost
    no signals above the trend filter.
    """
    frame = uptrend_with_dip()
    close = frame["close"]
    fast = ind.rsi(close, 2).iloc[100]
    slow = ind.rsi(close, 14).iloc[100]
    assert fast < 10.0, f"RSI(2) should collapse on the dip, got {fast:.1f}"
    assert slow > 30.0, f"RSI(14) should barely react, got {slow:.1f}"
