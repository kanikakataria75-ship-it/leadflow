"""Tests for the Sector-Relative Dislocation Reversion strategy.

The strategy's whole claim is that *relative* weakness and *absolute* weakness
are different things, so the tests check that distinction directly: a stock that
falls with its sector must not fire, and one that falls alone must.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.config import AppConfig
from nifty_swing_bot.research.data_audit import find_lookahead
from nifty_swing_bot.research.hypotheses import build_features, build_sector_aggregates
from nifty_swing_bot.strategy import srdr

N_BARS = 400


def make_series(
    *, drift: float = 0.0005, shock_bar: int | None = None, shock: float = -0.08,
    seed: int = 0, start: float = 100.0,
) -> pd.DataFrame:
    """A trending series with an optional single-bar shock."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=N_BARS, name="date")
    steps = rng.normal(drift, 0.012, N_BARS)
    if shock_bar is not None:
        steps[shock_bar] = shock
    close = start * np.exp(np.cumsum(steps))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.006,
            "low": close * 0.994,
            "close": close,
            "volume": np.full(N_BARS, 2_000_000.0),
        },
        index=idx,
    )


def peer_group(
    n: int, *, shock_bar: int | None = None, shock: float = -0.08, drift: float = 0.0005
) -> dict:
    """A set of peers; only the first optionally receives the shock.

    ``drift`` matters more than it looks. The strategy only buys names above
    their long moving average, so a peer group with negligible drift falls below
    its own average the moment it is shocked, and the trend filter -- not the
    signal -- decides the test. Tests that mean to exercise the signal use a
    drift strong enough for the stock to stay in an uptrend through the shock.
    """
    out = {}
    for i in range(n):
        out[f"PEER{i}"] = make_series(
            seed=i + 1,
            drift=drift,
            shock_bar=shock_bar if i == 0 else None,
            shock=shock,
        )
    return out


@pytest.fixture
def cfg() -> AppConfig:
    c = AppConfig()
    c.srdr.min_turnover_cr = 0.0      # synthetic volumes are arbitrary
    c.srdr.trend_ma = 50              # shorter series in tests
    return c


def test_fires_when_a_stock_falls_alone(cfg: AppConfig) -> None:
    """The core claim: dislocation from peers is the trigger.

    The shock is sized at roughly 2.5 ATRs, not 8. A -10% move on a stock whose
    daily range is 1.2% is 8 standard deviations, which the ``collapse_atr``
    filter correctly rejects as a repricing rather than an inventory shock --
    so testing with one would test the filter, not the signal.
    """
    shock_bar = 300
    prices = peer_group(8, shock_bar=shock_bar, shock=-0.035, drift=0.0025)
    industry_of = {s: "TestSector" for s in prices}

    features = srdr.build_signals(prices, industry_of, cfg=cfg)
    assert "PEER0" in features

    shocked = features["PEER0"]
    window = shocked.index[shock_bar : shock_bar + 3]
    assert shocked.loc[window, "signal"].any(), "the dislocated stock should be selected"

    # The dislocation is measured over three days, so a one-bar shock is only
    # fully inside the window on the FOLLOWING bar: at `shock_bar` two rising
    # days still offset it. Asserting on `shock_bar` itself tests the wrong bar.
    measured = shock_bar + 1
    assert shocked["rel_sector_3"].iloc[measured] < -0.02
    # The filter that rejects genuine collapses must not have fired here.
    assert shocked["ret_3_atr"].iloc[measured] > -cfg.srdr.collapse_atr


def test_a_collapse_is_rejected_even_though_it_is_dislocated(cfg: AppConfig) -> None:
    """A large enough solo fall is read as news, not as an inventory shock.

    This is the filter that made ranking on absolute weakness produce a NEGATIVE
    edge: the biggest decliner on any given day usually does have real news.
    """
    # Strong drift so the trend filter passes and the collapse filter is the
    # only thing that can reject it.
    prices = peer_group(8, shock_bar=300, shock=-0.12, drift=0.0025)
    industry_of = {s: "TestSector" for s in prices}
    features = srdr.build_signals(prices, industry_of, cfg=cfg)

    # As above, the three-day window only contains the whole shock one bar later.
    shocked = features["PEER0"]
    measured = 301
    assert shocked["rel_sector_3"].iloc[measured] < -0.05, "it is dislocated"
    assert shocked["ret_3_atr"].iloc[measured] < -cfg.srdr.collapse_atr, "but it is a collapse"
    assert not bool(shocked["eligible"].iloc[measured]), "so it must be rejected"

    # A fall this large also breaks the trend filter, so both guards reject it.
    # That overlap is by design rather than redundancy: they catch the same
    # failure mode -- a genuine repricing -- from different directions, and
    # either alone would leave a gap the other covers.
    assert not bool(shocked["above_trend"].iloc[measured])


def test_does_not_fire_when_the_whole_sector_falls(cfg: AppConfig) -> None:
    """A stock falling with its peers is not dislocated; it is just a sector move.

    This is the distinction the whole strategy rests on. A generic oversold rule
    would fire here; this one must not.
    """
    idx = pd.bdate_range("2020-01-01", periods=N_BARS, name="date")
    prices = {}
    for i in range(8):
        rng = np.random.default_rng(i + 50)
        steps = rng.normal(0.0006, 0.010, N_BARS)
        steps[300] = -0.09          # every peer drops together
        close = 100.0 * np.exp(np.cumsum(steps))
        prices[f"PEER{i}"] = pd.DataFrame(
            {"open": close * 0.999, "high": close * 1.006, "low": close * 0.994,
             "close": close, "volume": np.full(N_BARS, 2e6)},
            index=idx,
        )
    industry_of = {s: "TestSector" for s in prices}

    features = srdr.build_signals(prices, industry_of, cfg=cfg)
    window = slice(300, 303)
    fired = sum(int(f["signal"].iloc[window].any()) for f in features.values())
    assert fired == 0, "a sector-wide decline must not be read as dislocation"


def test_orphan_gap_is_excluded(cfg: AppConfig) -> None:
    """Unshared gap-downs were measured to keep falling, so they are filtered."""
    prices = peer_group(8)
    # Force PEER0 into a large unshared gap down on one bar.
    bar = 300
    frame = prices["PEER0"].copy()
    prev_close = float(frame["close"].iloc[bar - 1])
    frame.iloc[bar, frame.columns.get_loc("open")] = prev_close * 0.95
    frame.iloc[bar, frame.columns.get_loc("close")] = prev_close * 0.94
    frame.iloc[bar, frame.columns.get_loc("high")] = prev_close * 0.955
    frame.iloc[bar, frame.columns.get_loc("low")] = prev_close * 0.935
    prices["PEER0"] = frame

    industry_of = {s: "TestSector" for s in prices}
    features = srdr.build_signals(prices, industry_of, cfg=cfg)
    shocked = features["PEER0"]
    assert not bool(shocked["eligible"].iloc[bar]), "orphan gap-down must be ineligible"


def test_below_trend_filter_blocks_entry(cfg: AppConfig) -> None:
    """Below the long average, cheap-versus-peers may be a failing business."""
    prices = peer_group(8, shock_bar=300, shock=-0.10)
    # Give PEER0 a persistent downtrend so it sits below its own average.
    falling = make_series(drift=-0.004, shock_bar=300, shock=-0.10, seed=99)
    prices["PEER0"] = falling
    industry_of = {s: "TestSector" for s in prices}

    features = srdr.build_signals(prices, industry_of, cfg=cfg)
    shocked = features["PEER0"]
    assert not shocked["above_trend"].iloc[300]
    assert not bool(shocked["eligible"].iloc[300])


def test_rank_score_rises_with_dislocation(cfg: AppConfig) -> None:
    """The engine breaks ties by conviction, so the ordering must be right."""
    prices = peer_group(8, shock_bar=300, shock=-0.10)
    industry_of = {s: "TestSector" for s in prices}
    features = srdr.build_signals(prices, industry_of, cfg=cfg)

    shocked = features["PEER0"]
    more_dislocated = shocked["dislocation"].iloc[300]
    less_dislocated = shocked["dislocation"].iloc[250]
    assert more_dislocated < less_dislocated
    assert shocked["rank_score"].iloc[300] > shocked["rank_score"].iloc[250]


def test_only_top_n_are_selected_each_day(cfg: AppConfig) -> None:
    """Selection is cross-sectional: at most top_n names per date."""
    cfg.srdr.top_n = 2
    cfg.srdr.min_dislocation = 0.0
    prices = peer_group(12)
    industry_of = {s: "TestSector" for s in prices}
    features = srdr.build_signals(prices, industry_of, cfg=cfg)

    stacked = pd.DataFrame({s: f["signal"] for s, f in features.items()})
    per_day = stacked.sum(axis=1)
    assert per_day.max() <= cfg.srdr.top_n


def test_feature_builder_has_no_lookahead() -> None:
    """Structural audit: truncating history must not change any past value.

    This is the check that catches centred windows, missing shifts and
    full-period normalisation without having to reason about each formula.
    """
    frame = make_series(seed=7)
    leaks = find_lookahead(lambda f: build_features(f), frame, min_bar=250)
    assert leaks == [], f"look-ahead detected: {leaks[:3]}"


def test_sector_aggregates_are_knowable_at_the_time() -> None:
    """Peer medians must not move when future bars are removed."""
    prices = peer_group(6)
    industry_of = {s: "TestSector" for s in prices}

    full, _, _ = build_sector_aggregates(prices, industry_of)
    cut = 320
    truncated_prices = {s: f.iloc[:cut] for s, f in prices.items()}
    truncated, _, _ = build_sector_aggregates(truncated_prices, industry_of)

    stamp = prices["PEER0"].index[cut - 1]
    assert stamp in full["TestSector"].index
    assert stamp in truncated["TestSector"].index
    assert full["TestSector"].loc[stamp] == pytest.approx(
        truncated["TestSector"].loc[stamp], abs=1e-10
    )


def test_signal_is_not_set_without_sector_context(cfg: AppConfig) -> None:
    """With no peers there is no dislocation to measure, so nothing fires."""
    prices = {"LONELY": make_series(shock_bar=300, shock=-0.12)}
    features = srdr.build_signals(prices, {"LONELY": "TestSector"}, cfg=cfg)
    if features:
        # A single-member sector has zero relative return by construction.
        assert not features["LONELY"]["signal"].any()
