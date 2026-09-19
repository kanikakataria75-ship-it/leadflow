"""Tests for the transaction cost models.

The cost stack decides whether a measured edge is tradeable, so its arithmetic
is worth pinning down precisely — an error here silently moves the break-even
line for every strategy in the project.
"""

from __future__ import annotations

import pytest

from nifty_swing_bot.backtest.costs import (
    cost_breakdown_pct,
    entry_charges,
    exit_charges,
    round_trip_cost_pct,
)
from nifty_swing_bot.config import AppConfig


@pytest.fixture
def execp():
    """Default execution params — the realistic Indian delivery model."""
    return AppConfig().execution


def test_default_model_is_the_realistic_one(execp) -> None:
    assert execp.cost_model == "india_delivery"


def test_stt_is_charged_on_both_sides_for_delivery(execp) -> None:
    """The correction that matters most.

    The original model charged STT on the sell only (10bps). Delivery equity
    pays 0.1% on buy *and* sell, so the round-trip figure is 20bps. Getting this
    wrong understates costs by 10bps — larger than most of the edges measured
    in this project.
    """
    notional = 100_000.0
    buy = entry_charges(notional, execp)
    sell = exit_charges(notional, execp)

    assert buy.stt == pytest.approx(notional * 0.001)
    assert sell.stt == pytest.approx(notional * 0.001)
    assert (buy.stt + sell.stt) / notional == pytest.approx(0.002)


def test_flat_brokerage_does_not_scale_with_position_size(execp) -> None:
    """Rs 20 per order is flat, which is the whole point of the change."""
    small = entry_charges(20_000.0, execp)
    large = entry_charges(500_000.0, execp)
    assert small.brokerage == pytest.approx(large.brokerage)
    assert small.brokerage == pytest.approx(execp.brokerage_flat_inr)


def test_brokerage_cap_binds_on_tiny_orders(execp) -> None:
    """Brokers quote 'Rs 20 or 2.5%, whichever is lower'."""
    # 2.5% of Rs 400 is Rs 10, below the Rs 20 flat fee.
    charges = entry_charges(400.0, execp)
    assert charges.brokerage == pytest.approx(10.0)


def test_zero_brokerage_is_supported(execp) -> None:
    """Several Indian brokers charge nothing for delivery."""
    execp.brokerage_flat_inr = 0.0
    charges = entry_charges(100_000.0, execp)
    assert charges.brokerage == 0.0
    assert charges.gst == pytest.approx(charges.exchange * execp.gst_pct / 100.0)


def test_depository_charge_applies_only_on_the_sell(execp) -> None:
    assert entry_charges(100_000.0, execp).dp == 0.0
    assert exit_charges(100_000.0, execp).dp == pytest.approx(execp.dp_charge_inr)


def test_stamp_duty_applies_only_on_the_buy(execp) -> None:
    assert entry_charges(100_000.0, execp).stamp > 0
    assert exit_charges(100_000.0, execp).stamp == 0.0


def test_gst_applies_to_brokerage_and_exchange_only(execp) -> None:
    """GST is not charged on STT or stamp duty."""
    charges = entry_charges(100_000.0, execp)
    expected = (charges.brokerage + charges.exchange) * execp.gst_pct / 100.0
    assert charges.gst == pytest.approx(expected)


def test_flat_fees_make_small_positions_disproportionately_expensive(execp) -> None:
    """Percentage cost must fall as position size rises, and materially so."""
    small = round_trip_cost_pct(20_000.0, execp)
    large = round_trip_cost_pct(200_000.0, execp)
    assert small > large
    # The flat component is worth well over 10bps between these two sizes.
    assert small - large > 0.1


def test_round_trip_includes_slippage_on_both_sides(execp) -> None:
    execp.slippage_bps = 10.0
    with_slip = round_trip_cost_pct(100_000.0, execp)
    execp.slippage_bps = 0.0
    without = round_trip_cost_pct(100_000.0, execp)
    assert with_slip - without == pytest.approx(0.20)  # 10bps x 2 sides


def test_lower_slippage_is_the_largest_available_saving(execp) -> None:
    """Justifies the liquidity restriction: slippage dominates the controllables.

    STT is fixed by statute and brokerage is already near zero, so the only
    component that meaningfully responds to trading better-chosen names is
    slippage.
    """
    parts = cost_breakdown_pct(100_000.0, execp)
    controllable = {k: parts[k] for k in ("brokerage", "slippage", "dp")}
    assert max(controllable, key=controllable.get) == "slippage"


def test_simple_model_reproduces_the_original_assumptions(execp) -> None:
    """The old model, kept for comparison and for exact-arithmetic tests."""
    execp.cost_model = "simple_bps"
    notional = 100_000.0
    buy = entry_charges(notional, execp)
    sell = exit_charges(notional, execp)

    assert buy.stt == 0.0, "the original model charged no STT on the buy"
    assert sell.stt == pytest.approx(notional * 0.001)
    assert buy.brokerage == pytest.approx(notional * execp.brokerage_bps / 10_000)
    assert buy.dp == 0.0 and sell.dp == 0.0


def test_realistic_model_is_more_expensive_than_the_original_at_typical_size(execp) -> None:
    """The headline finding: 'realistic' is not automatically 'cheaper'.

    Flat brokerage saves a little, but correcting STT to both sides costs more
    than that saving, so at a typical position size the realistic model is
    *worse* than the original approximation. Only lowering slippage improves it.
    """
    realistic = round_trip_cost_pct(100_000.0, execp)
    execp.cost_model = "simple_bps"
    original = round_trip_cost_pct(100_000.0, execp)
    assert realistic > original


def test_zero_notional_is_free_and_does_not_divide_by_zero(execp) -> None:
    assert entry_charges(0.0, execp).total == 0.0
    assert exit_charges(0.0, execp).total == 0.0
    assert round_trip_cost_pct(0.0, execp) == 0.0
    assert cost_breakdown_pct(0.0, execp) == {}
