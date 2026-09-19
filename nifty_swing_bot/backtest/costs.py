"""Transaction cost models.

Two models are available:

``simple_bps``
    Flat basis-point charges on each side. Easy to reason about, and what the
    early backtests used, but it models brokerage as a percentage -- which no
    Indian discount broker actually charges any more.

``india_delivery``
    The real charge stack for a **delivery** (CNC) equity trade on an Indian
    discount broker. It differs from the simple model in two ways that pull in
    opposite directions:

    * **Brokerage is flat**, typically Rs 20 per order or zero, not a percentage.
      On a Rs 1 lakh position that is ~2bps instead of ~3bps; on a Rs 20,000
      position it is ~10bps. Flat fees punish small positions.
    * **STT is 0.1% on BOTH sides** for delivery, not just the sell. The earlier
      model charged 10bps on the sell alone; the correct figure is 20bps round
      trip. Intraday trades pay 0.025% on the sell only, but nothing here is
      intraday -- these are multi-day holds settled to demat.

    Plus stamp duty (buy only), exchange transaction charges, SEBI turnover
    fees, 18% GST on brokerage and exchange charges, and a flat depository
    charge on every sell.

Slippage is *not* included here. It is applied to the fill price by the engine,
because it is a price effect rather than a charge, and because it is the one
component that genuinely varies with how liquid the stock is.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ExecutionParams


@dataclass(frozen=True, slots=True)
class ChargeBreakdown:
    """Itemised charges for one side of a trade, in rupees."""

    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    stamp: float = 0.0
    sebi: float = 0.0
    gst: float = 0.0
    dp: float = 0.0

    @property
    def total(self) -> float:
        """Sum of every charge."""
        return (
            self.brokerage + self.stt + self.exchange
            + self.stamp + self.sebi + self.gst + self.dp
        )

    def as_dict(self) -> dict[str, float]:
        """Itemised view, plus the total, for reporting."""
        return {
            "brokerage": round(self.brokerage, 2),
            "stt": round(self.stt, 2),
            "exchange": round(self.exchange, 2),
            "stamp": round(self.stamp, 2),
            "sebi": round(self.sebi, 2),
            "gst": round(self.gst, 2),
            "dp": round(self.dp, 2),
            "total": round(self.total, 2),
        }


def _bps(value: float, notional: float) -> float:
    return notional * value / 10_000.0


def entry_charges(notional: float, execp: ExecutionParams) -> ChargeBreakdown:
    """Charges on the buy side of a delivery trade.

    Args:
        notional: Rupee value of the purchase (quantity x fill price).
        execp: Execution parameters.

    Returns:
        The itemised charges.
    """
    if notional <= 0:
        return ChargeBreakdown()

    if execp.cost_model == "simple_bps":
        return ChargeBreakdown(brokerage=_bps(execp.brokerage_bps, notional))

    # Flat brokerage, optionally capped at a percentage of turnover the way
    # several brokers quote it ("Rs 20 or 2.5%, whichever is lower").
    brokerage = execp.brokerage_flat_inr
    if execp.brokerage_pct_cap > 0:
        brokerage = min(brokerage, notional * execp.brokerage_pct_cap / 100.0)

    exchange = _bps(execp.exchange_txn_bps, notional)
    charges = ChargeBreakdown(
        brokerage=brokerage,
        stt=_bps(execp.stt_bps_buy, notional),
        exchange=exchange,
        stamp=_bps(execp.stamp_duty_bps_buy, notional),
        sebi=_bps(execp.sebi_bps, notional),
        # GST applies to brokerage and exchange charges, not to STT or stamp duty.
        gst=(brokerage + exchange) * execp.gst_pct / 100.0,
    )
    return charges


def exit_charges(notional: float, execp: ExecutionParams) -> ChargeBreakdown:
    """Charges on the sell side of a delivery trade.

    The depository (DP) charge is a flat per-scrip debit on every sell,
    regardless of quantity, which makes small positions disproportionately
    expensive to exit.
    """
    if notional <= 0:
        return ChargeBreakdown()

    if execp.cost_model == "simple_bps":
        return ChargeBreakdown(
            brokerage=_bps(execp.brokerage_bps, notional),
            stt=_bps(execp.stt_bps_sell, notional),
        )

    brokerage = execp.brokerage_flat_inr
    if execp.brokerage_pct_cap > 0:
        brokerage = min(brokerage, notional * execp.brokerage_pct_cap / 100.0)

    exchange = _bps(execp.exchange_txn_bps, notional)
    return ChargeBreakdown(
        brokerage=brokerage,
        stt=_bps(execp.stt_bps_sell, notional),
        exchange=exchange,
        sebi=_bps(execp.sebi_bps, notional),
        gst=(brokerage + exchange) * execp.gst_pct / 100.0,
        dp=execp.dp_charge_inr,
    )


def round_trip_cost_pct(notional: float, execp: ExecutionParams) -> float:
    """Total round-trip cost as a percentage of notional, including slippage.

    This is the number that has to be compared against a signal's measured edge.
    Because flat fees do not scale, it depends on position size: a Rs 20,000
    position pays a much higher percentage than a Rs 2,00,000 one.

    Args:
        notional: Position size in rupees.
        execp: Execution parameters.

    Returns:
        Round-trip cost as a percentage (0.46 means 0.46%).
    """
    if notional <= 0:
        return 0.0
    charges = entry_charges(notional, execp).total + exit_charges(notional, execp).total
    slippage = notional * (2 * execp.slippage_bps) / 10_000.0
    return (charges + slippage) / notional * 100.0


def cost_breakdown_pct(notional: float, execp: ExecutionParams) -> dict[str, float]:
    """Round-trip cost decomposed into components, each as a % of notional."""
    if notional <= 0:
        return {}
    entry = entry_charges(notional, execp)
    exit_ = exit_charges(notional, execp)
    pct = lambda v: round(v / notional * 100.0, 4)  # noqa: E731
    return {
        "brokerage": pct(entry.brokerage + exit_.brokerage),
        "stt": pct(entry.stt + exit_.stt),
        "exchange": pct(entry.exchange + exit_.exchange),
        "stamp": pct(entry.stamp),
        "sebi": pct(entry.sebi + exit_.sebi),
        "gst": pct(entry.gst + exit_.gst),
        "dp": pct(exit_.dp),
        "slippage": pct(notional * (2 * execp.slippage_bps) / 10_000.0),
        "TOTAL": round_trip_cost_pct(notional, execp),
    }


__all__ = [
    "ChargeBreakdown",
    "cost_breakdown_pct",
    "entry_charges",
    "exit_charges",
    "round_trip_cost_pct",
]
