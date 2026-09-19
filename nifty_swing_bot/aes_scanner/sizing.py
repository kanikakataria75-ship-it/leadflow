"""Trade plan: entry reference, ATR stop, risk-per-share sizing.

The arithmetic mirrors ``aes.portfolio.AESPortfolioBacktester`` deliberately,
so that what the scanner suggests and what the backtest measured are the same
rule. ``tests/test_aes_scanner.py`` asserts the two agree on a worked example
rather than trusting the comment.

One honest difference. The backtest fills at the *next* bar's open, which does
not exist yet when the scan runs after the close. The plan is therefore priced
off today's close and labelled a reference, not a fill: the stop distance and
the quantity move with the actual opening price, and the size should be
recomputed against it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..aes.params import PortfolioParams


@dataclass(frozen=True, slots=True)
class TradePlan:
    """What to do, in rupees, if this setup is taken."""

    entry_ref: float
    stop: float
    stop_basis: str          # "atr" | "big_box_bottom" | "min_risk_floor"
    risk_per_share: float
    stop_pct: float
    qty: int
    notional: float
    risk_amount: float
    capped_by_notional: bool
    #: Profit ladder as (trigger_price, qty_to_sell, trigger_pct), in the order
    #: they fire. Shown so the plan states where partial exits go *before* the
    #: trade is on, rather than leaving them to be remembered afterwards.
    tranches: tuple[tuple[float, int, float], ...] = ()
    #: Quantity still trailing on the ATR stop once every tranche has fired.
    runner_qty: int = 0


def tranche_levels(
    entry: float, qty: int, params: PortfolioParams
) -> tuple[tuple[tuple[float, int, float], ...], int]:
    """Ladder trigger prices and quantities for one position.

    Quantities are fractions of the ORIGINAL position, matching
    ``AESPortfolioBacktester``: a 50% tranche always means half the initial
    quantity, not half of whatever is left.
    """
    out: list[tuple[float, int, float]] = []
    sold = 0
    for trig, frac in (
        (params.ladder1_trigger_pct, params.ladder1_fraction),
        (params.ladder2_trigger_pct, params.ladder2_fraction),
        (params.ladder3_trigger_pct, params.ladder3_fraction),
    ):
        if frac <= 0 or qty <= 0:
            continue
        q = min(qty - sold, max(1, int(round(qty * frac))))
        if q <= 0:
            continue
        out.append((entry * (1.0 + trig), q, trig))
        sold += q
    return tuple(out), max(0, qty - sold)


def plan_trade(
    *,
    close: float,
    atr14: float,
    big_box_bottom: float,
    bucket: str,
    equity: float,
    params: PortfolioParams | None = None,
    regime_weak: bool = False,
) -> TradePlan:
    """Size a candidate at the configured risk per trade.

    Args:
        close: Latest close, used as the entry reference.
        atr14: ATR(14) at the same bar.
        big_box_bottom: The section 6.1 structural floor.
        bucket: Section 4 bucket, which scales size per section 7.
        equity: Account equity to size against.
        params: Portfolio parameters; defaults match the backtest.
        regime_weak: Whether the benchmark is below its own SMA(50).

    Returns:
        A ``TradePlan``. ``qty`` is 0 when the bucket is ``reject`` or the
        risk budget does not buy a single share.
    """
    p = params or PortfolioParams()
    mult = {
        "reject": p.size_mult_reject,
        "wait_and_watch": p.size_mult_wait_and_watch,
        "high_conviction": p.size_mult_high_conviction,
    }.get(bucket, 0.0)
    if regime_weak:
        mult *= p.regime_weak_size_mult

    # Whichever stop binds first as price falls -- the higher of the two --
    # matching the exit rule's own "either condition" logic.
    atr_level = close - (p.atr_stop_mult or 2.5) * atr14
    if atr_level >= big_box_bottom:
        stop, basis = atr_level, "atr"
    else:
        stop, basis = big_box_bottom, "big_box_bottom"

    floor = close * p.min_risk_per_share_pct
    raw = close - stop
    if raw < floor:
        risk_per_share, basis = floor, "min_risk_floor"
        stop = close - floor
    else:
        risk_per_share = raw

    if mult <= 0 or risk_per_share <= 0 or close <= 0:
        return TradePlan(round(close, 2), round(stop, 2), basis, round(risk_per_share, 2),
                         round(risk_per_share / close * 100, 2) if close else 0.0,
                         0, 0.0, 0.0, False)

    base_pct = p.base_position_pct_of_equity or (1.0 / p.target_concurrent_positions)
    cap_notional = equity * base_pct * mult
    risk_budget = equity * p.risk_per_trade_pct * mult
    want = risk_budget / risk_per_share * close
    notional = min(want, cap_notional)
    qty = int(notional // close)

    tr, runner = tranche_levels(close, qty, p)
    return TradePlan(
        entry_ref=round(close, 2),
        stop=round(stop, 2),
        stop_basis=basis,
        risk_per_share=round(risk_per_share, 2),
        stop_pct=round(risk_per_share / close * 100, 2),
        qty=qty,
        notional=round(qty * close, 0),
        risk_amount=round(qty * risk_per_share, 0),
        tranches=tuple((round(px, 2), q, pct) for px, q, pct in tr),
        runner_qty=runner,
        capped_by_notional=want > cap_notional,
    )


__all__ = ["TradePlan", "plan_trade", "tranche_levels"]
