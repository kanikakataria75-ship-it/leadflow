"""THE FROZEN LIVE CONFIGURATION. Do not tune this against historical data.

Frozen 2026-09-17, at the end of Phase 20. The 12-year dataset is exhausted:
2015-2021 was discovery from the start and 2022-2026 was spent as discovery by
Phase 16, so there is no in-sample test left that can inform a change here.
Everything from this point should come from the forward record the scanner
accumulates (``protocol.FORWARD_CHECK_FROM`` = 2026-09-17).

If a value below is changed, the forward record before the change and after it
are not the same experiment. Record the change and restart the record.

Why each value is what it is
----------------------------
geometry      Baseline ``BoxParams()``. Every loosening tested in Phase 15
              lowered fold return; the sloped/range/duration variants are
              implemented and default-off.
slots / risk  6 slots at 3% risk. The only cell above 10% annual in BOTH
              windows (§18): 10.60% and 15.38%, mean drawdown 8.79% / 9.02%.
scoring       Equal weight, 0.25 x 4. The calibrated weights are REJECTED #4.
conviction    Bucket multipliers left at 0.7 / 1.2. Phase 20's conviction test
              found score-to-edge is **not monotone** (monotone in 2 of 11
              folds), so no scaling was adopted -- and the configured 0.7/1.2
              already beat both a neutral 1.0/1.0 and a steeper 0.5/1.5 in
              both windows, so it is also not changed.
exits         **Variant (c)**: first tranche 50% at +12%, second 20% at +20%,
              remainder trails on ATR(14) x 2.5 with a hard 25-bar cap. The
              spec's +5%/+8% first tranches truncate the return distribution --
              zero winners above +30% across 398 positions, against 7 and 15
              without them (§20). Variant (c) is the only ladder better than
              the spec ladder in both windows (+4.04pp and +4.44pp) and has the
              best ex-best-fold figure of any variant. It is a **candidate**,
              not a proven improvement: best paired t is 1.98.
"""

from __future__ import annotations

from datetime import date

from ..aes.params import BoxParams, PortfolioParams, ScoreParams, ScreenerParams, WatchlistParams
from ..book.params import BookParams

#: The day the configuration below was frozen and the forward record begins.
FROZEN_ON: date = date(2026, 9, 17)

SCREENER = ScreenerParams()
BOX = BoxParams()
WATCHLIST = WatchlistParams()

#: Equal weight, 0.25 x 4 (UNPROVEN #3: the calibrated weights are overfit).
SCORING = ScoreParams(
    w_fast_resolution=0.25, w_absorbed=0.25, w_rs_capture=0.25, w_prior_cycles=0.25
)

#: Variant (c) exits on top of the §18 sizing cell.
PORTFOLIO = PortfolioParams(
    max_open_positions=6,
    target_concurrent_positions=5.0,        # 6 x the locked 2.5/3 ratio
    risk_per_trade_pct=0.03,
    atr_stop_mult=2.5,
    atr_stop_period=14,
    max_hold_bars=25,
    ladder1_trigger_pct=0.12, ladder1_fraction=0.50,
    ladder2_trigger_pct=0.20, ladder2_fraction=0.20,   # stage 2 carries the +20% tranche
    ladder3_fraction=0.0,                              # stage 3 disabled
)

#: 30 / 50 / 20 with annual rebalancing, gold never drawn on.
BOOK = BookParams(
    w_strategy=0.30, w_arbitrage=0.50, w_gold=0.20,
    arb_annual_rate=0.065, rebalance="annual",
    strategy_capital_basis="sleeve", gold_is_drawable=False,
)

#: The ladder as (trigger, fraction of ORIGINAL position), for display.
TRANCHES: tuple[tuple[float, float], ...] = (
    (PORTFOLIO.ladder1_trigger_pct, PORTFOLIO.ladder1_fraction),
    (PORTFOLIO.ladder2_trigger_pct, PORTFOLIO.ladder2_fraction),
)


def describe() -> str:
    t = "  ".join(f"{f:.0%} @ +{p:.0%}" for p, f in TRANCHES)
    rem = 1.0 - sum(f for _, f in TRANCHES)
    return "\n".join([
        f"  FROZEN {FROZEN_ON}  -- forward record only from here",
        "  geometry   baseline BoxParams (no loosening adopted)",
        f"  sizing     {PORTFOLIO.max_open_positions} slots, "
        f"{PORTFOLIO.risk_per_trade_pct:.0%} risk/trade, conviction x"
        f"{PORTFOLIO.size_mult_wait_and_watch}/{PORTFOLIO.size_mult_high_conviction}",
        f"  exits      ATR({PORTFOLIO.atr_stop_period}) x {PORTFOLIO.atr_stop_mult} trail, "
        f"{PORTFOLIO.max_hold_bars}-bar cap; tranches {t}; {rem:.0%} trails",
        f"  book       {BOOK.w_strategy:.0%} strategy / {BOOK.w_arbitrage:.0%} arbitrage "
        f"/ {BOOK.w_gold:.0%} gold, {BOOK.rebalance} rebalance",
    ])


__all__ = ["BOOK", "BOX", "FROZEN_ON", "PORTFOLIO", "SCORING", "SCREENER",
           "TRANCHES", "WATCHLIST", "describe"]
