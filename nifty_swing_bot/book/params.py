"""Configuration for the total-portfolio model. Every weight and rate lives here.

Nothing in ``book/`` hardcodes an allocation, a rate or a rebalance frequency;
they are all fields on ``BookParams`` so a different book can be modelled
without touching the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BookParams:
    """The whole book, not just the traded sleeve.

    Every return figure this project produced before Phase 19 described
    *deployed* capital only -- the AES sleeve at ~30% mean deployment -- and
    silently ignored that the rest of the book exists. These parameters make
    the rest explicit.
    """

    starting_capital: float = 1_000_000.0

    # --- sleeve weights (must sum to 1.0) ---------------------------------- #
    w_strategy: float = 0.30
    w_arbitrage: float = 0.50
    w_gold: float = 0.20

    # --- arbitrage sleeve --------------------------------------------------- #
    #: Annualised, accrued on actual calendar days (not trading days) so that
    #: weekends and holidays earn, which is what an arbitrage fund NAV does.
    arb_annual_rate: float = 0.065
    #: T+1 redemption. Cash lent to the strategy is requested one session
    #: before it is needed, so a same-day spike cannot be funded instantly.
    arb_redemption_lag_days: int = 1

    # --- gold sleeve -------------------------------------------------------- #
    #: A real INR price series, not an assumed yield. GOLDBEES is the NSE gold
    #: ETF: domestic INR gold including import duty, which is what an Indian
    #: book actually holds. Validated against GC=F x USDINR -- annual returns
    #: correlate 0.968 over 2014-2026.
    gold_symbol: str = "GOLDBEES"
    #: Gold is a diversifier, never a funding source. The strategy may draw on
    #: arbitrage and never on gold; this flag exists so that rule is visible
    #: in config rather than buried in the simulation.
    gold_is_drawable: bool = False

    # --- rebalancing -------------------------------------------------------- #
    #: "quarterly" | "annual" | "never".
    rebalance: str = "quarterly"

    # --- how the strategy is sized ------------------------------------------ #
    #: ``"book"``   risk-per-trade is a fraction of the WHOLE book, so mean
    #:              deployment lands near the 30% sleeve and peaks spill into
    #:              arbitrage. This is the only basis on which the
    #:              draw-from-arbitrage mechanic exists at all.
    #: ``"sleeve"`` risk-per-trade is a fraction of the strategy sleeve only.
    #:              Deployment then averages ~30% *of the sleeve* = ~9% of the
    #:              book, and arbitrage is never drawn on.
    #:
    #: Both are reported. Measured on this data, **"sleeve" is the one that
    #: matches the described mechanic**: deployment averages ~28% of the sleeve
    #: and exceeds it on 13 of 1,670 sessions, borrowing a peak of 8.6% of the
    #: book -- "mean ~30%, peaks higher when all 6 slots fill, draws from
    #: arbitrage". Under "book" the strategy borrows on 35-48% of all sessions
    #: and takes essentially the entire arbitrage sleeve at its peak (98-133%
    #: of the starting book), which is a leveraged book, not a 30% allocation.
    strategy_capital_basis: str = "sleeve"

    # --- conventions -------------------------------------------------------- #
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.065

    def __post_init__(self) -> None:
        total = self.w_strategy + self.w_arbitrage + self.w_gold
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"sleeve weights must sum to 1.0, got {total:.6f}")
        if self.rebalance not in ("quarterly", "annual", "never"):
            raise ValueError(f"rebalance must be quarterly|annual|never, got {self.rebalance!r}")
        if self.strategy_capital_basis not in ("book", "sleeve"):
            raise ValueError(
                f"strategy_capital_basis must be book|sleeve, got {self.strategy_capital_basis!r}"
            )
        for name in ("w_strategy", "w_arbitrage", "w_gold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


__all__ = ["BookParams"]
