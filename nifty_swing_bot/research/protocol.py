"""Research protocol: the rules this study binds itself to, as executable code.

Written and committed **before** any hypothesis was tested, so the split points
cannot drift to flatter a result. Every module in ``research/`` imports its
windows from here rather than defining its own.

Three date-separated periods:

``DISCOVERY``
    Where hypotheses are generated, features explored and thresholds chosen.
    Everything may be looked at as often as needed.

``VALIDATION``
    Used to *select* among candidates that already survived discovery. Looked at
    a bounded number of times. A candidate that fails here is rejected, not
    retuned.

``HOLDOUT``
    Touched exactly once, at the end, by whichever single candidate wins. No
    parameter, threshold, universe rule or cost assumption may be changed after
    reading it. If the result is bad, it is reported as bad.

The split is by **date**, not at random. Random splits leak: adjacent bars of
the same stock are highly correlated and overlapping forward-return windows
would straddle the boundary, so a random split tests memorisation rather than
generalisation.

Regime coverage was the criterion for where the boundaries fall, not
performance -- none of it had been measured when these were fixed:

======================  ====================================================
period                  regimes contained
======================  ====================================================
DISCOVERY 2015-2021     2015-16 correction, demonetisation, 2017 melt-up,
                        2018-19 small-cap bear, COVID crash and recovery,
                        2021 bull
VALIDATION 2022-2024H1  2022 rate-shock bear, 2023 recovery, 2024 H1 bull
HOLDOUT 2024H2-2026     2024 H2 peak, 2025 correction, 2026 recovery
======================  ====================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import pandas as pd


@dataclass(frozen=True, slots=True)
class Window:
    """A named, inclusive date range."""

    name: str
    start: date
    end: date
    purpose: str

    def contains(self, when: date | pd.Timestamp) -> bool:
        """True if a date falls inside this window."""
        stamp = pd.Timestamp(when).date()
        return self.start <= stamp <= self.end

    def mask(self, dates: pd.Series) -> pd.Series:
        """Boolean mask selecting rows whose date falls in this window."""
        stamps = pd.to_datetime(dates)
        return (stamps >= pd.Timestamp(self.start)) & (stamps <= pd.Timestamp(self.end))

    def __str__(self) -> str:
        return f"{self.name} [{self.start} .. {self.end}]"


DISCOVERY: Final[Window] = Window(
    "DISCOVERY",
    date(2015, 1, 1),
    date(2021, 12, 31),
    "Hypothesis generation and threshold selection. Unlimited looks.",
)

VALIDATION: Final[Window] = Window(
    "VALIDATION",
    date(2022, 1, 1),
    date(2024, 6, 30),
    "Candidate selection only. Failure means rejection, not retuning.",
)

HOLDOUT: Final[Window] = Window(
    "HOLDOUT",
    date(2024, 7, 1),
    date(2026, 12, 31),
    "Touched once, by one candidate, at the end. Nothing may change after.",
)

ALL_WINDOWS: Final[tuple[Window, ...]] = (DISCOVERY, VALIDATION, HOLDOUT)

#: Data fetched from here, giving the discovery window a warm-up buffer for
#: 200-day lookbacks without those lookbacks reaching outside the dataset.
DATA_START: Final[date] = date(2014, 1, 1)


# --------------------------------------------------------------------------- #
# Short-swing definition
# --------------------------------------------------------------------------- #
#: The holding-period framework, fixed before any candidate was measured so the
#: threshold cannot be reverse-engineered from whatever looked profitable.
#:
#: Rationale: a "swing" trade is conventionally understood as spanning days to
#: a couple of weeks -- long enough to capture a multi-day move, short enough
#: that the position is not a proxy for a trend or a factor exposure. Two weeks
#: of trading days is 10 bars, so:
MAX_HOLD_BARS: Final[int] = 10          # hard cap; nothing may exceed it
TARGET_MEAN_HOLD_BARS: Final[float] = 6.0   # average must land at or below this
TARGET_MEDIAN_HOLD_BARS: Final[float] = 5.0  # median at or below this
FAST_CLOSE_BARS: Final[int] = 5         # "closes quickly" means within a week
MIN_FAST_CLOSE_FRACTION: Final[float] = 0.50  # at least half must close that fast

#: Trade frequency floor. Below this the sample is too small to distinguish
#: skill from luck, regardless of how good the headline numbers look.
MIN_TRADES_PER_YEAR: Final[int] = 40
MIN_TOTAL_TRADES: Final[int] = 200


def qualifies_as_short_swing(
    mean_hold: float, median_hold: float, fast_close_fraction: float, max_hold: int
) -> tuple[bool, list[str]]:
    """Check a candidate against the short-swing framework.

    Args:
        mean_hold: Average bars held.
        median_hold: Median bars held.
        fast_close_fraction: Fraction of trades closed within ``FAST_CLOSE_BARS``.
        max_hold: Longest single hold observed.

    Returns:
        ``(passes, reasons_for_failure)``.
    """
    failures: list[str] = []
    if max_hold > MAX_HOLD_BARS:
        failures.append(f"max hold {max_hold} > {MAX_HOLD_BARS} bars")
    if mean_hold > TARGET_MEAN_HOLD_BARS:
        failures.append(f"mean hold {mean_hold:.1f} > {TARGET_MEAN_HOLD_BARS}")
    if median_hold > TARGET_MEDIAN_HOLD_BARS:
        failures.append(f"median hold {median_hold:.1f} > {TARGET_MEDIAN_HOLD_BARS}")
    if fast_close_fraction < MIN_FAST_CLOSE_FRACTION:
        failures.append(
            f"only {fast_close_fraction:.0%} closed within {FAST_CLOSE_BARS} bars "
            f"(need {MIN_FAST_CLOSE_FRACTION:.0%})"
        )
    return (not failures), failures


def qualifies_on_frequency(total_trades: int, years: float) -> tuple[bool, list[str]]:
    """Check a candidate against the trade-frequency floor."""
    failures: list[str] = []
    per_year = total_trades / years if years > 0 else 0.0
    if total_trades < MIN_TOTAL_TRADES:
        failures.append(f"{total_trades} trades < {MIN_TOTAL_TRADES} minimum")
    if per_year < MIN_TRADES_PER_YEAR:
        failures.append(f"{per_year:.0f} trades/year < {MIN_TRADES_PER_YEAR} minimum")
    return (not failures), failures


def describe() -> str:
    """Human-readable statement of the protocol, printed by every study."""
    lines = ["", "=" * 78, "  RESEARCH PROTOCOL (fixed before any result was measured)", "=" * 78]
    for window in ALL_WINDOWS:
        lines.append(f"  {window}")
        lines.append(f"      {window.purpose}")
    lines += [
        "",
        "  SHORT-SWING FRAMEWORK",
        f"      hard max hold        {MAX_HOLD_BARS} bars",
        f"      mean hold target     <= {TARGET_MEAN_HOLD_BARS} bars",
        f"      median hold target   <= {TARGET_MEDIAN_HOLD_BARS} bars",
        f"      closed within {FAST_CLOSE_BARS} bars  >= {MIN_FAST_CLOSE_FRACTION:.0%} of trades",
        "",
        "  FREQUENCY FLOOR",
        f"      >= {MIN_TRADES_PER_YEAR} trades/year and >= {MIN_TOTAL_TRADES} total",
        "=" * 78,
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ALL_WINDOWS",
    "DATA_START",
    "DISCOVERY",
    "FAST_CLOSE_BARS",
    "HOLDOUT",
    "MAX_HOLD_BARS",
    "MIN_FAST_CLOSE_FRACTION",
    "MIN_TOTAL_TRADES",
    "MIN_TRADES_PER_YEAR",
    "TARGET_MEAN_HOLD_BARS",
    "TARGET_MEDIAN_HOLD_BARS",
    "VALIDATION",
    "Window",
    "describe",
    "qualifies_as_short_swing",
    "qualifies_on_frequency",
]
