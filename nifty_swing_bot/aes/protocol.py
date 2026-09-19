"""AES's own research protocol -- a separate, explicit split from DAB/MR's.

The prior study's ``research/protocol.py`` windows are not reused here, on
purpose. Its own holdout (2024H2-2026) was "contaminated by repeated
consultation" (the original instructions' words), and AES needed a fresh
slice never touched by any study in this repository. The user's explicit
instruction (2026-09-12), given after being told the default proposal
(Jan 2025-today) sat mostly inside one bear/recovery regime: lock away
**2022-01-01 through today** instead, specifically because that span covers a
full cycle -- 2022 rate-shock bear, 2023 recovery, 2024 bull, 2025 correction,
2026 recovery -- so a strategy that only works in one regime cannot hide that
by getting evaluated in a window that never tests the other regimes.

This is a **two-way** split, not the prior study's three-way
discovery/validation/holdout. There is no separate validation stage: every
calibration choice in Phase 3 (volume multiplier, circuit threshold, scoring
weights) is made by looking *only* at ``AES_DISCOVERY``, and ``AES_HOLDOUT`` is
read exactly once, at the end, by whichever configuration that calibration
settles on. That is a deliberate trade against the prior study's protocol: it
gives up a dedicated selection-only stage in exchange for a bigger, more
regime-diverse final check. It also means AES's discovery window overlaps
prior-study's VALIDATION (2022-2024H1) and HOLDOUT (2024H2-2026) is now
*disjoint* from AES's own -- the AES discovery window (2015-2021) matches the
prior study's DISCOVERY exactly, and everything from 2022 onward is reserved.

**A caveat, stated rather than glossed over.** Phase 1/2 visual validation
(the rendered box-detection charts) used a handful of real examples that fall
inside 2022-2026 -- MINDACORP (2022), TORNTPOWER/APARINDS/SJVN/BSE (2024).
Those checks were purely geometric: does the drawn rectangle match the
consolidation a human would circle on the chart. No forward return, no edge
measurement, and no threshold in this codebase was set by looking at what
happened *after* those dates. That is a different thing from the contamination
the prior study suffered, which was tuning strategy parameters against
holdout-period backtest results repeatedly. But it is not nothing, either --
the detector's geometry rules (the anchor lookbacks, the isolated-wick trim,
the range bands) were shaped in part by how well they drew boxes on those
particular charts, so treat AES_HOLDOUT as index-untouched and
parameter-untouched, not eyes-never-seen-a-chart-from-this-period.
"""

from __future__ import annotations

from datetime import date
from typing import Final

from collections.abc import Iterable

from ..research.protocol import Window

AES_DISCOVERY: Final[Window] = Window(
    "AES_DISCOVERY",
    date(2015, 1, 1),
    date(2021, 12, 31),
    "All AES calibration -- volume multiplier, circuit threshold, scoring "
    "weights -- is chosen by looking only at this window. Unlimited looks.",
)

AES_HOLDOUT: Final[Window] = Window(
    "AES_HOLDOUT",
    # Moved from 2022-01-01 after Phase 6 found the boundary was already
    # porous: screener admissions are discovery-filtered, but the watchlist
    # walk can place an *entry* up to ~70 bars later, so 60 of 787 discovery
    # signals had entry dates between 2022-01-04 and 2022-07-04 and ten of
    # them became positions in the Phase 5 backtest. Starting the holdout
    # after that spill makes the window genuinely untouched rather than
    # nominally so.
    date(2022, 7, 5),
    date(2026, 12, 31),
    "Spans a full bear/recovery/bull/correction/recovery cycle by "
    "construction. Read exactly once, at the end, by the configuration "
    "AES_DISCOVERY calibration settles on. Nothing may change after.",
)

#: Same warm-up rationale as the prior study: 200+ day lookbacks (52-week
#: high, resistance search) need data before AES_DISCOVERY's own start to not
#: be truncated on day one of discovery itself.
DATA_START: Final[date] = date(2014, 1, 1)


def describe() -> str:
    lines = ["", "=" * 78, "  AES RESEARCH PROTOCOL", "=" * 78]
    for window in (AES_DISCOVERY, AES_HOLDOUT):
        lines.append(f"  {window}")
        lines.append(f"      {window.purpose}")
    lines += ["=" * 78, ""]
    return "\n".join(lines)


__all__ = [
    "AES_DISCOVERY",
    "AES_HOLDOUT",
    "DATA_START",
    "FORWARD_CHECK_FROM",
    "HOLDOUT_SPENT_ON",
    "VALIDATION_SALT",
    "VALIDATION_SYMBOL_FRACTION",
    "describe",
    "discovery_symbols",
    "validation_symbols",
]

# --------------------------------------------------------------------------- #
# Phase 16: the holdout is spent, and what replaces it
# --------------------------------------------------------------------------- #
#: **AES_HOLDOUT is no longer clean.** Phase 16 runs an outcome-first discovery
#: study (find every +30%-in-25-sessions move in 2022-2026, match controls,
#: measure what preceded them) directly on the holdout window. That is a
#: deliberate, user-authorised trade, made on the reasoning that 2015-2021 is
#: exhausted and the market actually being traded is this one.
#:
#: Nothing may now cite AES_HOLDOUT as an untouched out-of-sample check. It has
#: been read, and read exploratorily, over its whole span.
HOLDOUT_SPENT_ON = date(2026, 9, 16)

#: What replaces it is a **cross-sectional** slice rather than a temporal one,
#: because the temporal dimension is now used up in both directions: 2010-2021
#: is discovery, 2022-2026 is discovery as of Phase 16.
#:
#: ``AES_VALIDATION_SYMBOLS`` holds out 25% of the universe *by symbol*,
#: partitioned deterministically by a salted hash of the ticker so the split
#: cannot be nudged after seeing a result. Phase 16 discovery reads only the
#: other 75%. Any feature or rule that comes out of Phase 16 is then tested,
#: once, on the held-out names over the same 2022-2026 period.
#:
#: **The limitation, stated rather than glossed:** these names share the
#: calendar with the discovery names, so market-wide regime is common to both.
#: A cross-sectional holdout controls for "did I fit this stock" but not for
#: "did I fit this market". It is genuinely weaker than a fresh time window.
#: The only clean temporal check left is **forward** time -- bars dated after
#: ``FORWARD_CHECK_FROM``, which do not exist yet and cannot be peeked at.
VALIDATION_SYMBOL_FRACTION: Final[float] = 0.25
VALIDATION_SALT: Final[str] = "aes-phase16-symbol-split-v1"

#: The genuinely untouched check: live-forward bars. Anything Phase 16 produces
#: that is worth trading should be paper-run from here rather than backtested
#: into a window that has now been looked at.
FORWARD_CHECK_FROM: Final[date] = date(2026, 9, 17)


def validation_symbols(symbols: Iterable[str]) -> set[str]:
    """The held-out 25% of the universe, by deterministic salted hash.

    Deterministic so it reproduces exactly; salted and hashed so it is not a
    property anyone chose (no alphabetical block, no sector block, no
    liquidity block). Adding names to the universe later reassigns only the
    new names, never the existing ones.
    """
    import hashlib

    out = set()
    cut = int(VALIDATION_SYMBOL_FRACTION * 2**32)
    for sym in symbols:
        h = hashlib.sha256(f"{VALIDATION_SALT}:{sym}".encode()).digest()
        if int.from_bytes(h[:4], "big") < cut:
            out.add(sym)
    return out


def discovery_symbols(symbols: Iterable[str]) -> set[str]:
    """The 75% Phase 16 is allowed to look at."""
    syms = set(symbols)
    return syms - validation_symbols(syms)
