"""Phase 23: re-entry after a stop-out.

The case this addresses is real and the engine structurally cannot catch it: a
position closes below its stop and is exited, the name closes back above that
same level a few sessions later on expanding volume, and *then* makes its move.
Today the name is gone the moment the stop fires and never comes back, so the
second leg is unreachable no matter how good it is.

What is under test
------------------
After a **stop** exit -- ``atr_trail_stop`` or ``big_box_bottom_stop``, not a
ladder, time stop or hold cap -- the name enters a re-entry watch for ``N``
bars. It re-enters if it closes back above the level that stopped it, with
volume above ``vol_mult`` x its 20-day average. Re-entries per name per cycle
are capped so a chopping name cannot churn the account.

Why this is not simply the sixth signal-count attempt
-----------------------------------------------------
REJECTED 8/12/18/23 all failed the same way: raise the population, lose
ex-best-fold. This shares the risk -- more trades is more trades -- but the
mechanism differs in a way worth measuring separately. Those four all widened
**admission**, letting in names the system had not selected. Re-entry adds no
new names at all. It re-enters a name the pipeline already admitted, already
found a structure on, already scored and already chose to hold. The population
it draws from is the one that passed every existing gate.

That is a reason to test it, not a reason to expect it to work. The bar is
pre-registered in ``research/reentry_study.py`` and the verdict is computed.

No-lookahead convention
-----------------------
The re-entry condition is evaluated on the **close** of bar t and fills at the
**open** of bar t+1, matching the entry convention the rest of the engine uses
(``_assemble_signal`` sets ``entry_idx = entry_trigger_idx + 1``). Evaluating
and filling on the same bar would post an edge that cannot be traded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: Exit reasons that arm a re-entry watch. A ladder exit, a time stop or the
#: hold cap are not stop-outs and deliberately do not qualify: the thesis here
#: is specifically "the stop was wrong", not "the position ended".
STOP_REASONS: frozenset[str] = frozenset({"atr_trail_stop", "big_box_bottom_stop"})


@dataclass(frozen=True, slots=True)
class ReentryParams:
    """Re-entry configuration. Default-constructed, it is **disabled**.

    Disabled-by-default matters: threading this through the backtester must not
    be able to change a frozen result by accident, so every call site that does
    not explicitly ask for re-entry gets bit-identical behaviour.
    """

    enabled: bool = False
    #: Bars the name stays in the re-entry watch after the stop fires.
    watch_bars: int = 10
    #: Volume must exceed this multiple of its 20-day average. ``None`` = no
    #: volume condition at all.
    vol_mult: float | None = 1.5
    #: Rolling window for the volume average.
    vol_period: int = 20
    #: Maximum re-entries per name per cycle, where a cycle is the original
    #: signal. Prevents a chopping name from re-entering indefinitely.
    max_reentries: int = 1

    # --- eligibility gate (Phase 23b) ------------------------------------- #
    #: Only arm a watch when the stopped position carried this bucket at
    #: entry. ``None`` = any bucket, which is the Phase 23 behaviour.
    #:
    #: The narrow population exists because Phase 24b found the two groups are
    #: almost disjoint: among 23 high-conviction SMA10 stop-outs, those hit
    #: within 3 bars ran +20% or more in 7 of 11 cases (median max-in-40
    #: 27.87%), while slower grind-outs did so in 3 of 12 (median 8.36%), and
    #: zero of the fast group matched the compression pattern. Re-entry was
    #: previously tested against *every* stop-out, which pools those two.
    require_bucket: str | None = None
    #: Only arm a watch when the position was stopped within this many bars of
    #: entry. ``None`` = no limit.
    max_bars_held: int | None = None

    def eligible(self, bucket: str, bars_held: int) -> bool:
        """Whether a just-stopped position may arm a re-entry watch."""
        if self.require_bucket is not None and bucket != self.require_bucket:
            return False
        if self.max_bars_held is not None and bars_held > self.max_bars_held:
            return False
        return True

    def label(self) -> str:
        if not self.enabled:
            return "baseline"
        v = "none" if self.vol_mult is None else f"{self.vol_mult:g}x"
        gate = ""
        if self.require_bucket is not None or self.max_bars_held is not None:
            b = "hc" if self.require_bucket == "high_conviction" else (self.require_bucket or "any")
            gate = f"_{b}<={self.max_bars_held}b"
        return f"N{self.watch_bars}_vol{v}_cap{self.max_reentries}{gate}"


@dataclass(slots=True)
class ReentryWatch:
    """One name waiting to come back, with everything needed to re-open it."""

    symbol: str
    #: The level whose breach caused the exit -- the ATR trail level or the
    #: big-box floor, whichever actually fired. Re-entry requires a close back
    #: above *this*, not above the entry price.
    stop_level: float
    #: Calendar-bar index after which the watch expires.
    expires_idx: int
    #: Carried from the original signal so a re-entry sizes and scores exactly
    #: as the original did. Re-entry must not quietly become a different,
    #: better-informed trade.
    big_box_bottom: float
    entry_mode: int
    score: float
    bucket: str
    #: Re-entries already used in this cycle.
    used: int = 0


@dataclass(slots=True)
class EntryOrder:
    """What the portfolio loop needs to open a position, from either source.

    Normal signals and re-entries flow through one code path so the sizing,
    slot, regime, appetite and cash rules cannot drift apart between them.
    """

    symbol: str
    score: float
    bucket: str
    big_box_bottom: float
    entry_mode: int
    is_reentry: bool = False
    stop_level: float | None = None


def volume_ok(
    vol_series: pd.Series | None, day: pd.Timestamp, params: ReentryParams
) -> bool:
    """Whether today's volume clears the threshold.

    ``vol_mult=None`` means no volume condition, which is one of the swept
    cells rather than a disabled feature -- the point of sweeping it is to find
    out whether the volume confirmation is doing any work at all.
    """
    if params.vol_mult is None:
        return True
    if vol_series is None or day not in vol_series.index:
        return False
    v = vol_series.loc[day]
    return bool(pd.notna(v) and v >= params.vol_mult)


__all__ = [
    "STOP_REASONS", "EntryOrder", "ReentryParams", "ReentryWatch", "volume_ok",
]
