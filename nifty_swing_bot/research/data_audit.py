"""Data validation and look-ahead auditing.

Two jobs, both of which must pass before any hypothesis is taken seriously.

**Data validation.** Checks the raw price panel for the failure modes that
quietly corrupt a backtest: OHLC inconsistencies, duplicate or missing dates,
stale prints, zero-volume bars, and single-day moves large enough to suggest an
unadjusted corporate action. Nothing is repaired silently -- problems are
reported and the affected symbols excluded.

**Look-ahead auditing.** The decisive structural test: compute a feature on the
full series, then recompute it on the series truncated at bar *t*, and compare
the value at bar *t*. If a feature uses any information from after *t*, the two
disagree. This catches centred rolling windows, forgotten shifts, full-period
normalisation and future-return leakage in a single generic check, without
needing to reason about each formula by hand.

Survivorship bias cannot be *fixed* here -- NSE does not publish historical
constituent lists through the archive this project uses -- so it is measured and
reported instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: A single-bar move beyond this is more likely an unadjusted split/bonus than a
#: real price change, and warrants inspection.
EXTREME_MOVE_THRESHOLD: float = 0.35

#: Consecutive identical closes beyond this suggests a suspended or stale scrip.
STALE_RUN_THRESHOLD: int = 5


@dataclass(slots=True)
class SymbolAudit:
    """Per-symbol data quality findings."""

    symbol: str
    bars: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    duplicate_dates: int = 0
    ohlc_violations: int = 0
    non_positive: int = 0
    zero_volume: int = 0
    extreme_moves: int = 0
    longest_stale_run: int = 0
    calendar_gaps: int = 0

    @property
    def clean(self) -> bool:
        """True when nothing structural is wrong with this symbol's series."""
        return (
            self.duplicate_dates == 0
            and self.ohlc_violations == 0
            and self.non_positive == 0
            and self.longest_stale_run <= STALE_RUN_THRESHOLD
        )


@dataclass(slots=True)
class PanelAudit:
    """Whole-panel audit result."""

    symbols: list[SymbolAudit] = field(default_factory=list)
    calendar_days: int = 0
    universe_size: int = 0

    @property
    def frame(self) -> pd.DataFrame:
        if not self.symbols:
            return pd.DataFrame()
        # asdict, not vars: slots dataclasses have no __dict__.
        return pd.DataFrame([asdict(s) for s in self.symbols])

    def unclean(self) -> list[str]:
        """Symbols that failed a structural check."""
        return [s.symbol for s in self.symbols if not s.clean]


def audit_symbol(
    symbol: str, frame: pd.DataFrame, calendar: pd.DatetimeIndex | None = None
) -> SymbolAudit:
    """Validate one symbol's OHLCV series.

    Args:
        symbol: Symbol name.
        frame: OHLCV frame with a DatetimeIndex.
        calendar: Reference trading calendar (typically the benchmark's index)
            used to detect missing sessions.

    Returns:
        The findings for this symbol.
    """
    if frame is None or frame.empty:
        return SymbolAudit(symbol, 0, None, None)

    audit = SymbolAudit(
        symbol=symbol,
        bars=len(frame),
        first=frame.index.min(),
        last=frame.index.max(),
        duplicate_dates=int(frame.index.duplicated().sum()),
    )

    o, h, low, c = frame["open"], frame["high"], frame["low"], frame["close"]

    # High must bound open/close from above, low from below.
    audit.ohlc_violations = int(
        ((h < low) | (h < o) | (h < c) | (low > o) | (low > c)).sum()
    )
    audit.non_positive = int(((o <= 0) | (h <= 0) | (low <= 0) | (c <= 0)).sum())
    audit.zero_volume = int((frame["volume"].fillna(0) <= 0).sum())

    returns = c.pct_change()
    audit.extreme_moves = int((returns.abs() > EXTREME_MOVE_THRESHOLD).sum())

    # Longest run of identical closes.
    unchanged = (c.diff() == 0).astype(int)
    longest, current = 0, 0
    for flag in unchanged.to_numpy():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    audit.longest_stale_run = int(longest)

    if calendar is not None and len(frame) > 1:
        expected = calendar[(calendar >= audit.first) & (calendar <= audit.last)]
        audit.calendar_gaps = int(len(expected.difference(frame.index)))

    return audit


def audit_panel(
    prices: Mapping[str, pd.DataFrame], calendar: pd.DatetimeIndex | None = None
) -> PanelAudit:
    """Validate every symbol in a price panel."""
    result = PanelAudit(universe_size=len(prices))
    if calendar is not None:
        result.calendar_days = len(calendar)
    for symbol, frame in prices.items():
        result.symbols.append(audit_symbol(symbol, frame, calendar))
    return result


def survivorship_report(
    prices: Mapping[str, pd.DataFrame], checkpoints: list[pd.Timestamp]
) -> pd.DataFrame:
    """Quantify how much of today's universe simply did not exist in the past.

    This does not measure the whole of survivorship bias -- names that were
    *removed* from the indices, or delisted entirely, are absent from today's
    constituent list and therefore cannot be counted from it at all. What it
    does measure is the other half: how much of the backtested universe is made
    up of companies that were not listed (or not trading) at each past date.

    A high figure at an early date means the backtest over that period is run on
    a small, and by construction *surviving*, subset -- so early-period results
    deserve less weight.

    Args:
        prices: Symbol -> OHLCV.
        checkpoints: Dates at which to measure coverage.

    Returns:
        One row per checkpoint with the count and fraction of the universe that
        had begun trading by then.
    """
    starts = {
        symbol: frame.index.min()
        for symbol, frame in prices.items()
        if frame is not None and not frame.empty
    }
    total = len(starts)
    rows = []
    for when in checkpoints:
        live = sum(1 for first in starts.values() if first <= when)
        rows.append(
            {
                "date": pd.Timestamp(when).strftime("%Y-%m-%d"),
                "listed": live,
                "universe": total,
                "coverage_pct": round(live / total * 100, 1) if total else 0.0,
                "missing": total - live,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Look-ahead auditing
# --------------------------------------------------------------------------- #
def find_lookahead(
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    probes: int = 12,
    tolerance: float = 1e-8,
    min_bar: int = 220,
) -> list[dict[str, object]]:
    """Detect future-information leakage by truncation.

    For a set of probe bars, the features are recomputed on the series truncated
    immediately after that bar and compared with the values computed on the full
    series. A correct feature depends only on bars up to *t*, so the two must be
    identical. Any disagreement is leakage.

    Args:
        feature_fn: Maps an OHLCV frame to a feature frame.
        frame: The OHLCV series to test on.
        columns: Feature columns to check. Defaults to every numeric/boolean column.
        probes: How many bars to probe, spread across the usable range.
        tolerance: Absolute tolerance for float comparison.
        min_bar: First bar to probe, leaving room for the longest lookback to seed.

    Returns:
        One record per leaking (column, bar). Empty means the audit passed.
    """
    full = feature_fn(frame)
    if full is None or full.empty:
        return []

    if columns is None:
        columns = [
            c for c in full.columns
            if pd.api.types.is_numeric_dtype(full[c]) or pd.api.types.is_bool_dtype(full[c])
        ]

    usable = len(frame) - 2
    if usable <= min_bar:
        min_bar = max(50, usable // 2)
    if usable <= min_bar:
        return []

    positions = np.linspace(min_bar, usable, num=probes, dtype=int)
    findings: list[dict[str, object]] = []

    for pos in sorted(set(positions.tolist())):
        truncated = feature_fn(frame.iloc[: pos + 1])
        if truncated is None or truncated.empty:
            continue
        stamp = frame.index[pos]
        if stamp not in truncated.index or stamp not in full.index:
            continue

        for column in columns:
            if column not in truncated.columns:
                continue
            a, b = full.loc[stamp, column], truncated.loc[stamp, column]
            a_na, b_na = pd.isna(a), pd.isna(b)
            if a_na and b_na:
                continue
            if a_na != b_na:
                findings.append(
                    {"column": column, "date": stamp, "full": a, "truncated": b,
                     "reason": "NaN mismatch"}
                )
                continue
            try:
                if abs(float(a) - float(b)) > tolerance:
                    findings.append(
                        {"column": column, "date": stamp, "full": float(a),
                         "truncated": float(b), "reason": "value mismatch"}
                    )
            except (TypeError, ValueError):
                if a != b:
                    findings.append(
                        {"column": column, "date": stamp, "full": a, "truncated": b,
                         "reason": "value mismatch"}
                    )
    return findings


def format_audit(panel: PanelAudit, survivorship: pd.DataFrame) -> str:
    """Render the data audit as a report."""
    frame = panel.frame
    if frame.empty:
        return "No symbols audited."

    lines = ["", "=" * 84, "  DATA AUDIT", "=" * 84]
    lines.append(f"  Universe                    {panel.universe_size} symbols")
    lines.append(f"  Reference calendar          {panel.calendar_days} sessions")
    lines.append(f"  Median bars per symbol      {int(frame['bars'].median()):,}")
    lines.append(f"  Shortest / longest series   {int(frame['bars'].min()):,} / {int(frame['bars'].max()):,}")
    lines.append(f"  Earliest bar                {frame['first'].min()}")
    lines.append(f"  Latest bar                  {frame['last'].max()}")
    lines.append("")
    lines.append("  STRUCTURAL PROBLEMS (symbols affected / total occurrences)")
    for column, label in (
        ("duplicate_dates", "Duplicate dates"),
        ("ohlc_violations", "OHLC inconsistencies"),
        ("non_positive", "Non-positive prices"),
        ("zero_volume", "Zero-volume bars"),
        ("extreme_moves", f"Single-bar moves > {EXTREME_MOVE_THRESHOLD:.0%}"),
        ("calendar_gaps", "Missing sessions vs calendar"),
    ):
        affected = int((frame[column] > 0).sum())
        total = int(frame[column].sum())
        lines.append(f"    {label:<34}{affected:>5} symbols{total:>10,} occurrences")

    stale = int((frame["longest_stale_run"] > STALE_RUN_THRESHOLD).sum())
    lines.append(f"    {'Stale runs > ' + str(STALE_RUN_THRESHOLD) + ' bars':<34}{stale:>5} symbols")

    unclean = panel.unclean()
    lines.append("")
    lines.append(f"  Symbols failing a structural check: {len(unclean)}")
    if unclean:
        lines.append(f"    {', '.join(unclean[:12])}{' ...' if len(unclean) > 12 else ''}")

    lines += ["", "  SURVIVORSHIP EXPOSURE (how much of today's universe was trading then)"]
    lines.append(f"    {'date':<14}{'listed':>9}{'of':>6}{'coverage':>11}{'missing':>10}")
    for _, row in survivorship.iterrows():
        lines.append(
            f"    {row['date']:<14}{int(row['listed']):>9}{int(row['universe']):>6}"
            f"{row['coverage_pct']:>10.1f}%{int(row['missing']):>10}"
        )
    lines += [
        "",
        "    This counts only companies not yet listed. Names DELISTED or dropped",
        "    from the indices are absent from today's constituent list entirely and",
        "    cannot be counted from it -- that part of survivorship bias is real,",
        "    unmeasured, and biases early-period results optimistically.",
        "=" * 84,
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "EXTREME_MOVE_THRESHOLD",
    "STALE_RUN_THRESHOLD",
    "PanelAudit",
    "SymbolAudit",
    "audit_panel",
    "audit_symbol",
    "find_lookahead",
    "format_audit",
    "survivorship_report",
]
