"""Market breadth: reported on every run, never used as a gate.

Phase 10 found fold performance tracking breadth across annual means
(r=+0.89). Phases 11 and 12 then showed that relationship is not tradeable:
within-year it is r=+0.010 (p=0.773), the top breadth quartile is 84% a single
year even after extending the data to 2011, and 2014 -- the one independent
high-breadth year the extension added -- produced a *negative* fold return.

So breadth is not a filter here, and this module returns no signal and no
recommendation. What it returns is context: whether today looks like the
markets in which this strategy historically did well or the ones in which it
did not, so that a discretionary decision about aggression has that fact in
front of it rather than behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Trailing window over which distinct admitted names are counted. One
#: quarter, matching the definition Phase 11 tested rather than a new one.
BREADTH_WINDOW = 63


@dataclass(frozen=True, slots=True)
class BreadthReading:
    """Today's participation, and where it sits in this universe's history."""

    value: int
    percentile: float            # against all history available
    label: str
    nearest_year: int | None     # the historical year whose mean breadth is closest
    nearest_year_mean: float | None
    history: pd.Series           # the full daily series, for charting

    @property
    def summary(self) -> str:
        an = f", closest to {self.nearest_year}" if self.nearest_year else ""
        return (f"breadth {self.value} names ({self.percentile:.0f}th pct of history){an} "
                f"-- {self.label}")


def breadth_series(admissions: pd.DataFrame, window: int = BREADTH_WINDOW) -> pd.Series:
    """Distinct symbols admitted in the trailing ``window`` sessions, per day.

    Args:
        admissions: Long frame with ``symbol`` and ``date`` columns, one row
            per admission event (``aes.screener.screen_universe``'s output).
        window: Trailing session count.

    Returns:
        Daily series indexed by date. Empty if there are no admissions.
    """
    if admissions is None or admissions.empty:
        return pd.Series(dtype=float)
    wide = (
        admissions.assign(v=True)
        .pivot_table(index="date", columns="symbol", values="v", aggfunc="any")
        .fillna(False)
        .astype(bool)
        .sort_index()
    )
    # Reindex onto every session present so the rolling window counts sessions,
    # not admission days -- a quiet fortnight must lower breadth, not be skipped.
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="B")
    wide = wide.reindex(full, fill_value=False)
    return wide.rolling(window, min_periods=1).max().sum(axis=1)


def _label(pct: float) -> str:
    if pct >= 80:
        return "very broad -- the regime this strategy has done best in"
    if pct >= 60:
        return "broad"
    if pct >= 40:
        return "average participation"
    if pct >= 20:
        return "narrow"
    return "very narrow -- historically the worst regime for this strategy"


def current_breadth(admissions: pd.DataFrame, window: int = BREADTH_WINDOW) -> BreadthReading | None:
    """Today's reading plus its historical context. ``None`` if unmeasurable."""
    s = breadth_series(admissions, window)
    if s.empty:
        return None
    value = int(s.iloc[-1])
    pct = float((s <= value).mean() * 100)
    yearly = s.groupby(s.index.year).mean()
    nearest, nearest_mean = None, None
    if len(yearly) > 1:
        prior = yearly.iloc[:-1] if yearly.index[-1] == s.index[-1].year else yearly
        if len(prior):
            nearest = int((prior - value).abs().idxmin())
            nearest_mean = float(prior.loc[nearest])
    return BreadthReading(
        value=value, percentile=pct, label=_label(pct),
        nearest_year=nearest, nearest_year_mean=nearest_mean, history=s,
    )


__all__ = ["BREADTH_WINDOW", "BreadthReading", "breadth_series", "current_breadth"]
