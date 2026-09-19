"""INR gold series for the gold sleeve, with a documented data repair.

Gold is held as a volatile diversifier, so the sleeve is marked to a real price
series rather than an assumed yield. The instrument is **GOLDBEES**, the NSE
gold ETF: domestic INR gold including import duty, which is what an Indian book
actually holds. Validated against ``GC=F x USDINR`` -- annual returns correlate
**0.968** over 2014-2026.

**One repair is applied, and it is not optional.** Yahoo's GOLDBEES history
contains a two-session factor-of-100 misprint on 2019-12-19/20 (33.60 -> 0.336
-> 33.65). It is a bad print, not a split: the price returns to its prior level
immediately afterwards and the USD reference shows no such move. Left in, it
drives the gold sleeve to ~1% of its value and back, producing a -99% day and a
+9900% day, and every book statistic downstream is destroyed by it -- annualised
gold volatility reads 3,845%.

The repair rule is deliberately narrow: a move larger than ``jump_threshold``
that **reverses within ``max_gap`` sessions** is treated as a misprint and the
affected run is rescaled by the round factor implied. A genuine split does not
reverse, so this cannot silently undo one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class GoldRepair:
    start: pd.Timestamp
    end: pd.Timestamp
    factor: float
    n_bars: int

    def describe(self) -> str:
        return (f"{self.start.date()}..{self.end.date()}: "
                f"{self.n_bars} bar(s) rescaled by {self.factor:g}")


def repair_price_glitches(
    close: pd.Series,
    jump_threshold: float = 0.50,
    max_gap: int = 5,
) -> tuple[pd.Series, list[GoldRepair]]:
    """Repair isolated factor-of-N misprints; leave genuine splits alone.

    Returns the corrected series and a list of what was changed, so the repair
    is auditable rather than invisible.
    """
    s = close.astype(float).copy()
    r = s.pct_change()
    repairs: list[GoldRepair] = []
    i = 1
    while i < len(s):
        if not np.isfinite(r.iloc[i]) or abs(r.iloc[i]) < jump_threshold:
            i += 1
            continue
        # look ahead for a reversal that restores the prior level
        base = s.iloc[i - 1]
        j = i + 1
        while j < len(s) and j - i <= max_gap:
            if base > 0 and abs(s.iloc[j] / base - 1.0) < 0.15:
                ratio = base / s.iloc[i] if s.iloc[i] != 0 else np.nan
                if np.isfinite(ratio) and ratio > 0:
                    # snap to the nearest round decade (100, 10, 1/10, ...)
                    factor = 10.0 ** round(np.log10(ratio))
                    if abs(factor - 1.0) > 1e-9:
                        s.iloc[i:j] = s.iloc[i:j] * factor
                        repairs.append(GoldRepair(
                            start=s.index[i], end=s.index[j - 1],
                            factor=factor, n_bars=j - i))
                        r = s.pct_change()
                i = j
                break
            j += 1
        else:
            i += 1
            continue
        if j >= len(s) or j - i > max_gap:
            i += 1
    return s, repairs


def load_gold(
    path: str,
    reference: pd.Series | None = None,
    column: str = "close",
) -> tuple[pd.Series, dict]:
    """Load, repair and (optionally) cross-check the gold series.

    ``reference`` is a second series of the same economic quantity (GC=F x
    USDINR). When supplied, the correlation of annual returns is returned as a
    sanity figure; it is a check, not an input.
    """
    df = pd.read_parquet(path)
    raw = df[column].astype(float)
    fixed, repairs = repair_price_glitches(raw)
    info = {
        "bars": len(fixed),
        "start": str(fixed.index.min().date()),
        "end": str(fixed.index.max().date()),
        "repairs": [x.describe() for x in repairs],
        "max_abs_daily_move_pct": float(fixed.pct_change().abs().max() * 100),
    }
    if reference is not None:
        ref = reference.reindex(fixed.index).ffill()
        a = fixed.resample("YE").last().pct_change().dropna()
        b = ref.resample("YE").last().pct_change().dropna()
        j = pd.concat([a, b], axis=1).dropna()
        if len(j) > 3:
            info["annual_corr_vs_reference"] = round(float(j.corr().iloc[0, 1]), 3)
    return fixed, info


__all__ = ["GoldRepair", "load_gold", "repair_price_glitches"]
