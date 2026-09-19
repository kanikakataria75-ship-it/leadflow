"""Assembling one AES signal end-to-end: admission -> box -> breakout -> forward return.

This is the bridge between the per-feature modules (``boxes``, ``resistance``,
``timeframes``, ``relative_strength``) and Phase 3/4's calibration and edge
measurement. It does not decide whether to trade a signal -- that is the
scorer (Phase 3's other half) and the portfolio engine (Phase 5). It only
answers, for one screener admission: *if* a box forms and *if* it breaks out,
what did every measured factor say at that moment, and what happened next.

**Why entry is a genuine event, not every bar above the top.** Section 5's
"breakout momentum" mode fires once, on the first day price closes above the
operative top (the small box's, if one exists, else the big box's). Walking
forward and taking the *first* such day is what preserves the screener/entry
gap this whole project is built around: the box must already exist (found
within ``max_box_wait`` bars of admission) before the walk for a breakout even
starts.

**Why the breakout's volume ratio is recorded rather than filtered on.** The
spec's own volume multiplier is an unknown to calibrate (Phase 3), not a
constant to bake into signal construction. Baking in a guess here would make
every multiplier "sweep" downstream just re-filter the same fixed set of days
instead of asking what a *different* multiplier requirement would have done
to entry timing. So this module reports the realised ratio on the day price
actually crossed, and calibration buckets signals by it after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .boxes import BoxParams, NestedBox, detect_nested_box
from .params import RelativeStrengthParams, ResistanceParams, TimeframeParams
from .relative_strength import drawdown_relative_strength
from .resistance import resistance_headroom
from .timeframes import classify_breakout


@dataclass(frozen=True, slots=True)
class SignalParams:
    """Search bounds for the admission -> box -> breakout walk."""

    #: Bars after admission the box search runs before giving up.
    max_box_wait: int = 30
    #: Bars after box detection the breakout search runs before giving up.
    max_breakout_wait: int = 40
    #: The breakout close must clear the top by at least this margin.
    breakout_margin_pct: float = 0.001
    #: Bars used for the breakout-day volume baseline (excludes the breakout
    #: bar itself, matching this project's ``baseline_excludes_today``
    #: convention elsewhere).
    vol_baseline_bars: int = 20
    #: Forward horizons (bars from entry) at which return is measured.
    horizons: tuple[int, ...] = (5, 10, 15, 20)


@dataclass(frozen=True, slots=True)
class Signal:
    """One fully-assembled AES signal: everything measured, nothing decided."""

    symbol: str
    admission_date: pd.Timestamp
    box_date: pd.Timestamp
    breakout_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    breakout_volume_ratio: float
    used_small_top: bool

    nested: NestedBox
    resistance: dict[str, Any]
    breakout_tf: Any            # BreakoutClassification
    rel_strength: Any           # DrawdownRelativeStrength

    #: bars -> gross forward return (fraction, not %) from entry_price.
    forward_gross: dict[int, float]

    #: Which of the spec's three entry modes fired (section 5): 1 =
    #: breakout momentum, 2 = breakout + retest, 3 = box-bottom (pre-breakout).
    #: Defaults to 1 because every signal built before this field existed --
    #: including every path through ``build_signal`` -- is exactly that mode.
    entry_mode: int = 1

    #: Section 4 conviction score and bucket. Unset (0.0 / "reject") until
    #: ``scoring.attach_score`` runs -- a signal is never scored at the
    #: moment it is found, since scoring is deliberately a separate pass over
    #: already-built signals (``aes_calibration.py``'s own two-stage
    #: pipeline). The "reject" default means an unscored signal is excluded
    #: by ``AESPortfolioBacktester`` rather than silently traded.
    score: float = 0.0
    bucket: str = "reject"

    def as_row(self, cost_pct: float) -> dict[str, Any]:
        """Flat dict for a DataFrame, net-of-cost columns included.

        Args:
            cost_pct: Round-trip cost as a percentage of notional (e.g. 0.585
                for 0.585%), applied uniformly regardless of position size --
                the caller decides what notional that cost was priced at.
        """
        row: dict[str, Any] = {
            "symbol": self.symbol,
            "admission_date": self.admission_date,
            "box_date": self.box_date,
            "breakout_date": self.breakout_date,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "vol_ratio": self.breakout_volume_ratio,
            "used_small_top": self.used_small_top,
            "entry_mode": self.entry_mode,
            "score": self.score,
            "bucket": self.bucket,
            # --- box factors (Phase 1) ---
            "big_bars": self.nested.big.bars,
            "big_range_pct": self.nested.big.range_pct,
            "big_quality": self.nested.big.quality,
            "above_mid": self.nested.big.pct_closes_above_mid,
            "higher_lows": self.nested.big.higher_lows,
            "zone": self.nested.small_zone,
            "duration_vs_norm": self.nested.duration_vs_own_norm,
            "prior_boxes": self.nested.prior_boxes,
            "prior_cycles": self.nested.prior_cycles,
            "vol_dryup_ratio": self.nested.vol_dryup_ratio,
            # --- context factors (Phase 2) ---
            "headroom_pct": self.resistance["headroom_pct"],
            "clears_min_headroom": self.resistance["clears_min_headroom"],
            "resistance_age_strength": self.resistance["resistance_age_strength"],
            "resistance_rejected_count": self.resistance["resistance_rejected_count"],
            "resistance_absorbed_count": self.resistance["resistance_absorbed_count"],
            "tf_score": self.breakout_tf.score,
            "tf_daily": self.breakout_tf.daily,
            "tf_weekly": self.breakout_tf.weekly,
            "tf_monthly": self.breakout_tf.monthly,
            "rs_classification": self.rel_strength.classification,
            "rs_capture": self.rel_strength.downside_capture,
        }
        for bars, gross in self.forward_gross.items():
            row[f"fwd_gross_{bars}"] = gross
            row[f"fwd_net_{bars}"] = gross - cost_pct / 100.0
        return row


def _volume_ratio(volume: np.ndarray, idx: int, baseline_bars: int) -> float:
    start = max(0, idx - baseline_bars)
    baseline = volume[start:idx]
    if baseline.size == 0 or baseline.mean() <= 0:
        return float("nan")
    return float(volume[idx] / baseline.mean())


def find_box_after_admission(
    df: pd.DataFrame,
    admission_idx: int,
    box_params: BoxParams,
    signal_params: SignalParams,
    *,
    symbol: str = "",
    history: list | None = None,
) -> NestedBox | None:
    """Walk forward from admission looking for the first detected box.

    This is the gap the spec insists on: the screener only builds a
    watchlist, and nothing is evaluated for entry on admission day itself.
    """
    last = min(len(df) - 1, admission_idx + signal_params.max_box_wait)
    for i in range(admission_idx + 1, last + 1):
        nested = detect_nested_box(df, i, box_params, symbol=symbol, history=history)
        if nested is not None:
            return nested
    return None


def find_breakout(
    df: pd.DataFrame,
    box: NestedBox,
    signal_params: SignalParams,
) -> tuple[int, float, bool] | None:
    """First bar after the box that closes above the operative top.

    Returns:
        ``(breakout_idx, volume_ratio, used_small_top)``, or ``None`` if no
        qualifying close occurs within ``max_breakout_wait`` bars.
    """
    top = box.small.top if box.small is not None else box.big.top
    used_small = box.small is not None
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    last = min(len(df) - 1, box.as_of_idx + signal_params.max_breakout_wait)
    threshold = top * (1.0 + signal_params.breakout_margin_pct)

    for i in range(box.as_of_idx + 1, last + 1):
        if close[i] > threshold:
            ratio = _volume_ratio(volume, i, signal_params.vol_baseline_bars)
            return i, ratio, used_small
    return None


def build_signal(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    admission_idx: int,
    *,
    symbol: str,
    admission_date: pd.Timestamp,
    box_params: BoxParams | None = None,
    signal_params: SignalParams | None = None,
    resistance_params: ResistanceParams | None = None,
    timeframe_params: TimeframeParams | None = None,
    rs_params: RelativeStrengthParams | None = None,
    box_history: list | None = None,
) -> Signal | None:
    """Assemble one signal from a screener admission, or ``None`` if it never fires.

    Every context feature (resistance, timeframe, relative strength) is
    evaluated *at the breakout bar*, using only data up to and including it --
    the same look-ahead discipline as every other function in ``aes/``.
    Forward returns are read starting the bar after entry, which is itself the
    bar after the breakout (``entry_timing: next_open``, matching the rest of
    this project).
    """
    box_params = box_params or BoxParams()
    signal_params = signal_params or SignalParams()

    box = find_box_after_admission(
        df, admission_idx, box_params, signal_params, symbol=symbol, history=box_history
    )
    if box is None:
        return None
    found = find_breakout(df, box, signal_params)
    if found is None:
        return None
    breakout_idx, vol_ratio, used_small = found

    entry_idx = breakout_idx + 1
    if entry_idx >= len(df):
        return None
    entry_price = float(df["open"].iloc[entry_idx])
    if entry_price <= 0:
        return None

    resistance = resistance_headroom(
        df, breakout_idx, reference_price=entry_price, params=resistance_params
    )
    breakout_tf = classify_breakout(df, breakout_idx, timeframe_params)
    rel_strength = drawdown_relative_strength(df, index_df, breakout_idx, rs_params)

    close = df["close"].to_numpy(dtype=float)
    forward: dict[int, float] = {}
    for bars in signal_params.horizons:
        target = entry_idx + bars
        if target >= len(df):
            continue
        forward[bars] = float(close[target] / entry_price - 1.0)

    return Signal(
        symbol=symbol,
        admission_date=admission_date,
        box_date=box.as_of,
        breakout_date=df.index[breakout_idx],
        entry_date=df.index[entry_idx],
        entry_price=entry_price,
        breakout_volume_ratio=vol_ratio,
        used_small_top=used_small,
        nested=box,
        resistance=resistance,
        breakout_tf=breakout_tf,
        rel_strength=rel_strength,
        forward_gross=forward,
    )


def episode_starts(admission_idxs: list[int], signal_params: SignalParams) -> list[int]:
    """Collapse repeated admissions of the same name into distinct episodes.

    The screener fires on **every** day a name satisfies "20% up in 10 days
    and near its 52-week high" -- which, for a genuine multi-week move, is
    many consecutive sessions in a row. Treating each of those admissions as
    an independent trial is a real bug, not a modelling nuance: each one
    walks forward and finds the *same* box and the *same* breakout, so a
    single underlying event gets counted once for every day the stock stayed
    on the screener -- observed at 94% duplication in the discovery-period
    data, with the duplication count itself correlated with how strong the
    move was (a stronger, longer move satisfies the screener on more
    consecutive days). Left unfixed, that silently overweights exactly the
    outcomes most likely to look good.

    An admission starts a new episode only if it falls after the previous
    episode's full search horizon (box wait + breakout wait) has elapsed --
    i.e. only once the watchlist would genuinely be looking at this name
    fresh, not still mid-search on the last one.

    Args:
        admission_idxs: Bar indices of every admission for one symbol,
            ascending.
        signal_params: Supplies the search-horizon bar count.

    Returns:
        The subset of ``admission_idxs`` that start a new episode.
    """
    if not admission_idxs:
        return []
    horizon = signal_params.max_box_wait + signal_params.max_breakout_wait
    starts = [admission_idxs[0]]
    for idx in admission_idxs[1:]:
        if idx > starts[-1] + horizon:
            starts.append(idx)
    return starts


def build_signals_for_symbol(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    admissions: pd.DataFrame,
    *,
    symbol: str,
    box_params: BoxParams | None = None,
    signal_params: SignalParams | None = None,
    resistance_params: ResistanceParams | None = None,
    timeframe_params: TimeframeParams | None = None,
    rs_params: RelativeStrengthParams | None = None,
    box_history: list | None = None,
) -> list[Signal]:
    """Every genuine episode for one symbol, with repeated admissions collapsed.

    Sharper than ``episode_starts``'s fixed cooldown: the next episode may
    start as soon as the *previous one's actual outcome* clears -- its entry
    bar if a signal fired, or the full search horizon if it never did -- not
    after a flat window regardless of how quickly the previous search
    resolved. A box that breaks out and resolves in three weeks genuinely
    frees the name to form a new one sooner than one that never broke out.

    Args:
        df: The symbol's OHLCV frame.
        index_df: Benchmark frame for relative-strength.
        admissions: This symbol's admission rows (must have a ``date``
            column), any order.
        symbol, box_params, signal_params, resistance_params,
        timeframe_params, rs_params, box_history: Passed through to
            ``build_signal``.

    Returns:
        Signals in chronological order, at most one per episode.
    """
    signal_params = signal_params or SignalParams()
    dates = admissions["date"].sort_values().to_numpy()
    positions = df.index.get_indexer(dates)
    admission_idxs = sorted({int(p) for p in positions if p >= 0})

    signals: list[Signal] = []
    cooldown_until = -1
    for idx in admission_idxs:
        if idx <= cooldown_until:
            continue
        signal = build_signal(
            df, index_df, idx, symbol=symbol, admission_date=df.index[idx],
            box_params=box_params, signal_params=signal_params,
            resistance_params=resistance_params, timeframe_params=timeframe_params,
            rs_params=rs_params, box_history=box_history,
        )
        if signal is not None:
            signals.append(signal)
            entry_idx = df.index.get_indexer([signal.entry_date])[0]
            cooldown_until = entry_idx
        else:
            cooldown_until = idx + signal_params.max_box_wait + signal_params.max_breakout_wait
    return signals


__all__ = [
    "Signal",
    "SignalParams",
    "build_signal",
    "build_signals_for_symbol",
    "episode_starts",
    "find_box_after_admission",
    "find_breakout",
]
