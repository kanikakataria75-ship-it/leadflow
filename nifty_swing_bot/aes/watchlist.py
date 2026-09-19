"""The persistent watchlist state machine (AES spec section 1.3) and all
three entry modes (section 5).

``signals.build_signal`` (Phase 3/4's first pass) searched for one box and
one breakout starting from a single admission bar, within fixed wait windows.
That is not what section 1.3 describes: "Watchlist grows over time -- every
name is re-checked each cycle for a forming setup," and a name "stays until:
hard-delete triggered, OR box invalidated..., OR stale." A name that forms a
box, fails to break out, and later forms a *different* box without a fresh
admission event was invisible to the first pass. It should not be -- the
watchlist stays open on it the whole time.

This module replaces that per-admission search with an actual walk: for every
bar a symbol is on the watchlist, it looks for a box; once a box exists, it
checks all three entry modes, in the spec's own stated preference; a box that
breaks down or goes stale is dropped and scanning resumes; and being on the
list at all persists across admissions rather than being reset by each one.

**Entry modes (section 5), in the order the spec lists them:**

1. **Breakout momentum.** A fresh close above the operative top (the small
   box's, if nested; else the big box's) on volume the spec calls "clearly
   above-average". This module routes on the spec's own starting guess
   (2.0x) to decide *which mechanic applies* -- Phase 3 explicitly found no
   reliable multiplier to gate entry on at all, and reusing that same number
   as a hard gate here would contradict that finding. Choosing which of two
   entry mechanics to simulate is a materially weaker claim than deciding
   whether to trade.
2. **Breakout + retest.** Used when the initial breakout's volume is weak.
   Price must come back to the broken level and hold it (a close not more
   than ``retest_fail_pct`` below it) with a confirming bullish candle,
   within ``retest_window`` bars, or the attempt is abandoned.
3. **Box-bottom entry (pre-breakout).** Fires while the box is still intact,
   whenever price sits within ``box_bottom_zone_pct`` of the big-box bottom
   and closes above its own SMA(``sma_period``) -- "respecting SMA(10)".

At most one entry fires per box episode. Firing any of the three, or the box
invalidating, or the box going stale, all end that episode and return the
name to bare watching (no box) rather than removing it from the watchlist
outright -- only running past ``max_watch_without_box`` bars **since the last
admission** with no box currently active drops a name until a fresh
admission re-adds it.

That last clause matters more than it looks. The eligibility clock is tied
to admissions specifically, not to when an episode last ended -- an earlier
version reset it on every episode end, which meant any stock with ordinary
periodic volatility kept some box or other recurring often enough that the
clock never accumulated past the limit, and a name admitted once, years
earlier, stayed "on watch" for its entire subsequent history. Measured
directly: the admission-to-breakout gap came out with a **819-day median**.
Tying the clock to admissions instead fixes both that and a second problem
it caused -- a found signal being attributed to whichever admission first
opened the stay, however long ago, rather than the one that actually made it
relevant.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .boxes import BoxParams, NestedBox, detect_nested_box
from .gamma import GammaParams, detect_gamma_structure
from .params import RelativeStrengthParams, ResistanceParams, TimeframeParams, WatchlistParams
from .relative_strength import drawdown_relative_strength
from .resistance import resistance_headroom
from .signals import Signal, _volume_ratio
from .timeframes import classify_breakout


@dataclass
class _EpisodeState:
    """Mutable tracking for one box's journey through the state machine."""

    box: NestedBox
    first_seen_idx: int
    awaiting_retest: bool = False
    retest_level: float = 0.0
    retest_deadline_idx: int = 0
    used_small_top: bool = False
    #: Set once the box has passed ``box_stale_bars`` but is being held as a
    #: live entry candidate for ``box_carry_bars`` more bars. Only the
    #: breakout modes fire while this is set.
    carried: bool = False
    carry_deadline_idx: int = 0


def _operative_top(box: NestedBox) -> tuple[float, bool]:
    if box.small is not None:
        return box.small.top, True
    return box.big.top, False


def _assemble_signal(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    *,
    symbol: str,
    admission_date: pd.Timestamp,
    box: NestedBox,
    entry_trigger_idx: int,
    entry_mode: int,
    vol_ratio: float,
    used_small_top: bool,
    horizons: tuple[int, ...],
    resistance_params: ResistanceParams | None,
    timeframe_params: TimeframeParams | None,
    rs_params: RelativeStrengthParams | None,
) -> Signal | None:
    """Shared tail end for every entry mode: price the entry, measure context, look forward."""
    entry_idx = entry_trigger_idx + 1
    if entry_idx >= len(df):
        return None
    entry_price = float(df["open"].iloc[entry_idx])
    if entry_price <= 0:
        return None

    resistance = resistance_headroom(
        df, entry_trigger_idx, reference_price=entry_price, params=resistance_params
    )
    breakout_tf = classify_breakout(df, entry_trigger_idx, timeframe_params)
    rel_strength = drawdown_relative_strength(df, index_df, entry_trigger_idx, rs_params)

    close = df["close"].to_numpy(dtype=float)
    forward: dict[int, float] = {}
    for bars in horizons:
        target = entry_idx + bars
        if target < len(df):
            forward[bars] = float(close[target] / entry_price - 1.0)

    return Signal(
        symbol=symbol,
        admission_date=admission_date,
        box_date=box.as_of,
        breakout_date=df.index[entry_trigger_idx],
        entry_date=df.index[entry_idx],
        entry_price=entry_price,
        breakout_volume_ratio=vol_ratio,
        used_small_top=used_small_top,
        nested=box,
        resistance=resistance,
        breakout_tf=breakout_tf,
        rel_strength=rel_strength,
        forward_gross=forward,
        entry_mode=entry_mode,
    )


def run_watchlist_for_symbol(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    admission_dates: pd.Series | list[pd.Timestamp],
    *,
    symbol: str,
    box_params: BoxParams | None = None,
    watch_params: WatchlistParams | None = None,
    resistance_params: ResistanceParams | None = None,
    timeframe_params: TimeframeParams | None = None,
    rs_params: RelativeStrengthParams | None = None,
    box_history: list | None = None,
    gamma_params: "GammaParams | None" = None,
    trace: list | None = None,
) -> list[Signal]:
    """Walk one symbol's entire history as a persistent watchlist.

    Unlike ``signals.build_signals_for_symbol``, admissions here only ever
    *open* the watchlist (or extend how long it stays open with no box) --
    they do not each start an independent search. A single admission can
    therefore produce more than one signal, if a box resolves with no entry
    and a fresh one later forms while the name is still being watched, and
    two admissions close together produce no more signals than one would,
    exactly as intended.

    Args:
        df: The symbol's OHLCV frame.
        index_df: Benchmark frame for relative-strength.
        admission_dates: Every date this symbol was admitted by the screener.
        symbol, box_params, resistance_params, timeframe_params, rs_params,
            box_history: As in ``signals.build_signal``.
        watch_params: Watchlist and entry-mode thresholds.
        gamma_params: Phase 22 alternative structures. ``None`` or a
            ``GammaParams`` with no arms enabled leaves the frozen pipeline
            bit-identical; an enabled arm may arm an episode **only** on bars
            where the box detector returned nothing.

    Returns:
        Signals in chronological order.
    """
    box_params = box_params or BoxParams()
    watch_params = watch_params or WatchlistParams()
    dates = pd.Series(admission_dates).sort_values().to_numpy()
    positions = df.index.get_indexer(dates)
    admission_idxs = sorted({int(p) for p in positions if p >= 0})
    if not admission_idxs:
        return []

    arr_close = df["close"].to_numpy(dtype=float)
    arr_open = df["open"].to_numpy(dtype=float)
    arr_low = df["low"].to_numpy(dtype=float)
    arr_volume = df["volume"].to_numpy(dtype=float)
    sma = df["close"].rolling(watch_params.sma_period).mean().to_numpy(dtype=float)

    n = len(df)

    def _ev(kind: str, idx: int, **kw) -> None:
        """Record a state-machine event. Off unless a ``trace`` list is passed."""
        if trace is not None:
            trace.append({"kind": kind, "idx": idx, "date": df.index[idx], **kw})

    signals: list[Signal] = []
    admission_ptr = 0
    on_watchlist = False
    # The most recent admission at or before the current bar, updated on
    # EVERY admission regardless of watch state -- not just the one that
    # first opened the current stay. Both what a found signal is attributed
    # to, and the clock ``max_watch_without_box`` measures against, need
    # this to track the *latest* qualifying event, or a name admitted once
    # years ago and never dropped would misattribute everything it finds to
    # that first, increasingly stale admission.
    last_admission_idx = admission_idxs[0]
    episode: _EpisodeState | None = None
    # The start bar of the last box an episode ended on (by any route: entry
    # fired, invalidated, or stale). A freshly detected box whose own start
    # is no later than this is the *same* structure seen again, not a new
    # one -- without this guard, resolving an episode immediately re-detects
    # the box it just came from (the oscillation that produced it hasn't
    # gone anywhere) and can re-fire on the same few bars of ordinary noise.
    # A genuinely new episode must anchor on a box that starts later.
    last_box_start_idx = -1

    i = admission_idxs[0]
    while i < n:
        # Admissions extend/(re)open the watch; they never interrupt an
        # in-progress episode. Deliberately unconditional (not gated on
        # ``not on_watchlist``): a name that keeps re-qualifying stays fresh
        # even while a box is being tracked, and -- the bug this replaced --
        # a name that qualifies once and then coasts on recurring, quickly
        # resolved boxes must still eventually go stale rather than staying
        # "on watch" for its entire multi-year history. Measured directly:
        # before this fix, the admission-to-breakout gap had a **819-day**
        # median, because any stock with periodic ordinary volatility keeps
        # some box or other recurring often enough that a clock reset by
        # episode activity (rather than by admissions) never accumulates.
        while admission_ptr < len(admission_idxs) and admission_idxs[admission_ptr] <= i:
            last_admission_idx = admission_idxs[admission_ptr]
            on_watchlist = True
            admission_ptr += 1

        if not on_watchlist:
            i += 1
            continue

        if episode is None:
            nested = detect_nested_box(df, i, box_params, symbol=symbol, history=box_history)
            # GAMMA (Phase 22): only when the detector found nothing. A name
            # that forms a real box is armed by the box exactly as before, so
            # with `gamma_params.arms` empty -- the default -- this branch is
            # unreachable and behaviour is bit-identical to the frozen
            # pipeline. The alternative structure supplies the same two numbers
            # a box does, a breakout level and an invalidation floor, so every
            # line below this point runs unchanged on top of it.
            if nested is None and gamma_params is not None and gamma_params.arms:
                nested = detect_gamma_structure(
                    df, i, gamma_params, symbol=symbol,
                    history=box_history, box_params=box_params,
                )
            if nested is not None and nested.big.start_idx > last_box_start_idx:
                episode = _EpisodeState(box=nested, first_seen_idx=i)
                _ev("box_armed", i, box_start=nested.big.start_idx,
                    top=nested.big.top, bottom=nested.big.bottom,
                    small=nested.small is not None)
            elif nested is not None:
                _ev("box_suppressed_same_structure", i, box_start=nested.big.start_idx,
                    last_box_start=last_box_start_idx)
            elif i - last_admission_idx > watch_params.max_watch_without_box:
                on_watchlist = False   # nothing forming since the last admission; drop until re-admitted
                _ev("watch_dropped_stale", i, since_admission=i - last_admission_idx)
            i += 1
            continue

        box = episode.box
        # --- invalidation: a close below the big-box floor kills the setup ---
        if arr_close[i] < box.big.bottom:
            last_box_start_idx = box.big.start_idx
            episode = None
            _ev("box_invalidated", i)
            i += 1
            continue
        # --- staleness: tracked too long with no resolution ---
        if not episode.carried and i - episode.first_seen_idx > watch_params.box_stale_bars:
            if watch_params.box_carry_bars > 0:
                # Carry the resolved box forward as an entry candidate rather
                # than dropping it: the thrust out of a shelf typically comes
                # after the shelf stops measuring as one.
                episode.carried = True
                episode.carry_deadline_idx = i + watch_params.box_carry_bars
                episode.awaiting_retest = False
                _ev("box_carried", i, until=episode.carry_deadline_idx)
            else:
                held = i - episode.first_seen_idx
                last_box_start_idx = box.big.start_idx
                episode = None
                _ev("box_stale", i, held_bars=held)
                i += 1
                continue
        if episode.carried and i > episode.carry_deadline_idx:
            last_box_start_idx = box.big.start_idx
            episode = None
            _ev("carry_expired", i)
            i += 1
            continue

        top, used_small = _operative_top(box)
        threshold = top * (1.0 + watch_params.breakout_margin_pct)

        if episode.awaiting_retest:
            level = episode.retest_level
            if arr_close[i] < level * (1.0 - watch_params.retest_fail_pct):
                last_box_start_idx = box.big.start_idx
                episode = None       # retest failed: broke back down, abandon
                _ev("retest_failed", i)
                i += 1
                continue
            if i > episode.retest_deadline_idx:
                last_box_start_idx = box.big.start_idx
                episode = None       # retest window expired unresolved
                _ev("retest_expired", i)
                i += 1
                continue
            held = arr_low[i] <= level * (1.0 + watch_params.retest_hold_tol)
            confirming = arr_close[i] > arr_open[i]
            if held and confirming:
                vol_ratio = _volume_ratio(arr_volume, i, watch_params.vol_baseline_bars)
                sig = _assemble_signal(
                    df, index_df, symbol=symbol,
                    admission_date=df.index[last_admission_idx], box=box,
                    entry_trigger_idx=i, entry_mode=2, vol_ratio=vol_ratio,
                    used_small_top=episode.used_small_top, horizons=watch_params.horizons,
                    resistance_params=resistance_params, timeframe_params=timeframe_params,
                    rs_params=rs_params,
                )
                if sig is not None:
                    signals.append(sig)
                last_box_start_idx = box.big.start_idx
                episode = None
                i += 1
                continue
            i += 1
            continue

        # --- mode 1 / mode 2 trigger: a fresh close above the operative top ---
        prev_close = arr_close[i - 1] if i > 0 else arr_close[i]
        fresh_cross = arr_close[i] > threshold and prev_close <= threshold
        if fresh_cross:
            vol_ratio = _volume_ratio(arr_volume, i, watch_params.vol_baseline_bars)
            if vol_ratio == vol_ratio and vol_ratio >= watch_params.mode1_vol_threshold:
                sig = _assemble_signal(
                    df, index_df, symbol=symbol,
                    admission_date=df.index[last_admission_idx], box=box,
                    entry_trigger_idx=i, entry_mode=1, vol_ratio=vol_ratio,
                    used_small_top=used_small, horizons=watch_params.horizons,
                    resistance_params=resistance_params, timeframe_params=timeframe_params,
                    rs_params=rs_params,
                )
                if sig is not None:
                    signals.append(sig)
                last_box_start_idx = box.big.start_idx
                episode = None
                i += 1
                continue
            _ev("breakout_weak_vol_to_retest", i, vol_ratio=vol_ratio)
            episode.awaiting_retest = True
            episode.retest_level = top
            episode.retest_deadline_idx = i + watch_params.retest_window
            episode.used_small_top = used_small
            i += 1
            continue

        # --- mode 3: box-bottom entry, only while nothing else has fired ---
        near_bottom = arr_close[i] <= box.big.bottom * (1.0 + watch_params.box_bottom_zone_pct)
        respecting_sma = sma[i] == sma[i] and arr_close[i] > sma[i]
        if near_bottom and respecting_sma and not episode.carried:
            sig = _assemble_signal(
                df, index_df, symbol=symbol,
                admission_date=df.index[last_admission_idx], box=box,
                entry_trigger_idx=i, entry_mode=3, vol_ratio=float("nan"),
                used_small_top=False, horizons=watch_params.horizons,
                resistance_params=resistance_params, timeframe_params=timeframe_params,
                rs_params=rs_params,
            )
            if sig is not None:
                signals.append(sig)
            last_box_start_idx = box.big.start_idx
            episode = None
            i += 1
            continue

        i += 1

    return signals


__all__ = ["run_watchlist_for_symbol"]
