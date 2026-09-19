"""GAMMA (Phase 22): alternative daily continuation structures at the box stage.

The change under test, and nothing else
---------------------------------------
Admission is untouched -- +20%/10d, within 5% of the 52-week high, the price
and turnover floors, the hard deletes. §14.1 proved both admission conditions
load-bearing and REJECTED #11 measured the 20/5 cell as the best in its grid.
Everything downstream of the structure is untouched too: equal-weight scoring,
6 slots, 3% risk, the 0.7/1.2 conviction multipliers, ATR(14) x 2.5 with the
25-bar cap, the frozen ladder, and the 30/50/20 book on annual rebalance.

The single change is at the **box stage**. Today a watched name that never
forms a nested box is dropped once ``max_watch_without_box`` elapses. GAMMA
lets an alternative bullish daily structure arm the episode instead. Each
structure yields the same two numbers a box yields -- a breakout level and an
invalidation floor -- so the state machine, the entry modes, the retest logic
and the exits all run unchanged on top of it.

Why this is worth running despite four failures before it
--------------------------------------------------------
REJECTED #8, #12, #18 and the geometry sweep in #14 all raised signal count and
all lowered fold return, so the prior is bad and is stated here rather than
discovered later. What makes GAMMA a different question from those four is
**§3a, the standing contradiction**: the detector's admission gate runs at
11.13% against a 4.38% base rate, but within a turnover- and volatility-matched
comparison essentially all residual power separating runners from non-runners
(AUC 0.598 of 0.599) sits in *twelve features the detector does not compute*.
The four prior attempts all widened the same detector. GAMMA asks whether a
different structure sees something the box shape cannot express. That is a new
reason, which is what the STATE.md reading rule requires.

The three arms
--------------
a) ``hhhl``      A confirmed higher-high / higher-low sequence. ``min_swings``
                 is swept at 2 / 3 / 4. Pivots are fractal and confirmed, so a
                 pivot at bar j is only visible from bar j + ``swing_window``.

b) ``rising_ma`` Price holding above a rising moving average for a minimum
                 window -- the plainest continuation structure there is, and
                 the one most likely to be what a discretionary trader means
                 by "it's trending".

c) ``pullback``  **Shallow-pullback continuation, and it is not a free choice
                 -- it is read off this project's own runner study.** In
                 ``results/runner_composite.json`` (validation AUC 0.612, sign
                 agreement 21/21) the runner profile is: ``dd_from_40b_peak``
                 median -9.1% with sign -1, ``ema20_dist`` median -1.2% with
                 sign -1, ``close_position_40`` median 0.45 with sign -1, and
                 ``max_dd_40`` / ``mean_dd_40`` both sign -1. Read together
                 that is a literal description of a name sitting in a shallow
                 pullback inside an uptrend, a little under its EMA20, around
                 mid-range of its 40-bar span. It is also precisely a structure
                 the box detector does not compute, which is why it is the
                 third arm rather than something invented for the occasion.

Nothing here is adopted on this module's own say-so. It exists to be measured
fold by fold against the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from .boxes import BoxGeometry, BoxParams, NestedBox, _frame_arrays, _resolved_upward

ArmName = Literal["hhhl", "rising_ma", "pullback"]

#: Every arm, in the order they are attempted when several are enabled.
ALL_ARMS: tuple[str, ...] = ("hhhl", "rising_ma", "pullback")


@dataclass(frozen=True, slots=True)
class GammaParams:
    """Thresholds for the alternative structures. Empty ``arms`` is a no-op.

    Default-constructed, this object disables GAMMA entirely, so importing it
    or threading it through a call site cannot change frozen behaviour. An arm
    only runs when it is named.
    """

    #: Which arms may arm an episode. Empty tuple = baseline behaviour.
    arms: tuple[str, ...] = ()

    # --- shared ---
    #: Half-width of the fractal pivot. A pivot at j is confirmed at j + this.
    swing_window: int = 3
    #: How far back a structure may look for its own start.
    lookback: int = 60
    #: A structure must clear this much range to be tradeable at all, matching
    #: the spirit of the box range floor that REJECTED #18 proved load-bearing.
    min_range_pct: float = 0.05

    # --- a) higher highs / higher lows ---
    hhhl_min_swings: int = 3

    # --- b) rising moving average ---
    ma_kind: str = "ema"           # "ema" | "sma"
    ma_period: int = 20
    #: Bars price must have held above the average, and the average risen, for.
    ma_min_hold_bars: int = 20
    #: Minimum slope of the average over the hold window, as a fraction of price.
    ma_min_slope_pct: float = 0.0
    #: Fraction of the hold window whose closes must sit above the average.
    ma_min_containment: float = 0.90

    # --- c) shallow-pullback continuation ---
    pb_peak_lookback: int = 40
    pb_min_depth: float = 0.03
    pb_max_depth: float = 0.12
    pb_min_hold_bars: int = 5

    def with_arms(self, *arms: str) -> "GammaParams":
        return replace(self, arms=tuple(arms))


# --------------------------------------------------------------------------- #
# Pivots
# --------------------------------------------------------------------------- #
def _confirmed_pivots(
    high: np.ndarray, low: np.ndarray, i: int, w: int, lookback: int
) -> tuple[list[int], list[int]]:
    """Fractal swing highs and lows that are *confirmed* by bar ``i``.

    A pivot at bar j needs ``w`` bars either side to be a pivot at all, so it
    is not knowable until bar ``j + w``. Only pivots with ``j + w <= i`` are
    returned. Getting this wrong is the standard way a structure detector
    smuggles in lookahead and posts an edge that cannot be traded.
    """
    lo_bound = max(w, i - lookback)
    hi_bound = i - w
    highs: list[int] = []
    lows: list[int] = []
    for j in range(lo_bound, hi_bound + 1):
        seg_h = high[j - w : j + w + 1]
        seg_l = low[j - w : j + w + 1]
        if seg_h.size and high[j] == seg_h.max() and (seg_h.argmax() == w):
            highs.append(j)
        if seg_l.size and low[j] == seg_l.min() and (seg_l.argmin() == w):
            lows.append(j)
    return highs, lows


# --------------------------------------------------------------------------- #
# Arms. Each returns (start_idx, top, bottom, label) or None.
# --------------------------------------------------------------------------- #
def _arm_hhhl(arr: dict[str, np.ndarray], i: int, p: GammaParams):
    """a) A confirmed ascending sequence of swing highs and swing lows."""
    highs, lows = _confirmed_pivots(arr["high"], arr["low"], i, p.swing_window, p.lookback)
    need = p.hhhl_min_swings
    if len(highs) < need or len(lows) < need:
        return None

    hh = highs[-need:]
    hl = lows[-need:]
    hv = [arr["high"][j] for j in hh]
    lv = [arr["low"][j] for j in hl]
    if not all(b > a for a, b in zip(hv, hv[1:])):
        return None
    if not all(b > a for a, b in zip(lv, lv[1:])):
        return None

    # The breakout level is the most recent confirmed swing high; the floor is
    # the most recent confirmed higher low. Price must not already have run
    # away above the level, or the structure is being armed after the move.
    top = float(hv[-1])
    bottom = float(lv[-1])
    start = int(min(hh[0], hl[0]))
    if bottom <= 0 or top <= bottom:
        return None
    if arr["close"][i] < bottom:
        return None
    return start, top, bottom, f"hhhl{need}"


def _arm_rising_ma(arr: dict[str, np.ndarray], i: int, p: GammaParams):
    """b) Price holding above a rising moving average over a minimum window."""
    close = arr["close"]
    n = p.ma_period
    hold = p.ma_min_hold_bars
    if i < n + hold:
        return None

    s = pd.Series(close[: i + 1])
    ma = (s.ewm(span=n, adjust=False).mean() if p.ma_kind == "ema"
          else s.rolling(n).mean()).to_numpy()

    win = slice(i - hold + 1, i + 1)
    ma_win = ma[win]
    if np.isnan(ma_win).any():
        return None

    # rising, by a stated amount rather than merely not-falling
    slope = (ma_win[-1] - ma_win[0]) / ma_win[0] if ma_win[0] > 0 else 0.0
    if slope <= p.ma_min_slope_pct:
        return None

    contained = float((close[win] > ma_win).mean())
    if contained < p.ma_min_containment:
        return None

    start = i - hold + 1
    top = float(arr["high"][start : i + 1].max())
    bottom = float(ma_win[-1])
    if bottom <= 0 or top <= bottom:
        return None
    return start, top, bottom, "rising_ma"


def _arm_pullback(arr: dict[str, np.ndarray], i: int, p: GammaParams):
    """c) Shallow pullback from a recent peak, held for a minimum number of bars.

    The floor is the pullback low rather than the peak: that is the level whose
    loss says the continuation thesis was wrong, and it is what the ATR trail
    and the big-box-bottom stop downstream both expect to be handed.
    """
    lb = p.pb_peak_lookback
    if i < lb + p.pb_min_hold_bars:
        return None

    seg_hi = arr["high"][i - lb + 1 : i + 1]
    peak_off = int(seg_hi.argmax())
    peak_idx = i - lb + 1 + peak_off
    peak = float(seg_hi[peak_off])
    if peak <= 0:
        return None

    bars_since_peak = i - peak_idx
    if bars_since_peak < p.pb_min_hold_bars:
        return None

    trough = float(arr["low"][peak_idx : i + 1].min())
    depth = (peak - trough) / peak
    if not (p.pb_min_depth <= depth <= p.pb_max_depth):
        return None

    # still holding: the current close must sit above the pullback low, and
    # must not already have taken out the peak (that is a breakout, not a
    # structure waiting for one).
    if arr["close"][i] <= trough or arr["close"][i] >= peak:
        return None

    return int(peak_idx), peak, trough, "pullback"


_ARM_FUNCS = {"hhhl": _arm_hhhl, "rising_ma": _arm_rising_ma, "pullback": _arm_pullback}


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _synth_geometry(
    df: pd.DataFrame, arr: dict[str, np.ndarray], start: int, i: int,
    top: float, bottom: float, label: str,
) -> BoxGeometry:
    """A ``BoxGeometry`` describing a GAMMA structure.

    The behaviour metrics a real box carries (containment, oscillation, top
    touches) are *measured* on the structure's own span rather than faked, so
    anything downstream that reads them reads something true. ``quality`` is
    set to NaN rather than 0.0: these structures are not boxes and do not have
    a box quality, and a NaN cannot be silently averaged into one.
    """
    close = arr["close"][start : i + 1]
    high = arr["high"][start : i + 1]
    low = arr["low"][start : i + 1]
    height = top - bottom
    mid = (top + bottom) / 2.0
    bars = i - start + 1

    return BoxGeometry(
        kind=f"gamma:{label}",
        start_idx=start,
        end_idx=i,
        start_date=df.index[start],
        end_date=df.index[i],
        top=float(top),
        bottom=float(bottom),
        bars=int(bars),
        range_pct=float(height / bottom) if bottom > 0 else 0.0,
        higher_lows=0,
        swing_lows=0,
        lows_tilt=0.0,
        pct_closes_above_mid=float((close > mid).mean()) if close.size else 0.0,
        top_touches=int((high >= top * 0.995).sum()),
        bottom_touches=int((low <= bottom * 1.005).sum()),
        close_containment=float(((close >= bottom) & (close <= top)).mean())
                          if close.size else 0.0,
        drift_ratio=float((close[-1] - close[0]) / height) if close.size and height > 0 else 0.0,
        oscillation=0.0,
        bars_since_top=int(i - (start + int(high.argmax()))) if high.size else 0,
        quality=float("nan"),
    )


def detect_gamma_structure(
    df: pd.DataFrame,
    i: int,
    params: GammaParams | None = None,
    *,
    symbol: str = "",
    history: list[BoxGeometry] | None = None,
    box_params: BoxParams | None = None,
) -> NestedBox | None:
    """The first enabled arm that finds a structure at bar ``i``, or ``None``.

    Only bars ``[0 .. i]`` are read. The returned object is a ``NestedBox`` with
    ``small=None`` and a ``kind`` of ``gamma:<arm>`` on the big geometry, so a
    GAMMA-armed episode is always distinguishable from a box-armed one in any
    downstream frame or trace.

    ``prior_cycles`` is computed from the stock's **box** history, exactly as
    ``detect_nested_box`` does. That is deliberate: it is a property of the
    stock ("has this name completed accumulation cycles before"), not of the
    structure being armed now. Zeroing it for GAMMA signals would hand them a
    systematically lower score on a 0.25-weighted factor and quietly bias every
    arm downward against the baseline.
    """
    p = params or GammaParams()
    if not p.arms:
        return None
    bp = box_params or BoxParams()
    if i < bp.min_history_bars - 1:
        return None

    arr = _frame_arrays(df)
    found = None
    for arm in p.arms:
        fn = _ARM_FUNCS.get(arm)
        if fn is None:
            continue
        got = fn(arr, i, p)
        if got is not None:
            found = got
            break
    if found is None:
        return None

    start, top, bottom, label = found
    if bottom <= 0 or (top - bottom) / bottom < p.min_range_pct:
        return None

    big = _synth_geometry(df, arr, start, i, top, bottom, label)

    leg_start = max(0, big.start_idx - bp.prior_leg_lookback)
    if big.start_idx > leg_start:
        leg_low = float(arr["low"][leg_start : big.start_idx].min())
        prior_leg_pct = (big.top - leg_low) / leg_low if leg_low > 0 else 0.0
        pre_vol = float(arr["volume"][leg_start : big.start_idx].mean())
    else:
        prior_leg_pct = 0.0
        pre_vol = float("nan")
    body_vol = float(arr["volume"][big.start_idx : i + 1].mean())
    vol_dryup = body_vol / pre_vol if pre_vol and pre_vol > 0 else float("nan")

    own_norm: float | None = None
    dur_vs_norm: float | None = None
    prior_boxes = prior_cycles = 0
    if history:
        past = [b for b in history if b.end_idx < big.start_idx]
        prior_boxes = len(past)
        if past:
            own_norm = float(np.median([b.bars for b in past]))
            if own_norm > 0:
                dur_vs_norm = big.bars / own_norm
            prior_cycles = sum(1 for b in past if _resolved_upward(arr, b, i, bp))

    return NestedBox(
        symbol=symbol,
        as_of=df.index[i],
        as_of_idx=i,
        big=big,
        small=None,
        small_position=None,
        small_zone="none",
        prior_leg_pct=float(prior_leg_pct),
        vol_dryup_ratio=float(vol_dryup),
        own_norm_bars=own_norm,
        duration_vs_own_norm=dur_vs_norm,
        prior_boxes=prior_boxes,
        prior_cycles=prior_cycles,
    )


def is_gamma(box: NestedBox) -> bool:
    """Whether this structure was armed by GAMMA rather than by the detector."""
    return str(box.big.kind).startswith("gamma:")


def gamma_arm(box: NestedBox) -> str | None:
    """Which arm armed it, or ``None`` for a real box."""
    k = str(box.big.kind)
    return k.split(":", 1)[1] if k.startswith("gamma:") else None


__all__ = [
    "ALL_ARMS", "ArmName", "GammaParams", "detect_gamma_structure",
    "gamma_arm", "is_gamma",
]
