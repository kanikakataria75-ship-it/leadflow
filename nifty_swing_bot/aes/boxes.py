"""Nested accumulation box detection (AES spec section 2).

The unit of the strategy is not a bar, it is a *box*: a stretch of sideways
price action that follows an expansion leg and precedes the next one. This
module finds those boxes and describes their internal health. It does not
decide whether to trade one -- that is the scorer's job (section 4).

Geometry
--------
A detection anchored at bar ``i`` consists of:

``big box``
    The outer consolidation, 15-25% from low to high. Its bottom is the hard
    minimum low of the window because section 6.1 uses it as a stop level.
    Found by sweeping every window length in ``[big_min_bars, big_max_bars]``
    that ends at ``i`` and keeping the best-formed one.

``small box``
    A tighter 5-10% range nested inside, also ending at ``i``. Its edges are
    high/low quantiles rather than extremes, because section 2.1 explicitly
    allows wicks to exceed it while requiring closing bodies to stay in.

Where the small box sits inside the big box is itself a signal: near the big
box bottom is a red flag, middle or upper half is a positive (section 2.1).

Look-ahead discipline
---------------------
Every function here takes a frame and an integer bar index, and reads only
``[0 .. i]``. Nothing uses ``i + 1``. This is enforced structurally by
``test_aes_boxes.py``, which runs the detector on a full series and on the
series truncated at ``i`` and requires identical output. That test is the
reason these functions take an index rather than being written as rolling
operations over the whole frame.

A consequence worth stating: because the big box top is the window maximum,
and a box whose top is today's bar is a breakout rather than a consolidation,
the detector *refuses* to return a box on the breakout bar itself
(``min_bars_since_top``). That is deliberate. Boxes are frozen when found and
carried forward on the watchlist; the breakout is then evaluated against the
frozen box, not against a box redrawn to include the breakout.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .params import BoxParams

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BoxGeometry:
    """One rectangle, with the structure metrics that say how healthy it is.

    Positional fields are bar indices into the frame the box was found in;
    dates are carried alongside so a box survives being moved between frames.
    """

    kind: str                    # "big" or "small"
    start_idx: int
    end_idx: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    top: float
    bottom: float
    bars: int
    range_pct: float

    # --- internal behaviour (section 2.4) ---
    higher_lows: int             # ascending transitions between swing lows
    swing_lows: int
    lows_tilt: float             # regression slope through swing lows / height
    pct_closes_above_mid: float
    top_touches: int
    bottom_touches: int
    close_containment: float     # fraction of closes inside [bottom, top]
    drift_ratio: float           # net close move / height
    oscillation: float           # midpoint crossings, normalised
    bars_since_top: int
    quality: float               # composite "is this box-shaped at all"

    @property
    def mid(self) -> float:
        """Box midpoint, the level section 2.4 measures time-above against."""
        return (self.top + self.bottom) / 2.0

    @property
    def height(self) -> float:
        """Box height in rupees."""
        return self.top - self.bottom

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for assembling feature frames."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NestedBox:
    """A big box with its nested small box, as seen from one evaluation bar."""

    symbol: str
    as_of: pd.Timestamp
    as_of_idx: int
    big: BoxGeometry
    small: BoxGeometry | None

    # --- nesting (section 2.1) ---
    small_position: float | None     # 0 = big box bottom, 1 = big box top
    small_zone: str                  # "bottom" | "middle" | "upper" | "none"

    # --- formation context (section 2.2, 2.5) ---
    prior_leg_pct: float             # expansion leg that built the box top
    vol_dryup_ratio: float           # box volume / pre-box volume; < 1 is dry-up

    # --- per-stock adaptive duration and history (section 2.3, 0) ---
    own_norm_bars: float | None      # median duration of this stock's past boxes
    duration_vs_own_norm: float | None
    prior_boxes: int
    prior_cycles: int                # past boxes that resolved upward

    def as_dict(self) -> dict[str, Any]:
        """Flat dict with ``big_``/``small_`` prefixed geometry columns."""
        row: dict[str, Any] = {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "as_of_idx": self.as_of_idx,
            "small_position": self.small_position,
            "small_zone": self.small_zone,
            "prior_leg_pct": self.prior_leg_pct,
            "vol_dryup_ratio": self.vol_dryup_ratio,
            "own_norm_bars": self.own_norm_bars,
            "duration_vs_own_norm": self.duration_vs_own_norm,
            "prior_boxes": self.prior_boxes,
            "prior_cycles": self.prior_cycles,
        }
        for prefix, box in (("big", self.big), ("small", self.small)):
            if box is None:
                continue
            for key, value in box.as_dict().items():
                if key == "kind":
                    continue
                row[f"{prefix}_{key}"] = value
        return row


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def swing_low_indices(low: np.ndarray, k: int) -> np.ndarray:
    """Indices of local minima with ``k`` higher bars on both sides.

    Args:
        low: Array of lows.
        k: Bars required either side.

    Returns:
        Ascending array of positions into ``low``. The last ``k`` bars can
        never qualify, which is correct: a swing low is not confirmed until
        ``k`` bars have printed after it.
    """
    n = low.size
    if n < 2 * k + 1:
        return np.empty(0, dtype=np.int64)
    found: list[int] = []
    for j in range(k, n - k):
        left = low[j - k : j]
        right = low[j + 1 : j + k + 1]
        if low[j] < left.min() and low[j] <= right.min():
            found.append(j)
    return np.asarray(found, dtype=np.int64)


def _count_separated(hits: np.ndarray, separation: int) -> int:
    """Count hits, collapsing any that fall within ``separation`` bars.

    Three consecutive bars poking at the box top are one visit to the level,
    not three, and counting them as three would flatter a trend that simply
    ran into the edge and stopped.
    """
    if hits.size == 0:
        return 0
    count = 1
    last = int(hits[0])
    for h in hits[1:]:
        if int(h) - last >= separation:
            count += 1
            last = int(h)
    return count


def _trim_isolated_extreme(
    values: np.ndarray, side: str, gap_pct: float, max_trim: int
) -> float:
    """The outermost value that is not an isolated outlier.

    Walks inward from the extreme (highest for ``side="max"``, lowest for
    ``side="min"``). A candidate is dropped only if the next value in is more
    than ``gap_pct`` away from it in price -- i.e. it is genuinely alone.
    The moment a neighbour is found within tolerance, that candidate is kept
    as the edge, because it is now a *level* (two or more bars near the same
    price), not a single bar's tail.

    This is the rule stated directly after chart review: a lone wick outside
    the visible range is noise and should be excluded, but if several wicks
    sit together out there, that is a real level and the box should extend to
    include it -- the edge becomes the far side of that cluster, not the near
    side.

    Args:
        values: The candidate window's highs (for ``side="max"``) or lows
            (for ``side="min"``).
        side: ``"max"`` or ``"min"``.
        gap_pct: Relative-price gap above which two points count as unrelated.
        max_trim: Ceiling on isolated points that may be discarded, so a
            window with no clustering anywhere cannot be trimmed away almost
            entirely.

    Returns:
        The trimmed extreme. Falls back to the raw extreme if the array has
        only one element.
    """
    order = np.sort(values)
    ordered = order[::-1] if side == "max" else order
    limit = min(max_trim, ordered.size - 1)
    idx = 0
    while idx < limit:
        cur, nxt = ordered[idx], ordered[idx + 1]
        reference = nxt if nxt != 0 else cur
        if reference == 0 or abs(cur - nxt) / abs(reference) <= gap_pct:
            break
        idx += 1
    return float(ordered[idx])


def _measure(
    kind: str,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    top: float,
    bottom: float,
    params: BoxParams,
) -> dict[str, Any]:
    """Structure metrics for one candidate window. Pure arithmetic."""
    bars = close.size
    height = top - bottom
    if height <= 0 or bars == 0:
        return {}

    mid = (top + bottom) / 2.0
    tol = params.edge_touch_tol * height

    top_hits = np.flatnonzero(high >= top - tol)
    bottom_hits = np.flatnonzero(low <= bottom + tol)
    top_touches = _count_separated(top_hits, params.min_touch_separation)
    bottom_touches = _count_separated(bottom_hits, params.min_touch_separation)

    lows_idx = swing_low_indices(low, params.swing_k)
    if lows_idx.size >= 2:
        values = low[lows_idx]
        higher_lows = int(np.sum(np.diff(values) > 0))
        # Theil-Sen (median of pairwise slopes) rather than least squares.
        # A box is anchored at its own high, so its first swing low is often
        # far above the rest; under OLS that single point dominates the fit and
        # reports a *falling* tilt for a series of plainly ascending troughs.
        # The median slope ignores it.
        xs = lows_idx.astype(float)
        dx = xs[None, :] - xs[:, None]
        dy = values[None, :] - values[:, None]
        forward = dx > 0
        slope = float(np.median(dy[forward] / dx[forward])) if forward.any() else 0.0
        lows_tilt = slope * bars / height
    else:
        higher_lows = 0
        lows_tilt = 0.0

    pct_above_mid = float(np.mean(close > mid))
    containment = float(np.mean((close >= bottom) & (close <= top)))
    drift_ratio = float((close[-1] - close[0]) / height)

    side = np.sign(close - mid)
    side = side[side != 0]
    crossings = int(np.sum(np.diff(side) != 0)) if side.size >= 2 else 0
    oscillation = min(1.0, crossings / max(1.0, bars / 4.0))

    bars_since_top = int(bars - 1 - int(np.argmax(high)))

    # Every structure metric is expressed as a *rate* rather than a count.
    # Counts grade length, not shape: a 45-bar box has more swing lows and more
    # chances to touch an edge than a 12-bar box, so a count-based grade picks
    # the longest candidate almost regardless of how it looks. Measured
    # directly, count-based weights gave corr(quality, bars) = 0.60.
    n_swings = int(lows_idx.size)
    hl_score = higher_lows / (n_swings - 1) if n_swings >= 2 else 0.0
    per_10 = max(1.0, bars / 10.0)
    touch_score = (
        0.6 * min(1.0, top_touches / per_10) + 0.4 * min(1.0, bottom_touches / per_10)
    )

    # Containment is only a measurement for the small box. The big box's edges
    # are the window's own extremes of high and low, so every close is inside
    # it by construction and the term would contribute a constant. Drop it
    # there and renormalise, rather than let a tautology carry 25% of the
    # grade.
    terms = [
        (params.w_above_mid, pct_above_mid),
        (params.w_higher_lows, hl_score),
        (params.w_touch, touch_score),
    ]
    if kind != "big":
        terms.append((params.w_containment, containment))
    total_weight = sum(w for w, _ in terms)
    quality = sum(w * v for w, v in terms) / total_weight if total_weight else 0.0

    return {
        "kind": kind,
        "top": float(top),
        "bottom": float(bottom),
        "bars": int(bars),
        "range_pct": float(height / bottom),
        "higher_lows": higher_lows,
        "swing_lows": int(lows_idx.size),
        "lows_tilt": float(lows_tilt),
        "pct_closes_above_mid": pct_above_mid,
        "top_touches": int(top_touches),
        "bottom_touches": int(bottom_touches),
        "close_containment": containment,
        "drift_ratio": drift_ratio,
        "oscillation": float(oscillation),
        "bars_since_top": bars_since_top,
        "quality": float(quality),
    }


def _frame_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    return {c: df[c].to_numpy(dtype=float) for c in REQUIRED_COLUMNS}


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _basis_arrays(
    arr: dict[str, np.ndarray], params: BoxParams
) -> tuple[np.ndarray, np.ndarray]:
    """The two series the big box reads as "the high" and "the low".

    One basis throughout: the bar that anchors the box, the edges, the touch
    counts and ``bars_since_top`` must all agree about what a high is.
    """
    if params.edge_basis == "body":
        return np.maximum(arr["open"], arr["close"]), np.minimum(arr["open"], arr["close"])
    if params.edge_basis == "close":
        c = arr["close"]
        return c, c
    return arr["high"], arr["low"]


def _shelf_start(
    close: np.ndarray, top: float, start: int, i: int, params: BoxParams
) -> int:
    """Where the base begins, once the leg that built ``top`` is excluded.

    Only moves the start when the window actually contains a leg too big to be
    a box -- price dropping more than ``big_range_max`` below the anchor top
    somewhere inside it. In that case the base is taken to begin after the last
    bar that closed below the shelf zone, so the advance back up to the top is
    context (it is what ``NestedBox.prior_leg_pct`` measures) rather than box
    height. A window with no such leg is returned untouched.
    """
    floor = top * (1.0 - params.big_range_max)
    below = np.flatnonzero(close[start : i + 1] < floor)
    if below.size == 0:
        return start
    return start + int(below[-1]) + 1


def _sloped_candidate(
    upper: np.ndarray,
    lower: np.ndarray,
    close: np.ndarray,
    start: int,
    i: int,
    params: BoxParams,
) -> dict[str, Any] | None:
    """A rising channel over ``[start, i]``, measured on detrended price.

    Fits a Theil-Sen line through the closes -- median of pairwise slopes, the
    same estimator ``_measure`` already uses for the swing-low tilt and for the
    same reason -- then measures the band as the residual spread around it. The
    reported edges are the channel's value **at bar i**, which is the level an
    entry or a §6.1 stop would actually reference.

    ``sloped_max_drift`` is what keeps this from swallowing every quiet
    uptrend: a candidate whose trend rise is large relative to its own band is
    a trend, not a base, and is refused.
    """
    n = i - start + 1
    if n < params.big_min_bars or n > params.big_max_bars:
        return None
    y = close[start : i + 1]
    x = np.arange(n, dtype=float)
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    fwd = dx > 0
    if not fwd.any():
        return None
    slope = float(np.median(dy[fwd] / dx[fwd]))
    intercept = float(np.median(y - slope * x))
    trend = intercept + slope * x
    last = float(trend[-1])

    hi = upper[start : i + 1] - trend + last
    lo = lower[start : i + 1] - trend + last
    cl = y - trend + last

    top = float(hi.max())
    bottom = _trim_isolated_extreme(
        lo, "min", params.isolated_wick_gap_pct, params.max_isolated_trim
    )
    if bottom <= 0 or top <= bottom:
        return None
    height = top - bottom
    range_pct = height / bottom
    if not params.big_range_min <= range_pct <= params.big_range_max:
        return None
    drift = slope * (n - 1) / height if height > 0 else 0.0
    if not params.sloped_min_drift <= drift <= params.sloped_max_drift:
        return None
    m = _measure("big", hi, lo, cl, top, bottom, params)
    if not m or m["bars_since_top"] < params.min_bars_since_top:
        return None
    return m | {"start_idx": start, "end_idx": i}


def detect_big_box(
    df: pd.DataFrame, i: int, params: BoxParams | None = None
) -> BoxGeometry | None:
    """Best-formed 15-25% consolidation ending at bar ``i``, if there is one.

    The box is *anchored*, not swept. For each lookback in
    ``params.anchor_lookbacks`` the running-maximum bar is taken as a candidate
    box top -- the high price stalled under (§2.2) -- and the box runs from that
    bar to ``i``. Duration therefore comes out of price structure rather than
    out of a search, which is what makes the §2.3 duration norm mean anything.

    Candidates failing the height band, the duration bounds, or the
    "today's high is the top" test are dropped. Among the survivors the one
    with the **highest top** wins, ties broken by §2.4 health.

    Selecting on the top rather than on health is deliberate. Nested anchors
    often yield a shorter box whose ceiling is a fraction lower -- a high made
    *inside* the consolidation rather than the older high that created it.
    Picking by health let that shorter box win on small scoring differences,
    which made the box top jitter between adjacent sessions and contradicted
    both §2.2 ("box top is typically the recent/52-week high where price
    stalled") and §3.2 ("resistance strength = age"). The operative ceiling is
    the highest one that still forms a valid 15-25% box.

    Args:
        df: OHLCV frame, ascending date index.
        i: Bar index to evaluate at. Only bars ``[0 .. i]`` are read.
        params: Geometry thresholds.

    Returns:
        The box, or ``None`` if no anchor qualifies.
    """
    params = params or BoxParams()
    arr = _frame_arrays(df)
    if i < params.big_min_bars - 1:
        return None

    best: dict[str, Any] | None = None
    best_key: tuple[float, float] = (-np.inf, -np.inf)
    seen: set[int] = set()

    # One basis throughout: the bar that anchors the box, the edges, the touch
    # counts and ``bars_since_top`` must all agree about what "the high" is.
    upper, lower = _basis_arrays(arr, params)

    for lookback in params.anchor_lookbacks:
        window_start = max(0, i - lookback + 1)
        if i - window_start + 1 < params.big_min_bars:
            continue
        start = window_start + int(np.argmax(upper[window_start : i + 1]))
        if params.exclude_prior_leg:
            anchor_top = float(upper[start])
            start = _shelf_start(arr["close"], anchor_top, start, i, params)
        if start in seen:
            continue
        seen.add(start)
        bars = i - start + 1
        if not params.big_min_bars <= bars <= params.big_max_bars:
            continue
        high = upper[start : i + 1]
        low = lower[start : i + 1]
        close = arr["close"][start : i + 1]
        # The top is anchor-defined -- the start bar is, by construction, the
        # window's running maximum -- so only the floor is a free extreme an
        # isolated wick can distort.
        top = float(high.max())
        bottom = _trim_isolated_extreme(
            low, "min", params.isolated_wick_gap_pct, params.max_isolated_trim
        )
        if bottom <= 0 or top <= bottom:
            continue
        range_pct = (top - bottom) / bottom
        if not params.big_range_min <= range_pct <= params.big_range_max:
            continue
        m = _measure("big", high, low, close, top, bottom, params)
        if not m or m["bars_since_top"] < params.min_bars_since_top:
            continue
        key = (top, m["quality"])
        if key > best_key:
            best_key = key
            best = m | {"start_idx": start, "end_idx": i}

    # Sloped channels are a *fallback*, never a competitor: a horizontal box
    # found at this bar always wins. That keeps the arm purely additive, so
    # "how many does it recover" is answerable, and it means nothing already
    # detected can be redrawn as a channel and quietly change its edges.
    if best is None and params.allow_sloped:
        best_key = (-np.inf, -np.inf)
        for lookback in params.anchor_lookbacks:
            window_start = max(0, i - lookback + 1)
            if i - window_start + 1 < params.big_min_bars:
                continue
            m = _sloped_candidate(upper, lower, arr["close"], window_start, i, params)
            if m is None:
                continue
            key = (m["top"], m["quality"])
            if key > best_key:
                best_key = key
                best = m

    if best is None:
        return None
    return BoxGeometry(
        start_date=df.index[best["start_idx"]],
        end_date=df.index[best["end_idx"]],
        **best,
    )


def detect_small_box(
    df: pd.DataFrame, i: int, big: BoxGeometry, params: BoxParams | None = None
) -> BoxGeometry | None:
    """Best 5-10% range ending at bar ``i`` and nested inside ``big``.

    Edges are trimmed extremes (``_trim_isolated_extreme``), not the raw
    high/low, so a single isolated wick does not define the box -- but if
    several bars' wicks cluster beyond the visible range, the box extends to
    cover them, since that is a real level rather than noise. Section 2.1's
    "wicks may exceed the small box, closing bodies should mostly stay
    inside" still holds for whatever residual wicks remain outside the
    trimmed edge, and is what makes ``close_containment`` a real measurement
    rather than a tautology.
    """
    params = params or BoxParams()
    arr = _frame_arrays(df)
    slack = params.small_wick_tolerance * big.height
    best: dict[str, Any] | None = None
    best_score = -np.inf
    upper = min(params.small_max_bars, big.bars)
    span = max(1, upper - params.small_min_bars)

    for length in range(params.small_min_bars, upper + 1):
        start = i - length + 1
        if start < 0:
            break
        high = arr["high"][start : i + 1]
        low = arr["low"][start : i + 1]
        close = arr["close"][start : i + 1]
        top = _trim_isolated_extreme(
            high, "max", params.isolated_wick_gap_pct, params.max_isolated_trim
        )
        bottom = _trim_isolated_extreme(
            low, "min", params.isolated_wick_gap_pct, params.max_isolated_trim
        )
        if bottom <= 0 or top <= bottom:
            continue
        range_pct = (top - bottom) / bottom
        if not params.small_range_min <= range_pct <= params.small_range_max:
            continue
        if top > big.top + slack or bottom < big.bottom - slack:
            continue
        m = _measure("small", high, low, close, top, bottom, params)
        if not m:
            continue
        score = m["quality"] + params.duration_pref * (length - params.small_min_bars) / span
        if score > best_score:
            best_score = score
            best = m | {"start_idx": start, "end_idx": i}

    if best is None:
        return None
    return BoxGeometry(
        start_date=df.index[best["start_idx"]],
        end_date=df.index[best["end_idx"]],
        **best,
    )


def _zone(position: float, params: BoxParams) -> str:
    if position < params.zone_bottom_max:
        return "bottom"
    if position > params.zone_upper_min:
        return "upper"
    return "middle"


def detect_nested_box(
    df: pd.DataFrame,
    i: int,
    params: BoxParams | None = None,
    *,
    symbol: str = "",
    history: list[BoxGeometry] | None = None,
) -> NestedBox | None:
    """Full section 2 detection at bar ``i``: big box, small box, and context.

    Args:
        df: OHLCV frame, ascending date index.
        i: Bar index to evaluate at. Only bars ``[0 .. i]`` are read.
        params: Geometry thresholds.
        symbol: Carried through onto the result.
        history: Previously detected box episodes for this stock, used for the
            per-stock adaptive duration norm (section 2.3). Episodes that end
            at or after the current box's start are ignored, so passing the
            full history is safe and does not leak.

    Returns:
        The nested box, or ``None`` if no big box qualifies at ``i``.
    """
    params = params or BoxParams()
    if i < params.min_history_bars - 1:
        return None

    big = detect_big_box(df, i, params)
    if big is None:
        return None
    small = detect_small_box(df, i, big, params)

    position: float | None = None
    if small is not None and big.height > 0:
        position = float((small.mid - big.bottom) / big.height)
        zone = _zone(position, params)
    else:
        zone = "none"

    arr = _frame_arrays(df)
    leg_start = max(0, big.start_idx - params.prior_leg_lookback)
    if big.start_idx > leg_start:
        leg_low = float(arr["low"][leg_start : big.start_idx].min())
        prior_leg_pct = (big.top - leg_low) / leg_low if leg_low > 0 else 0.0
        pre_vol = float(arr["volume"][leg_start : big.start_idx].mean())
    else:
        prior_leg_pct = 0.0
        pre_vol = float("nan")
    box_vol = float(arr["volume"][big.start_idx : i + 1].mean())
    vol_dryup = box_vol / pre_vol if pre_vol and pre_vol > 0 else float("nan")

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
            prior_cycles = sum(1 for b in past if _resolved_upward(arr, b, i, params))

    return NestedBox(
        symbol=symbol,
        as_of=df.index[i],
        as_of_idx=i,
        big=big,
        small=small,
        small_position=position,
        small_zone=zone,
        prior_leg_pct=float(prior_leg_pct),
        vol_dryup_ratio=float(vol_dryup),
        own_norm_bars=own_norm,
        duration_vs_own_norm=dur_vs_norm,
        prior_boxes=prior_boxes,
        prior_cycles=prior_cycles,
    )


def _resolved_upward(
    arr: dict[str, np.ndarray], box: BoxGeometry, i: int, params: BoxParams
) -> bool:
    """Did a past box break out upward, using only bars at or before ``i``?

    A box whose resolution window has not fully elapsed by ``i`` is judged on
    the bars available so far: a clear break already seen counts, and the
    absence of one does not, because the rest of the window is in the future.
    """
    start = box.end_idx + 1
    stop = min(box.end_idx + params.resolve_window, i)
    if stop < start:
        return False
    window = arr["close"][start : stop + 1]
    return bool(window.size and window.max() >= box.top * (1.0 + params.resolve_pct))


# --------------------------------------------------------------------------- #
# Historical episodes (per-stock adaptive norm, section 2.3)
# --------------------------------------------------------------------------- #
def scan_box_history(
    df: pd.DataFrame,
    params: BoxParams | None = None,
    *,
    start_idx: int | None = None,
    end_idx: int | None = None,
    step: int = 1,
) -> list[BoxGeometry]:
    """Distinct accumulation episodes across a stock's history.

    The detector is anchored at every bar in the range, which produces many
    overlapping views of the same consolidation. Those are collapsed into
    episodes: detections sharing at least ``episode_overlap`` of their bars are
    the same box seen from different days, and the highest-quality view wins.

    This is what gives section 2.3 its per-stock norm -- "weight the current
    box against its own history, not a universal constant" -- and section 0 its
    count of completed accumulation-to-expansion cycles.

    Args:
        df: OHLCV frame.
        params: Geometry thresholds.
        start_idx: First bar to evaluate. Defaults to ``big_min_bars - 1``.
        end_idx: Last bar to evaluate, inclusive. Defaults to the final bar.
        step: Evaluate every ``step`` bars. A step above 1 trades a little
            episode-boundary precision for speed.

    Returns:
        Episodes in ascending order of end date.
    """
    params = params or BoxParams()
    first = params.big_min_bars - 1 if start_idx is None else start_idx
    last = len(df) - 1 if end_idx is None else end_idx
    episodes: list[BoxGeometry] = []

    for i in range(max(first, params.big_min_bars - 1), last + 1, step):
        box = detect_big_box(df, i, params)
        if box is None:
            continue
        if episodes and _overlap_fraction(episodes[-1], box) >= params.episode_overlap:
            if box.quality > episodes[-1].quality:
                episodes[-1] = box
            continue
        episodes.append(box)
    return episodes


def _overlap_fraction(a: BoxGeometry, b: BoxGeometry) -> float:
    """Shared bars as a fraction of the shorter box."""
    lo = max(a.start_idx, b.start_idx)
    hi = min(a.end_idx, b.end_idx)
    shared = max(0, hi - lo + 1)
    return shared / min(a.bars, b.bars)


__all__ = [
    "REQUIRED_COLUMNS",
    "BoxGeometry",
    "NestedBox",
    "detect_big_box",
    "detect_nested_box",
    "detect_small_box",
    "scan_box_history",
    "swing_low_indices",
]
