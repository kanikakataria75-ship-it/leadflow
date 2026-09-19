"""Age-weighted resistance and prior-visit reaction (AES spec sections 3.2, 3.3).

Two related ideas live here.

**Resistance strength is age, not just price.** Section 3.2 is explicit: "Older
levels are stronger. Weight resistance levels by how far back they formed." A
level is found by clustering swing highs -- price points where the market
turned away before -- and its age is measured from the *oldest* touch in the
cluster, because a level first made three years ago and retested last month is
an old level, not a recent one.

**A level's history is a track record, not just a price.** Section 3.3 asks
whether price has visited a zone before, and if so, what happened: was the
visit met with volume and a sharp rejection, or did price grind through it? A
level touched several times with strong volume and no clean break is a wall; a
level ground through repeatedly on rising volume is being absorbed.

Both read only bars up to the evaluation index, matched to the rest of ``aes/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .boxes import REQUIRED_COLUMNS
from .params import ResistanceParams


@dataclass(frozen=True, slots=True)
class ResistanceLevel:
    """One clustered resistance level, as seen from an evaluation bar."""

    price: float                 # cluster's representative level (its highest touch)
    first_touch_idx: int
    last_touch_idx: int
    touches: tuple[int, ...]     # bar indices of every touch in the cluster
    age_bars: int                # bars since the *first* touch -- section 3.2
    age_strength: float          # 0-1, saturating with age
    #: Per-touch reaction (section 3.3), aligned with ``touches``.
    touch_volume_ratio: tuple[float, ...]
    touch_reaction: tuple[str, ...]     # "rejected" | "absorbed" | "unresolved"

    def elevated_volume_touches(self, mult: float) -> int:
        """Touches where volume was at least ``mult`` times its own baseline."""
        return int(sum(1 for r in self.touch_volume_ratio if r == r and r >= mult))

    @property
    def rejected_count(self) -> int:
        return int(sum(1 for r in self.touch_reaction if r == "rejected"))

    @property
    def absorbed_count(self) -> int:
        return int(sum(1 for r in self.touch_reaction if r == "absorbed"))


def _swing_high_indices(high: np.ndarray, k: int) -> np.ndarray:
    """Mirror of ``boxes.swing_low_indices`` for highs.

    Kept local rather than shared, because the two are not quite symmetric: a
    swing high should be strict on both neighbours (a plateau's *first* peak
    still counts as resistance the moment it forms), which is the opposite
    convention from the swing-low plateau rule pinned in
    ``test_swing_low_plateau_convention``. Getting this backward would silently
    duplicate resistance levels on flat-topped consolidations.
    """
    n = high.size
    if n < 2 * k + 1:
        return np.empty(0, dtype=np.int64)
    found: list[int] = []
    for j in range(k, n - k):
        left = high[j - k : j]
        right = high[j + 1 : j + k + 1]
        if high[j] > left.max() and high[j] >= right.max():
            found.append(j)
    return np.asarray(found, dtype=np.int64)


def _classify_reaction(
    close: np.ndarray, touch_idx: int, params: ResistanceParams
) -> str:
    """What happened after a touch: rejected, absorbed, or not yet resolved."""
    start = touch_idx + 1
    stop = min(touch_idx + params.reaction_window, close.size - 1)
    if stop < start:
        return "unresolved"
    window = close[start : stop + 1]
    touch_price = close[touch_idx]
    if touch_price <= 0:
        return "unresolved"
    if window.min() <= touch_price * (1.0 - params.rejection_drop_pct):
        return "rejected"
    return "absorbed"


def find_resistance_levels(
    df: pd.DataFrame,
    i: int,
    *,
    reference_price: float | None = None,
    params: ResistanceParams | None = None,
) -> list[ResistanceLevel]:
    """Resistance levels above ``reference_price``, as of bar ``i``.

    Args:
        df: OHLCV frame, ascending date index.
        i: Evaluation bar. Only bars ``[0 .. i]`` are read.
        reference_price: Price to measure headroom from. Defaults to the
            close at ``i``.
        params: Thresholds.

    Returns:
        Levels within ``max_headroom_pct`` above the reference price, nearest
        first, each carrying age strength and per-touch reaction history.
    """
    params = params or ResistanceParams()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")

    start = max(0, i - params.lookback_bars + 1)
    high = df["high"].to_numpy(dtype=float)[start : i + 1]
    close = df["close"].to_numpy(dtype=float)[start : i + 1]
    volume = df["volume"].to_numpy(dtype=float)[start : i + 1]
    if high.size < 2 * params.swing_k + 1:
        return []

    ref = float(close[-1]) if reference_price is None else float(reference_price)
    if ref <= 0:
        return []

    swings = _swing_high_indices(high, params.swing_k)
    swings = swings[high[swings] > ref]
    swings = swings[high[swings] <= ref * (1.0 + params.max_headroom_pct)]
    if swings.size == 0:
        return []
    swings = swings[np.argsort(high[swings])]   # ascending price -> cluster low to high

    clusters: list[list[int]] = []
    for idx in swings:
        price = high[idx]
        placed = False
        for cluster in clusters:
            cluster_price = high[cluster[-1]]
            if abs(price - cluster_price) / cluster_price <= params.cluster_tol_pct:
                cluster.append(int(idx))
                placed = True
                break
        if not placed:
            clusters.append([int(idx)])

    levels: list[ResistanceLevel] = []
    for cluster in clusters:
        cluster.sort()
        touches = _dedupe_touches(cluster, params.min_touch_separation)
        if not touches:
            continue
        level_price = float(high[touches].max())
        first_local, last_local = touches[0], touches[-1]

        vol_ratios: list[float] = []
        reactions: list[str] = []
        for t in touches:
            baseline_start = max(0, t - params.touch_vol_lookback)
            baseline = volume[baseline_start:t]
            ratio = float(volume[t] / baseline.mean()) if baseline.size and baseline.mean() > 0 else np.nan
            vol_ratios.append(ratio)
            reactions.append(_classify_reaction(close, t, params))

        age_bars = int((i - start) - first_local)
        age_strength = float(min(1.0, np.log1p(age_bars) / np.log1p(params.age_saturation_bars)))

        levels.append(
            ResistanceLevel(
                price=level_price,
                first_touch_idx=start + first_local,
                last_touch_idx=start + last_local,
                touches=tuple(start + t for t in touches),
                age_bars=age_bars,
                age_strength=age_strength,
                touch_volume_ratio=tuple(vol_ratios),
                touch_reaction=tuple(reactions),
            )
        )

    levels.sort(key=lambda lv: lv.price)
    return levels


def _dedupe_touches(sorted_idx: list[int], separation: int) -> list[int]:
    """Collapse touches within ``separation`` bars into their first occurrence.

    Mirrors ``boxes._count_separated``'s intent: several consecutive bars
    poking the same level are one visit, not several.
    """
    if not sorted_idx:
        return []
    kept = [sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if idx - kept[-1] >= separation:
            kept.append(idx)
    return kept


def nearest_resistance(
    df: pd.DataFrame,
    i: int,
    *,
    reference_price: float | None = None,
    params: ResistanceParams | None = None,
) -> ResistanceLevel | None:
    """The single nearest qualifying resistance level above the reference price."""
    levels = find_resistance_levels(df, i, reference_price=reference_price, params=params)
    return levels[0] if levels else None


def resistance_headroom(
    df: pd.DataFrame,
    i: int,
    *,
    reference_price: float | None = None,
    params: ResistanceParams | None = None,
) -> dict[str, Any]:
    """Section 3.2's headroom check, as a flat dict for feature frames.

    Args:
        df: OHLCV frame.
        i: Evaluation bar.
        reference_price: Entry reference (e.g. the big-box bottom for a
            box-bottom entry). Defaults to the close at ``i``.
        params: Thresholds.

    Returns:
        ``headroom_pct`` (``None`` if no level found within range -- read as
        "no nearby resistance", a positive fact, not a missing one),
        ``clears_min_headroom``, the level's ``age_bars``/``age_strength``, and
        the section 3.3 reaction summary for that level.
    """
    params = params or ResistanceParams()
    ref = float(df["close"].iloc[i]) if reference_price is None else float(reference_price)
    level = nearest_resistance(df, i, reference_price=ref, params=params)
    if level is None:
        return {
            "headroom_pct": None,
            "clears_min_headroom": None,
            "resistance_age_bars": None,
            "resistance_age_strength": None,
            "resistance_touches": 0,
            "resistance_rejected_count": 0,
            "resistance_absorbed_count": 0,
        }
    headroom_pct = (level.price - ref) / ref
    return {
        "headroom_pct": float(headroom_pct),
        "clears_min_headroom": bool(headroom_pct >= params.min_headroom_pct),
        "resistance_age_bars": level.age_bars,
        "resistance_age_strength": level.age_strength,
        "resistance_touches": len(level.touches),
        "resistance_rejected_count": level.rejected_count,
        "resistance_absorbed_count": level.absorbed_count,
    }


__all__ = [
    "ResistanceLevel",
    "find_resistance_levels",
    "nearest_resistance",
    "resistance_headroom",
]
