"""Soft ranking: no hard rejects, every gate becomes a visible penalty.

Every binary gate in this pipeline has now been wrong at its own boundary.
Colgate's 8.8% base was the cleanest structure in the 41-trade set and the 15%
range floor threw it away; Radhika at Rs 95 was below the price floor and
returned +10.6%; across the five discretionary overrides the rule-breakers
beat the rule-followers (+2.30% vs +1.27%). A boundary that is wrong in both
directions is not a boundary, it is a preference expressed too strongly.

So nothing is dropped here. Each rule becomes a **penalty** -- zero inside the
rule, growing smoothly outside it -- and the output is one ranked list with
every penalty itemised, so a name that ranks low *because of one gate it
barely missed* is visible rather than absent.

Two deliberate constraints on the construction:

* **No fitted weights.** Penalties are summed, not weighted. Weighting them
  would be a new free parameter set on the same data that already produced an
  overfit score (Phase 6), and there is no evidence to set them from.
* **Tolerances come from the rule, not from a sweep.** A band's tolerance is
  its own width; a floor's tolerance is the floor. Being one band-width
  outside costs 1.0. Nothing here was chosen by looking at a return.

``rank_score = quality - sum(penalties)``, where ``quality`` is the same
equal-weight four-factor score the scanner already uses. The score is
unbounded below, which is fine: the job is to order, not to classify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..aes.boxes import BoxGeometry, _frame_arrays, _measure, _trim_isolated_extreme
from ..aes.params import (BoxParams, RelativeStrengthParams, ResistanceParams,
                          ScoreParams, ScreenerParams, WatchlistParams)
from ..aes.relative_strength import drawdown_relative_strength
from ..aes.resistance import resistance_headroom
from ..aes.screener import circuit_hits, screen
from ..aes.scoring import score_setup

EQUAL_WEIGHTS = ScoreParams(
    w_fast_resolution=0.25, w_absorbed=0.25, w_rs_capture=0.25, w_prior_cycles=0.25
)


#: No single rule may dominate the ordering. Two tolerances outside a rule,
#: the rule is simply failed and the exact magnitude stops carrying
#: information -- a name 1,100 sessions stale is not 12x worse than one 180
#: sessions stale, it is just stale. Stated, not tuned.
PENALTY_CAP = 2.0


@dataclass(frozen=True, slots=True)
class Penalty:
    """One rule, softened. ``value`` is 0 inside the rule, capped at ``PENALTY_CAP``."""

    name: str
    value: float
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(min(max(self.value, 0.0), PENALTY_CAP)))


def _band(x: float | None, lo: float, hi: float, name: str, unit: str = "",
          tol: float | None = None) -> Penalty:
    """Distance outside ``[lo, hi]``, normalised by the band's own width."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return Penalty(name, 1.0, f"{name}: not measurable")
    t = tol if tol is not None else max(hi - lo, 1e-9)
    if x < lo:
        v = (lo - x) / t
        return Penalty(name, v, f"{x:.1f}{unit} is below the {lo:.0f}-{hi:.0f}{unit} band")
    if x > hi:
        v = (x - hi) / t
        return Penalty(name, v, f"{x:.1f}{unit} is above the {lo:.0f}-{hi:.0f}{unit} band")
    return Penalty(name, 0.0, f"{x:.1f}{unit} inside {lo:.0f}-{hi:.0f}{unit}")


def _floor(x: float | None, lo: float, name: str, unit: str = "",
           tol: float | None = None) -> Penalty:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return Penalty(name, 1.0, f"{name}: not measurable")
    t = tol if tol is not None else max(abs(lo), 1e-9)
    if x < lo:
        return Penalty(name, (lo - x) / t, f"{x:,.1f}{unit} is under {lo:,.0f}{unit}")
    return Penalty(name, 0.0, f"{x:,.1f}{unit} clears {lo:,.0f}{unit}")


def _ceiling(x: float | None, hi: float, name: str, unit: str = "",
             tol: float | None = None) -> Penalty:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return Penalty(name, 1.0, f"{name}: not measurable")
    t = tol if tol is not None else max(abs(hi), 1e-9)
    if x > hi:
        return Penalty(name, (x - hi) / t, f"{x:,.1f}{unit} is over {hi:,.0f}{unit}")
    return Penalty(name, 0.0, f"{x:,.1f}{unit} under {hi:,.0f}{unit}")


def relaxed_big_box(df: pd.DataFrame, i: int, params: BoxParams) -> BoxGeometry | None:
    """The best consolidation ending at ``i``, *ignoring* the range and duration bands.

    Same anchor construction as ``aes.boxes.detect_big_box`` -- the running-max
    bar of each lookback becomes a candidate top -- but a candidate is no
    longer thrown away for being 8.8% deep or 90 bars long. Those become
    penalties instead, which is the whole point: you cannot mark a box down
    for being too tight if the detector refuses to show it to you.
    """
    arr = _frame_arrays(df)
    if i < params.big_min_bars - 1:
        return None
    if params.edge_basis == "body":
        upper = np.maximum(arr["open"], arr["close"])
        lower = np.minimum(arr["open"], arr["close"])
    else:
        upper, lower = arr["high"], arr["low"]

    best = None
    best_key = (-np.inf, -np.inf)
    seen: set[int] = set()
    for lookback in params.anchor_lookbacks:
        ws = max(0, i - lookback + 1)
        if i - ws + 1 < 4:                      # need *some* window to measure
            continue
        start = ws + int(np.argmax(upper[ws : i + 1]))
        if start in seen:
            continue
        seen.add(start)
        bars = i - start + 1
        if bars < 4:
            continue
        high, low = upper[start : i + 1], lower[start : i + 1]
        close = arr["close"][start : i + 1]
        top = float(high.max())
        bottom = _trim_isolated_extreme(
            low, "min", params.isolated_wick_gap_pct, params.max_isolated_trim)
        if bottom <= 0 or top <= bottom:
            continue
        m = _measure("big", high, low, close, top, bottom, params)
        if not m:
            continue
        key = (top, m["quality"])
        if key > best_key:
            best_key = key
            best = m | {"start_idx": start, "end_idx": i}
    if best is None:
        return None
    return BoxGeometry(start_date=df.index[best["start_idx"]],
                       end_date=df.index[best["end_idx"]], **best)


@dataclass
class RankedCandidate:
    """One symbol's soft rank, with every mark-down kept separate."""

    symbol: str
    rank_score: float
    quality: float
    penalties: list[Penalty] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def penalty_total(self) -> float:
        return float(sum(p.value for p in self.penalties))

    @property
    def hard_gated(self) -> bool:
        """Would the current pipeline have thrown this away outright?"""
        return any(p.value > 0 for p in self.penalties
                   if p.name in ("freshness", "near_52w_high", "price_floor",
                                 "price_ceiling", "turnover", "circuits",
                                 "box_range", "box_duration"))

    def as_row(self) -> dict[str, Any]:
        row = {"symbol": self.symbol, "rank_score": round(self.rank_score, 4),
               "quality": round(self.quality, 4),
               "penalty_total": round(self.penalty_total, 4),
               "hard_gated": int(self.hard_gated)}
        for p in self.penalties:
            row[f"pen_{p.name}"] = round(p.value, 3)
        row.update(self.facts)
        return row


def rank_symbol(
    symbol: str,
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    i: int | None = None,
    *,
    screener_params: ScreenerParams | None = None,
    box_params: BoxParams | None = None,
    watch_params: WatchlistParams | None = None,
    score_params: ScoreParams | None = None,
) -> RankedCandidate | None:
    """Score and penalise one symbol as of bar ``i`` (default: the last bar)."""
    SP = screener_params or ScreenerParams()
    BP = box_params or BoxParams()
    WP = watch_params or WatchlistParams()
    SCP = score_params or EQUAL_WEIGHTS
    if df is None or len(df) < 60:
        return None
    i = len(df) - 1 if i is None else i
    if i < 60:
        return None

    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    px = float(c[i])
    if not np.isfinite(px) or px <= 0:
        return None

    # --- screener rules, softened ------------------------------------------ #
    move = (px / float(c[i - SP.move_lookback]) - 1.0) * 100 if i >= SP.move_lookback else None
    h52 = float(c[max(0, i - SP.high_lookback + 1) : i + 1].max())
    below = (1.0 - px / h52) * 100 if h52 > 0 else None
    tv = px * float(v[max(0, i - SP.turnover_lookback + 1) : i + 1].mean())
    try:
        hits = int(circuit_hits(df.iloc[: i + 1], SP).iloc[i])
    except Exception:
        hits = 0

    # No same-day momentum penalty. Section 1.1's +20%/10-session move is an
    # *event* that starts a watch, not a state a name must hold: a stock
    # coiling 30 sessions after its spike is behaving exactly as the strategy
    # intends, and marking it down for not spiking again today would punish
    # the thesis. How long ago it last qualified is carried by ``freshness``
    # below, which is the same information without the double count.
    pens = [
        _ceiling(below, SP.high_proximity * 100, "near_52w_high", "%", tol=15.0),
        _floor(px, 100.0, "price_floor", " Rs", tol=100.0),
        _ceiling(px, 5000.0, "price_ceiling", " Rs", tol=5000.0),
        _floor(tv / 1e7, SP.min_turnover_inr / 1e7, "turnover", " cr",
               tol=SP.min_turnover_inr / 1e7),
        _ceiling(float(hits), 3.0, "circuits", "", tol=3.0),
    ]

    # --- how stale the last admission is ------------------------------------ #
    adm = screen(df.iloc[: i + 1], SP)
    idx = np.flatnonzero(adm.to_numpy())
    age = int(i - idx[-1]) if len(idx) else None
    if age is None:
        pens.append(Penalty("freshness", PENALTY_CAP,
                            "has never passed the section 1.1 screener"))
    else:
        pens.append(_ceiling(float(age), float(WP.max_watch_without_box),
                             "freshness", " sessions",
                             tol=float(WP.max_watch_without_box)))

    # --- geometry, measured without the bands ------------------------------- #
    big = relaxed_big_box(df.iloc[: i + 1], i, BP)
    if big is None:
        return None
    rng = float(big.range_pct) * 100
    pens.append(_band(rng, BP.big_range_min * 100, BP.big_range_max * 100,
                      "box_range", "%"))
    pens.append(_band(float(big.bars), float(BP.big_min_bars), float(BP.big_max_bars),
                      "box_duration", " bars", tol=float(BP.big_max_bars - BP.big_min_bars)))
    # Containment is a quality measure in [0,1]; the shortfall is the penalty.
    pens.append(Penalty("containment", float(max(0.0, 1.0 - big.close_containment)),
                        f"{big.close_containment*100:.0f}% of closes inside the box"))
    pos = (px - big.bottom) / (big.top - big.bottom) if big.top > big.bottom else None
    # Buying the very top of a base is a different, worse trade than buying
    # inside it; buying below the floor means the base has already broken.
    pen_pos = 0.0 if pos is None else (max(0.0, pos - 0.85) / 0.15 + max(0.0, -pos) / 0.15)
    pens.append(Penalty("box_position", float(pen_pos),
                        "outside the base" if pos is None or pos < 0 or pos > 1
                        else f"{pos:.2f} of the way up the base"))

    # --- the same four scoring factors the scanner uses --------------------- #
    res = resistance_headroom(df.iloc[: i + 1], i, reference_price=px, params=ResistanceParams())
    rs = drawdown_relative_strength(df.iloc[: i + 1], index_df, i, RelativeStrengthParams())
    gap = None if age is None else float(age)
    # ``prior_cycles`` needs the nested-box history, which the relaxed detector
    # does not build. It is passed as 0 for every symbol, so it is a constant
    # offset that cannot change the ordering -- but it does mean ``quality``
    # here tops out at 0.75, not 1.0, and is not comparable in level with the
    # scanner's own score.
    sc = score_setup(gap_days=gap, resistance_absorbed_count=res["resistance_absorbed_count"],
                     rs_capture=rs.downside_capture, prior_cycles=0, params=SCP,
                     rs_params=RelativeStrengthParams())

    facts = {
        "close": round(px, 2), "move_10d_pct": None if move is None else round(move, 1),
        "below_52wh_pct": None if below is None else round(below, 1),
        "turnover_cr": round(tv / 1e7, 2), "admission_age": age,
        "box_bars": int(big.bars), "box_range_pct": round(rng, 1),
        "box_position": None if pos is None else round(pos, 2),
        "higher_lows": int(big.higher_lows),
        "containment_pct": round(float(big.close_containment) * 100, 0),
        "rs_capture": None if rs.downside_capture is None else round(float(rs.downside_capture), 2),
        "absorbed": int(res["resistance_absorbed_count"]),
    }
    q = float(sc.score)
    total = float(sum(p.value for p in pens))
    return RankedCandidate(symbol=symbol, rank_score=q - total, quality=q,
                           penalties=pens, facts=facts)


def rank_universe(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    **kw: Any,
) -> pd.DataFrame:
    """Rank every symbol that has data, best first. Nothing is excluded."""
    rows = []
    for sym, df in frames.items():
        if df is None or df.empty:
            continue
        if as_of is not None:
            df = df.loc[:as_of]
            if len(df) < 60:
                continue
        try:
            rc = rank_symbol(sym, df, index_df, **kw)
        except Exception:
            continue
        if rc is not None:
            rows.append(rc.as_row())
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("rank_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


__all__ = ["EQUAL_WEIGHTS", "Penalty", "RankedCandidate", "rank_symbol",
           "rank_universe", "relaxed_big_box"]
