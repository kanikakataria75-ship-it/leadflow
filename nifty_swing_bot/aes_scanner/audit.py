"""Why every symbol in the universe ended up where it did.

The scan itself only reports names that produced a box, which is the short
list worth looking at -- but it means the other ~95% of the universe is
dropped silently, and a silent drop cannot be checked. This module walks the
same pipeline and records a verdict and a *reason* for every symbol, so the
ranking can be audited by hand rather than taken on trust.

The reason strings are deliberately specific. "no box" is not a reason; "every
anchor was too wide -- best was 30 bars at 34.2% against a 15-25% band" is,
because it tells you whether the detector or the band is the thing that is
wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..aes.boxes import _frame_arrays, _measure, _trim_isolated_extreme, detect_nested_box
from ..aes.params import BoxParams, ScoreParams, ScreenerParams, WatchlistParams

#: Ordered worst-to-best, so a sort on this column reads as a funnel.
STAGES = [
    "insufficient_history",
    "never_admitted",
    "watch_expired",
    "hard_deleted",
    "no_box",
    "scored_reject",
    "wait_and_watch",
    "high_conviction",
]

VERDICT = {
    "insufficient_history": "REJECTED",
    "never_admitted": "REJECTED",
    "watch_expired": "REJECTED",
    "hard_deleted": "REJECTED",
    "no_box": "REJECTED",
    "scored_reject": "REJECTED",
    "wait_and_watch": "WAIT & WATCH",
    "high_conviction": "ACCEPTED",
}


@dataclass(frozen=True, slots=True)
class BoxDiagnosis:
    """Why ``detect_big_box`` returned nothing, in the detector's own terms."""

    anchors_tried: int
    reason: str
    best_bars: int | None = None
    best_range_pct: float | None = None


def diagnose_big_box(df: pd.DataFrame, i: int, params: BoxParams) -> BoxDiagnosis:
    """Re-walk the anchor loop, counting which constraint rejected each anchor.

    Mirrors ``aes.boxes.detect_big_box`` exactly. It is a duplicate of that
    loop, which is a real cost -- but the alternative is threading a diagnostic
    channel through the detector itself, and the detector is settled code that
    the research results depend on. A test asserts the two agree on whether a
    box exists.
    """
    arr = _frame_arrays(df)
    if i < params.big_min_bars - 1:
        return BoxDiagnosis(0, "not enough bars to form the shortest allowed box")

    if params.edge_basis == "body":
        upper = np.maximum(arr["open"], arr["close"])
        lower = np.minimum(arr["open"], arr["close"])
    else:
        upper, lower = arr["high"], arr["low"]

    fails: dict[str, int] = {}
    tried = 0
    widest: tuple[float, int] | None = None
    tightest: tuple[float, int] | None = None
    seen: set[int] = set()

    for lookback in params.anchor_lookbacks:
        window_start = max(0, i - lookback + 1)
        if i - window_start + 1 < params.big_min_bars:
            continue
        start = window_start + int(np.argmax(upper[window_start : i + 1]))
        if start in seen:
            continue
        seen.add(start)
        tried += 1
        bars = i - start + 1
        if bars < params.big_min_bars:
            fails["too_short"] = fails.get("too_short", 0) + 1
            continue
        if bars > params.big_max_bars:
            fails["too_long"] = fails.get("too_long", 0) + 1
            continue
        high = upper[start : i + 1]
        low = lower[start : i + 1]
        close = arr["close"][start : i + 1]
        top = float(high.max())
        bottom = _trim_isolated_extreme(
            low, "min", params.isolated_wick_gap_pct, params.max_isolated_trim
        )
        if bottom <= 0 or top <= bottom:
            fails["degenerate"] = fails.get("degenerate", 0) + 1
            continue
        rng = (top - bottom) / bottom
        if rng > params.big_range_max:
            fails["too_wide"] = fails.get("too_wide", 0) + 1
            if widest is None or rng > widest[0]:
                widest = (rng, bars)
            continue
        if rng < params.big_range_min:
            fails["too_tight"] = fails.get("too_tight", 0) + 1
            if tightest is None or rng < tightest[0]:
                tightest = (rng, bars)
            continue
        m = _measure("big", high, low, close, top, bottom, params)
        if not m:
            fails["unmeasurable"] = fails.get("unmeasurable", 0) + 1
            continue
        if m["bars_since_top"] < params.min_bars_since_top:
            fails["top_is_today"] = fails.get("top_is_today", 0) + 1
            continue
        return BoxDiagnosis(tried, "a valid box exists", bars, rng * 100)

    if not tried:
        return BoxDiagnosis(0, "no usable anchor window")
    lo, hi = params.big_range_min * 100, params.big_range_max * 100
    top_fail = max(fails, key=lambda k: fails[k])
    detail = {
        "too_wide": (f"every anchor was too wide for the {lo:.0f}-{hi:.0f}% band"
                     + (f" -- widest {widest[0]*100:.1f}% over {widest[1]} bars" if widest else "")),
        "too_tight": (f"every anchor was too tight for the {lo:.0f}-{hi:.0f}% band"
                      + (f" -- tightest {tightest[0]*100:.1f}% over {tightest[1]} bars" if tightest else "")),
        "too_long": f"the consolidation runs longer than the {params.big_max_bars}-bar maximum",
        "too_short": f"shorter than the {params.big_min_bars}-bar minimum",
        "top_is_today": "the box top is today's own high -- nothing has stalled under it yet",
        "degenerate": "no usable high/low separation",
        "unmeasurable": "the window could not be measured",
    }.get(top_fail, top_fail)
    best_r = widest[0] * 100 if widest else (tightest[0] * 100 if tightest else None)
    best_b = widest[1] if widest else (tightest[1] if tightest else None)
    return BoxDiagnosis(tried, detail, best_b, best_r)


def _screen_detail(df: pd.DataFrame, i: int, p: ScreenerParams) -> tuple[bool, list[str], dict]:
    """Which of the four section-1.1 conditions pass on bar ``i``, and by how much."""
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    px = float(c[i])
    fails: list[str] = []

    moved = None
    if i >= p.move_lookback:
        moved = px / float(c[i - p.move_lookback]) - 1.0
        if moved < p.move_pct:
            fails.append(f"only {moved*100:+.1f}% in {p.move_lookback} sessions "
                         f"(needs {p.move_pct*100:.0f}%)")
    else:
        fails.append("not enough history for the move test")

    lo = max(0, i - p.high_lookback + 1)
    h52 = float(c[lo : i + 1].max())
    # How far price sits BELOW the high, as a share of the high -- so it is
    # bounded by 100%. ``h52/px - 1`` is the other direction (how far the high
    # is above price) and is unbounded, which produced nonsense like "131%
    # below the 52-week high".
    below = (1.0 - px / h52) * 100.0 if h52 > 0 else 0.0
    if px < h52 * (1 - p.high_proximity):
        fails.append(f"{below:.1f}% below its 52-week high "
                     f"(needs to be within {p.high_proximity*100:.0f}%)")
    if px <= p.min_price:
        fails.append(f"price Rs {px:.1f} is at or below the Rs {p.min_price:.0f} floor")
    tv = px * float(v[max(0, i - p.turnover_lookback + 1) : i + 1].mean())
    if tv < p.min_turnover_inr:
        fails.append(f"20-day turnover Rs {tv/1e7:.2f} cr is below the "
                     f"Rs {p.min_turnover_inr/1e7:.2f} cr floor")
    return (not fails), fails, {"move_10d_pct": None if moved is None else round(moved * 100, 1),
                                "below_52wh_pct": round(below, 1),
                                "turnover_cr": round(tv / 1e7, 2), "price": round(px, 2)}



def _plain_score_reason(r, params) -> str:
    """Say what is actually missing, in the terms a trader would use.

    "zero on fast resolution" names a variable. What a human needs is the fact
    behind it: the stock qualified 137 days ago and still has not broken out.
    Every clause below is the fact, with the threshold it missed.
    """
    def g(key, default=None):
        v = r.get(key, default)
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
        except TypeError:
            pass
        return v

    score = float(g("score", 0.0) or 0.0)
    bars, rng = g("box_bars"), g("box_range_pct")
    pos = g("box_position")
    gap = g("admission_gap_days")
    cycles = g("prior_cycles")
    absorbed = g("resistance_absorbed")
    cap = g("rs_capture")
    hl = g("higher_lows")
    above_mid = g("pct_closes_above_mid")

    # What it has going for it.
    good: list[str] = []
    if bars and rng:
        where = ("near the floor" if pos is not None and pos < 0.25 else
                 "near the top" if pos is not None and pos > 0.7 else "mid-range")
        good.append(f"a valid {int(bars)}-session base {rng:.0f}% deep, price {where}")
    if hl:
        good.append(f"{int(hl)} higher low(s)")
    if above_mid is not None and above_mid >= 50:
        good.append(f"{above_mid:.0f}% of closes in the upper half")

    # What is missing, with the fact and the bar it missed.
    miss: list[str] = []
    fast_days = params.fast_resolution_days
    if float(g("sc_fast", 0) or 0) <= 0:
        if gap is None:
            miss.append("it has not broken out yet, so there is no breakout to time")
        else:
            miss.append(f"it qualified on the screener {int(gap)} days ago and still has not "
                        f"broken out (a setup only counts as 'fast' inside {fast_days} days) "
                        f"-- this is drifting, not coiling")
    if float(g("sc_cycles", 0) or 0) <= 0:
        miss.append(f"no completed accumulation-to-expansion cycle anywhere in its history "
                    f"(wants {params.prior_cycles_cap}) -- this pattern has never worked on "
                    f"this name before")
    elif cycles is not None and cycles < params.prior_cycles_cap:
        miss.append(f"only {int(cycles)} prior cycle (wants {params.prior_cycles_cap})")
    if float(g("sc_absorbed", 0) or 0) <= 0:
        miss.append(f"price has not chewed through any overhead supply yet "
                    f"({int(absorbed or 0)} absorbed levels, wants {params.absorbed_count_cap})")
    if float(g("sc_rs", 0) or 0) <= 0.34:
        if cap is None:
            miss.append("relative strength could not be measured")
        else:
            miss.append(f"on the days the index fell, this fell {cap:.2f}x as much -- "
                        f"no relative strength (wants 0.70x or better)")

    head = (f"Score {score:.2f} of 1.00, needs 0.33 to be tradeable. "
            if score < 0.33 else f"Score {score:.2f} of 1.00 -- tradeable, but not high conviction "
                                 f"(needs 0.66). ")
    body = ""
    if good:
        body += "Has: " + "; ".join(good) + ". "
    if miss:
        body += "Missing: " + "; ".join(miss) + "."
    return (head + body).strip()


def _plain_delete_reason(df, i, p) -> str:
    """Which of section 1.2's three hard-delete conditions actually fired."""
    close = float(df["close"].iloc[i])
    why = []
    if close < p.delete_below_price:
        why.append(f"price Rs {close:.0f} is under the Rs {p.delete_below_price:.0f} floor "
                   f"-- too cheap to swing-trade cleanly")
    if close > p.delete_above_price:
        why.append(f"price Rs {close:.0f} is over the Rs {p.delete_above_price:.0f} ceiling")
    tv = close * float(df["volume"].iloc[max(0, i - p.turnover_lookback + 1): i + 1].mean())
    if tv < p.min_turnover_inr:
        why.append(f"turnover has stayed under Rs {p.min_turnover_inr/1e7:.1f} cr for "
                   f"{p.illiquid_sustained_bars}+ sessions (now Rs {tv/1e7:.2f} cr) "
                   f"-- you would move the price getting in")
    if not why:
        why.append(f"it hits circuit limits too often (more than {p.max_circuit_hits} times) "
                   f"-- gaps make the stop unreliable")
    return "Removed from the watchlist for good: " + "; ".join(why) + "."


def _screen_prose(fails: list[str]) -> list[str]:
    """Turn the raw condition failures into a sentence with a verdict."""
    out = [f"{f[0].upper()}{f[1:]}." for f in fails[:3]]
    joined = " ".join(fails).lower()
    if "sessions (needs" in joined and "below the 52-week high" in joined:
        out.append("It is neither moving nor near its highs -- there is no breakout story here.")
    elif "below the 52-week high" in joined:
        out.append("It is moving, but too far below its highs to be an accumulation setup.")
    elif "sessions (needs" in joined:
        out.append("It is near its highs but has not made the sharp move the screener looks for.")
    return out


def audit_universe(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    scanned: pd.DataFrame,
    *,
    screener_params: ScreenerParams | None = None,
    box_params: BoxParams | None = None,
    watch_params: WatchlistParams | None = None,
    score_params: ScoreParams | None = None,
    min_history: int | None = None,
) -> pd.DataFrame:
    """One row per universe symbol: stage, verdict, and a specific reason.

    Args:
        frames: The same ``{symbol: OHLCV}`` the scan ran on.
        index_df: Benchmark, unused here but kept for signature symmetry.
        scanned: ``ScanResult.candidates`` -- the names that reached scoring.

    Returns:
        A frame ordered worst-to-best by pipeline stage.
    """
    SP = screener_params or ScreenerParams()
    BP = box_params or BoxParams()
    WP = watch_params or WatchlistParams()
    #: Only the normalisation caps are read (how many prior cycles or absorbed
    #: levels count as "full marks"), so the weights are irrelevant here.
    SCP = score_params or ScoreParams()
    min_hist = min_history if min_history is not None else BP.min_history_bars

    from ..aes.screener import hard_delete_mask, screen

    scored = scanned.set_index("symbol") if scanned is not None and len(scanned) else pd.DataFrame()
    rows: list[dict[str, Any]] = []

    for sym in sorted(frames):
        df = frames[sym]
        rec: dict[str, Any] = {"symbol": sym}

        if df is None or len(df) < min_hist:
            rows.append({**rec, "stage": "insufficient_history",
                         "reason": f"Too new to judge: only "
                                   f"{0 if df is None else len(df)} sessions of price history, "
                                   f"and the 52-week-high test alone needs {min_hist}."})
            continue

        i = len(df) - 1
        adm = screen(df, SP)
        hits = np.flatnonzero(adm.to_numpy())
        ok_today, fails, nums = _screen_detail(df, i, SP)
        rec.update(nums)

        # --- already scored? then the reason is the score itself ----------- #
        if len(scored) and sym in scored.index:
            r = scored.loc[sym]
            bucket = str(r["bucket"])
            why = _plain_score_reason(r, SCP)
            if bucket == "reject":
                stage = "scored_reject"
            else:
                stage = bucket
                why += (" An entry trigger has fired -- it is actionable now."
                        if int(r.get("actionable") or 0)
                        else " No entry trigger yet: it is still inside the base.")
            rows.append({**rec, "stage": stage, "reason": why,
                         "score": round(float(r["score"]), 3),
                         "sc_fast": r.get("sc_fast"), "sc_absorbed": r.get("sc_absorbed"),
                         "sc_rs": r.get("sc_rs"), "sc_cycles": r.get("sc_cycles"),
                         "box_bars": r.get("box_bars"), "box_range_pct": r.get("box_range_pct"),
                         "box_position": r.get("box_position"),
                         "higher_lows": r.get("higher_lows"),
                         "prior_cycles": r.get("prior_cycles"),
                         "resistance_absorbed": r.get("resistance_absorbed"),
                         "rs_capture": r.get("rs_capture"),
                         "actionable": int(r.get("actionable") or 0),
                         "qty": r.get("qty")})
            continue

        # --- not scored: find where it fell out ---------------------------- #
        if not len(hits):
            rows.append({**rec, "stage": "never_admitted",
                         "reason": "Never got onto the watchlist. " + " ".join(
                             _screen_prose(fails))})
            continue

        last_adm = int(hits[-1])
        age = i - last_adm
        rec["last_admission"] = df.index[last_adm].strftime("%Y-%m-%d")
        rec["admission_age_bars"] = int(age)

        if bool(hard_delete_mask(df, SP).iloc[i]):
            rows.append({**rec, "stage": "hard_deleted",
                         "reason": _plain_delete_reason(df, i, SP)})
            continue
        if age > WP.max_watch_without_box:
            rows.append({**rec, "stage": "watch_expired",
                         "reason": f"Was on the watchlist but timed out. It last qualified on "
                                   f"{df.index[last_adm]:%d %b %Y} ({age} sessions ago) and never "
                                   f"formed a tradeable base in the {WP.max_watch_without_box} "
                                   f"sessions after, so it was dropped. To come back it needs a "
                                   f"fresh +20% move in 10 sessions near its 52-week high."})
            continue

        d = diagnose_big_box(df, i, BP)
        nested = detect_nested_box(df, i, BP, symbol=sym)
        if nested is None:
            rows.append({**rec, "stage": "no_box",
                         "reason": ("On the watchlist and still being checked daily, but there is "
                                    "no tradeable base yet: " + d.reason
                                    + ". Until price settles into a 15-25% range for 10-45 "
                                      "sessions there is nothing to buy a breakout out of."),
                         "box_bars": d.best_bars,
                         "box_range_pct": None if d.best_range_pct is None else round(d.best_range_pct, 1)})
            continue
        rows.append({**rec, "stage": "no_box",
                     "reason": ("A wide base formed but the tight coil inside it did not -- the "
                                "last few weeks are still too loose to call it a spring.")})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["verdict"] = out["stage"].map(VERDICT)
    out["stage_rank"] = out["stage"].map({s: n for n, s in enumerate(STAGES)})
    # On a universe where nothing reaches scoring there is no ``score`` column
    # at all, and sorting by a missing label raises. Declare the optional
    # columns so the frame has one shape regardless of how far the funnel got.
    for col in ("score", "sc_fast", "sc_absorbed", "sc_rs", "sc_cycles",
                "box_bars", "box_range_pct", "box_position", "actionable", "qty"):
        if col not in out.columns:
            out[col] = None
    return out.sort_values(["stage_rank", "score"], ascending=[False, False],
                           na_position="last").reset_index(drop=True)


__all__ = ["STAGES", "VERDICT", "BoxDiagnosis", "audit_universe", "diagnose_big_box"]
