"""Phase 24b: does multi-day compression before a stop exit predict a missed run?

Diagnostic only. Nothing is adopted, nothing is compared against a baseline.

THE DEFINITION, FIXED HERE BEFORE THE RUN
=========================================
The pattern described is: price hugging near the stop level for several
sessions -- tight range, small candle bodies, repeated touches -- and then a
marginal close below it, as opposed to a decisive break away from the average.
That is four separate claims, so four things are measured, all in units of the
name's own **ATR(14) on the exit bar**. Normalising by ATR is the whole point:
a 1% drift is hugging for a 4%-ATR name and a decisive break for a 0.5%-ATR
name, and an unnormalised threshold would silently sort names by volatility
instead of by behaviour.

Window ``W`` = the **8 sessions ending on the bar before the exit bar**. The
exit bar itself is excluded: the hug is supposed to *precede* the break, and
including the breaking bar would let the break contaminate the measure of the
thing that came before it. W=5 and W=10 are reported as robustness.

    prox_frac   fraction of the W closes with |close - stop| <= 1.0 x ATR14
    range_atr   (max high - min low) over W, divided by ATR14
    body_atr    mean |close - open| over W, divided by ATR14
    touches     bars in W whose [low, high] straddles the stop level

Why those constants, set a priori from what ATR means rather than tuned:

* **1.0 ATR proximity.** One ATR is by construction the name's typical daily
  travel. A close more than a full day's range away from the stop is not
  hugging it, in any name.
* **5.0 ATR for an 8-bar range.** A name travelling its full ATR in one
  direction every bar would cover ~8 ATR over the window. 5 is a deliberately
  generous "ordinary" reference, so the range term only saturates for windows
  that are genuinely wide rather than merely normal.
* **1.0 ATR body.** A full-ATR candle body is a decisive session. Indecision
  candles are a fraction of it.

The three terms are combined into one score, each clipped to [0, 1] so no
single term can dominate by scale:

    prox_term  = prox_frac
    range_term = clip(1 - range_atr / 5.0, 0, 1)
    body_term  = clip(1 - body_atr  / 1.0, 0, 1)
    score      = (prox_term + range_term + body_term) / 3

Three cuts are reported, and all three are declared here so none of them is a
threshold found after seeing the answer:

1. **PRIMARY -- median split** of ``score`` within each window's own
   population. Threshold-free, and keeps both groups at ~43, above the n=30
   floor this project reports below.
2. **STRICT** -- the literal reading of the description, all three at once:
   ``prox_frac >= 0.6 AND range_atr <= 2.5 AND body_atr <= 0.6``. Faithful but
   expected to be small; its n is reported and read accordingly.
3. **QUARTILE** -- top vs bottom quartile of ``score``, a sharper contrast at
   n~21. Below the floor, so reported and labelled inconclusive.

Which stop this runs on
-----------------------
The live configuration stops on the **ATR(14) x 2.5 trail**; the SMA(10) close
stop is unreachable under it (PROVEN #1). The population here is therefore the
ATR-trail exits. ``stop_level`` for each exit is the trail level the engine
actually breached, recomputed in ``stop_forensics.harvest_exits``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from ..strategy.indicators import atr
from .stop_forensics import HORIZONS, THRESHOLDS, describe

WINDOWS_W = (5, 8, 10)
PRIMARY_W = 8

PROX_ATR = 1.0
RANGE_REF_ATR = 5.0
BODY_REF_ATR = 1.0

STRICT_PROX = 0.60
STRICT_RANGE = 2.50
STRICT_BODY = 0.60


def compression_features(
    df: pd.DataFrame, exit_idx: int, stop_level: float, atr_val: float, w: int
) -> dict[str, float]:
    """The four measures over the ``w`` bars ending BEFORE ``exit_idx``."""
    lo = exit_idx - w
    if lo < 0 or not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(stop_level):
        return {}
    seg = df.iloc[lo:exit_idx]
    if len(seg) < w:
        return {}
    c = seg["close"].to_numpy(float)
    o = seg["open"].to_numpy(float)
    h = seg["high"].to_numpy(float)
    l = seg["low"].to_numpy(float)

    prox_frac = float((np.abs(c - stop_level) <= PROX_ATR * atr_val).mean())
    range_atr = float((h.max() - l.min()) / atr_val)
    body_atr = float(np.abs(c - o).mean() / atr_val)
    touches = int(((l <= stop_level) & (h >= stop_level)).sum())

    prox_term = min(max(prox_frac, 0.0), 1.0)
    range_term = min(max(1.0 - range_atr / RANGE_REF_ATR, 0.0), 1.0)
    body_term = min(max(1.0 - body_atr / BODY_REF_ATR, 0.0), 1.0)
    return {
        "prox_frac": round(prox_frac, 4),
        "range_atr": round(range_atr, 4),
        "body_atr": round(body_atr, 4),
        "touches": touches,
        "comp_score": round((prox_term + range_term + body_term) / 3.0, 4),
        "comp_strict": bool(prox_frac >= STRICT_PROX
                       and range_atr <= STRICT_RANGE
                       and body_atr <= STRICT_BODY),
    }


def attach(ex: pd.DataFrame, frames: dict[str, pd.DataFrame], w: int) -> pd.DataFrame:
    """Add the compression measures to a harvested exit table."""
    rows = []
    for r in ex.itertuples():
        df = frames.get(r.symbol)
        if df is None:
            rows.append({})
            continue
        ed = pd.Timestamp(r.exit_date)
        if ed not in df.index:
            rows.append({})
            continue
        i = int(df.index.get_loc(ed))
        rows.append(compression_features(df, i, float(r.stop_level), float(r.atr14), w))
    feats = pd.DataFrame(rows, index=ex.index)
    dupes = [c for c in feats.columns if c in ex.columns]
    if dupes:
        raise ValueError(f'compression columns collide with exit table: {dupes}')
    return pd.concat([ex, feats], axis=1)


def cuts(ex: pd.DataFrame) -> dict[str, Any]:
    """The three declared splits, each with its own sample sizes."""
    ok = ex[ex["comp_score"].notna()].copy()
    out: dict[str, Any] = {
        "n_total": int(len(ex)),
        "n_measurable": int(len(ok)),
        "comp_score_median": round(float(ok["comp_score"].median()), 4) if len(ok) else None,
    }
    if ok.empty:
        return out

    med = float(ok["comp_score"].median())
    out["primary_median_split"] = {
        "compressed_upper_half": describe(ok[ok["comp_score"] >= med]),
        "decisive_lower_half": describe(ok[ok["comp_score"] < med]),
    }
    out["strict"] = {
        "compressed_strict": describe(ok[ok["comp_strict"].astype(bool)]),
        "everything_else": describe(ok[~ok["comp_strict"].astype(bool)]),
        "rule": (f"prox>={STRICT_PROX} AND range<={STRICT_RANGE}atr "
                 f"AND body<={STRICT_BODY}atr"),
    }
    q1, q3 = ok["comp_score"].quantile(0.25), ok["comp_score"].quantile(0.75)
    out["quartile"] = {
        "top_quartile_compressed": describe(ok[ok["comp_score"] >= q3]),
        "bottom_quartile_decisive": describe(ok[ok["comp_score"] <= q1]),
    }
    out["feature_medians"] = {
        k: round(float(ok[k].median()), 4)
        for k in ("prox_frac", "range_atr", "body_atr", "touches")
    }
    return out


def real_trade_stop_attribution(
    real: list[dict[str, Any]], atr_mult: float = 2.5, sma_period: int = 10,
    atr_period: int = 14, max_bars: int = 60,
) -> pd.DataFrame:
    """Which stop would fire FIRST on each real trade, and on what bar.

    Answers "which mechanism am I actually describing" without needing the
    portfolio simulator: walk each trade forward from its own entry bar and
    record the first bar on which each rule would have closed the position.
    """
    rows = []
    for t in real:
        df, i0 = t["df"], int(t["idx"])
        if i0 + 2 >= len(df):
            continue
        close = df["close"]
        sma = close.rolling(sma_period).mean()
        a = atr(df["high"], df["low"], close, atr_period)
        entry = float(df["open"].iloc[i0 + 1]) if i0 + 1 < len(df) else float(close.iloc[i0])
        peak = entry
        sma_bar = atr_bar = None
        for k in range(i0 + 1, min(len(df), i0 + 1 + max_bars)):
            c = float(close.iloc[k])
            peak = max(peak, c)
            s = float(sma.iloc[k]) if np.isfinite(sma.iloc[k]) else None
            av = float(a.iloc[k]) if np.isfinite(a.iloc[k]) else None
            if sma_bar is None and s is not None and c < s:
                sma_bar = k - i0
            if atr_bar is None and av is not None and c < peak - atr_mult * av:
                atr_bar = k - i0
            if sma_bar is not None and atr_bar is not None:
                break
        if sma_bar is None and atr_bar is None:
            first = "neither within 60 bars"
        elif atr_bar is None:
            first = "sma10"
        elif sma_bar is None:
            first = "atr_trail"
        else:
            first = "sma10" if sma_bar < atr_bar else (
                "atr_trail" if atr_bar < sma_bar else "same bar")
        rows.append({
            "stock": t["stock"], "ticker": t["ticker"], "entry_date": t["date"],
            "sma10_bar": sma_bar, "atr_trail_bar": atr_bar, "fires_first": first,
        })
    return pd.DataFrame(rows)


def main(out_name: str = "compression_forensics.json") -> dict[str, Any]:
    from ..config import get_config
    from .aes_calibration import load_universe
    from .geometry_sweep import load_real_trades

    cfg = get_config()
    results = str(cfg.paths.results_dir)
    frames = load_universe(str(cfg.paths.ohlcv_dir))

    res: dict[str, Any] = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "stop_under_test": "atr_trail (the live stop; SMA10 is unreachable under frozen config)",
        "definition": {
            "window_primary": PRIMARY_W,
            "prox_atr": PROX_ATR, "range_ref_atr": RANGE_REF_ATR, "body_ref_atr": BODY_REF_ATR,
            "strict_rule": (f"prox>={STRICT_PROX} AND range<={STRICT_RANGE}atr "
                            f"AND body<={STRICT_BODY}atr"),
        },
        "windows": {},
        "robustness": {},
    }

    for wname in ("2015-2021", "2022-2026"):
        path = os.path.join(results, f"stop_exits_{wname}_frozen_atr.csv")
        if not os.path.exists(path):
            continue
        ex = pd.read_csv(path)
        res["windows"][wname] = cuts(attach(ex, frames, PRIMARY_W))
        res["robustness"][wname] = {}
        for w in WINDOWS_W:
            if w == PRIMARY_W:
                continue
            c = cuts(attach(ex, frames, w))
            res["robustness"][wname][f"W{w}"] = {
                k: {kk: {"n": vv.get("n"),
                         "fwd30_mean": vv.get("h30", {}).get("mean_pct"),
                         "fwd30_median": vv.get("h30", {}).get("median_pct")}
                    for kk, vv in v.items() if isinstance(vv, dict) and "n" in vv}
                for k, v in c.items() if k in ("primary_median_split", "strict")
            }

    # real-trade stop attribution
    real = load_real_trades(
        os.path.join(str(cfg.paths.cache_dir), "recs"),
        os.path.join(results, "recs_mapping.csv"),
    )
    att = real_trade_stop_attribution(real)
    att.to_csv(os.path.join(results, "real_trade_stop_attribution.csv"), index=False)
    res["real_trade_attribution"] = {
        "n": int(len(att)),
        "fires_first": att["fires_first"].value_counts().to_dict(),
        "median_sma10_bar": float(att["sma10_bar"].dropna().median()) if att["sma10_bar"].notna().any() else None,
        "median_atr_trail_bar": float(att["atr_trail_bar"].dropna().median()) if att["atr_trail_bar"].notna().any() else None,
    }

    path = os.path.join(results, out_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    res["_path"] = path
    return res


__all__ = ["attach", "compression_features", "cuts", "main", "real_trade_stop_attribution"]
