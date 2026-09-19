"""Phase 16: outcome-first. What precedes a +30%-in-25-sessions move?

Every prior phase in this project started from a stated rule and asked whether
it worked. This starts from the outcome and asks what preceded it, with a
matched control group so that "what runners had" is never confused with "what
separates runners from everything else".

Three deliberate choices, all consequential:

1. **The window is 2022-2026, which is AES_HOLDOUT.** This spends the holdout
   as discovery data. Authorised explicitly on the reasoning that 2015-2021 is
   exhausted and the traded market is this one. See ``aes.protocol`` --
   nothing may cite AES_HOLDOUT as clean out-of-sample again.

2. **Discovery reads only 75% of the universe.** The other 25%
   (``protocol.validation_symbols``) is held out by salted hash, fixed before
   any result was seen, because the temporal dimension is now used up in both
   directions and a cross-sectional slice is the only holdout left that exists
   today.

3. **Controls are matched on date and turnover bucket, not sampled freely.**
   Same calendar day means market regime is held constant by construction,
   which is the confound that would otherwise dominate: runners cluster in
   rallies, so an unmatched control group makes every bullish feature look
   discriminating.

Point-in-time market cap does not exist in this cache (``ScreenerParams.
market_cap_applied`` is False and records why), so turnover is the size proxy.
That is a real limitation: turnover conflates size with activity, and activity
is plausibly part of what precedes a run.

Usage::

    python -m nifty_swing_bot.research.runner_study
"""

from __future__ import annotations


import numpy as np
import pandas as pd

RUN_THRESHOLD = 0.30
RUN_WINDOW = 25
FEATURE_LOOKBACK = 40
MIN_HISTORY = 252  # a full year, so 52-week and 12-month features are real


# --------------------------------------------------------------------------- #
# 1. Runners
# --------------------------------------------------------------------------- #
def find_runners(
    frames: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    threshold: float = RUN_THRESHOLD,
    window: int = RUN_WINDOW,
) -> pd.DataFrame:
    """Every bar from which price gains ``threshold`` within ``window`` sessions.

    The recorded bar is the **last bar before the run** -- the close the move is
    measured from, so every feature computed at or before it is causal and an
    entry would be at the next open.

    Episodes are de-duplicated: once a run starts at ``t``, the next candidate
    for that symbol is not considered until the run window has elapsed.
    Without this a single 40% move contributes twenty overlapping "runners"
    and the sample becomes a handful of episodes counted many times.
    """
    rows = []
    for sym, df in frames.items():
        close = df["close"].to_numpy(dtype=float)
        n = len(close)
        if n < MIN_HISTORY + window + 1:
            continue
        # forward max close over the next `window` bars
        fwd = np.full(n, np.nan)
        for k in range(1, window + 1):
            shifted = np.concatenate([close[k:], np.full(k, np.nan)])
            fwd = np.fmax(fwd, shifted)
        gain = fwd / close - 1.0

        idx = df.index
        lo = int(idx.searchsorted(start))
        hi = int(idx.searchsorted(end, side="right"))
        lo = max(lo, MIN_HISTORY)
        t = lo
        while t < min(hi, n - 1):
            if np.isfinite(gain[t]) and gain[t] >= threshold:
                rows.append({
                    "symbol": sym, "idx": t, "date": idx[t],
                    "gain": float(gain[t]),
                    "bars_to_peak": int(np.nanargmax(
                        close[t + 1 : t + window + 1]) + 1),
                })
                t += window          # skip the run itself
            else:
                t += 1
    return pd.DataFrame(rows)


def mark_runner_bars(runners: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    return set(zip(runners["symbol"], runners["date"]))


# --------------------------------------------------------------------------- #
# 2. Matched controls
# --------------------------------------------------------------------------- #
def _turnover_panel(frames: dict[str, pd.DataFrame], lookback: int = 20) -> pd.DataFrame:
    """20-day mean turnover per symbol per date, as a wide panel."""
    cols = {}
    for sym, df in frames.items():
        to = (df["close"] * df["volume"]).rolling(lookback).mean()
        cols[sym] = to
    return pd.DataFrame(cols)


def build_controls(
    frames: dict[str, pd.DataFrame],
    runners: pd.DataFrame,
    *,
    threshold: float = RUN_THRESHOLD,
    window: int = RUN_WINDOW,
    per_runner: int = 2,
    n_buckets: int = 5,
    seed: int = 20260916,
    vol_panel: pd.DataFrame | None = None,
    n_vol_buckets: int = 5,
) -> pd.DataFrame:
    """For each runner, ``per_runner`` same-day, same-turnover-bucket non-runners.

    A control must, on the same date, (a) have enough history for every
    feature, (b) sit in the same cross-sectional turnover quintile, and (c)
    **not** gain ``threshold`` within the next ``window`` sessions -- the
    explicit negative of the runner test, not merely "was not sampled as a
    runner".
    """
    rng = np.random.default_rng(seed)
    turnover = _turnover_panel(frames)

    # forward gain per symbol, cached once
    gains: dict[str, pd.Series] = {}
    for sym, df in frames.items():
        close = df["close"].to_numpy(dtype=float)
        n = len(close)
        fwd = np.full(n, np.nan)
        for k in range(1, window + 1):
            fwd = np.fmax(fwd, np.concatenate([close[k:], np.full(k, np.nan)]))
        gains[sym] = pd.Series(fwd / close - 1.0, index=df.index)

    rows = []
    for date, grp in runners.groupby("date"):
        day_to = turnover.loc[date].dropna() if date in turnover.index else pd.Series(dtype=float)
        if day_to.empty:
            continue
        ranks = day_to.rank(pct=True)
        buckets = np.ceil(ranks * n_buckets).clip(1, n_buckets)

        # Optional second matching axis: realised volatility. A +30% move in 25
        # sessions is mechanically easier for a volatile name, so without this
        # the study cannot distinguish "this setup precedes runs" from "this
        # stock moves a lot". Matching on it asks the sharper question: among
        # names equally volatile and equally traded on the same day, what
        # separates the ones that ran?
        if vol_panel is not None and date in vol_panel.index:
            dv = vol_panel.loc[date].reindex(day_to.index)
            vbuckets = np.ceil(dv.rank(pct=True) * n_vol_buckets).clip(1, n_vol_buckets)
        else:
            vbuckets = pd.Series(1.0, index=day_to.index)

        eligible: dict[tuple[int, int], list[str]] = {}
        for sym in day_to.index:
            df = frames[sym]
            pos = int(df.index.searchsorted(date))
            if pos >= len(df) or df.index[pos] != date:
                continue
            if pos < MIN_HISTORY or pos >= len(df) - window:
                continue
            g = gains[sym].iloc[pos]
            if not np.isfinite(g) or g >= threshold:
                continue
            vb = vbuckets.get(sym, 1.0)
            if not np.isfinite(vb):
                continue
            eligible.setdefault((int(buckets[sym]), int(vb)), []).append(sym)

        for r in grp.itertuples():
            if r.symbol not in buckets.index:
                continue
            vb = vbuckets.get(r.symbol, 1.0)
            if not np.isfinite(vb):
                continue
            b = (int(buckets[r.symbol]), int(vb))
            pool = [s for s in eligible.get(b, []) if s != r.symbol]
            if not pool:
                continue
            take = rng.choice(pool, size=min(per_runner, len(pool)), replace=False)
            for sym in np.atleast_1d(take):
                df = frames[sym]
                pos = int(df.index.searchsorted(date))
                rows.append({
                    "symbol": str(sym), "idx": pos, "date": date,
                    "gain": float(gains[sym].iloc[pos]),
                    "matched_to": r.symbol,
                    "bucket_turnover": b[0], "bucket_vol": b[1],
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Features, all causal
# --------------------------------------------------------------------------- #
def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def _slope_norm(x: np.ndarray) -> float:
    """OLS slope over the window, divided by the window mean -- a per-bar
    fractional trend, so it is comparable across price and volume levels."""
    n = len(x)
    if n < 3:
        return np.nan
    m = float(np.nanmean(x))
    if not np.isfinite(m) or m == 0:
        return np.nan
    t = np.arange(n, dtype=float)
    t -= t.mean()
    denom = float((t * t).sum())
    if denom == 0:
        return np.nan
    return float((t * (x - np.nanmean(x))).sum() / denom / m)


def feature_frame(df: pd.DataFrame, index_close: pd.Series, sector_close: pd.Series | None) -> pd.DataFrame:
    """Every feature as a full series, vectorised, causal at each bar.

    Computing per-bar series once per symbol and indexing into them is ~100x
    faster than recomputing a 40-bar window per sample, and guarantees runners
    and controls are measured by identical code.
    """
    L = FEATURE_LOOKBACK
    c, h, lw, v = df["close"], df["high"], df["low"], df["volume"]
    o = df["open"]
    out = pd.DataFrame(index=df.index)

    ret1 = c.pct_change()
    tr = pd.concat([h - lw, (h - c.shift()).abs(), (lw - c.shift()).abs()], axis=1).max(axis=1)

    # --- compression / volatility ---
    atr10, atr40 = tr.rolling(10).mean(), tr.rolling(L).mean()
    out["atr_ratio_10_40"] = atr10 / atr40
    out["atr_pct_40"] = atr40 / c
    rng40 = (h.rolling(L).max() - lw.rolling(L).min()) / c
    rng10 = (h.rolling(10).max() - lw.rolling(10).min()) / c
    out["range_pct_40"] = rng40
    out["range_pct_10"] = rng10
    out["compression_10_40"] = rng10 / rng40
    rv20, rv60 = ret1.rolling(20).std(), ret1.rolling(60).std()
    out["realized_vol_20"] = rv20 * np.sqrt(252)
    out["vol_ratio_20_60"] = rv20 / rv60
    out["bb_width_20"] = 2 * c.rolling(20).std() / c.rolling(20).mean()

    # --- volume / turnover ---
    v10, v40 = v.rolling(10).mean(), v.rolling(L).mean()
    out["vol_ratio_10_40"] = v10 / v40
    out["vol_zscore_5"] = (v.rolling(5).mean() - v40) / v.rolling(L).std()
    logv = np.log1p(v)
    out["vol_trend_40"] = logv.rolling(L).apply(_slope_norm, raw=True)
    to = c * v
    out["turnover_log_10"] = np.log1p(to.rolling(10).mean())
    out["turnover_ratio_10_40"] = to.rolling(10).mean() / to.rolling(L).mean()
    out["turnover_trend_40"] = np.log1p(to).rolling(L).apply(_slope_norm, raw=True)

    # --- location / trend ---
    out["dist_52w_high"] = c / h.rolling(252).max() - 1.0
    out["dist_52w_low"] = c / lw.rolling(252).min() - 1.0
    out["ret_63"] = c / c.shift(63) - 1.0
    out["ret_126"] = c / c.shift(126) - 1.0
    out["ret_252"] = c / c.shift(252) - 1.0
    e20, e50, e200 = _ema(c, 20), _ema(c, 50), _ema(c, 200)
    out["ema20_dist"] = c / e20 - 1.0
    out["ema50_dist"] = c / e50 - 1.0
    out["ema200_dist"] = c / e200 - 1.0
    out["ema_stacked"] = ((e20 > e50) & (e50 > e200)).astype(float)
    out["ema20_slope_10"] = e20 / e20.shift(10) - 1.0
    out["ema50_slope_20"] = e50 / e50.shift(20) - 1.0
    out["pct_above_ema20_40"] = (c > e20).rolling(L).mean()

    # --- structure within the window ---
    # Drawdown must be measured against a *rolling* peak, not the running
    # all-time max. Against c.cummax() nearly every bar of nearly every name
    # sits more than 5% below its lifetime high, so the pullback count
    # saturates at the window length for both groups and measures nothing.
    roll_peak = c.rolling(L, min_periods=2).max()
    dd = c / roll_peak - 1.0
    out["dd_from_40b_peak"] = dd
    out["n_pullbacks_5pct_40"] = (dd < -0.05).rolling(L).sum()
    out["mean_dd_40"] = dd.rolling(L).mean()
    # True max drawdown inside the window: worst peak-to-trough where the
    # trough follows the peak, rather than (window min / window max), which
    # ignores order and so reports a range rather than a drawdown.
    out["max_dd_40"] = dd.rolling(L, min_periods=2).min()
    out["close_position_40"] = (c - lw.rolling(L).min()) / (h.rolling(L).max() - lw.rolling(L).min())

    # --- gaps ---
    gap = o / c.shift() - 1.0
    out["gap_freq_40"] = (gap.abs() > 0.02).rolling(L).mean()
    out["up_gap_freq_40"] = (gap > 0.02).rolling(L).mean()

    # --- relative strength ---
    idx = index_close.reindex(df.index).ffill()
    for w in (20, 60):
        out[f"rs_{w}_vs_index"] = (c / c.shift(w)) - (idx / idx.shift(w))
    if sector_close is not None:
        sec = sector_close.reindex(df.index).ffill()
        for w in (20, 60):
            out[f"rs_{w}_vs_sector"] = (c / c.shift(w)) - (sec / sec.shift(w))
    else:
        out["rs_20_vs_sector"] = np.nan
        out["rs_60_vs_sector"] = np.nan

    # --- days since the last 20%-in-10-sessions move ---
    gain10 = c / c.shift(10) - 1.0
    had = (gain10 >= 0.20).to_numpy()
    since = np.full(len(df), np.nan)
    last = -1
    for i in range(len(df)):
        if had[i]:
            last = i
        since[i] = (i - last) if last >= 0 else np.nan
    out["days_since_20pct_move"] = since

    return out


FEATURE_ORDER: list[str] = []


# --------------------------------------------------------------------------- #
# 4. Discrimination
# --------------------------------------------------------------------------- #
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC -- P(random runner scores above random control), ties at 0.5.

    Chosen as the headline effect size because it is unit-free, robust to the
    heavy tails these features have, and reads directly as discriminating
    power: 0.5 is nothing, and the distance from 0.5 is the whole signal.
    """
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    rp = r[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    n1, n2 = len(pos), len(neg)
    s = np.sqrt(((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2))
    return float((pos.mean() - neg.mean()) / s) if s > 0 else np.nan


def compare(run_f: pd.DataFrame, ctl_f: pd.DataFrame) -> pd.DataFrame:
    """Per-feature discrimination, with both base rates shown.

    ``base_rate_*`` are the fraction of each group above the **pooled median**
    of that feature, which makes "how common was it in each group" directly
    readable and comparable across features on different scales.
    """
    rows = []
    for col in run_f.columns:
        p, n = run_f[col].to_numpy(dtype=float), ctl_f[col].to_numpy(dtype=float)
        pooled = np.concatenate([p, n])
        med = float(np.nanmedian(pooled))
        a = auc(p, n)
        rows.append({
            "feature": col,
            "auc": a,
            "auc_edge": abs(a - 0.5) if np.isfinite(a) else np.nan,
            "cohens_d": cohens_d(p, n),
            "runner_median": float(np.nanmedian(p)),
            "control_median": float(np.nanmedian(n)),
            "runner_mean": float(np.nanmean(p)),
            "control_mean": float(np.nanmean(n)),
            "base_rate_runners": float(np.nanmean(p > med)),
            "base_rate_controls": float(np.nanmean(n > med)),
            "n_runners": int(np.isfinite(p).sum()),
            "n_controls": int(np.isfinite(n).sum()),
        })
    out = pd.DataFrame(rows).sort_values("auc_edge", ascending=False).reset_index(drop=True)
    return out


__all__ = [
    "FEATURE_LOOKBACK", "MIN_HISTORY", "RUN_THRESHOLD", "RUN_WINDOW",
    "auc", "build_controls", "cohens_d", "compare", "feature_frame",
    "find_runners", "mark_runner_bars",
]
