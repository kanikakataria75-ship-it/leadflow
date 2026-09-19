"""AES Phase 3/4: calibration and bar-level edge, discovery period only.

Four questions, in the order the build plan asked for them:

1. **Breakout volume multiplier.** The spec describes it only visually
   ("larger than surrounding bars") and asks for a sweep from 1.5x to 3.0x of
   SMA(volume, 20) to find where forward edge concentrates.
2. **Circuit-mover threshold.** Starting guess 3 hits in 60 sessions; sweep it.
3. **Scoring-factor weights.** Every AES section 2-4 factor is tested for a
   univariate relationship with net-of-cost forward return. Report which
   carry real weight and which are noise -- bluntly, per instruction.
4. **Bar-level edge vs a random entry in the same universe.** The same
   yardstick the DAB/MR study used (``signal_lab.py``): excess return over the
   *index-relative* universe base rate, net of the realistic 0.585% round-trip
   cost. This is the honesty checkpoint -- if it fails here, a portfolio
   backtest cannot be trusted to show something different.

**A structural fix that came before any of this could be measured.** The
screener re-admits a name on every consecutive day it satisfies "20% up in 10
days and near its 52-week high" -- which, for a genuine multi-week move, is
many sessions in a row. Treating each admission as an independent trial
counted the same underlying box+breakout up to six times, at a rate
correlated with how strong the move was (a stronger move satisfies the
screener longer). Measured directly on this data: 94% of naively-generated
signals were such duplicates. ``aes.signals.build_signals_for_symbol``
collapses repeated admissions into one signal per genuine episode; every
number in this module is computed on the deduplicated set.

Discovery-only, by design (``aes.protocol.AES_DISCOVERY``, 2015-2021).
``AES_HOLDOUT`` (2022-today) is not read by this module at all.

Usage::

    python -m nifty_swing_bot.research.aes_calibration
    python -m nifty_swing_bot.research.aes_calibration --min-turnover-cr 1.0
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..aes.boxes import scan_box_history
from ..aes.params import BoxParams, ScreenerParams, WatchlistParams
from ..aes.protocol import AES_DISCOVERY, describe
from ..aes.screener import circuit_hits, screen_universe
from ..aes.signals import SignalParams, build_signals_for_symbol
from ..aes.watchlist import run_watchlist_for_symbol
from ..config import get_config

logger = logging.getLogger(__name__)

COST_PCT_DEFAULT = 0.585   # RESEARCH.md section 8: realistic round trip at a Rs 1L position
HORIZONS = (5, 10, 15, 20)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_universe(ohlcv_dir: str | Path, min_bars: int = 400) -> dict[str, pd.DataFrame]:
    """Every cached symbol with enough history, keyed by symbol."""
    frames: dict[str, pd.DataFrame] = {}
    for f in sorted(glob.glob(os.path.join(str(ohlcv_dir), "*.parquet"))):
        sym = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        if len(df) >= min_bars:
            frames[sym] = df
    return frames


def build_discovery_signals(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    *,
    screener_params: ScreenerParams | None = None,
    box_params: BoxParams | None = None,
    signal_params: SignalParams | None = None,
) -> pd.DataFrame:
    """Every deduplicated AES episode admitted during AES_DISCOVERY.

    Returns a flat frame (one row per signal) with box/context/forward-return
    columns and the index-relative excess added for each horizon.
    """
    SP = screener_params or ScreenerParams()
    BP = box_params or BoxParams()
    sig_params = signal_params or SignalParams()

    admissions = screen_universe(frames, SP)
    mask = admissions["date"].dt.date.between(AES_DISCOVERY.start, AES_DISCOVERY.end)
    admissions = admissions[mask]

    all_signals = []
    for sym in sorted(admissions["symbol"].unique()):
        df = frames[sym]
        sym_adm = admissions[admissions["symbol"] == sym]
        last_idx = df.index.get_indexer([sym_adm["date"].max()])[0]
        history = scan_box_history(df, BP, end_idx=min(len(df) - 1, last_idx + 100), step=3)
        all_signals.extend(
            build_signals_for_symbol(
                df, index_df, sym_adm, symbol=sym, box_params=BP,
                signal_params=sig_params, box_history=history,
            )
        )

    return _finish_signal_frame(all_signals, frames, index_df, SP)


def discovery_signals_raw(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    *,
    screener_params: ScreenerParams | None = None,
    box_params: BoxParams | None = None,
    watch_params: WatchlistParams | None = None,
) -> list:
    """Every AES ``Signal`` object admitted during AES_DISCOVERY, via the
    persistent watchlist -- the live objects, not the flattened frame.

    Unlike ``build_discovery_signals``, admissions here only open or extend a
    watchlist stay -- ``aes.watchlist.run_watchlist_for_symbol`` does the
    actual per-bar walk, checking all three entry modes (section 5) for as
    long as the name is watched, which can yield more than one signal per
    admission if a box resolves with no entry and a fresh one later forms.

    Kept separate from ``build_discovery_signals_watchlist`` (which flattens
    to a DataFrame for calibration) because Phase 5's portfolio backtest
    needs the live objects -- ``signal.nested.big.bottom`` for the section
    6.1 stop, in particular -- which a flattened row does not carry.
    """
    SP = screener_params or ScreenerParams()
    BP = box_params or BoxParams()
    WP = watch_params or WatchlistParams()

    admissions = screen_universe(frames, SP)
    mask = admissions["date"].dt.date.between(AES_DISCOVERY.start, AES_DISCOVERY.end)
    admissions = admissions[mask]

    all_signals = []
    for sym in sorted(admissions["symbol"].unique()):
        df = frames[sym]
        sym_adm = admissions[admissions["symbol"] == sym]
        last_idx = df.index.get_indexer([sym_adm["date"].max()])[0]
        history = scan_box_history(df, BP, end_idx=min(len(df) - 1, last_idx + 100), step=3)
        all_signals.extend(
            run_watchlist_for_symbol(
                df, index_df, sym_adm["date"], symbol=sym, box_params=BP,
                watch_params=WP, box_history=history,
            )
        )
    return all_signals


def build_discovery_signals_watchlist(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    *,
    screener_params: ScreenerParams | None = None,
    box_params: BoxParams | None = None,
    watch_params: WatchlistParams | None = None,
) -> pd.DataFrame:
    """``discovery_signals_raw``, flattened to a frame for calibration."""
    SP = screener_params or ScreenerParams()
    all_signals = discovery_signals_raw(
        frames, index_df, screener_params=SP, box_params=box_params, watch_params=watch_params
    )
    return _finish_signal_frame(all_signals, frames, index_df, SP)


def _finish_signal_frame(
    all_signals: list, frames: dict[str, pd.DataFrame], index_df: pd.DataFrame, SP: ScreenerParams
) -> pd.DataFrame:
    """Shared tail for both builders: cost columns, index-relative excess, circuit hits."""
    rows = [s.as_row(cost_pct=COST_PCT_DEFAULT) for s in all_signals]
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    idx_close = index_df["close"]
    circuit_cache: dict[str, pd.Series] = {}
    for h in HORIZONS:
        idx_fwd = []
        for r in out.itertuples():
            pos = idx_close.index.get_indexer([r.entry_date])[0]
            if pos < 0 or pos + h >= len(idx_close):
                idx_fwd.append(np.nan)
            else:
                idx_fwd.append(idx_close.iloc[pos + h] / idx_close.iloc[pos] - 1.0)
        out[f"idx_fwd_{h}"] = idx_fwd
        out[f"excess_{h}"] = out[f"fwd_gross_{h}"] - out[f"idx_fwd_{h}"]

    hits = []
    for r in out.itertuples():
        if r.symbol not in circuit_cache:
            circuit_cache[r.symbol] = circuit_hits(frames[r.symbol], SP)
        s = circuit_cache[r.symbol]
        pos = frames[r.symbol].index.get_indexer([r.admission_date])[0]
        hits.append(int(s.iloc[pos]) if pos >= 0 else np.nan)
    out["circuit_hits_60d"] = hits
    return out


# --------------------------------------------------------------------------- #
# 1. Volume multiplier sweep
# --------------------------------------------------------------------------- #
def volume_multiplier_sweep(signals: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Bucket signals by realised breakout-day volume ratio; report net edge per bucket.

    Not a re-triggering search over candidate multipliers -- the breakout
    date is a fixed, well-defined event (first close above the operative
    top), and this asks where its *observed* volume ratio concentrates edge.
    """
    d = signals.dropna(subset=["vol_ratio"]).copy()
    bins = [0, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
    labels = ["<1.0x", "1.0-1.5x", "1.5-2.0x", "2.0-2.5x", "2.5-3.0x", "3.0x+"]
    d["bucket"] = pd.cut(d["vol_ratio"], bins=bins, labels=labels)
    col = f"fwd_net_{horizon}"
    rows = []
    for lab in labels:
        x = d.loc[d["bucket"] == lab, col].dropna()
        if len(x) < 3:
            rows.append({"bucket": lab, "n": len(x)})
            continue
        t, p = stats.ttest_1samp(x, 0.0)
        rows.append({
            "bucket": lab, "n": len(x), "mean_net_pct": x.mean() * 100,
            "median_net_pct": x.median() * 100, "t": t, "p": p,
            "win_rate_pct": (x > 0).mean() * 100,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Circuit-mover threshold sweep
# --------------------------------------------------------------------------- #
def circuit_threshold_sweep(signals: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """How many signals a given N-hit threshold would exclude, and their edge."""
    col = f"fwd_net_{horizon}"
    rows = []
    for n_thresh in range(0, 6):
        excluded = signals[signals["circuit_hits_60d"] > n_thresh][col].dropna()
        kept = signals[signals["circuit_hits_60d"] <= n_thresh][col].dropna()
        rows.append({
            "n_threshold": n_thresh,
            "excluded_n": len(excluded),
            "excluded_mean_net_pct": excluded.mean() * 100 if len(excluded) else np.nan,
            "kept_n": len(kept),
            "kept_mean_net_pct": kept.mean() * 100 if len(kept) else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Scoring-factor univariate calibration
# --------------------------------------------------------------------------- #
CONTINUOUS_FACTORS = [
    "above_mid", "big_quality", "higher_lows", "duration_vs_norm", "prior_boxes",
    "prior_cycles", "vol_dryup_ratio", "headroom_pct", "resistance_age_strength",
    "resistance_rejected_count", "resistance_absorbed_count", "tf_score",
    "rs_capture", "vol_ratio",
]
CATEGORICAL_FACTORS = ["zone", "used_small_top", "clears_min_headroom", "tf_monthly", "rs_classification"]


def calibrate_factors(signals: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Univariate Pearson r and tercile split for every measured factor.

    This is Phase 3's core deliverable: which of the spec's section 4 factors
    show a real, individually-measurable relationship with net forward
    return, and which do not. No factor surviving here does not mean the
    factor is wrong -- it means this sample cannot distinguish it from noise,
    which is a fact about statistical power as much as about the factor.
    """
    col = f"fwd_net_{horizon}"
    rows = []
    for name in CONTINUOUS_FACTORS:
        x = signals[[col, name]].dropna()
        if len(x) < 20:
            rows.append({"factor": name, "n": len(x), "kind": "continuous"})
            continue
        r, p = stats.pearsonr(x[name], x[col])
        rows.append({"factor": name, "n": len(x), "kind": "continuous", "pearson_r": r, "p": p})
    for name in CATEGORICAL_FACTORS:
        x = signals[[col, name]].dropna()
        if len(x) < 20:
            rows.append({"factor": name, "n": len(x), "kind": "categorical"})
            continue
        groups = [v.to_numpy() for _, v in x.groupby(name)[col] if len(v) >= 3]
        f, p = stats.f_oneway(*groups) if len(groups) >= 2 else (np.nan, np.nan)
        rows.append({"factor": name, "n": len(x), "kind": "categorical", "anova_f": f, "p": p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Bar-level edge vs a random entry in the same universe (Phase 4)
# --------------------------------------------------------------------------- #
def universe_base_rate(
    frames: dict[str, pd.DataFrame], index_df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS
) -> dict[int, pd.Series]:
    """The excess-over-index forward return of every evaluable discovery-period bar.

    This is the "random entry in the same universe" benchmark: what a
    next-open entry on *any* bar of *any* stock would have returned, relative
    to the index over the same window. Matches ``signal_lab.py``'s exact
    methodology so the two are on the same footing.
    """
    disc_start, disc_end = pd.Timestamp(AES_DISCOVERY.start), pd.Timestamp(AES_DISCOVERY.end)
    idx_close = index_df["close"]
    per_horizon: dict[int, list[pd.Series]] = {h: [] for h in horizons}
    for df in frames.values():
        mask = (df.index >= disc_start) & (df.index <= disc_end)
        if mask.sum() < 50:
            continue
        entry = df["open"].shift(-1)
        idx_aligned = idx_close.reindex(df.index).ffill()
        for h in horizons:
            raw_fwd = df["close"].shift(-h) / entry - 1.0
            idx_fwd = idx_aligned.shift(-h) / idx_aligned.shift(-1) - 1.0
            per_horizon[h].append((raw_fwd - idx_fwd)[mask])
    return {h: pd.concat(v).dropna() for h, v in per_horizon.items()}


def bar_level_edge_report(
    signals: pd.DataFrame, base_rate: dict[int, pd.Series], cost_pct: float = COST_PCT_DEFAULT
) -> pd.DataFrame:
    """AES signals vs the universe base rate, net of cost -- Phase 4's headline table."""
    rows = []
    for h in base_rate:
        exc_col = f"excess_{h}"
        aes = signals[exc_col].dropna()
        base_mean = base_rate[h].mean()
        gross_excess = aes.mean() - base_mean
        net_excess = gross_excess - cost_pct / 100.0
        sd = aes.std(ddof=1)
        n = len(aes)
        t = (aes.mean() - base_mean) / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan
        rows.append({
            "horizon": h, "n_signals": n, "n_base_bars": len(base_rate[h]),
            "aes_idxrel_pct": aes.mean() * 100, "base_idxrel_pct": base_mean * 100,
            "gross_excess_pct": gross_excess * 100, "net_excess_pct": net_excess * 100,
            "t": t,
        })
    return pd.DataFrame(rows)


def bar_level_edge_by_mode(
    signals: pd.DataFrame, base_rate: dict[int, pd.Series], horizon: int, cost_pct: float = COST_PCT_DEFAULT
) -> pd.DataFrame:
    """The same report, split by which of the three entry modes (section 5) fired."""
    rows = []
    mode_names = {1: "1 breakout", 2: "2 retest", 3: "3 box-bottom"}
    for mode, group in signals.groupby("entry_mode"):
        rpt = bar_level_edge_report(group, {horizon: base_rate[horizon]}, cost_pct)
        r = rpt.iloc[0]
        rows.append({
            "entry_mode": mode_names.get(mode, str(mode)), "n": int(r["n_signals"]),
            "net_excess_pct": r["net_excess_pct"], "t": r["t"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, default=20, help="Primary horizon for factor/bucket tables.")
    parser.add_argument(
        "--engine", choices=["watchlist", "per_admission"], default="watchlist",
        help="'watchlist' (default) uses the persistent state machine with all three "
             "entry modes; 'per_admission' is the earlier, narrower pass (mode 1 only, "
             "one search window per admission) -- kept for comparison.",
    )
    args = parser.parse_args(argv)

    print(describe())
    cfg = get_config()
    t0 = time.time()
    frames = load_universe(cfg.paths.ohlcv_dir)
    index_df = frames.pop("CRSLDX", None)
    if index_df is None:
        index_df = pd.read_parquet(os.path.join(str(cfg.paths.ohlcv_dir), "CRSLDX.parquet"))
    print(f"Loaded {len(frames)} symbols in {time.time() - t0:.0f}s")

    t0 = time.time()
    if args.engine == "watchlist":
        signals = build_discovery_signals_watchlist(frames, index_df)
    else:
        signals = build_discovery_signals(frames, index_df)
    print(f"Built {len(signals)} signals via the '{args.engine}' engine "
          f"({signals['symbol'].nunique()} symbols) in {time.time() - t0:.0f}s")
    if "entry_mode" in signals.columns:
        print("  entry mode counts:", signals["entry_mode"].value_counts().sort_index().to_dict())

    print("\n=== 1. Breakout volume multiplier sweep ===")
    print(volume_multiplier_sweep(signals, args.horizon).to_string(index=False))

    print("\n=== 2. Circuit-mover threshold sweep ===")
    print(circuit_threshold_sweep(signals, args.horizon).to_string(index=False))

    print("\n=== 3. Scoring-factor calibration ===")
    print(calibrate_factors(signals, args.horizon).to_string(index=False))

    print("\n=== 4. Bar-level edge vs universe base rate (net of cost) ===")
    base_rate = universe_base_rate(frames, index_df)
    print(bar_level_edge_report(signals, base_rate).to_string(index=False))
    if "entry_mode" in signals.columns and signals["entry_mode"].nunique() > 1:
        print(f"\n  by entry mode, horizon={args.horizon}:")
        print(bar_level_edge_by_mode(signals, base_rate, args.horizon).to_string(index=False))

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "aes_discovery_signals.parquet" if args.engine == "watchlist" else "aes_discovery_signals_per_admission.parquet"
    signals.to_parquet(out_dir / out_name)
    print(f"\nSaved signals to {out_dir / out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
