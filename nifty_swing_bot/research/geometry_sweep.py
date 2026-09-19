"""Phase 15: box-geometry sweeps -- the four deferred in REJECTED #14, plus edge_basis.

Why this is being run after being rejected, stated first as the reading rule
in ``STATE.md`` requires. REJECTED #14 deferred these on the premise that the
detector "already sees these setups": it finds a box for 62% of the 41 real
trades somewhere within +/-60 bars, and trades them earlier without measurably
worse results. That premise is now falsified on the axis that matters. Five
admission rules have been run (§14) and ``box_at_entry`` is **3 of 39 under
every one of them**, while watchlist coverage moves 25 -> 33. Whatever reaches
the detector, it recognises the structure at the entry bar in 3 cases. Seeing
a box 16 bars earlier is not the same property as seeing one now, and only the
second one can produce a signal.

Admission is held at the V0 20/5 baseline throughout (§14.1 proved both
conditions load-bearing). Only ``BoxParams`` changes.

Reported per variant, on the same footing as ``results/admission_variants.json``:
box-at-entry on the 41 real trades, signals/year, per-trade edge and t, and
fold-by-fold return and drawdown over the seven 2015-2021 annual folds.

Usage::

    python -m nifty_swing_bot.research.geometry_sweep --stage a
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..aes.boxes import detect_nested_box
from ..aes.locked import assert_locked
from ..aes.params import (
    BoxParams,
    PortfolioParams,
    ScoreParams,
    ScreenerParams,
    WatchlistParams,
)
from ..aes.portfolio import AESPortfolioBacktester
from ..aes.scoring import attach_score
from ..config import get_config
from .aes_calibration import (
    COST_PCT_DEFAULT,
    _finish_signal_frame,
    discovery_signals_raw,
    load_universe,
    universe_base_rate,
)

FOLDS = list(range(2015, 2022))
EDGE_HORIZON = 20
BUCKET_ORDER = {"reject": 0, "wait_and_watch": 1, "high_conviction": 2}

#: The locked configuration scores **equal weight, 0.25 x 4**. ``ScoreParams``
#: still *defaults* to the calibrated 0.40/0.25/0.20/0.15 weights, which are
#: REJECTED #4 -- three of the four factors were selected on the sample they
#: score and none holds its pooled sign in a majority of folds. Passing this
#: explicitly is what keeps a sweep comparable to §9.1 and §14; taking the
#: dataclass default silently re-introduces the rejected scorer.
EQUAL_WEIGHTS = ScoreParams(
    w_fast_resolution=0.25, w_absorbed=0.25, w_rs_capture=0.25, w_prior_cycles=0.25
)

#: Likewise for the exit. ``PortfolioParams`` defaults to ``atr_stop_mult=None``
#: (the spec SMA(10) stop) and ``max_hold_bars=None``, but the locked
#: configuration is the ATR(14) x 2.5 chandelier stop with a hard 25-bar cap --
#: PROVEN #1, the one component of this system with fold-by-fold evidence
#: behind it. Taking the dataclass defaults silently walk-forwards the *spec*
#: stop instead, which reproduces §9.2's "spec" column rather than §9.1.
LOCKED_PORTFOLIO = PortfolioParams(atr_stop_mult=2.5, max_hold_bars=25)

#: Phase 7's 41 discretionary trades. Two (RMC Switchgears, NSE SME; Dhani
#: Services, delisted) have no usable series and are excluded throughout, so
#: every coverage figure here is out of 39 -- the same denominator §7 and §14
#: use.
RECS_EXTRA_TICKERS = {"Modisons": "MODISONLTD.NS", "Akzo Nobel India": "JSWDULUX.NS"}
RECS_EXCLUDE = {"RMC Switchgears", "Dhani Services"}


# --------------------------------------------------------------------------- #
# Real-trade coverage
# --------------------------------------------------------------------------- #
def load_real_trades(recs_dir: str | Path, mapping_csv: str | Path) -> list[dict[str, Any]]:
    """The 39 testable real trades, each with its price frame and entry bar.

    ``recs_dir`` holds series fetched for these names specifically. They are
    deliberately kept out of ``cache/ohlcv``: most are outside the 402-name
    discovery universe (four are BSE lines), and adding them there would change
    the universe every other result in this project was measured on.
    """
    m = pd.read_csv(mapping_csv)
    out = []
    for r in m.itertuples():
        if r.stock in RECS_EXCLUDE:
            continue
        tk = RECS_EXTRA_TICKERS.get(r.stock) or (r.ticker if isinstance(r.ticker, str) else None)
        if not tk:
            continue
        path = os.path.join(str(recs_dir), tk.replace(".", "_") + ".parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        pos = int(df.index.searchsorted(pd.Timestamp(r.date)))
        if pos >= len(df):
            continue
        out.append({"stock": r.stock, "ticker": tk, "df": df, "idx": pos, "date": r.date})
    return out


def coverage(real: list[dict[str, Any]], params: BoxParams, admitted: set[str]) -> dict[str, Any]:
    """Box-at-entry over the real trades under ``params``.

    Two figures, because they answer different questions and this project has
    quoted both:

    ``box_geom_at_entry``  a nested box forms on the entry bar itself -- pure
        detector, and the only quantity a geometry change can move. Baseline
        6/39 (§7.2).
    ``box_at_entry``  that, **and** the name was screener-admitted around the
        entry -- the pipeline figure §14 tracks, which is what a live signal
        would actually require. Baseline 3/39.
    """
    geom, both, names = 0, 0, []
    for t in real:
        nb = detect_nested_box(t["df"], t["idx"], params, symbol=t["ticker"])
        if nb is None:
            continue
        geom += 1
        names.append(t["stock"])
        if t["stock"] in admitted:
            both += 1
    return {
        "testable": len(real),
        "box_geom_at_entry": geom,
        "box_at_entry": both,
        "names": sorted(names),
    }


# --------------------------------------------------------------------------- #
# Edge and folds
# --------------------------------------------------------------------------- #
def per_trade_edge(
    frame: pd.DataFrame, base_rate: dict[int, pd.Series], horizon: int = EDGE_HORIZON
) -> dict[str, float]:
    """Per-trade return at ``horizon``, under all three definitions.

    ``results/admission_variants.json`` reports a field called ``edge`` that is
    ``mean(fwd_gross_20) * 100 - cost`` -- the **absolute** net return, not an
    edge over anything. It is reproduced here as ``ret_abs``, and carried as
    ``edge`` too, because the four admission variants were compared on it and
    a new number has to sit on the same footing to be read against them.

    But PROVEN #6 is explicit that this quantity is not an edge: of the 20-bar
    +3.15%, +1.26% was index drift and +0.68% the universe base rate. So the
    two deflated versions are reported alongside, and ``edge_vs_base`` is the
    one that answers "is this better than a random entry in the same
    universe". Geometry changes which *kind* of structure is admitted, so the
    three can move apart in a way that four admission-width changes never
    forced them to.
    """
    gross = frame[f"fwd_gross_{horizon}"].dropna()
    exc = frame[f"excess_{horizon}"].dropna()
    if gross.empty or exc.empty:
        return {
            "edge": float("nan"), "edge_t": float("nan"), "edge_n": 0,
            "ret_abs": float("nan"), "edge_vs_index": float("nan"),
            "edge_vs_base": float("nan"), "edge_vs_base_t": float("nan"),
        }
    base_mean = float(base_rate[horizon].mean())
    g_sd, g_n = float(gross.std(ddof=1)), len(gross)
    e_sd, e_n = float(exc.std(ddof=1)), len(exc)
    g_t = gross.mean() / (g_sd / np.sqrt(g_n)) if g_sd > 0 else float("nan")
    b_t = (exc.mean() - base_mean) / (e_sd / np.sqrt(e_n)) if e_sd > 0 else float("nan")
    return {
        # the admission_variants definition, kept for comparability
        "edge": float(gross.mean() * 100 - COST_PCT_DEFAULT),
        "edge_t": float(g_t),
        "edge_n": int(g_n),
        "ret_abs": float(gross.mean() * 100 - COST_PCT_DEFAULT),
        # deflated
        "edge_vs_index": float(exc.mean() * 100 - COST_PCT_DEFAULT),
        "edge_vs_base": float((exc.mean() - base_mean) * 100 - COST_PCT_DEFAULT),
        "edge_vs_base_t": float(b_t),
        "base_rate_pct": round(base_mean * 100, 3),
    }


def _truncate(
    frames: dict[str, pd.DataFrame], end: pd.Timestamp, tail: int = 40
) -> dict[str, pd.DataFrame]:
    """Frames cut at ``end`` plus a ``tail``-bar run-off, so a 25-bar-capped
    position can close naturally and no fold can earn anything from a later
    period. Phase 9's isolation convention, reused unchanged."""
    out = {}
    for sym, df in frames.items():
        pos = int(df.index.searchsorted(end, side="right"))
        if pos <= 0:
            continue
        out[sym] = df.iloc[: min(len(df), pos + tail)]
    return out


def edge_by_population(
    frame: pd.DataFrame,
    entered: set[tuple[str, pd.Timestamp]],
    base_rate: dict[int, pd.Series],
    horizon: int = EDGE_HORIZON,
) -> dict[str, Any]:
    """Edge over the universe base rate, split by whether the signal was traded.

    This is the direct test of PROVEN #5's *mechanism*. That finding says the
    marginal signal is worse than the average one "because score-ranking was
    already taking the good ones". If that is true, the signals the portfolio
    actually entered should carry a higher ``edge_vs_base`` than the ones it
    passed over. If entered and skipped signals have the same edge, then the
    three-slot cap is discarding signal at random and the constraint is
    capacity, not signal quality.
    """
    key = list(zip(frame["symbol"], frame["entry_date"]))
    mask = np.array([k in entered for k in key], dtype=bool)
    base_mean = float(base_rate[horizon].mean())
    col = f"excess_{horizon}"

    def one(sub: pd.DataFrame) -> dict[str, Any]:
        a = sub[col].dropna()
        if len(a) < 2:
            return {"n": int(len(a)), "edge": float("nan"), "t": float("nan")}
        sd = float(a.std(ddof=1))
        t = (a.mean() - base_mean) / (sd / np.sqrt(len(a))) if sd > 0 else float("nan")
        return {
            "n": int(len(a)),
            "edge": round(float((a.mean() - base_mean) * 100 - COST_PCT_DEFAULT), 3),
            "t": round(float(t), 3),
        }

    ent, skip = one(frame[mask]), one(frame[~mask])
    gap = (
        round(ent["edge"] - skip["edge"], 3)
        if np.isfinite(ent["edge"]) and np.isfinite(skip["edge"])
        else None
    )
    return {"entered": ent, "skipped": skip, "selection_gap": gap}


def run_folds(
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    scored: list,
    pp: PortfolioParams,
    entered_out: set[tuple[str, pd.Timestamp]] | None = None,
) -> list[dict[str, Any]]:
    """Seven annual folds, nothing refitted -- the §9.1 walk-forward.

    ``entered_out``, if given, collects every (symbol, entry_date) the
    portfolio actually opened, for ``edge_by_population``.
    """
    rows = []
    for year in FOLDS:
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        sub = [s for s in scored if start <= s.entry_date <= end]
        if not sub:
            rows.append({
                "fold": year, "n": 0, "positions": 0, "ret": 0.0,
                "alpha": 0.0, "dd": 0.0, "deploy": 0.0, "top1": None,
            })
            continue
        f_trunc = _truncate(frames, end)
        idx_cut = int(index_df.index.searchsorted(end, side="right")) + 40
        idx_trunc = index_df.iloc[:idx_cut]
        bt = AESPortfolioBacktester(params=pp)
        res = bt.run(f_trunc, idx_trunc, sub, min_bucket="wait_and_watch")
        st = res.stats
        tf = res.trades_frame
        if entered_out is not None and not tf.empty:
            for sym, ed in tf.groupby(["symbol", "entry_date"]).groups:
                entered_out.add((sym, pd.Timestamp(ed)))
        deploy = (
            float(res.equity_curve["exposure"].mean() * 100)
            if "exposure" in res.equity_curve
            else 0.0
        )
        top1 = None
        if not tf.empty:
            by_sym = tf.groupby("symbol")["net_pnl"].sum()
            total = float(by_sym.sum())
            if abs(total) > 1e-9:
                top1 = round(float(by_sym.max() / total * 100), 1)
        rows.append({
            "fold": year,
            "n": len(sub),
            # One position = one (symbol, entry_date). The ladder writes a row
            # per partial exit, and a name can be traded more than once in a
            # fold, so neither len(tf) nor nunique(symbol) is the count.
            "positions": (
                int(tf.groupby(["symbol", "entry_date"]).ngroups) if not tf.empty else 0
            ),
            "ret": float(st.get("total_return_pct", 0.0)),
            "alpha": float(st.get("alpha_annual_pct", 0.0)),
            "dd": float(st.get("max_drawdown_pct", 0.0)),
            "deploy": round(deploy, 2),
            "top1": top1,
        })
    return rows


def summarise_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy import stats as sps

    rets = np.array([f["ret"] for f in folds], dtype=float)
    dds = np.array([f["dd"] for f in folds], dtype=float)
    n = len(rets)
    sd = float(rets.std(ddof=1)) if n > 1 else 0.0
    t = float(rets.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan")
    p = float(2 * (1 - sps.t.cdf(abs(t), n - 1))) if np.isfinite(t) and n > 1 else float("nan")
    return {
        "mean_fold_ret": round(float(rets.mean()), 2),
        "fold_t": round(t, 2) if np.isfinite(t) else None,
        "fold_p": round(p, 3) if np.isfinite(p) else None,
        "pos_folds": f"{int((rets > 0).sum())}/{n}",
        "mean_dd": round(float(dds.mean()), 2),
        "worst_dd": round(float(dds.max()), 2),
    }


# --------------------------------------------------------------------------- #
# One variant, end to end
# --------------------------------------------------------------------------- #
def run_variant(
    name: str,
    bp: BoxParams,
    *,
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    base_rate: dict[int, pd.Series],
    real: list[dict[str, Any]],
    admitted: set[str],
    sp: ScreenerParams | None = None,
    wp: WatchlistParams | None = None,
    pp: PortfolioParams | None = None,
    with_folds: bool = True,
    portfolio_deviations: tuple[str, ...] = (),
    deviation_reason: str = "",
) -> dict[str, Any]:
    sp = sp or ScreenerParams()
    wp = wp or WatchlistParams()
    pp = pp or LOCKED_PORTFOLIO
    # Loud by construction: any deviation from the locked exit has to be named
    # by the caller, with a reason, or this raises. See aes/locked.py.
    assert_locked(
        score=EQUAL_WEIGHTS, portfolio=pp,
        deviations=portfolio_deviations, reason=deviation_reason,
    )
    t0 = time.time()

    out: dict[str, Any] = {"variant": name}
    # Coverage first: it is cheap, needs no universe run, and is the quantity
    # the whole phase is about. A variant that moves nothing here has already
    # answered the question.
    out["coverage"] = coverage(real, bp, admitted)

    signals = discovery_signals_raw(
        frames, index_df, screener_params=sp, box_params=bp, watch_params=wp
    )
    scored = [attach_score(s, EQUAL_WEIGHTS) for s in signals]
    traded = [s for s in scored if BUCKET_ORDER.get(s.bucket, 0) >= 1]
    out["signals"] = len(signals)
    out["traded"] = len(traded)
    out["sig_per_year"] = round(len(signals) / len(FOLDS), 1)

    if traded:
        frame = _finish_signal_frame(traded, frames, index_df, sp)
        out.update(per_trade_edge(frame, base_rate))
    else:
        out.update({"edge": float("nan"), "edge_t": float("nan"), "edge_n": 0})

    if with_folds:
        entered: set[tuple[str, pd.Timestamp]] = set()
        folds = run_folds(frames, index_df, scored, pp, entered_out=entered)
        out.update(summarise_folds(folds))
        out["folds"] = folds
        out["n_entered"] = len(entered)
        if traded:
            out["by_population"] = edge_by_population(frame, entered, base_rate)
    out["secs"] = round(time.time() - t0, 1)
    return out


def load_context():
    """Universe, index and the base rate. The base rate does not depend on box
    geometry, so it is computed once and shared across every variant."""
    cfg = get_config()
    frames = load_universe(cfg.paths.ohlcv_dir)
    index_df = frames.pop("CRSLDX", None)
    if index_df is None:
        index_df = pd.read_parquet(os.path.join(str(cfg.paths.ohlcv_dir), "CRSLDX.parquet"))
    base_rate = universe_base_rate(frames, index_df)
    return cfg, frames, index_df, base_rate


__all__ = [
    "BUCKET_ORDER",
    "FOLDS",
    "coverage",
    "edge_by_population",
    "load_context",
    "load_real_trades",
    "per_trade_edge",
    "run_folds",
    "run_variant",
    "summarise_folds",
]
