"""The once-daily scan: refresh, screen, watch, detect, score, size, persist.

Run order matters and mirrors the spec's own sections: admission (§1.1)
decides who is watched; the watchlist state machine (§1.3) decides who stays;
box detection (§2) describes what is forming; the context features (§3) and
the equal-weight score (§4) rank it; §6.1/§7 turn a ranked setup into rupees.

Nothing here decides to trade. Every component value that moved a candidate
up or down is carried through to the output so the ranking can be disagreed
with, which is the whole point of the tool.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..aes.boxes import NestedBox, detect_nested_box, scan_box_history
from ..aes.params import (BoxParams, PortfolioParams, RelativeStrengthParams, ResistanceParams,
                          ScoreParams, ScreenerParams, TimeframeParams, WatchlistParams)
from ..aes.relative_strength import drawdown_relative_strength
from ..aes.resistance import resistance_headroom
from ..aes.screener import hard_delete_mask, screen, screen_universe
from ..aes.scoring import score_setup
from ..aes.timeframes import classify_breakout
from ..aes.watchlist import run_watchlist_for_symbol
from ..config import AppConfig, get_config
from ..strategy.indicators import atr
from .breadth import BreadthReading, current_breadth
from .sizing import plan_trade
from .store import AESScanStore

logger = logging.getLogger(__name__)

#: Equal weight, not the Phase 3 calibrated weights. Phase 6 found three of
#: the four in-sample-selected and none stable fold-to-fold; equal weighting
#: is not better, it is merely not fitted.
EQUAL_WEIGHTS = ScoreParams(
    w_fast_resolution=0.25, w_absorbed=0.25, w_rs_capture=0.25, w_prior_cycles=0.25
)

#: Bars of history each symbol's watchlist walk replays. Long enough to
#: rebuild current state deterministically from price alone -- the machine is
#: a pure function of history, so nothing needs to be stored between runs and
#: nothing can drift.
WALK_BARS = 520


@dataclass
class ScanResult:
    """One evening's output."""

    scan_date: date
    candidates: pd.DataFrame
    breadth: BreadthReading | None
    universe_size: int
    watchlist_size: int
    duration_s: float
    regime_weak: bool
    charts: dict[str, str] = field(default_factory=dict)
    #: One row per universe symbol with a stage, a verdict and a reason.
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Equity the plans were sized against, and rupees currently in open
    #: positions if the caller supplied it. Carried so the report can state
    #: sleeve usage rather than leaving deployment to be worked out by hand.
    equity: float | None = None
    deployed_notional: float | None = None

    @property
    def actionable(self) -> pd.DataFrame:
        if self.candidates.empty:
            return self.candidates
        return self.candidates[self.candidates["actionable"] == 1]

    def bucket(self, name: str) -> pd.DataFrame:
        if self.candidates.empty:
            return self.candidates
        return self.candidates[self.candidates["bucket"] == name]


def _atr14(df: pd.DataFrame, i: int, period: int = 14) -> float:
    a = atr(df["high"], df["low"], df["close"], period)
    v = a.iloc[i] if i < len(a) else np.nan
    return float(v) if np.isfinite(v) else float("nan")


def _describe(
    symbol: str,
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    i: int,
    nested: NestedBox,
    *,
    gap_days: float | None,
    box_params: BoxParams,
    resistance_params: ResistanceParams,
    timeframe_params: TimeframeParams,
    rs_params: RelativeStrengthParams,
    score_params: ScoreParams,
) -> dict[str, Any]:
    """Everything measurable about one setup as of bar ``i``."""
    close = float(df["close"].iloc[i])
    big, small = nested.big, nested.small

    res = resistance_headroom(df, i, reference_price=close, params=resistance_params)
    tf = classify_breakout(df, i, timeframe_params)
    rs = drawdown_relative_strength(df, index_df, i, rs_params)

    sc = score_setup(
        gap_days=gap_days,
        resistance_absorbed_count=res["resistance_absorbed_count"],
        rs_capture=rs.downside_capture,
        prior_cycles=nested.prior_cycles,
        params=score_params,
        rs_params=rs_params,
    )

    height = big.top - big.bottom
    hp = res.get("headroom_pct")
    return {
        "symbol": symbol,
        "bucket": sc.bucket,
        "score": round(sc.score, 4),
        "box_date": nested.as_of.strftime("%Y-%m-%d"),
        "box_bars": int(big.bars),
        "box_range_pct": round(float(big.range_pct) * 100, 2),
        "box_top": round(float(big.top), 2),
        "box_bottom": round(float(big.bottom), 2),
        "box_position": round((close - big.bottom) / height, 3) if height > 0 else None,
        "higher_lows": int(big.higher_lows),
        "pct_closes_above_mid": round(float(big.pct_closes_above_mid) * 100, 1),
        "box_quality": round(float(big.quality), 3),
        "small_zone": nested.small_zone,
        "small_top": round(float(small.top), 2) if small is not None else None,
        "small_bottom": round(float(small.bottom), 2) if small is not None else None,
        "small_bars": int(small.bars) if small is not None else None,
        "prior_cycles": int(nested.prior_cycles),
        "vol_dryup_ratio": round(float(nested.vol_dryup_ratio), 3)
        if nested.vol_dryup_ratio is not None else None,
        "resistance_level": round(close * (1 + hp), 2) if hp is not None else None,
        "resistance_dist_pct": round(hp * 100, 2) if hp is not None else None,
        "resistance_age": res.get("resistance_age_bars"),
        "resistance_absorbed": int(res["resistance_absorbed_count"]),
        "resistance_rejected": int(res["resistance_rejected_count"]),
        "tf_daily": int(bool(tf.daily)),
        "tf_weekly": int(bool(tf.weekly)),
        "tf_monthly": int(bool(tf.monthly)),
        "rs_capture": round(float(rs.downside_capture), 3)
        if rs.downside_capture is not None else None,
        "rs_class": rs.classification,
        "sc_fast": round(sc.subscores["fast_resolution"], 3),
        "sc_absorbed": round(sc.subscores["resistance_absorbed"], 3),
        "sc_rs": round(sc.subscores["rs_capture"], 3),
        "sc_cycles": round(sc.subscores["prior_cycles"], 3),
        "close": round(close, 2),
    }


def run_scan(
    cfg: AppConfig | None = None,
    *,
    frames: dict[str, pd.DataFrame] | None = None,
    index_df: pd.DataFrame | None = None,
    equity: float | None = None,
    recent_bars: int = 3,
    render: bool = True,
    chart_dir: Path | str | None = None,
    store: AESScanStore | None = None,
    persist: bool = True,
    audit: bool = True,
    universe_meta: pd.DataFrame | None = None,
    screener_params: ScreenerParams | None = None,
    box_params: BoxParams | None = None,
    watch_params: WatchlistParams | None = None,
    portfolio_params: PortfolioParams | None = None,
    score_params: ScoreParams | None = None,
) -> ScanResult:
    """Run one evening's scan.

    Args:
        frames: ``{symbol: OHLCV}``. Fetched from the configured universe when
            omitted, which is the normal production path; passing them in is
            what makes this testable and what the API reuses.
        index_df: Benchmark OHLCV for relative strength and the §8 regime.
        equity: Account equity to size against. Defaults to configured capital.
        recent_bars: A watchlist entry trigger this many bars old or newer
            counts as actionable today.
        render: Write a box chart per actionable/high-conviction candidate.
        persist: Write the scan and its candidates to the store.
        audit: Also record a verdict and a reason for *every* universe symbol,
            not just the ones that produced a box. Without it the ~95% that
            drop out do so silently, and a silent drop cannot be checked.

    Returns:
        A ``ScanResult``. ``candidates`` is one row per watchlist name with a
        detectable box, ranked by score.
    """
    t0 = time.time()
    cfg = cfg or get_config()
    SP = screener_params or ScreenerParams()
    BP = box_params or BoxParams()
    WP = watch_params or WatchlistParams()
    PP = portfolio_params or PortfolioParams()
    SCP = score_params or EQUAL_WEIGHTS
    RP, TP, RSP = ResistanceParams(), TimeframeParams(), RelativeStrengthParams()
    equity = float(equity if equity is not None else cfg.risk.starting_capital)

    if frames is None:
        frames, index_df = _load_live(cfg)
    if index_df is None:
        raise ValueError("index_df is required when frames are supplied directly")

    frames = {s: d for s, d in frames.items() if d is not None and len(d) >= BP.min_history_bars}
    scan_date = max(d.index.max() for d in frames.values()).date() if frames else date.today()

    # -- §8 regime: benchmark below its own SMA(50) on a closing basis ------- #
    ic = index_df["close"]
    regime_weak = bool(len(ic) > PP.regime_sma_period
                       and ic.iloc[-1] < ic.rolling(PP.regime_sma_period).mean().iloc[-1])

    # -- §1.1 admissions across the universe, and breadth -------------------- #
    admissions = screen_universe(frames, SP)
    reading = current_breadth(admissions)

    # -- §1.2/§1.3 who is on the watchlist right now ------------------------- #
    watch: dict[str, int] = {}
    for sym, df in frames.items():
        adm = screen(df, SP)
        hits = np.flatnonzero(adm.to_numpy())
        if not len(hits):
            continue
        last_adm = int(hits[-1])
        age = len(df) - 1 - last_adm
        if age > WP.max_watch_without_box:
            continue
        deleted = hard_delete_mask(df, SP)
        if bool(deleted.iloc[-1]):
            continue                      # §1.2 hard delete
        watch[sym] = last_adm

    # -- §2 current structure --------------------------------------------- #
    live: dict[str, tuple[NestedBox, int]] = {}
    for sym, last_adm in watch.items():
        df = frames[sym]
        i = len(df) - 1
        nested = detect_nested_box(df, i, BP, symbol=sym)
        if nested is not None:
            live[sym] = (nested, last_adm)

    # -- §5 did an entry trigger fire in the last few sessions? -------------- #
    # Run before scoring, because the fast-resolution factor is defined as the
    # admission-to-*breakout* gap: it is a property of how a breakout happened,
    # and crediting it to a setup that has not broken out yet would invent
    # information. Pre-breakout names pass ``None`` and score its minimum,
    # which is what ``score_setup`` documents.
    fired_by: dict[str, tuple[int, pd.Timestamp]] = {}
    for sym in live:
        df = frames[sym]
        adm = screen(df, SP)
        adm_dates = list(adm[adm].index)
        if not adm_dates:
            continue
        sub = df.iloc[max(0, len(df) - WALK_BARS):]
        sub_adm = [d for d in adm_dates if d >= sub.index[0]]
        if not sub_adm:
            continue
        try:
            hist = scan_box_history(sub, BP, step=3)
            sigs = run_watchlist_for_symbol(sub, index_df, pd.Series(sub_adm), symbol=sym,
                                            box_params=BP, watch_params=WP, box_history=hist)
        except Exception:                       # one bad series must not kill the scan
            logger.warning("watchlist walk failed for %s", sym, exc_info=True)
            continue
        cutoff = sub.index[max(0, len(sub) - 1 - recent_bars)]
        recent = [s for s in sigs if s.entry_date >= cutoff]
        if recent:
            latest = max(recent, key=lambda s: s.entry_date)
            fired_by[sym] = (int(latest.entry_mode), latest.breakout_date)

    # -- §3 context, §4 score, §7 size --------------------------------------- #
    rows: list[dict[str, Any]] = []
    boxes: dict[str, tuple[NestedBox, pd.DataFrame]] = {}
    for sym, (nested, last_adm) in live.items():
        df = frames[sym]
        i = len(df) - 1
        mode_date = fired_by.get(sym)
        gap = float((mode_date[1] - df.index[last_adm]).days) if mode_date else None
        row = _describe(sym, df, index_df, i, nested, gap_days=gap, box_params=BP,
                        resistance_params=RP, timeframe_params=TP, rs_params=RSP,
                        score_params=SCP)
        a = _atr14(df, i)
        row["atr14"] = round(a, 2) if np.isfinite(a) else None
        plan = plan_trade(close=row["close"], atr14=a if np.isfinite(a) else 0.0,
                          big_box_bottom=row["box_bottom"], bucket=row["bucket"],
                          equity=equity, params=PP, regime_weak=regime_weak)
        row.update(entry_ref=plan.entry_ref, stop=plan.stop, risk_per_share=plan.risk_per_share,
                   stop_pct=plan.stop_pct, qty=plan.qty, notional=plan.notional,
                   risk_amount=plan.risk_amount, stop_basis=plan.stop_basis)
        row["actionable"] = 1 if mode_date else 0
        row["entry_mode"] = mode_date[0] if mode_date else None
        row["admission_gap_days"] = (
            float((df.index[i] - df.index[last_adm]).days))
        rows.append(row)
        boxes[sym] = (nested, df)

    out = pd.DataFrame(rows)
    if not out.empty:
        if universe_meta is not None and "symbol" in universe_meta:
            meta = universe_meta.set_index("symbol")
            out["company"] = out["symbol"].map(meta.get("company", pd.Series(dtype=str)))
            out["industry"] = out["symbol"].map(meta.get("industry", pd.Series(dtype=str)))
        out = out.sort_values(["actionable", "score"], ascending=[False, False]).reset_index(drop=True)

    charts: dict[str, str] = {}
    if render and not out.empty:
        charts = _render_charts(out, boxes, cfg, chart_dir, scan_date, BP)
        out["chart_path"] = out["symbol"].map(charts)

    audit_frame = pd.DataFrame()
    if audit:
        from .audit import audit_universe
        try:
            audit_frame = audit_universe(
                frames, index_df, out, screener_params=SP, box_params=BP, watch_params=WP)
            if len(audit_frame) and universe_meta is not None and "symbol" in universe_meta:
                meta = universe_meta.set_index("symbol")
                for col in ("company", "industry"):
                    if col in meta:
                        audit_frame[col] = audit_frame["symbol"].map(meta[col])
        except Exception:
            logger.exception("Universe audit failed; the scan itself is unaffected.")

    result = ScanResult(
        scan_date=scan_date, candidates=out, breadth=reading, audit=audit_frame,
        universe_size=len(frames), watchlist_size=len(watch),
        duration_s=round(time.time() - t0, 1), regime_weak=regime_weak, charts=charts,
    )

    if persist:
        store = store or AESScanStore(cfg=cfg)
        store.record_scan(
            scan_date, universe_size=len(frames), watchlist_size=len(watch),
            candidate_count=int(len(out)),
            actionable_count=int(out["actionable"].sum()) if not out.empty else 0,
            breadth=reading.value if reading else None,
            breadth_pctile=round(reading.percentile, 1) if reading else None,
            breadth_label=reading.label if reading else None,
            duration_s=result.duration_s,
        )
        if not out.empty:
            store.add_candidates(out.to_dict(orient="records"), scan_date)
        if len(audit_frame):
            store.add_audit(audit_frame.to_dict(orient="records"), scan_date)
    return result


def _render_charts(out, boxes, cfg, chart_dir, scan_date, box_params) -> dict[str, str]:
    """One box overlay per candidate worth looking at -- the Phase 1 render."""
    from ..aes.render import render_box

    base = Path(chart_dir) if chart_dir else Path(cfg.paths.results_dir) / "aes_scan_charts"
    day = base / scan_date.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    want = out[(out["actionable"] == 1) | (out["bucket"] == "high_conviction")]
    charts: dict[str, str] = {}
    for r in want.itertuples():
        nested, df = boxes.get(r.symbol, (None, None))
        if nested is None:
            continue
        note = (f"{r.symbol} | {r.bucket} {r.score:.2f} | box {r.box_bars}b "
                f"{r.box_range_pct:.0f}% | pos {r.box_position} | "
                f"{'ACTIONABLE mode ' + str(r.entry_mode) if r.actionable else 'watching'}")
        try:
            p = render_box(df, nested, day / f"{r.symbol}.png", params=box_params,
                           context_bars=70, forward_bars=0, note=note)
            charts[r.symbol] = str(p)
        except Exception:
            logger.warning("render failed for %s", r.symbol, exc_info=True)
    return charts


def _load_live(cfg: AppConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Refresh the universe's OHLCV and the benchmark."""
    from ..data.fetch_ohlcv import fetch_benchmark, fetch_ohlcv
    from ..data.universe import build_universe

    uni = build_universe(cfg=cfg)
    symbols = uni["symbol"].tolist()
    frames = fetch_ohlcv(symbols, cfg=cfg, show_progress=False)
    index_df = fetch_benchmark(cfg=cfg)
    return frames, index_df


def track_outcomes(
    store: AESScanStore,
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame | None = None,
    *,
    params: PortfolioParams | None = None,
    max_age_days: int = 120,
) -> int:
    """Extend the forward record for everything flagged actionable.

    Replays the adopted exit -- ATR 2.5x trailing stop with the 25-bar cap --
    from the entry reference, so the live record is measured the same way the
    backtest measured, not by a looser rule that would flatter it.
    """
    p = params or PortfolioParams()
    mult = p.atr_stop_mult or 2.5
    cap = p.max_hold_bars or 25
    pending = store.pending_outcomes(max_age_days=max_age_days)
    updated = 0
    unresolved: list[str] = []
    for r in pending.itertuples():
        df = frames.get(r.symbol)
        if df is None or not len(df):
            # A candidate whose symbol does not resolve can never accumulate a
            # forward record, and the failure is invisible unless it is said
            # out loud. This bites when two code paths key the same stock
            # differently -- ``fetch_ohlcv`` returns the true NSE symbol
            # (``M&MFIN``) while anything globbing the parquet cache sees the
            # filesystem-sanitised stem (``M_MFIN``).
            unresolved.append(str(r.symbol))
            continue
        sd = pd.Timestamp(r.scan_date)
        idx = df.index[df.index > sd]
        if not len(idx):
            continue
        i0 = int(df.index.get_indexer([idx[0]])[0])      # first bar after the flag
        entry = float(r.entry_ref) if r.entry_ref else float(df["close"].iloc[i0])
        if entry <= 0:
            continue
        c = df["close"].to_numpy(dtype=float)
        a = atr(df["high"], df["low"], df["close"], 14).to_numpy(dtype=float)
        n = len(c)
        peak = entry
        stop_hit, stop_bar, realised, reason = 0, None, None, "open"
        rets: dict[int, float | None] = {h: None for h in (5, 10, 15, 20, 25)}
        mfe = mae = 0.0
        bars = 0
        for k in range(0, min(cap, n - i0)):
            j = i0 + k
            px = c[j]
            bars = k + 1
            peak = max(peak, px)
            ret = (px / entry - 1) * 100
            mfe, mae = max(mfe, ret), min(mae, ret)
            if bars in rets:
                rets[bars] = round(ret, 2)
            if not stop_hit and np.isfinite(a[j]) and px < peak - mult * a[j]:
                stop_hit, stop_bar = 1, bars
                realised, reason = round(ret, 2), "atr_trail_stop"
                break
        if realised is None and bars >= cap:
            realised, reason = round((c[min(i0 + cap - 1, n - 1)] / entry - 1) * 100, 2), "max_hold"
        ex20 = None
        if index_df is not None and rets[20] is not None:
            ii = index_df.index.get_indexer([df.index[i0]], method="nearest")[0]
            if ii >= 0 and ii + 20 < len(index_df):
                bench = (float(index_df["close"].iloc[ii + 20]) / float(index_df["close"].iloc[ii]) - 1) * 100
                ex20 = round(rets[20] - bench, 2)
        store.upsert_outcome(int(r.id), {
            "as_of": df.index[-1].strftime("%Y-%m-%d"), "bars_elapsed": bars,
            "ret_5": rets[5], "ret_10": rets[10], "ret_15": rets[15],
            "ret_20": rets[20], "ret_25": rets[25], "excess_20": ex20,
            "mfe_pct": round(mfe, 2), "mae_pct": round(mae, 2),
            "stop_hit": stop_hit, "stop_hit_bar": stop_bar,
            "exit_reason": reason, "realised_pct": realised,
        })
        updated += 1
    if unresolved:
        logger.warning(
            "%d flagged candidate(s) had no price series and were not tracked: %s",
            len(unresolved), ", ".join(sorted(set(unresolved))[:20]),
        )
    return updated


__all__ = ["EQUAL_WEIGHTS", "ScanResult", "run_scan", "track_outcomes"]
