"""Edge decomposition: does the signal predict anything, or was it the tape?

A losing backtest has several possible causes, and they demand opposite fixes:

1. **The market fell.** A long-only breakout system loses in a declining
   small-cap tape regardless of entry quality. Fix: a regime filter, or trade
   the short side. Nothing about the signal needs changing.
2. **The signal has no predictive content.** Entries are no better than random
   picks from the same universe. Fix: change or drop the rules.
3. **The signal predicts, but the exits give it back.** Trades go favourable and
   then die. Fix: exits, not entries.

This module separates those. The central measurement is **excess forward
return**: the stock's return from the next open over N bars, minus the
benchmark's return over the same window. Subtracting the index strips out
market direction, so what remains is the signal's own skill. A rule that
produces positive excess return is finding something even if the raw backtest
lost money.

Everything here is measured on the same next-open entry the backtester uses, so
these numbers are directly comparable to it.

Usage::

    python -m nifty_swing_bot.research.diagnostics
    python -m nifty_swing_bot.research.diagnostics --limit 150 --horizons 5,10,20
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.engine import PortfolioBacktester
from ..config import AppConfig, get_config

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 15, 20)

#: The individual DAB rule columns, plus the composite.
RULE_COLUMNS: tuple[str, ...] = (
    "cond_delivery", "cond_volume", "cond_close", "cond_rs", "cond_setup", "signal",
)


def build_event_panel(
    features: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Stack every evaluable bar into one panel with forward returns attached.

    For each bar the entry is the **next** bar's open, matching the backtester,
    and the exit is the close N bars later. The benchmark return is measured
    over the identical window so the difference is a clean excess.

    Args:
        features: Symbol -> feature frame from ``compute_features``.
        benchmark: Benchmark OHLCV.
        horizons: Forward horizons in bars.

    Returns:
        Long frame with one row per (symbol, bar), carrying the rule booleans,
        the raw forward returns ``fwd_{h}`` and the excess returns ``exc_{h}``.
    """
    bench_close = benchmark["close"].copy()
    bench_close.index = pd.DatetimeIndex(bench_close.index).tz_localize(None).normalize()
    bench_close = bench_close[~bench_close.index.duplicated(keep="last")].sort_index()

    chunks: list[pd.DataFrame] = []
    for symbol, feats in features.items():
        if feats is None or feats.empty or "signal" not in feats:
            continue

        entry = feats["open"].shift(-1)          # tradeable next-open entry
        bench = bench_close.reindex(feats.index).ffill()
        bench_entry = bench.shift(-1)

        frame = pd.DataFrame(index=feats.index)
        frame["symbol"] = symbol
        for col in RULE_COLUMNS:
            frame[col] = feats[col].fillna(False).astype(bool) if col in feats else False
        for col in ("deliv_spike_ratio", "vol_ratio", "range_position",
                    "rs_excess", "atr_ratio", "dist_from_high", "is_breakout"):
            frame[col] = feats[col] if col in feats else np.nan

        for h in horizons:
            fwd = feats["close"].shift(-h) / entry - 1.0
            bfwd = bench.shift(-h) / bench_entry - 1.0
            frame[f"fwd_{h}"] = fwd
            frame[f"exc_{h}"] = fwd - bfwd

        # A bar is only evaluable once every rule has a real value behind it.
        frame["evaluable"] = (
            feats.get("deliv_ratio", pd.Series(np.nan, index=feats.index)).notna()
            & feats.get("vol_ratio", pd.Series(np.nan, index=feats.index)).notna()
            & feats.get("ret_bench", pd.Series(np.nan, index=feats.index)).notna()
            & feats.get("high_n", pd.Series(np.nan, index=feats.index)).notna()
        )
        chunks.append(frame)

    if not chunks:
        return pd.DataFrame()
    panel = pd.concat(chunks)
    panel.index.name = "date"
    return panel.reset_index()


def summarise(
    panel: pd.DataFrame,
    mask: pd.Series,
    label: str,
    *,
    horizons: Sequence[int],
    baseline: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Summarise forward performance for a subset of bars.

    Args:
        panel: The event panel.
        mask: Boolean selector over ``panel``.
        label: Name for the row.
        horizons: Horizons to report.
        baseline: Base-rate excess returns to compare against, per horizon.

    Returns:
        A dict with the count and, per horizon, mean excess return, hit rate and
        a t-statistic on the excess return.
    """
    subset = panel[mask]
    out: dict[str, Any] = {"label": label, "n": int(len(subset))}
    for h in horizons:
        col = f"exc_{h}"
        values = subset[col].dropna()
        if values.empty:
            out[f"exc_{h}"] = np.nan
            out[f"hit_{h}"] = np.nan
            out[f"t_{h}"] = np.nan
            continue
        mean = float(values.mean())
        out[f"exc_{h}"] = mean * 100
        out[f"hit_{h}"] = float((values > 0).mean()) * 100
        sd = float(values.std(ddof=1))
        # A t-stat on overlapping windows overstates significance, so treat it
        # as a rough ranking device rather than a formal test.
        out[f"t_{h}"] = mean / (sd / np.sqrt(len(values))) if sd > 0 else np.nan
        if baseline is not None and h in baseline:
            out[f"vs_base_{h}"] = (mean - baseline[h]) * 100
    return out


def rule_ablation(
    panel: pd.DataFrame, *, horizons: Sequence[int] = DEFAULT_HORIZONS
) -> pd.DataFrame:
    """Forward performance of each rule alone, and of the full signal.

    The comparison that matters is each row against ``UNIVERSE BASE RATE``. A
    rule whose excess return is no better than the base rate is contributing
    nothing except a reduction in sample size.
    """
    evaluable = panel[panel["evaluable"]]
    if evaluable.empty:
        return pd.DataFrame()

    base = {h: float(evaluable[f"exc_{h}"].dropna().mean()) for h in horizons}
    rows = [summarise(evaluable, pd.Series(True, index=evaluable.index),
                      "UNIVERSE BASE RATE", horizons=horizons)]

    pretty = {
        "cond_delivery": "1. Delivery accumulation",
        "cond_volume": "2. Volume surge",
        "cond_close": "3. Strong close",
        "cond_rs": "4. Relative strength",
        "cond_setup": "5. Breakout / VCP",
        "signal": "ALL FIVE (DAB signal)",
    }
    for col, label in pretty.items():
        if col in evaluable:
            rows.append(
                summarise(evaluable, evaluable[col], label, horizons=horizons, baseline=base)
            )

    # The four non-delivery rules together, to isolate what rule 1 adds.
    others = (
        evaluable["cond_volume"] & evaluable["cond_close"]
        & evaluable["cond_rs"] & evaluable["cond_setup"]
    )
    rows.append(summarise(evaluable, others, "Rules 2-5 (no delivery)",
                          horizons=horizons, baseline=base))
    rows.append(summarise(evaluable, others & evaluable["cond_delivery"],
                          "Rules 2-5 + delivery", horizons=horizons, baseline=base))
    return pd.DataFrame(rows)


def regime_check(
    panel: pd.DataFrame, benchmark: pd.DataFrame, *, horizons: Sequence[int] = DEFAULT_HORIZONS
) -> dict[str, Any]:
    """Was the tape itself the problem?

    Reports the universe's own average forward return (unadjusted) against the
    benchmark's move over the same period. If the universe base rate is
    negative, every long-only system in it starts from behind.
    """
    evaluable = panel[panel["evaluable"]]
    out: dict[str, Any] = {}
    for h in horizons:
        raw = evaluable[f"fwd_{h}"].dropna()
        exc = evaluable[f"exc_{h}"].dropna()
        out[f"universe_raw_{h}"] = float(raw.mean()) * 100 if not raw.empty else np.nan
        out[f"universe_exc_{h}"] = float(exc.mean()) * 100 if not exc.empty else np.nan
        out[f"universe_hit_{h}"] = float((raw > 0).mean()) * 100 if not raw.empty else np.nan

    close = benchmark["close"].dropna()
    if len(close) > 1:
        out["benchmark_total_pct"] = float(close.iloc[-1] / close.iloc[0] - 1) * 100
    return out


def hold_period_sweep(
    panel: pd.DataFrame, *, horizons: Sequence[int] = (3, 5, 10, 15, 20, 30, 40)
) -> pd.DataFrame:
    """How signal performance evolves with holding period.

    The backtest exited 55% of trades on the 10-bar clock. If excess return is
    still climbing at 20-30 bars, the hold is simply too short and the exit rule
    is cutting the thesis off before it plays out.
    """
    evaluable = panel[panel["evaluable"]]
    if evaluable.empty:
        return pd.DataFrame()
    signal = evaluable[evaluable["signal"]]

    rows = []
    for h in horizons:
        col = f"exc_{h}"
        if col not in evaluable:
            continue
        sig = signal[col].dropna()
        base = evaluable[col].dropna()
        rows.append(
            {
                "bars": h,
                "n": int(len(sig)),
                "signal_exc_pct": float(sig.mean()) * 100 if not sig.empty else np.nan,
                "base_exc_pct": float(base.mean()) * 100 if not base.empty else np.nan,
                "edge_pct": (float(sig.mean()) - float(base.mean())) * 100
                if not sig.empty and not base.empty else np.nan,
                "hit_pct": float((sig > 0).mean()) * 100 if not sig.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def exit_analysis(trades_csv: pd.DataFrame) -> dict[str, Any]:
    """Was there profit on the table that the exits failed to capture?

    Uses MFE (maximum favourable excursion) from the trade log. If the average
    trade reaches a healthy positive MFE but closes near zero or negative, the
    entries were finding moves and the exits were surrendering them.
    """
    if trades_csv.empty:
        return {}
    mfe = pd.to_numeric(trades_csv.get("mfe_r"), errors="coerce").dropna()
    mae = pd.to_numeric(trades_csv.get("mae_r"), errors="coerce").dropna()
    r = pd.to_numeric(trades_csv.get("r_multiple"), errors="coerce").dropna()

    out: dict[str, Any] = {
        "trades": int(len(trades_csv)),
        "avg_mfe_r": float(mfe.mean()) if not mfe.empty else np.nan,
        "avg_mae_r": float(mae.mean()) if not mae.empty else np.nan,
        "avg_realised_r": float(r.mean()) if not r.empty else np.nan,
        "pct_reaching_1r": float((mfe >= 1.0).mean()) * 100 if not mfe.empty else np.nan,
        "pct_reaching_2r": float((mfe >= 2.0).mean()) * 100 if not mfe.empty else np.nan,
        "pct_reaching_3r": float((mfe >= 3.0).mean()) * 100 if not mfe.empty else np.nan,
    }
    if not mfe.empty and not r.empty:
        # How much of the best unrealised gain actually reached the account.
        reached_1r = trades_csv[mfe >= 1.0] if len(mfe) == len(trades_csv) else pd.DataFrame()
        if not reached_1r.empty:
            out["avg_r_given_reached_1r"] = float(
                pd.to_numeric(reached_1r["r_multiple"], errors="coerce").mean()
            )
        out["capture_ratio"] = float(r.mean() / mfe.mean()) if mfe.mean() else np.nan

    if "exit_reason" in trades_csv:
        by_reason = (
            trades_csv.assign(r=pd.to_numeric(trades_csv["r_multiple"], errors="coerce"))
            .groupby("exit_reason")["r"]
            .agg(["count", "mean"])
            .sort_values("count", ascending=False)
        )
        out["by_exit_reason"] = {
            str(k): {"n": int(v["count"]), "avg_r": round(float(v["mean"]), 3)}
            for k, v in by_reason.iterrows()
        }
    return out


def _fmt_table(frame: pd.DataFrame, horizons: Sequence[int]) -> str:
    """Render an ablation table as fixed-width text."""
    if frame.empty:
        return "  (no data)"
    header = f"  {'':<26}{'N':>8}"
    for h in horizons:
        header += f"{f'{h}d exc%':>10}{'hit%':>8}"
    lines = [header, "  " + "-" * (26 + 8 + 18 * len(horizons))]
    for _, row in frame.iterrows():
        line = f"  {row['label']:<26}{int(row['n']):>8,}"
        for h in horizons:
            exc = row.get(f"exc_{h}")
            hit = row.get(f"hit_{h}")
            line += f"{exc:>10.3f}" if pd.notna(exc) else f"{'—':>10}"
            line += f"{hit:>8.1f}" if pd.notna(hit) else f"{'—':>8}"
        lines.append(line)
    return "\n".join(lines)


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    limit: int | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Run the full diagnostic suite and print a report."""
    from ..backtest.run_backtest import load_market_data
    from ..data.fetch_delivery import cached_delivery_days

    cfg = cfg or get_config()
    available = cached_delivery_days(cfg)
    if not available:
        raise RuntimeError("No delivery data cached. Run the backfill first.")
    start = start or available[min(35, len(available) - 1)]
    end = end or available[-1]

    logger.info("Diagnostics window: %s to %s", start, end)
    _, prices, delivery, benchmark = load_market_data(
        start=start, end=end, limit=limit, cfg=cfg
    )
    features = PortfolioBacktester(cfg).compute_all_features(prices, delivery, benchmark)
    panel = build_event_panel(features, benchmark, horizons=tuple(set(horizons) | {3, 30, 40}))

    if panel.empty:
        raise RuntimeError("Event panel is empty; check the data window.")

    regime = regime_check(panel, benchmark, horizons=horizons)
    ablation = rule_ablation(panel, horizons=horizons)
    sweep = hold_period_sweep(panel)

    trades = pd.DataFrame()
    for name in ("trades_full.csv", "trades.csv"):
        path = cfg.paths.results_dir / name
        if path.exists():
            trades = pd.read_csv(path)
            break
    exits = exit_analysis(trades)

    # ---------------------------------------------------------------- report --
    print("\n" + "=" * 78)
    print("  DAB EDGE DECOMPOSITION")
    print(f"  {start} to {end} · {len(features)} symbols · {len(panel):,} bars")
    print("=" * 78)

    print("\n  1. WAS IT THE TAPE?")
    print(f"     Benchmark over the window        {regime.get('benchmark_total_pct', float('nan')):+.2f}%")
    for h in horizons:
        print(
            f"     Universe avg {h:>2}-bar return      "
            f"{regime.get(f'universe_raw_{h}', float('nan')):+.3f}%   "
            f"(vs index {regime.get(f'universe_exc_{h}', float('nan')):+.3f}%, "
            f"{regime.get(f'universe_hit_{h}', float('nan')):.1f}% positive)"
        )

    print("\n  2. DOES ANY RULE PREDICT? (excess return vs the index, in %)")
    print(_fmt_table(ablation, horizons))
    print("\n     Compare every row against UNIVERSE BASE RATE. A rule that does not")
    print("     beat it is only shrinking the sample, not adding information.")

    print("\n  3. IS THE 10-BAR HOLD TOO SHORT?")
    if not sweep.empty:
        print(f"     {'bars':>6}{'N':>9}{'signal exc%':>14}{'base exc%':>12}{'edge%':>9}{'hit%':>8}")
        print("     " + "-" * 58)
        for _, row in sweep.iterrows():
            print(
                f"     {int(row['bars']):>6}{int(row['n']):>9,}"
                f"{row['signal_exc_pct']:>14.3f}{row['base_exc_pct']:>12.3f}"
                f"{row['edge_pct']:>9.3f}{row['hit_pct']:>8.1f}"
            )

    if exits:
        print("\n  4. ARE THE EXITS LEAVING MONEY BEHIND?")
        print(f"     Trades                          {exits.get('trades')}")
        print(f"     Avg max favourable excursion    {exits.get('avg_mfe_r', float('nan')):+.3f}R")
        print(f"     Avg max adverse excursion       {exits.get('avg_mae_r', float('nan')):+.3f}R")
        print(f"     Avg realised                    {exits.get('avg_realised_r', float('nan')):+.3f}R")
        print(f"     Capture ratio (realised/MFE)    {exits.get('capture_ratio', float('nan')):.3f}")
        print(f"     Reached +1R at some point       {exits.get('pct_reaching_1r', float('nan')):.1f}%")
        print(f"     Reached +2R at some point       {exits.get('pct_reaching_2r', float('nan')):.1f}%")
        print(f"     Reached +3R at some point       {exits.get('pct_reaching_3r', float('nan')):.1f}%")
        if "avg_r_given_reached_1r" in exits:
            print(
                "     Avg R of trades that hit +1R    "
                f"{exits['avg_r_given_reached_1r']:+.3f}R"
            )
        if "by_exit_reason" in exits:
            print("\n     By exit reason:")
            for reason, stats in exits["by_exit_reason"].items():
                print(f"       {reason:<24}{stats['n']:>5}   avg {stats['avg_r']:+.3f}R")

    print("\n" + "=" * 78 + "\n")

    return {"regime": regime, "ablation": ablation, "sweep": sweep, "exits": exits, "panel": panel}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Decompose the DAB strategy's edge.")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--horizons", default="5,10,20",
                        help="Comma-separated forward horizons in bars.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    horizons = tuple(int(h) for h in args.horizons.split(","))
    run(start=args.start, end=args.end, limit=args.limit, horizons=horizons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
