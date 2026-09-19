"""Where does SRDR cross break-even, and is that region robust or a knife edge?

The first portfolio run lost money while the underlying signal measured a real
+0.54% five-day edge. The two are consistent: at 20 equal-weight positions on a
Rs 10 lakh book each position is only Rs 50,000, and flat fees plus two-sided STT
plus slippage cost 0.65% round trip there. The edge is real and smaller than the
bill.

That makes the decisive question a sensitivity question rather than a
performance question: **under what execution assumptions, if any, does this
mechanism pay -- and is the profitable region broad enough to trade, or one
lucky corner?** A strategy that only works at a single slippage assumption is
not a strategy, it is a rounding error with a backtest.

The sweep varies the things that genuinely differ between traders -- book size,
concurrency, slippage -- plus the strategy's own two structural choices (hold
length, selection width). Reporting the whole surface rather than its best cell
is the point; a broad plateau of mildly positive results is worth more than one
spectacular cell surrounded by losses.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..backtest.costs import round_trip_cost_pct
from ..config import AppConfig, get_config
from ..strategy import srdr
from .protocol import DISCOVERY, VALIDATION, Window
from .srdr_study import load_everything, make_config, run_window, summarise

logger = logging.getLogger(__name__)


def sweep(
    prices: dict,
    industry_of: dict,
    benchmark: pd.DataFrame,
    base: AppConfig,
    *,
    windows: Sequence[Window] = (DISCOVERY, VALIDATION),
    positions: Sequence[int] = (10, 15, 20),
    slippages: Sequence[float] = (5.0, 8.0, 12.0, 15.0),
    holds: Sequence[int] = (3, 5, 8),
    top_ns: Sequence[int] = (2, 3, 5),
    capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Run the grid and return one row per (window, configuration).

    Features are rebuilt only when ``top_n`` changes, since that is the only
    swept parameter that alters signal generation; everything else is an
    execution or portfolio choice applied to the same signals.
    """
    rows: list[dict[str, Any]] = []

    for top_n in top_ns:
        feat_cfg = make_config(base, max_positions=20, top_n=top_n)
        features = srdr.build_signals(prices, industry_of, cfg=feat_cfg)
        if not features:
            continue

        for n_pos, slip, hold in itertools.product(positions, slippages, holds):
            cfg = make_config(
                base, max_positions=n_pos, top_n=top_n,
                hold_bars=hold, slippage_bps=slip,
            )
            cfg.risk.starting_capital = capital
            position_size = capital / n_pos
            cost_pct = round_trip_cost_pct(position_size, cfg.execution)

            for window in windows:
                result = run_window(features, prices, benchmark, window, cfg)
                summary = summarise(result, benchmark, window, cfg)
                if not summary.get("trades"):
                    continue
                rows.append(
                    {
                        "window": window.name,
                        "top_n": top_n,
                        "positions": n_pos,
                        "slippage_bps": slip,
                        "hold": hold,
                        "position_rs": position_size,
                        "cost_pct": cost_pct,
                        "trades": summary["trades"],
                        "per_year": summary["trades_per_year"],
                        "cagr": summary["cagr_pct"],
                        "bench_cagr": summary["benchmark_cagr_pct"],
                        "excess_cagr": summary["excess_cagr_pct"],
                        "sharpe": summary["sharpe"],
                        "maxdd": summary["max_drawdown_pct"],
                        "pf": summary["profit_factor"],
                        "avg_r": summary["avg_r"],
                        "exposure": summary["exposure_pct"],
                        "mean_hold": summary["mean_hold"],
                        "short_swing_ok": summary["short_swing_ok"],
                    }
                )
    return pd.DataFrame(rows)


def robustness_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Fraction of configurations that are profitable, per axis value.

    A mechanism with a real edge shows a broad band of mildly positive cells. A
    fitted one shows a scatter of extremes. This reduces the grid to that
    question, axis by axis.
    """
    rows = []
    for axis in ("top_n", "positions", "slippage_bps", "hold"):
        for value, group in frame.groupby(axis):
            rows.append(
                {
                    "axis": axis,
                    "value": value,
                    "configs": len(group),
                    "pct_positive_excess": float((group["excess_cagr"] > 0).mean() * 100),
                    "median_excess_cagr": float(group["excess_cagr"].median()),
                    "median_sharpe": float(group["sharpe"].median()),
                }
            )
    return pd.DataFrame(rows)


def run(
    *,
    cfg: AppConfig | None = None,
    capital: float = 1_000_000.0,
    windows: Sequence[Window] = (DISCOVERY, VALIDATION),
) -> pd.DataFrame:
    """Run the sensitivity sweep and print the surface."""
    base = cfg or get_config()
    prices, industry_of, benchmark = load_everything(base)

    frame = sweep(prices, industry_of, benchmark, base, windows=windows, capital=capital)
    if frame.empty:
        print("No configurations produced trades.")
        return frame

    print("\n" + "=" * 104)
    print(f"  SRDR SENSITIVITY SURFACE   (book Rs {capital:,.0f})")
    print("=" * 104)

    for window in windows:
        scoped = frame[frame["window"] == window.name]
        if scoped.empty:
            continue
        positive = int((scoped["excess_cagr"] > 0).sum())
        print(f"\n  {window.name}: {positive}/{len(scoped)} configurations beat the benchmark "
              f"({positive / len(scoped) * 100:.0f}%)")
        best = scoped.nlargest(8, "excess_cagr")
        print(f"    {'topN':>5}{'pos':>5}{'slip':>6}{'hold':>6}{'posRs':>9}{'cost%':>7}"
              f"{'/yr':>6}{'CAGR':>8}{'bench':>8}{'excess':>8}{'Sharpe':>8}{'DD%':>7}{'PF':>6}")
        for _, row in best.iterrows():
            print(
                f"    {int(row['top_n']):>5}{int(row['positions']):>5}{row['slippage_bps']:>6.0f}"
                f"{int(row['hold']):>6}{row['position_rs']:>9,.0f}{row['cost_pct']:>7.3f}"
                f"{row['per_year']:>6.0f}{row['cagr']:>8.2f}{row['bench_cagr']:>8.2f}"
                f"{row['excess_cagr']:>8.2f}{row['sharpe']:>8.2f}{row['maxdd']:>7.1f}{row['pf']:>6.2f}"
            )

    print("\n  ROBUSTNESS BY AXIS (share of configurations beating the benchmark)")
    scores = robustness_score(frame[frame["window"] == windows[0].name])
    print(f"    {'axis':<14}{'value':>8}{'configs':>9}{'% positive':>12}{'median excess':>15}{'median Sharpe':>15}")
    for _, row in scores.iterrows():
        print(
            f"    {row['axis']:<14}{row['value']:>8}{int(row['configs']):>9}"
            f"{row['pct_positive_excess']:>11.0f}%{row['median_excess_cagr']:>15.2f}"
            f"{row['median_sharpe']:>15.2f}"
        )

    print("\n  BREAK-EVEN: excess CAGR vs assumed slippage (median across configs)")
    for window in windows:
        scoped = frame[frame["window"] == window.name]
        if scoped.empty:
            continue
        line = f"    {window.name:<12}"
        for slip, group in scoped.groupby("slippage_bps"):
            line += f"  {slip:.0f}bps: {group['excess_cagr'].median():+6.2f}%"
        print(line)

    print("=" * 104 + "\n")

    out = (cfg or get_config()).paths.results_dir / "srdr_sensitivity.csv"
    frame.to_csv(out, index=False)
    logger.info("Sweep written to %s", out)
    return frame


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SRDR sensitivity sweep.")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    run(capital=args.capital)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
