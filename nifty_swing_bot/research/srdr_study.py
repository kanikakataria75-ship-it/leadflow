"""Portfolio evaluation of the SRDR strategy, window by window.

Runs the strategy through the shared-capital portfolio engine and reports the
full metric set, the holding-period distribution that decides whether it
qualifies as short-swing, trade frequency by year, and the benchmark comparison.

Position sizing is **equal weight**, achieved through the engine's existing
caps: the per-position notional cap is set to ``1 / max_open_positions`` and the
risk budget is set wide enough that the cap always binds. Every position is
therefore the same size, which is the right construction for a cross-sectional
strategy -- the signal ranks names against each other, so letting volatility
decide position size would silently re-weight the ranking.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult, PortfolioBacktester
from ..config import AppConfig, get_config
from ..strategy import srdr
from .protocol import (
    DISCOVERY,
    HOLDOUT,
    VALIDATION,
    FAST_CLOSE_BARS,
    Window,
    qualifies_as_short_swing,
    qualifies_on_frequency,
)

logger = logging.getLogger(__name__)


def make_config(
    base: AppConfig,
    *,
    max_positions: int = 20,
    hold_bars: int | None = None,
    top_n: int | None = None,
    stop_atr: float | None = None,
    slippage_bps: float | None = None,
) -> AppConfig:
    """Build a config for one SRDR run, with equal-weight sizing enforced."""
    cfg = base.model_copy(deep=True)
    if top_n is not None:
        cfg.srdr.top_n = top_n
    if hold_bars is not None:
        cfg.srdr.hold_bars = hold_bars
    if stop_atr is not None:
        cfg.srdr.stop_atr = stop_atr
    if slippage_bps is not None:
        cfg.execution.slippage_bps = slippage_bps

    cfg.risk.max_open_positions = max_positions
    # Equal weight: cap every position at 1/N of equity and make the risk
    # budget wide enough that the cap, not the volatility, decides the size.
    cfg.risk.max_position_pct = 1.0 / max_positions
    cfg.risk.risk_per_trade_pct = 0.10
    cfg.risk.max_portfolio_heat_pct = 10.0
    cfg.risk.max_hold_days = cfg.srdr.hold_bars
    cfg.risk.atr_stop_mult = cfg.srdr.stop_atr
    cfg.risk.trail_activate_r = 1e9      # no trailing: the exit is the clock
    cfg.risk.min_hold_days = 0
    return cfg


def run_window(
    features: Mapping[str, pd.DataFrame],
    prices: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    window: Window,
    cfg: AppConfig,
) -> BacktestResult:
    """Run the portfolio backtest inside one window."""
    return PortfolioBacktester(cfg).run(
        prices, {}, benchmark,
        start=pd.Timestamp(window.start), end=pd.Timestamp(window.end),
        features=dict(features),
    )


def holding_profile(result: BacktestResult) -> dict[str, Any]:
    """Holding-period distribution, which decides the short-swing question."""
    if not result.trades:
        return {}
    bars = np.array([t.bars_held for t in result.trades], dtype=float)
    return {
        "mean": float(bars.mean()),
        "median": float(np.median(bars)),
        "max": int(bars.max()),
        "p25": float(np.percentile(bars, 25)),
        "p75": float(np.percentile(bars, 75)),
        "fast_fraction": float((bars <= FAST_CLOSE_BARS).mean()),
        "histogram": {int(b): int((bars == b).sum()) for b in np.unique(bars)},
    }


def yearly_breakdown(result: BacktestResult, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Year-by-year strategy return, trade count and benchmark comparison."""
    if result.equity_curve.empty:
        return pd.DataFrame()

    equity = result.equity_curve["equity"]
    bench = benchmark["close"].copy()
    bench.index = pd.DatetimeIndex(bench.index).tz_localize(None).normalize()
    bench = bench.reindex(equity.index).ffill()

    trades = pd.DataFrame(
        [{"year": t.exit_date.year, "r": t.r_multiple, "net": t.net_pnl}
         for t in result.trades]
    )

    rows = []
    for year, group in equity.groupby(equity.index.year):
        if len(group) < 2:
            continue
        strat = float(group.iloc[-1] / group.iloc[0] - 1) * 100
        bench_group = bench.loc[group.index].dropna()
        bench_ret = (
            float(bench_group.iloc[-1] / bench_group.iloc[0] - 1) * 100
            if len(bench_group) > 1 else np.nan
        )
        year_trades = trades[trades["year"] == year] if not trades.empty else pd.DataFrame()
        rows.append(
            {
                "year": int(year),
                "strategy_pct": strat,
                "benchmark_pct": bench_ret,
                "excess_pct": strat - bench_ret if pd.notna(bench_ret) else np.nan,
                "trades": int(len(year_trades)),
                "win_pct": float((year_trades["r"] > 0).mean() * 100) if len(year_trades) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarise(
    result: BacktestResult,
    benchmark: pd.DataFrame,
    window: Window,
    cfg: AppConfig,
) -> dict[str, Any]:
    """Condense a run into the metric set the research brief asks for."""
    stats = result.stats
    if not stats.get("trade_count"):
        return {"window": window.name, "trades": 0}

    hold = holding_profile(result)
    years = (window.end - window.start).days / 365.25

    equity = result.equity_curve["equity"]
    bench = benchmark["close"].copy()
    bench.index = pd.DatetimeIndex(bench.index).tz_localize(None).normalize()
    bench = bench.reindex(equity.index).ffill().dropna()
    bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1) * 100 if len(bench) > 1 else np.nan
    bench_cagr = (
        (float(bench.iloc[-1] / bench.iloc[0]) ** (1 / max(years, 1e-9)) - 1) * 100
        if len(bench) > 1 else np.nan
    )
    bench_dd = 0.0
    if len(bench) > 1:
        peak = bench.cummax()
        bench_dd = float(-(bench / peak - 1).min()) * 100

    exposure = stats.get("avg_exposure_pct", 0.0)
    alpha = stats["total_return_pct"] - (bench_total * exposure / 100 if pd.notna(bench_total) else 0)

    short_swing_ok, ss_reasons = qualifies_as_short_swing(
        hold.get("mean", 99), hold.get("median", 99),
        hold.get("fast_fraction", 0.0), hold.get("max", 99),
    )
    freq_ok, freq_reasons = qualifies_on_frequency(stats["trade_count"], years)

    return {
        "window": window.name,
        "years": years,
        "trades": stats["trade_count"],
        "trades_per_year": stats["trade_count"] / years,
        "total_return_pct": stats["total_return_pct"],
        "cagr_pct": stats["cagr_pct"],
        "benchmark_total_pct": bench_total,
        "benchmark_cagr_pct": bench_cagr,
        "excess_cagr_pct": stats["cagr_pct"] - bench_cagr if pd.notna(bench_cagr) else np.nan,
        "alpha_pct": alpha,
        "sharpe": stats["sharpe"],
        "sortino": stats["sortino"],
        "calmar": stats["calmar"],
        "max_drawdown_pct": stats["max_drawdown_pct"],
        "benchmark_dd_pct": bench_dd,
        "win_rate_pct": stats["win_rate_pct"],
        "profit_factor": stats["profit_factor"],
        "avg_r": stats["avg_r_multiple"],
        "median_r": stats["median_r_multiple"],
        "expectancy_rs": stats["expectancy_rs"],
        "exposure_pct": exposure,
        "costs_rs": stats["total_costs_rs"],
        "mean_hold": hold.get("mean"),
        "median_hold": hold.get("median"),
        "max_hold": hold.get("max"),
        "fast_fraction": hold.get("fast_fraction"),
        "histogram": hold.get("histogram"),
        "short_swing_ok": short_swing_ok,
        "short_swing_reasons": ss_reasons,
        "frequency_ok": freq_ok,
        "frequency_reasons": freq_reasons,
        "exit_reasons": stats.get("exit_reasons", {}),
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    """Render one window's results."""
    if not summary.get("trades"):
        return f"  {summary.get('window')}: no trades."

    s = summary
    lines = [
        "",
        "-" * 84,
        f"  {s['window']}   ({s['years']:.1f} years)",
        "-" * 84,
        "  RETURNS                              RISK",
        f"    Total return     {s['total_return_pct']:>9.2f}%        Max drawdown     {s['max_drawdown_pct']:>8.2f}%",
        f"    CAGR             {s['cagr_pct']:>9.2f}%        Benchmark DD     {s['benchmark_dd_pct']:>8.2f}%",
        f"    Benchmark CAGR   {s['benchmark_cagr_pct']:>9.2f}%        Sharpe           {s['sharpe']:>8.2f}",
        f"    Excess CAGR      {s['excess_cagr_pct']:>9.2f}%        Sortino          {s['sortino']:>8.2f}",
        f"    Alpha (exp-adj)  {s['alpha_pct']:>9.2f}%        Calmar           {s['calmar']:>8.2f}",
        "",
        "  TRADES                               HOLDING PERIOD",
        f"    Count            {s['trades']:>9}         Mean             {s['mean_hold']:>8.2f} bars",
        f"    Per year         {s['trades_per_year']:>9.0f}         Median           {s['median_hold']:>8.1f} bars",
        f"    Win rate         {s['win_rate_pct']:>9.2f}%        Max              {s['max_hold']:>8} bars",
        f"    Profit factor    {s['profit_factor']:>9.2f}         Closed <= {FAST_CLOSE_BARS}d    {s['fast_fraction']:>8.0%}",
        f"    Avg R            {s['avg_r']:>9.3f}         Exposure         {s['exposure_pct']:>8.1f}%",
        f"    Expectancy    Rs {s['expectancy_rs']:>9,.0f}         Costs         Rs {s['costs_rs']:>8,.0f}",
    ]
    if s.get("histogram"):
        total = sum(s["histogram"].values())
        bars = " ".join(
            f"{k}d:{v * 100 // total}%" for k, v in sorted(s["histogram"].items())
        )
        lines.append(f"    Hold distribution  {bars}")
    if s.get("exit_reasons"):
        mix = " ".join(f"{k}:{v}" for k, v in sorted(
            s["exit_reasons"].items(), key=lambda kv: -kv[1]))
        lines.append(f"    Exit mix           {mix}")

    lines.append("")
    ss = "PASS" if s["short_swing_ok"] else "FAIL: " + "; ".join(s["short_swing_reasons"])
    fr = "PASS" if s["frequency_ok"] else "FAIL: " + "; ".join(s["frequency_reasons"])
    lines.append(f"    Short-swing check  {ss}")
    lines.append(f"    Frequency check    {fr}")
    lines.append("-" * 84)
    return "\n".join(lines)


def load_everything(cfg: AppConfig) -> tuple[dict, dict, pd.DataFrame]:
    """Load prices, industry map and benchmark once for reuse across windows."""
    from .event_study import load_panel

    prices, industry_of, benchmark = load_panel(cfg=cfg)
    return prices, industry_of, benchmark


def run(
    *,
    windows: Sequence[Window] = (DISCOVERY, VALIDATION),
    max_positions: int = 20,
    cfg: AppConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run SRDR through the portfolio engine on the given windows."""
    base = cfg or get_config()
    prices, industry_of, benchmark = load_everything(base)

    run_cfg = make_config(base, max_positions=max_positions)
    features = srdr.build_signals(prices, industry_of, cfg=run_cfg)
    if not features:
        raise RuntimeError("SRDR produced no features.")

    results: dict[str, Any] = {}
    for window in windows:
        result = run_window(features, prices, benchmark, window, run_cfg)
        summary = summarise(result, benchmark, window, run_cfg)
        results[window.name] = {"result": result, "summary": summary}
        if verbose:
            print(format_summary(summary))
            yearly = yearly_breakdown(result, benchmark)
            if not yearly.empty:
                print(f"\n    {'year':>6}{'strategy':>11}{'benchmark':>12}{'excess':>10}"
                      f"{'trades':>9}{'win%':>8}")
                for _, row in yearly.iterrows():
                    print(
                        f"    {int(row['year']):>6}{row['strategy_pct']:>10.2f}%"
                        f"{row['benchmark_pct']:>11.2f}%{row['excess_pct']:>9.2f}%"
                        f"{int(row['trades']):>9}"
                        f"{row['win_pct']:>7.1f}%" if pd.notna(row['win_pct'])
                        else f"    {int(row['year']):>6}{row['strategy_pct']:>10.2f}%"
                             f"{row['benchmark_pct']:>11.2f}%{row['excess_pct']:>9.2f}%"
                             f"{int(row['trades']):>9}{'—':>8}"
                    )
                print()
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SRDR portfolio study.")
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--holdout", action="store_true",
                        help="Include the holdout window. Use once, at the end.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    windows = [DISCOVERY, VALIDATION] + ([HOLDOUT] if args.holdout else [])
    run(windows=windows, max_positions=args.positions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
