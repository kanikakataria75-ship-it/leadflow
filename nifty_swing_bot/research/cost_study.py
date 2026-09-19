"""Does the mean-reversion edge survive realistic costs, and does it grow with the hold?

Two questions, answered side by side:

1. **Cost realism.** The early model charged brokerage as a percentage and STT
   on the sell side only. Neither is right for an Indian delivery trade. This
   re-prices the strategy with flat brokerage, STT on *both* sides, stamp duty,
   exchange and SEBI fees, GST and the depository charge -- and, separately,
   with the lower slippage a liquidity-restricted universe justifies.

2. **Hold period.** Charges are paid once per round trip regardless of how long
   the position is held, so a longer hold amortises them. The question is
   whether the signal's edge grows with the horizon faster than it decays into
   the market's own drift.

Both are reported as **net edge**: the signal's excess forward return over the
universe base rate, minus the round-trip cost of capturing it. Net edge above
zero is the necessary condition for the strategy to be worth trading; the
portfolio backtest at the bottom is the sufficient one.

Usage::

    python -m nifty_swing_bot.research.cost_study
    python -m nifty_swing_bot.research.cost_study --holds 10,15,20 --quantile 0.75
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.costs import cost_breakdown_pct, round_trip_cost_pct
from ..config import AppConfig, ExecutionParams, get_config
from ..strategy import mr_strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CostScenario:
    """A named combination of cost assumptions and universe restriction."""

    name: str
    slippage_bps: float
    cost_model: str
    liquidity_quantile: float | None
    note: str


def default_scenarios(quantile: float) -> tuple[CostScenario, ...]:
    """The scenarios worth comparing, from the original model to the realistic one."""
    return (
        CostScenario(
            "1. Original model",
            slippage_bps=15.0,
            cost_model="simple_bps",
            liquidity_quantile=None,
            note="% brokerage, STT on sell only -- what the first backtests used",
        ),
        CostScenario(
            "2. Realistic charges",
            slippage_bps=15.0,
            cost_model="india_delivery",
            liquidity_quantile=None,
            note="Flat Rs 20 brokerage, but STT on BOTH sides + stamp/GST/DP",
        ),
        CostScenario(
            "3. + liquid quartile",
            slippage_bps=6.0,
            cost_model="india_delivery",
            liquidity_quantile=quantile,
            note="Top-quartile turnover justifies 6bps slippage instead of 15",
        ),
        CostScenario(
            "4. + zero brokerage",
            slippage_bps=6.0,
            cost_model="india_delivery",
            liquidity_quantile=quantile,
            note="Several brokers charge Rs 0 on delivery; best realistic case",
        ),
    )


def _scenario_config(base: AppConfig, scenario: CostScenario) -> AppConfig:
    """Apply a scenario's assumptions to a copy of the config."""
    cfg = base.model_copy(deep=True)
    cfg.execution.cost_model = scenario.cost_model  # type: ignore[assignment]
    cfg.execution.slippage_bps = scenario.slippage_bps
    cfg.universe.liquidity_quantile = scenario.liquidity_quantile
    if scenario.name.startswith("4."):
        cfg.execution.brokerage_flat_inr = 0.0
    return cfg


def measure_edges(
    features: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    *,
    holds: Sequence[int],
    split_frac: float = 0.6,
) -> pd.DataFrame:
    """Signal edge over the universe base rate, per holding period.

    The edge is the mean excess forward return of signal bars minus the mean
    excess forward return of *all* evaluable bars, so it measures selection
    skill rather than market direction. Reported on a train period and a
    date-separated holdout.

    Args:
        features: Symbol -> PBR feature frame.
        benchmark: Benchmark OHLCV.
        holds: Holding periods in bars.
        split_frac: Fraction of the date range used as train.

    Returns:
        One row per holding period with train/holdout edges, sample sizes and
        t-statistics.
    """
    bench = benchmark["close"].copy()
    bench.index = pd.DatetimeIndex(bench.index).tz_localize(None).normalize()
    bench = bench[~bench.index.duplicated(keep="last")].sort_index()

    chunks: list[pd.DataFrame] = []
    for symbol, f in features.items():
        if f is None or f.empty or "signal" not in f or len(f) < 80:
            continue
        entry = f["open"].shift(-1)
        b = bench.reindex(f.index).ffill()
        frame = pd.DataFrame(index=f.index)
        frame["symbol"] = symbol
        frame["signal"] = f["signal"].fillna(False).astype(bool)
        for h in holds:
            fwd = f["close"].shift(-h) / entry - 1.0
            bfwd = b.shift(-h) / b.shift(-1) - 1.0
            frame[f"exc_{h}"] = fwd - bfwd
        chunks.append(frame)

    if not chunks:
        return pd.DataFrame()

    panel = pd.concat(chunks)
    panel.index.name = "date"
    panel = panel.reset_index()

    dates = np.sort(panel["date"].unique())
    cut = dates[int(len(dates) * split_frac)]
    train, holdout = panel[panel["date"] < cut], panel[panel["date"] >= cut]

    rows: list[dict[str, Any]] = []
    for h in holds:
        col = f"exc_{h}"
        row: dict[str, Any] = {"hold": h}
        for label, subset in (("train", train), ("hold", holdout)):
            values = subset[col].dropna()
            sig = subset.loc[subset["signal"], col].dropna()
            if values.empty or sig.empty:
                row[f"{label}_n"] = 0
                row[f"{label}_edge"] = np.nan
                row[f"{label}_t"] = np.nan
                continue
            base = float(values.mean())
            mean = float(sig.mean())
            sd = float(sig.std(ddof=1)) if len(sig) > 1 else np.nan
            row[f"{label}_n"] = int(len(sig))
            row[f"{label}_base"] = base * 100
            row[f"{label}_gross"] = mean * 100
            row[f"{label}_edge"] = (mean - base) * 100
            row[f"{label}_t"] = (
                (mean - base) / (sd / np.sqrt(len(sig))) if sd and sd > 0 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def cost_table(notionals: Sequence[float], execp: ExecutionParams) -> pd.DataFrame:
    """Round-trip cost as a percentage, for a range of position sizes."""
    return pd.DataFrame(
        [{"notional": n, **cost_breakdown_pct(n, execp)} for n in notionals]
    )


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    holds: Sequence[int] = (10, 15, 20),
    quantile: float = 0.75,
    typical_notional: float = 100_000.0,
    cfg: AppConfig | None = None,
    run_backtests: bool = True,
    only: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run the full cost and hold-period study and print the report."""
    from ..backtest.engine import PortfolioBacktester
    from ..backtest.run_backtest import load_market_data
    from ..data.fetch_delivery import cached_delivery_days

    base = cfg or get_config()
    available = cached_delivery_days(base)
    if not available:
        raise RuntimeError("No delivery data cached. Run the backfill first.")
    start = start or available[min(35, len(available) - 1)]
    end = end or available[-1]

    scenarios = default_scenarios(quantile)
    if only:
        # 1-based selection so it matches the numbering in the report.
        scenarios = tuple(scenarios[i - 1] for i in only if 1 <= i <= len(scenarios))
    results: dict[str, Any] = {"scenarios": [], "start": start, "end": end}

    print("\n" + "=" * 100)
    print("  COST REALISM & HOLDING-PERIOD STUDY -- Pullback Reversion")
    print(f"  {start} to {end}")
    print("=" * 100)

    # ---------------------------------------------------------------- costs --
    print("\n  ROUND-TRIP COST BY POSITION SIZE (%, including slippage)")
    print(f"  {'scenario':<24}{'Rs 25k':>9}{'Rs 50k':>9}{'Rs 1L':>9}{'Rs 2L':>9}   composition at Rs 1L")
    print("  " + "-" * 96)
    for scenario in scenarios:
        scfg = _scenario_config(base, scenario)
        sizes = [25_000.0, 50_000.0, 100_000.0, 200_000.0]
        costs = [round_trip_cost_pct(n, scfg.execution) for n in sizes]
        parts = cost_breakdown_pct(typical_notional, scfg.execution)
        comp = (
            f"stt {parts['stt']:.2f} slip {parts['slippage']:.2f} "
            f"brok {parts['brokerage']:.2f} other "
            f"{parts['TOTAL'] - parts['stt'] - parts['slippage'] - parts['brokerage']:.2f}"
        )
        print(
            f"  {scenario.name:<24}" + "".join(f"{c:>9.3f}" for c in costs) + f"   {comp}"
        )
    print("  " + "-" * 96)
    for scenario in scenarios:
        print(f"    {scenario.name:<24} {scenario.note}")

    # ------------------------------------------------------- edge vs costs --
    for scenario in scenarios:
        scfg = _scenario_config(base, scenario)
        _, prices, delivery, benchmark = load_market_data(
            start=start, end=end, cfg=scfg
        )
        feats: dict[str, pd.DataFrame] = {}
        for symbol, ohlcv in prices.items():
            if ohlcv is None or len(ohlcv) < scfg.universe.min_history_days:
                continue
            frame = mr_strategy.compute_features(ohlcv, delivery.get(symbol), cfg=scfg)
            if not frame.empty:
                feats[symbol] = frame

        edges = measure_edges(feats, benchmark, holds=holds)
        rt_cost = round_trip_cost_pct(typical_notional, scfg.execution)

        print("\n" + "=" * 100)
        print(f"  {scenario.name}   |   {len(feats)} symbols   |   "
              f"round-trip cost at Rs 1L = {rt_cost:.3f}%")
        print("=" * 100)
        print(f"  {'hold':>5}{'signals':>10}{'gross%':>10}{'base%':>9}{'EDGE%':>9}{'t':>7}"
              f"{'cost%':>8}{'NET EDGE%':>11}   verdict")
        print("  " + "-" * 96)

        scenario_rows = []
        for _, row in edges.iterrows():
            for label, tag in (("hold", "OOS"), ("train", "IS ")):
                edge = row.get(f"{label}_edge", np.nan)
                if not np.isfinite(edge):
                    continue
                net = edge - rt_cost
                verdict = "POSITIVE" if net > 0 else "negative"
                if label == "train":
                    verdict += " (in-sample)"
                print(
                    f"  {int(row['hold']):>5}{int(row[f'{label}_n']):>10,}"
                    f"{row.get(f'{label}_gross', np.nan):>10.3f}"
                    f"{row.get(f'{label}_base', np.nan):>9.3f}{edge:>9.3f}"
                    f"{row.get(f'{label}_t', np.nan):>7.2f}{rt_cost:>8.3f}{net:>11.3f}"
                    f"   {tag} {verdict}"
                )
                scenario_rows.append(
                    {"scenario": scenario.name, "hold": int(row["hold"]), "period": tag.strip(),
                     "edge": edge, "cost": rt_cost, "net_edge": net}
                )
        results["scenarios"].append({"scenario": scenario.name, "rows": scenario_rows})

        # ------------------------------------------------ portfolio backtest --
        if run_backtests:
            print("\n  PORTFOLIO BACKTEST (same scenario, all costs applied by the engine)")
            print(f"  {'hold':>5}{'trades':>8}{'win%':>7}{'W/L':>6}{'PF':>6}"
                  f"{'ret%':>8}{'bench%':>8}{'alpha':>8}{'DD%':>7}{'exp%':>6}")
            print("  " + "-" * 74)
            bclose = benchmark["close"].copy()
            bclose.index = pd.DatetimeIndex(bclose.index).tz_localize(None).normalize()
            window = bclose.loc[
                (bclose.index >= pd.Timestamp(start)) & (bclose.index <= pd.Timestamp(end))
            ]
            bench_ret = (
                float(window.iloc[-1] / window.iloc[0] - 1) * 100 if len(window) > 1 else np.nan
            )

            for hold in holds:
                rcfg = scfg.model_copy(deep=True)
                rcfg.risk.max_hold_days = hold
                rcfg.risk.atr_stop_mult = 2.5
                rcfg.risk.trail_activate_r = 1e9
                rcfg.risk.max_open_positions = 20
                rcfg.risk.max_portfolio_heat_pct = 0.12
                hold_feats = {
                    s: f.drop(columns=["exit_signal"], errors="ignore")
                    for s, f in feats.items()
                }
                res = PortfolioBacktester(rcfg).run(
                    prices, delivery, benchmark, start=start, end=end, features=hold_feats
                )
                st = res.stats
                if not st.get("trade_count"):
                    print(f"  {hold:>5}   no trades")
                    continue
                alpha = st["total_return_pct"] - bench_ret * st["avg_exposure_pct"] / 100
                print(
                    f"  {hold:>5}{st['trade_count']:>8}{st['win_rate_pct']:>7.1f}"
                    f"{st['win_loss_ratio']:>6.2f}{st['profit_factor']:>6.2f}"
                    f"{st['total_return_pct']:>8.2f}{bench_ret:>8.2f}{alpha:>8.2f}"
                    f"{st['max_drawdown_pct']:>7.2f}{st['avg_exposure_pct']:>6.1f}"
                )
            print("  " + "-" * 74)
            print("  alpha = return minus (benchmark x average exposure): what is left")
            print("          after paying for the market exposure the strategy carried.")

    print("\n" + "=" * 100 + "\n")
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Re-price the mean-reversion strategy with realistic Indian costs."
    )
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--holds", default="10,15,20")
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--notional", type=float, default=100_000.0)
    parser.add_argument("--only", default=None,
                        help="Comma-separated scenario numbers to run, e.g. 2,3.")
    parser.add_argument("--no-backtest", action="store_true",
                        help="Only measure bar-level edges, skip the portfolio runs.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    run(
        start=args.start, end=args.end,
        holds=tuple(int(h) for h in args.holds.split(",")),
        quantile=args.quantile,
        typical_notional=args.notional,
        run_backtests=not args.no_backtest,
        only=[int(x) for x in args.only.split(",")] if args.only else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
