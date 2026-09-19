"""Backtest orchestration: assemble the data, run the engine, save the results.

Usage::

    python -m nifty_swing_bot.backtest.run_backtest --start 2025-04-01
    python -m nifty_swing_bot.backtest.run_backtest --limit 100 --no-save

Results land in ``cache/../results/`` as three files: ``stats.json``,
``trades.csv`` and ``equity_curve.csv``, which the FastAPI layer serves
directly.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import AppConfig, get_config
from ..data.fetch_delivery import cached_delivery_days, fetch_delivery_history, to_symbol_panel
from ..data.fetch_ohlcv import fetch_benchmark, fetch_ohlcv
from ..data.universe import apply_liquidity_filter, build_universe
from .engine import BacktestResult, PortfolioBacktester
from .metrics import monthly_returns, r_multiple_distribution

logger = logging.getLogger(__name__)

#: Calendar days of price history fetched before the simulation start.
#: Must exceed ``universe.min_history_days`` *trading* bars once weekends and
#: holidays are removed (roughly 1.45 calendar days per trading day), otherwise
#: the liquidity filter rejects every symbol for having too little history --
#: which is invisible in a long backtest but empties the universe in the daily
#: scanner, whose whole window is this buffer. Price data is cheap: one
#: yfinance request covers years, so the buffer is generous.
PRICE_WARMUP_DAYS = 280

#: Calendar days of delivery history fetched before the simulation start.
#: Deliberately tight -- each day is a separate rate-limited NSE request. Needs
#: to cover the 20-bar delivery baseline plus the 10-bar spike window (~30
#: trading days), with margin for holiday clusters.
DELIVERY_WARMUP_DAYS = 75

#: Cached delivery days skipped when auto-selecting a start date, so the
#: first simulated bar has a fully seeded delivery baseline.
DELIVERY_SEED_DAYS = 35



#: Selectable strategies. Each entry is a callable with the signature
#: ``(ohlcv, delivery, benchmark, *, cfg) -> features``.
STRATEGIES: dict[str, str] = {
    "dab": "Delivery Accumulation Breakout (the original brief)",
    "pbr": "Pullback Reversion (what the research supports)",
}


def compute_strategy_features(
    strategy: str,
    prices: dict[str, pd.DataFrame],
    delivery: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    *,
    cfg: AppConfig,
) -> dict[str, pd.DataFrame]:
    """Build feature frames for the named strategy.

    Args:
        strategy: ``"dab"`` or ``"pbr"``.
        prices: Symbol -> OHLCV.
        delivery: Symbol -> delivery data.
        benchmark: Benchmark OHLCV.
        cfg: Config.

    Returns:
        Symbol -> feature frame, ready to hand to the backtester.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")

    if strategy == "dab":
        return PortfolioBacktester(cfg).compute_all_features(prices, delivery, benchmark)

    from ..strategy import mr_strategy

    out: dict[str, pd.DataFrame] = {}
    for symbol, ohlcv in prices.items():
        if ohlcv is None or len(ohlcv) < cfg.universe.min_history_days:
            continue
        try:
            frame = mr_strategy.compute_features(ohlcv, delivery.get(symbol), cfg=cfg)
        except (KeyError, ValueError) as exc:
            logger.debug("PBR features failed for %s: %s", symbol, exc)
            continue
        if not frame.empty:
            out[symbol] = frame
    logger.info("Computed PBR features for %d/%d symbols.", len(out), len(prices))
    return out


def load_market_data(
    *,
    start: date,
    end: date,
    limit: int | None = None,
    refresh: bool = False,
    refresh_delivery: bool = False,
    cfg: AppConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Assemble everything the backtester needs.

    Prices are fetched with a warm-up buffer before ``start`` so that the 20-day
    rolling baselines and ATR windows are fully seeded on the first simulated
    bar, rather than the backtest opening with a stretch of NaN features.

    Args:
        start: First simulated bar.
        end: Last simulated bar.
        limit: Keep only the N most liquid symbols. Useful for quick runs.
        refresh: Re-download OHLCV.
        refresh_delivery: Re-download bhavcopy files.
        cfg: Config override.

    Returns:
        ``(universe, prices, delivery, benchmark)``.
    """
    cfg = cfg or get_config()
    # Price history is cheap (one yfinance request covers years), so it gets a
    # generous buffer to seed ATR and the 20-bar high.
    warmup_start = pd.Timestamp(start) - pd.Timedelta(days=PRICE_WARMUP_DAYS)
    # Delivery data costs one NSE request per trading day, and its longest
    # lookback is the 20-bar baseline plus the 10-bar spike window. 60 calendar
    # days covers that comfortably. Using the price buffer here instead would
    # download ~60 extra bhavcopy files per run for no analytical gain -- and
    # because the default start date is derived from what is cached, it would
    # also walk the window further back on every single run.
    delivery_warmup = pd.Timestamp(start) - pd.Timedelta(days=DELIVERY_WARMUP_DAYS)

    universe = build_universe(refresh=refresh, cfg=cfg)
    if universe.empty:
        raise RuntimeError("Universe is empty; NSE constituent lists could not be loaded.")
    logger.info("Universe before liquidity filter: %d symbols.", len(universe))

    symbols = universe["symbol"].tolist()
    prices = fetch_ohlcv(symbols, start=warmup_start, end=end, refresh=refresh, cfg=cfg)
    universe = apply_liquidity_filter(universe, prices, cfg=cfg, as_of=pd.Timestamp(end))
    if limit is not None:
        universe = universe.head(limit)
        logger.info("Limited to the %d most liquid symbols.", len(universe))

    kept = set(universe["symbol"])
    prices = {s: f for s, f in prices.items() if s in kept and not f.empty}

    delivery_hist = fetch_delivery_history(
        pd.Timestamp(delivery_warmup).date(),
        end,
        symbols=kept,
        refresh=refresh_delivery,
        cfg=cfg,
    )
    delivery = to_symbol_panel(delivery_hist)
    logger.info("Delivery data available for %d/%d symbols.", len(delivery), len(prices))

    benchmark = fetch_benchmark(start=warmup_start, end=end, refresh=refresh, cfg=cfg)
    return universe, prices, delivery, benchmark


def save_results(result: BacktestResult, cfg: AppConfig | None = None, tag: str = "") -> Path:
    """Persist stats, trades and the equity curve for the API and the UI."""
    cfg = cfg or get_config()
    out_dir = cfg.paths.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": result.stats,
        "rejected": result.rejected,
        "params": result.params,
        "r_distribution": r_multiple_distribution(result.trades),
    }
    if not result.equity_curve.empty:
        payload["monthly_returns"] = (
            monthly_returns(result.equity_curve["equity"])
            .reset_index()
            .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records")
        )

    (out_dir / f"stats{suffix}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    result.trades_frame.to_csv(out_dir / f"trades{suffix}.csv", index=False)
    if not result.equity_curve.empty:
        result.equity_curve.to_csv(out_dir / f"equity_curve{suffix}.csv")

    logger.info("Results written to %s", out_dir)
    return out_dir


def format_report(result: BacktestResult) -> str:
    """Human-readable summary for the terminal."""
    s = result.stats
    if not s or not s.get("trade_count"):
        return "No trades were generated. Check delivery-data coverage and the date window."

    def line(label: str, value: object, unit: str = "") -> str:
        return f"  {label:<26}{value}{unit}"

    rows = [
        "",
        "=" * 62,
        f"  DAB BACKTEST  {s.get('start_date')} to {s.get('end_date')}",
        "=" * 62,
        "  RETURNS",
        line("Starting capital", f"Rs {s['start_equity']:,.0f}"),
        line("Final equity", f"Rs {s['final_equity']:,.0f}"),
        line("Total return", f"{s['total_return_pct']:+.2f}", "%"),
        line("CAGR", f"{s['cagr_pct']:+.2f}", "%"),
    ]
    if "benchmark_return_pct" in s:
        rows += [
            line("Benchmark (buy & hold)", f"{s['benchmark_return_pct']:+.2f}", "%"),
            line("Excess return", f"{s['excess_return_pct']:+.2f}", "%"),
        ]
    rows += [
        "",
        "  RISK",
        line("Max drawdown", f"{s['max_drawdown_pct']:.2f}", "%"),
        line("Sharpe", f"{s['sharpe']:.2f}"),
        line("Sortino", f"{s['sortino']:.2f}"),
        line("Calmar", f"{s['calmar']:.2f}"),
        line("Annualised volatility", f"{s['volatility_pct']:.2f}", "%"),
        line("Avg exposure", f"{s['avg_exposure_pct']:.1f}", "%"),
        "",
        "  TRADES",
        line("Trade count", s["trade_count"]),
        line("Win rate", f"{s['win_rate_pct']:.2f}", "%"),
        line("Profit factor", f"{s['profit_factor']:.2f}"),
        line("Avg R multiple", f"{s['avg_r_multiple']:+.3f}", "R"),
        line("Median R multiple", f"{s['median_r_multiple']:+.3f}", "R"),
        line("Expectancy", f"Rs {s['expectancy_rs']:,.0f}"),
        line("Avg win / avg loss", f"{s['win_loss_ratio']:.2f}"),
        line("Best / worst trade", f"{s['best_trade_r']:+.2f}R / {s['worst_trade_r']:+.2f}R"),
        line("Avg bars held", f"{s['avg_bars_held']:.1f}"),
        line("Total costs", f"Rs {s['total_costs_rs']:,.0f}"),
    ]
    if "exit_reasons" in s:
        rows += ["", "  EXIT MIX"]
        for reason, count in sorted(s["exit_reasons"].items(), key=lambda kv: -kv[1]):
            pct = count / s["trade_count"] * 100
            rows.append(line(reason, f"{count:>4}  ({pct:.1f}%)"))
        rows.append(line("Adverse fills", f"{s.get('adverse_fill_pct', 0):.1f}", "%"))
    if result.rejected:
        skipped = {k: v for k, v in result.rejected.items() if v}
        if skipped:
            rows += ["", "  SIGNALS SKIPPED"]
            for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
                rows.append(line(reason, count))
    rows += ["=" * 62, ""]
    return "\n".join(rows)


def run(
    *,
    strategy: str = "dab",
    start: date | None = None,
    end: date | None = None,
    limit: int | None = None,
    refresh: bool = False,
    refresh_delivery: bool = False,
    save: bool = True,
    tag: str = "",
    cfg: AppConfig | None = None,
) -> BacktestResult:
    """Load data, run the backtest, optionally persist the output."""
    cfg = cfg or get_config()

    if start is None or end is None:
        available = cached_delivery_days(cfg)
        if not available:
            raise RuntimeError(
                "No delivery data cached. Run the backfill first:\n"
                "  python -m nifty_swing_bot.data.backfill --start 2023-01-01"
            )
        # Skip the earliest cached days so the first simulated bar has a fully
        # seeded delivery baseline rather than one built from NaNs.
        start = start or available[min(DELIVERY_SEED_DAYS, len(available) - 1)]
        end = end or available[-1]
        logger.info(
            "Auto-selected window from %d cached delivery days (%s..%s).",
            len(available), available[0], available[-1],
        )

    logger.info("Backtest window: %s to %s", start, end)
    universe, prices, delivery, benchmark = load_market_data(
        start=start, end=end, limit=limit,
        refresh=refresh, refresh_delivery=refresh_delivery, cfg=cfg,
    )
    logger.info("Simulating %d symbols.", len(prices))

    engine = PortfolioBacktester(cfg)
    features = compute_strategy_features(strategy, prices, delivery, benchmark, cfg=cfg)
    result = engine.run(
        prices, delivery, benchmark, start=start, end=end, features=features
    )
    if save:
        save_results(result, cfg=cfg, tag=tag)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the DAB portfolio backtest.")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap the universe size.")
    parser.add_argument("--refresh", action="store_true", help="Re-download OHLCV.")
    parser.add_argument("--refresh-delivery", action="store_true", help="Re-download bhavcopy.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--tag", default="", help="Suffix for the output filenames.")
    parser.add_argument(
        "--strategy", default="dab", choices=sorted(STRATEGIES),
        help="; ".join(f"{k}: {v}" for k, v in STRATEGIES.items()),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run(
        strategy=args.strategy,
        start=args.start, end=args.end, limit=args.limit,
        refresh=args.refresh, refresh_delivery=args.refresh_delivery,
        save=not args.no_save, tag=args.tag or args.strategy,
    )
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
