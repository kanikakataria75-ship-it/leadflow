"""Walk-forward validation.

A single in-sample backtest tells you almost nothing about whether a strategy
will keep working: with five rules and a dozen parameters it is easy to fit the
window you happened to test on. Walk-forward analysis answers the sharper
question -- *if I had re-tuned the parameters using only data available at the
time, how would the next quarter have gone?*

Each fold:

1. Optimise the DAB parameters over a ``wf_train_months`` training window.
2. Apply those exact parameters, unchanged, to the following
   ``wf_test_months`` of out-of-sample data.
3. Step forward by ``wf_step_months`` and repeat.

The out-of-sample segments are then chained into a single equity curve. Because
each fold's simulation restarts at the configured capital, the segments are
joined by compounding their daily *returns* rather than concatenating rupee
values, which preserves the shape of the combined curve.

The headline diagnostic is the **walk-forward efficiency**: out-of-sample
performance divided by in-sample performance. A ratio near or above 1.0 means
the edge survived tuning; well below 0.5 means most of the in-sample result was
curve fitting.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig, get_config
from .engine import BacktestResult, PortfolioBacktester, Trade
from .metrics import compute_stats

logger = logging.getLogger(__name__)

#: Default optimisation grid. Deliberately small: every extra axis multiplies
#: runtime *and* the number of ways to fool yourself.
DEFAULT_GRID: dict[str, Sequence[Any]] = {
    "delivery_spike_mult": (1.3, 1.5, 1.8),
    "volume_surge_mult": (1.5, 2.0),
    "delivery_spike_window": (5, 10),
}


@dataclass(slots=True)
class Fold:
    """One train/test split and its results."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    train_stats: dict[str, Any]
    test_stats: dict[str, Any]
    test_equity: pd.Series
    trades: list[Trade] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary for the API."""
        return {
            "index": self.index,
            "train_start": self.train_start.strftime("%Y-%m-%d"),
            "train_end": self.train_end.strftime("%Y-%m-%d"),
            "test_start": self.test_start.strftime("%Y-%m-%d"),
            "test_end": self.test_end.strftime("%Y-%m-%d"),
            "best_params": self.best_params,
            "train": {
                k: self.train_stats.get(k)
                for k in ("trade_count", "total_return_pct", "avg_r_multiple",
                          "win_rate_pct", "profit_factor", "sharpe", "max_drawdown_pct")
            },
            "test": {
                k: self.test_stats.get(k)
                for k in ("trade_count", "total_return_pct", "avg_r_multiple",
                          "win_rate_pct", "profit_factor", "sharpe", "max_drawdown_pct")
            },
        }


@dataclass(slots=True)
class WalkForwardResult:
    """The full walk-forward study."""

    folds: list[Fold]
    combined_equity: pd.DataFrame
    combined_stats: dict[str, Any]
    efficiency: dict[str, float | None]
    grid: dict[str, Sequence[Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "folds": [f.as_dict() for f in self.folds],
            "combined_stats": self.combined_stats,
            "efficiency": self.efficiency,
            "grid": {k: list(v) for k, v in self.grid.items()},
        }


def make_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate rolling ``(train_start, train_end, test_start, test_end)`` windows."""
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    train_start = pd.Timestamp(start)
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > pd.Timestamp(end):
            # Keep a final partial fold only if it has a usable test stub.
            if train_end < pd.Timestamp(end) - pd.Timedelta(days=20):
                folds.append((train_start, train_end, train_end, pd.Timestamp(end)))
            break
        folds.append((train_start, train_end, train_end, test_end))
        train_start = train_start + pd.DateOffset(months=step_months)
    return folds


def objective_value(stats: Mapping[str, Any], trades: Sequence[Trade], *, kind: str) -> float:
    """Score a parameter set. Higher is better; ``-inf`` disqualifies it.

    ``expectancy_t`` (the default) is a t-statistic on the R-multiple series:
    mean R divided by its standard error. It rewards a genuine edge with enough
    trades to believe it, rather than a handful of lucky ones -- which is the
    failure mode a plain return or profit-factor objective walks straight into.
    """
    n = int(stats.get("trade_count", 0) or 0)
    if n < 5:
        return float("-inf")

    if kind == "expectancy_t":
        rs = np.array([t.r_multiple for t in trades if np.isfinite(t.r_multiple)])
        if rs.size < 5:
            return float("-inf")
        sd = rs.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return float("-inf")
        return float(rs.mean() / (sd / np.sqrt(rs.size)))
    if kind == "sharpe":
        return float(stats.get("sharpe", 0.0) or 0.0)
    if kind == "calmar":
        return float(stats.get("calmar", 0.0) or 0.0)
    if kind == "profit_factor":
        pf = float(stats.get("profit_factor", 0.0) or 0.0)
        return pf if np.isfinite(pf) else 0.0
    if kind == "total_return":
        return float(stats.get("total_return_pct", 0.0) or 0.0)
    raise ValueError(f"Unknown objective: {kind}")


def _apply_params(cfg: AppConfig, params: Mapping[str, Any]) -> AppConfig:
    """Return a copy of the config with DAB parameters overridden."""
    out = cfg.model_copy(deep=True)
    for key, value in params.items():
        setattr(out.dab, key, value)
    return out


def _grid_combos(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of concrete parameter dicts."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _chain_equity(segments: Sequence[pd.Series], starting_capital: float) -> pd.Series:
    """Chain per-fold equity segments into one continuous curve.

    Each fold restarts at the configured capital, so the segments are joined by
    compounding daily returns. Concatenating rupee values instead would produce
    a sawtooth that resets at every fold boundary.
    """
    if not segments:
        return pd.Series(dtype=float)
    returns: list[pd.Series] = []
    for seg in segments:
        if seg is None or len(seg) < 2:
            continue
        returns.append(seg.pct_change().dropna())
    if not returns:
        return pd.Series(dtype=float)
    joined = pd.concat(returns).sort_index()
    joined = joined[~joined.index.duplicated(keep="first")]
    return starting_capital * (1.0 + joined).cumprod()


def run_walk_forward(
    prices: Mapping[str, pd.DataFrame],
    delivery: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame | None,
    *,
    start: date | pd.Timestamp,
    end: date | pd.Timestamp,
    cfg: AppConfig | None = None,
    grid: Mapping[str, Sequence[Any]] | None = None,
    objective: str = "expectancy_t",
) -> WalkForwardResult:
    """Run the full rolling walk-forward study.

    Args:
        prices: Symbol -> OHLCV.
        delivery: Symbol -> delivery data.
        benchmark: Benchmark OHLCV.
        start: First day of the first training window.
        end: Last day of the last test window.
        cfg: Config override.
        grid: Parameter grid. Defaults to :data:`DEFAULT_GRID`.
        objective: Selection criterion, see :func:`objective_value`.

    Returns:
        A :class:`WalkForwardResult` with per-fold detail, the chained
        out-of-sample equity curve and the efficiency diagnostics.
    """
    cfg = cfg or get_config()
    grid = dict(grid or (DEFAULT_GRID if cfg.backtest.wf_optimise else {}))
    windows = make_folds(
        pd.Timestamp(start), pd.Timestamp(end),
        train_months=cfg.backtest.wf_train_months,
        test_months=cfg.backtest.wf_test_months,
        step_months=cfg.backtest.wf_step_months,
    )
    if not windows:
        raise ValueError(
            f"No walk-forward folds fit between {start} and {end}. "
            f"Need at least {cfg.backtest.wf_train_months + cfg.backtest.wf_test_months} "
            "months of data, or reduce wf_train_months."
        )
    logger.info("Walk-forward: %d folds, grid of %d combinations.",
                len(windows), int(np.prod([len(v) for v in grid.values()])) if grid else 1)

    combos = _grid_combos(grid)

    # Feature frames depend only on the parameters, not on the fold window, and
    # computing them is by far the most expensive step. Iterating combinations
    # in the OUTER loop means each one's features are built once and reused for
    # every fold, then released -- turning combos*folds feature builds into
    # combos, without ever holding more than one set in memory.
    train_scores: list[dict[int, float]] = [{} for _ in windows]
    train_stats_by: list[dict[int, dict[str, Any]]] = [{} for _ in windows]
    test_results: list[dict[int, BacktestResult]] = [{} for _ in windows]

    for c_idx, params in enumerate(combos):
        trial_cfg = _apply_params(cfg, params)
        engine = PortfolioBacktester(trial_cfg)
        logger.info(
            "Parameter set %d/%d: %s", c_idx + 1, len(combos), params or "(config defaults)"
        )
        feats = engine.compute_all_features(prices, delivery, benchmark)

        for f_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            train = engine.run(
                prices, delivery, benchmark, start=tr_s, end=tr_e, features=feats
            )
            train_scores[f_idx][c_idx] = objective_value(
                train.stats, train.trades, kind=objective
            )
            train_stats_by[f_idx][c_idx] = train.stats
            test_results[f_idx][c_idx] = engine.run(
                prices, delivery, benchmark, start=te_s, end=te_e, features=feats
            )
        del feats  # release before building the next parameter set's features

    folds: list[Fold] = []
    segments: list[pd.Series] = []
    all_trades: list[Trade] = []

    for f_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        scores = train_scores[f_idx]
        best_idx = max(scores, key=lambda k: scores[k])
        if not np.isfinite(scores[best_idx]):
            logger.warning(
                "Fold %d: no parameter set produced enough trades in training; "
                "falling back to config defaults.", f_idx + 1,
            )
            best_idx = 0
        best_params = combos[best_idx]
        test_result = test_results[f_idx][best_idx]
        equity = (
            test_result.equity_curve["equity"]
            if not test_result.equity_curve.empty else pd.Series(dtype=float)
        )
        logger.info(
            "Fold %d/%d  train %s..%s -> test %s..%s | best %s (%s=%.3f) "
            "| OOS: %d trades, %+.2f%%, avg %+.3fR",
            f_idx + 1, len(windows), tr_s.date(), tr_e.date(), te_s.date(), te_e.date(),
            best_params or "defaults", objective, scores[best_idx],
            test_result.stats.get("trade_count", 0),
            test_result.stats.get("total_return_pct", 0.0),
            test_result.stats.get("avg_r_multiple", 0.0),
        )
        folds.append(
            Fold(
                index=f_idx, train_start=tr_s, train_end=tr_e,
                test_start=te_s, test_end=te_e,
                best_params=best_params,
                train_stats=train_stats_by[f_idx][best_idx],
                test_stats=test_result.stats, test_equity=equity,
                trades=test_result.trades,
            )
        )
        segments.append(equity)
        all_trades.extend(test_result.trades)

    combined = _chain_equity(segments, cfg.risk.starting_capital)
    combined_frame = combined.to_frame("equity") if not combined.empty else pd.DataFrame()
    combined_stats = (
        compute_stats(all_trades, combined_frame, cfg=cfg, benchmark=benchmark)
        if not combined_frame.empty else {"trade_count": len(all_trades)}
    )
    return WalkForwardResult(
        folds=folds,
        combined_equity=combined_frame,
        combined_stats=combined_stats,
        efficiency=_efficiency(folds),
        grid=grid,
    )


def _efficiency(folds: Sequence[Fold]) -> dict[str, float | None]:
    """Out-of-sample versus in-sample performance ratios.

    Interpretation: ~1.0 means out-of-sample matched in-sample and the edge
    likely survives tuning; ~0.5 means half the in-sample result was fitting;
    at or below 0 the strategy did not generalise at all.
    """
    def mean_of(key: str, which: str) -> float:
        vals = [
            float(getattr(f, which).get(key, 0.0) or 0.0)
            for f in folds
            if (getattr(f, which).get("trade_count", 0) or 0) > 0
        ]
        return float(np.mean(vals)) if vals else 0.0

    # An efficiency ratio is only meaningful when the in-sample value is far
    # enough from zero to be a stable denominator. Near zero the ratio explodes
    # and reads as a spectacular result when it actually means "in-sample had no
    # edge either" -- so it is reported as None instead of a large number.
    min_denominator = {
        "avg_r_multiple": 0.05,
        "win_rate_pct": 5.0,
        "profit_factor": 0.10,
        "total_return_pct": 1.0,
    }

    out: dict[str, float | None] = {}
    for key in ("avg_r_multiple", "win_rate_pct", "profit_factor", "total_return_pct"):
        is_v, oos_v = mean_of(key, "train_stats"), mean_of(key, "test_stats")
        out[f"is_{key}"] = round(is_v, 3)
        out[f"oos_{key}"] = round(oos_v, 3)
        out[f"efficiency_{key}"] = (
            round(oos_v / is_v, 3) if abs(is_v) >= min_denominator[key] else None
        )

    profitable = sum(
        1 for f in folds if float(f.test_stats.get("total_return_pct", 0.0) or 0.0) > 0
    )
    out["folds"] = float(len(folds))
    out["profitable_folds"] = float(profitable)
    out["profitable_fold_pct"] = round(profitable / len(folds) * 100, 1) if folds else 0.0
    return out


def format_report(result: WalkForwardResult) -> str:
    """Human-readable walk-forward summary."""
    rows = ["", "=" * 78, "  WALK-FORWARD VALIDATION", "=" * 78]
    header = (
        f"  {'#':<3}{'test window':<26}{'trades':>7}{'return':>10}"
        f"{'avg R':>9}{'win%':>8}{'PF':>7}"
    )
    rows.append(header)
    rows.append("  " + "-" * 74)
    for f in result.folds:
        t = f.test_stats
        window = f"{f.test_start.date()} to {f.test_end.date()}"
        pf = t.get("profit_factor", 0.0) or 0.0
        rows.append(
            f"  {f.index + 1:<3}{window:<26}{t.get('trade_count', 0):>7}"
            f"{t.get('total_return_pct', 0.0):>9.2f}%{t.get('avg_r_multiple', 0.0):>9.3f}"
            f"{t.get('win_rate_pct', 0.0):>7.1f}%{pf if np.isfinite(pf) else 0.0:>7.2f}"
        )

    e = result.efficiency
    c = result.combined_stats
    rows += [
        "  " + "-" * 74,
        "",
        "  IN-SAMPLE vs OUT-OF-SAMPLE (fold averages)",
        f"    {'metric':<20}{'in-sample':>14}{'out-of-sample':>16}{'efficiency':>14}",
    ]
    for key, label in (
        ("avg_r_multiple", "Avg R multiple"),
        ("win_rate_pct", "Win rate %"),
        ("profit_factor", "Profit factor"),
        ("total_return_pct", "Return %"),
    ):
        eff = e.get(f"efficiency_{key}")
        eff_text = f"{eff:>14.2f}" if eff is not None else f"{'n/a':>14}"
        rows.append(
            f"    {label:<20}{e.get(f'is_{key}', 0):>14.3f}"
            f"{e.get(f'oos_{key}', 0):>16.3f}{eff_text}"
        )
    if any(e.get(f"efficiency_{k}") is None for k in
           ("avg_r_multiple", "win_rate_pct", "profit_factor", "total_return_pct")):
        rows.append(
            "    (n/a = the in-sample value was too close to zero for the ratio to"
        )
        rows.append(
            "     mean anything, which itself says in-sample found no edge to keep.)"
        )
    rows += [
        "",
        f"    Profitable folds     {int(e.get('profitable_folds', 0))}/{int(e.get('folds', 0))} "
        f"({e.get('profitable_fold_pct', 0):.0f}%)",
        "",
        "  COMBINED OUT-OF-SAMPLE EQUITY",
        f"    Trades               {c.get('trade_count', 0)}",
        f"    Total return         {c.get('total_return_pct', 0.0):+.2f}%",
        f"    CAGR                 {c.get('cagr_pct', 0.0):+.2f}%",
        f"    Max drawdown         {c.get('max_drawdown_pct', 0.0):.2f}%",
        f"    Sharpe               {c.get('sharpe', 0.0):.2f}",
        f"    Avg R multiple       {c.get('avg_r_multiple', 0.0):+.3f}",
        f"    Win rate             {c.get('win_rate_pct', 0.0):.1f}%",
        "=" * 78,
        "",
    ]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    from ..data.fetch_delivery import cached_delivery_days
    from .run_backtest import load_market_data

    parser = argparse.ArgumentParser(description="Walk-forward validate the DAB strategy.")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-months", type=int, default=None)
    parser.add_argument("--test-months", type=int, default=None)
    parser.add_argument("--step-months", type=int, default=None)
    parser.add_argument("--objective", default="expectancy_t",
                        choices=["expectancy_t", "sharpe", "calmar", "profit_factor", "total_return"])
    parser.add_argument("--no-optimise", action="store_true",
                        help="Run fixed parameters out-of-sample, skipping the grid search.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    )
    cfg = get_config()
    if args.train_months:
        cfg.backtest.wf_train_months = args.train_months
    if args.test_months:
        cfg.backtest.wf_test_months = args.test_months
    if args.step_months:
        cfg.backtest.wf_step_months = args.step_months

    available = cached_delivery_days(cfg)
    if not available:
        raise SystemExit("No delivery data cached. Run: python -m nifty_swing_bot.data.backfill")
    start = args.start or available[min(30, len(available) - 1)]
    end = args.end or available[-1]

    _, prices, delivery, benchmark = load_market_data(
        start=start, end=end, limit=args.limit, cfg=cfg
    )
    result = run_walk_forward(
        prices, delivery, benchmark,
        start=start, end=end, cfg=cfg,
        grid=None if not args.no_optimise else {},
        objective=args.objective,
    )
    print(format_report(result))

    out = cfg.paths.results_dir / "walk_forward.json"
    out.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    if not result.combined_equity.empty:
        result.combined_equity.to_csv(cfg.paths.results_dir / "walk_forward_equity.csv")
    logger.info("Walk-forward results written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
