"""Performance statistics for a completed backtest.

All ratios are computed from the **daily portfolio equity curve**, not from
trade returns, because trade returns overlap in time and would understate
volatility. Return figures are net of the slippage, brokerage and STT charged
by the engine.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..config import AppConfig, get_config

if TYPE_CHECKING:  # avoids a circular import at runtime
    from .engine import Trade

logger = logging.getLogger(__name__)


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None, int]:
    """Deepest peak-to-trough decline.

    Returns:
        ``(depth_fraction, peak_date, trough_date, longest_underwater_days)``
        where depth is a positive fraction (0.25 == a 25% drawdown).
    """
    if equity.empty:
        return 0.0, None, None, 0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    trough = drawdown.idxmin()
    depth = float(-drawdown.min())
    peak = equity.loc[:trough].idxmax() if trough is not None else None

    underwater = (drawdown < 0).astype(int)
    longest, current = 0, 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return depth, peak, trough, longest


def sharpe_ratio(returns: pd.Series, risk_free: float, periods: int) -> float:
    """Annualised Sharpe from daily returns, using an excess-return series."""
    if returns.empty or len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free) ** (1 / periods) - 1
    excess = returns - daily_rf
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free: float, periods: int) -> float:
    """Like Sharpe but penalising only downside deviation."""
    if returns.empty or len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free) ** (1 / periods) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    dd = downside.std(ddof=1)
    if not np.isfinite(dd) or dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods))


def alpha_beta(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series, periods: int
) -> tuple[float, float, int]:
    """CAPM alpha (annualised) and beta from daily returns, by OLS.

    Deliberately separate from the plain "total return minus buy-and-hold"
    figure ``compute_stats`` already reports. That figure cannot tell a real
    edge from beta: a strategy that is simply invested most of the time in a
    rising market beats a buy-and-hold benchmark on capital not deployed the
    whole period, with no selection skill at all. Regressing daily portfolio
    returns on daily benchmark returns splits the two apart -- beta is the
    market exposure actually carried, and alpha is what is left after
    removing exactly that much of the benchmark's own return.

    Args:
        portfolio_returns: Daily equity curve returns (``pct_change``).
        benchmark_returns: Daily benchmark returns, any index.
        periods: Trading days per year, for annualising the daily intercept.

    Returns:
        ``(annualised_alpha, beta, n_days)``. ``n_days`` is the aligned
        sample size actually regressed on; with too few overlapping days
        (under 20) alpha and beta are both returned as 0.0 rather than fit
        to noise.
    """
    aligned = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
    n = len(aligned)
    if n < 20 or aligned["b"].std(ddof=1) == 0:
        return 0.0, 0.0, n

    b = aligned["b"].to_numpy()
    p = aligned["p"].to_numpy()
    beta, daily_alpha = np.polyfit(b, p, 1)
    annual_alpha = float((1.0 + daily_alpha) ** periods - 1.0)
    return annual_alpha, float(beta), n


def cagr(equity: pd.Series, periods: int) -> float:
    """Compound annual growth rate, derived from the number of bars elapsed."""
    if len(equity) < 2:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    years = len(equity) / periods
    if years <= 0:
        return 0.0
    return float((end / start) ** (1 / years) - 1)


def profit_factor(pnls: Sequence[float]) -> float:
    """Gross profit divided by gross loss.

    Returns ``inf`` when there are no losing trades, which is a red flag for
    too small a sample rather than a good result.
    """
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def compute_stats(
    trades: Sequence["Trade"],
    equity_curve: pd.DataFrame,
    *,
    cfg: AppConfig | None = None,
    benchmark: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Full statistics bundle for a backtest run.

    Args:
        trades: Completed trades.
        equity_curve: Frame with an ``equity`` column on a DatetimeIndex.
        cfg: Config override.
        benchmark: Optional benchmark OHLCV for a buy-and-hold comparison.

    Returns:
        A flat dict of JSON-serialisable statistics.
    """
    cfg = cfg or get_config()
    bp = cfg.backtest
    periods = bp.trading_days_per_year

    if equity_curve is None or equity_curve.empty or "equity" not in equity_curve:
        return {"trade_count": 0}

    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().dropna()
    start_equity = cfg.risk.starting_capital
    final_equity = float(equity.iloc[-1])

    depth, peak_date, trough_date, underwater = max_drawdown(equity)

    pnls = [t.net_pnl for t in trades]
    r_multiples = [t.r_multiple for t in trades if np.isfinite(t.r_multiple)]
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    n = len(trades)

    ann_return = cagr(equity, periods)
    stats: dict[str, Any] = {
        # Headline
        "start_equity": round(start_equity, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity / start_equity - 1) * 100, 2) if start_equity else 0.0,
        "cagr_pct": round(ann_return * 100, 2),
        "max_drawdown_pct": round(depth * 100, 2),
        "calmar": round(ann_return / depth, 3) if depth > 0 else 0.0,
        "sharpe": round(sharpe_ratio(returns, bp.risk_free_rate, periods), 3),
        "sortino": round(sortino_ratio(returns, bp.risk_free_rate, periods), 3),
        "volatility_pct": round(float(returns.std(ddof=1) * np.sqrt(periods) * 100), 2)
        if len(returns) > 1 else 0.0,
        # Trades
        "trade_count": n,
        "win_rate_pct": round(len(wins) / n * 100, 2) if n else 0.0,
        "profit_factor": round(profit_factor(pnls), 3) if n else 0.0,
        "avg_r_multiple": round(float(np.mean(r_multiples)), 3) if r_multiples else 0.0,
        "median_r_multiple": round(float(np.median(r_multiples)), 3) if r_multiples else 0.0,
        "expectancy_rs": round(float(np.mean(pnls)), 2) if pnls else 0.0,
        "avg_win_rs": round(float(np.mean([t.net_pnl for t in wins])), 2) if wins else 0.0,
        "avg_loss_rs": round(float(np.mean([t.net_pnl for t in losses])), 2) if losses else 0.0,
        "win_loss_ratio": round(
            abs(np.mean([t.net_pnl for t in wins]) / np.mean([t.net_pnl for t in losses])), 3
        ) if wins and losses else 0.0,
        "best_trade_r": round(max(r_multiples), 3) if r_multiples else 0.0,
        "worst_trade_r": round(min(r_multiples), 3) if r_multiples else 0.0,
        "avg_bars_held": round(float(np.mean([t.bars_held for t in trades])), 2) if n else 0.0,
        "total_costs_rs": round(sum(t.costs for t in trades), 2),
        # Risk / exposure
        "underwater_days": int(underwater),
        "max_dd_peak": peak_date.strftime("%Y-%m-%d") if peak_date is not None else None,
        "max_dd_trough": trough_date.strftime("%Y-%m-%d") if trough_date is not None else None,
        "avg_exposure_pct": round(float(equity_curve.get("exposure", pd.Series(dtype=float)).mean() * 100), 2)
        if "exposure" in equity_curve and not equity_curve["exposure"].empty else 0.0,
        "max_open_positions": int(equity_curve["open_positions"].max())
        if "open_positions" in equity_curve and not equity_curve.empty else 0,
        "bars": int(len(equity)),
        "start_date": equity.index[0].strftime("%Y-%m-%d"),
        "end_date": equity.index[-1].strftime("%Y-%m-%d"),
    }

    # Exit-reason mix: a high share of gap/circuit exits is the tell that the
    # strategy is paying for illiquidity rather than being wrong on direction.
    if trades:
        reasons: dict[str, int] = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        stats["exit_reasons"] = reasons
        adverse = sum(
            v for k, v in reasons.items() if k in {"gap_through_stop", "circuit_locked_stop"}
        )
        stats["adverse_fill_pct"] = round(adverse / len(trades) * 100, 2)

    if benchmark is not None and not benchmark.empty and "close" in benchmark:
        bench = benchmark["close"].reindex(equity.index).ffill().dropna()
        if len(bench) > 1:
            bh_return = float(bench.iloc[-1] / bench.iloc[0] - 1)
            stats["benchmark_return_pct"] = round(bh_return * 100, 2)
            stats["excess_return_pct"] = round(
                stats["total_return_pct"] - bh_return * 100, 2
            )
            bench_dd, *_ = max_drawdown(bench)
            stats["benchmark_max_drawdown_pct"] = round(bench_dd * 100, 2)

    return stats


def r_multiple_distribution(trades: Sequence["Trade"], bins: int = 20) -> dict[str, list[float]]:
    """Histogram of R multiples, for the front-end chart."""
    values = [t.r_multiple for t in trades if np.isfinite(t.r_multiple)]
    if not values:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Month-by-month percentage returns, for the heatmap on the results page."""
    if equity.empty:
        return pd.DataFrame()
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        return pd.DataFrame()
    frame = monthly.to_frame("return_pct")
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    return frame


__all__ = [
    "alpha_beta",
    "cagr",
    "compute_stats",
    "max_drawdown",
    "monthly_returns",
    "profit_factor",
    "r_multiple_distribution",
    "sharpe_ratio",
    "sortino_ratio",
]
