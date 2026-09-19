"""The full metric set, computed identically for the strategy sleeve and the book.

One implementation, applied to both series, so any difference between the two
columns is diversification and not a difference in how they were measured.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _ann_factor(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 1.0
    years = (idx[-1] - idx[0]).days / 365.25
    return max(years, 1e-9)


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def drawdown_episodes(equity: pd.Series) -> pd.DataFrame:
    """Every peak-to-recovery episode, with duration and recovery split out.

    ``duration`` is peak to trough, ``recovery`` is trough back to the prior
    peak. An episode still underwater at the end has ``recovered=False`` and its
    recovery is censored, which is reported rather than silently treated as zero.
    """
    dd = drawdown_series(equity)
    rows, in_dd, peak_i = [], False, 0
    for i in range(len(dd)):
        if not in_dd and dd.iloc[i] < -1e-12:
            in_dd, peak_i = True, i - 1 if i > 0 else 0
        elif in_dd and dd.iloc[i] >= -1e-12:
            seg = dd.iloc[peak_i:i + 1]
            t = int(seg.values.argmin())
            rows.append({
                "start": dd.index[peak_i], "trough": seg.index[t], "end": dd.index[i],
                "depth_pct": float(seg.min() * 100),
                "duration_days": int((seg.index[t] - dd.index[peak_i]).days),
                "recovery_days": int((dd.index[i] - seg.index[t]).days),
                "recovered": True,
            })
            in_dd = False
    if in_dd:
        seg = dd.iloc[peak_i:]
        t = int(seg.values.argmin())
        rows.append({
            "start": dd.index[peak_i], "trough": seg.index[t], "end": dd.index[-1],
            "depth_pct": float(seg.min() * 100),
            "duration_days": int((seg.index[t] - dd.index[peak_i]).days),
            "recovery_days": int((dd.index[-1] - seg.index[t]).days),
            "recovered": False,
        })
    return pd.DataFrame(rows)


def omega_ratio(r: pd.Series, threshold: float = 0.0) -> float:
    """Probability-weighted ratio of gains to losses above/below ``threshold``.

    Omega uses the whole return distribution rather than collapsing it to two
    moments, so it does not assume returns are close to normal -- useful here
    precisely because variant (c) is being tested for whether it fattens the
    right tail, which a Sharpe-style ratio is built to be insensitive to.
    """
    excess = r - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan


def ulcer_index(equity: pd.Series) -> float:
    """RMS of the drawdown series -- penalises depth AND duration of drawdowns.

    Two curves with the same max drawdown are told apart by the Ulcer Index if
    one recovers quickly and the other stays underwater; Calmar cannot do this
    because it only ever looks at the single worst trough.
    """
    dd = drawdown_series(equity) * 100
    return float(np.sqrt((dd ** 2).mean()))


def var_cvar(r: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Historical (non-parametric) VaR and CVaR at ``level``, in percent.

    Historical rather than parametric: the return series here is exactly the
    kind of fat-tailed, laddered distribution a normal-distribution VaR would
    misstate, which is the same reason Omega is reported instead of relying on
    Sharpe alone.
    """
    if len(r) < 20:
        return np.nan, np.nan
    q = float(np.percentile(r, (1 - level) * 100))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else q
    return q * 100, cvar * 100


def compounded_return_on_dates(daily_returns: pd.Series, dates: pd.DatetimeIndex) -> float:
    """Geometric return earned ONLY on the given dates, in percent.

    For a non-contiguous date set this is NOT the same as
    ``equity[dates[-1]] / equity[dates[0]] - 1``: that ratio silently folds in
    whatever happened on every day *between* the first and last date, including
    days excluded from the set entirely. This compounds each included day's own
    return and skips the rest, which is what "the return earned specifically on
    these days" has to mean once the set isn't contiguous -- e.g. reporting the
    return of "narrow-breadth years" when those years are not adjacent.
    """
    r = daily_returns.reindex(dates).dropna()
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0) * 100


def performance(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free: float = 0.065,
) -> dict[str, Any]:
    """CAGR through correlation, for one equity series."""
    eq = equity.dropna().astype(float)
    if len(eq) < 3:
        return {}
    r = eq.pct_change().dropna()
    yrs = _ann_factor(pd.DatetimeIndex(eq.index))
    total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    downside = r[r < 0]
    dvol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else np.nan
    dd = drawdown_series(eq)
    maxdd = float(dd.min())
    eps = drawdown_episodes(eq)
    var95, cvar95 = var_cvar(r, 0.95)

    out: dict[str, Any] = {
        "total_return_pct": total * 100,
        "cagr_pct": cagr * 100,
        "volatility_pct": vol * 100,
        "downside_deviation_pct": dvol * 100 if np.isfinite(dvol) else np.nan,
        "sharpe": (cagr - risk_free) / vol if vol > 0 else np.nan,
        "sortino": (cagr - risk_free) / dvol if dvol and dvol > 0 else np.nan,
        "calmar": cagr / abs(maxdd) if maxdd < 0 else np.nan,
        "omega": omega_ratio(r, threshold=risk_free / TRADING_DAYS),
        "ulcer_index": ulcer_index(eq),
        "var_95_pct_daily": var95,
        "cvar_95_pct_daily": cvar95,
        "skew": float(r.skew()) if len(r) > 2 else np.nan,
        "kurtosis": float(r.kurtosis()) if len(r) > 3 else np.nan,   # excess kurtosis (pandas convention)
        "max_drawdown_pct": maxdd * 100,
        "avg_drawdown_pct": float(eps["depth_pct"].mean()) if not eps.empty else 0.0,
        "longest_dd_days": int(eps["duration_days"].max()) if not eps.empty else 0,
        "longest_recovery_days": int(eps["recovery_days"].max()) if not eps.empty else 0,
        "unrecovered": bool((~eps["recovered"]).any()) if not eps.empty else False,
        "positive_days_pct": float((r > 0).mean() * 100),
        "n_days": int(len(eq)),
        "years": yrs,
    }

    annual = eq.resample("YE").last()
    annual = pd.concat([eq.iloc[:1], annual]).drop_duplicates()
    ar = annual.pct_change().dropna() * 100
    if len(ar):
        out["best_year_pct"] = float(ar.max())
        out["worst_year_pct"] = float(ar.min())
        out["positive_years_pct"] = float((ar > 0).mean() * 100)

    if benchmark is not None:
        b = benchmark.reindex(eq.index).ffill().pct_change().dropna()
        j = pd.concat([r, b], axis=1).dropna()
        j.columns = ["p", "b"]
        if len(j) > 20 and j["b"].var() > 0:
            beta = float(j.cov().iloc[0, 1] / j["b"].var())
            out["beta"] = beta
            out["alpha_pct"] = (cagr - risk_free - beta * (
                float((1 + j["b"].mean()) ** TRADING_DAYS - 1) - risk_free)) * 100
            out["corr_to_index"] = float(j["p"].corr(j["b"]))
            active = j["p"] - j["b"]
            te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
            out["tracking_error_pct"] = te * 100
            out["information_ratio"] = (
                float(active.mean() * TRADING_DAYS / te) if te > 0 else np.nan
            )
    return out


def positions_from_rows(trades: pd.DataFrame) -> pd.DataFrame:
    """Collapse ladder-split trade ROWS to POSITIONS.

    Phase 20's finding, applied everywhere rather than in one script: a profit
    ladder splits one position into 2-4 trade rows, and every ladder row is
    profitable by construction (a tranche sells only because price reached its
    trigger). A statistic averaged over rows therefore counts a position's
    winning fragments repeatedly and its losing remainder once -- this is
    exactly how the "0.65-0.72 average R multiple" number in earlier phases of
    this project was produced, and it is wrong. Position-level R under the same
    ladder is ~0.17-0.18. Everything downstream of this function is
    position-level unless explicitly labelled a row figure.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    need = {"symbol", "entry_date", "net_pnl", "qty", "entry_price"}
    if not need.issubset(trades.columns):
        return trades.copy()   # already position-level, or missing join keys
    g = trades.groupby(["symbol", "entry_date"])
    out = pd.DataFrame({
        "net_pnl": g["net_pnl"].sum(),
        "qty": g["qty"].sum(),
        "entry_price": g["entry_price"].first(),
        "bars_held": g["bars_held"].max() if "bars_held" in trades else np.nan,
    }).reset_index()
    if "r_multiple" in trades.columns:
        rmult = g.apply(
            lambda d: float((d["r_multiple"] * d["qty"]).sum() / d["qty"].sum())
            if d["qty"].sum() else np.nan,
            include_groups=False,
        )
        out["r_multiple"] = rmult.to_numpy()
    out["ret_pct"] = out["net_pnl"] / (out["entry_price"] * out["qty"]) * 100
    return out


def trade_stats(trades: pd.DataFrame) -> dict[str, Any]:
    """Win rate, profit factor, expectancy, R multiple, best/worst -- at the
    POSITION level (see ``positions_from_rows``), not the trade-row level."""
    if trades is None or trades.empty:
        return {}
    pos = positions_from_rows(trades)
    if pos.empty:
        return {}
    pnl = pos["net_pnl"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_w, gross_l = float(wins.sum()), float(abs(losses.sum()))
    out = {
        "n_trades": int(len(pnl)),          # positions, not rows
        "n_trade_rows": int(len(trades)),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "profit_factor": gross_w / gross_l if gross_l > 0 else np.nan,
        "expectancy_inr": float(pnl.mean()),
        "avg_win_inr": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_inr": float(losses.mean()) if len(losses) else 0.0,
        "payoff_ratio": (float(wins.mean()) / abs(float(losses.mean()))
                         if len(losses) and losses.mean() != 0 else np.nan),
        "best_trade_inr": float(pnl.max()),
        "worst_trade_inr": float(pnl.min()),
        "best_trade_ret_pct": float(pos["ret_pct"].max()),
        "worst_trade_ret_pct": float(pos["ret_pct"].min()),
        "median_hold_bars": float(pos["bars_held"].median()) if "bars_held" in pos else np.nan,
    }
    if "r_multiple" in pos.columns:
        out["avg_r_multiple"] = float(pos["r_multiple"].mean())          # POSITION-level
        out["median_r_multiple"] = float(pos["r_multiple"].median())
        if "r_multiple" in trades.columns:
            out["avg_r_multiple_row_mean_DO_NOT_USE"] = float(
                trades["r_multiple"].astype(float).mean())
    if "costs" in trades.columns and gross_w + gross_l > 0:
        gross_pnl_abs = float((pnl.abs()).sum())
        out["total_costs_inr"] = float(trades["costs"].astype(float).sum())
        gross_before_cost = gross_pnl_abs + out["total_costs_inr"]
        out["cost_pct_of_gross_pnl"] = (
            out["total_costs_inr"] / gross_before_cost * 100 if gross_before_cost > 0 else np.nan
        )
    return out


def summarise(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    deployment: pd.Series | None = None,
    skipped: int | None = None,
    total_signals: int | None = None,
    risk_free: float = 0.065,
) -> dict[str, Any]:
    out = performance(equity, benchmark, risk_free)
    out.update(trade_stats(trades) if trades is not None else {})
    if deployment is not None and len(deployment):
        out["mean_deployment_pct"] = float(deployment.mean() * 100)
        out["time_in_market_pct"] = float((deployment > 0).mean() * 100)
    if skipped is not None and total_signals:
        out["skipped_signal_pct"] = skipped / total_signals * 100
        out["skipped_signals"] = int(skipped)
    return out


__all__ = ["compounded_return_on_dates", "drawdown_episodes", "drawdown_series", "omega_ratio",
           "performance", "positions_from_rows", "summarise", "trade_stats", "ulcer_index", "var_cvar"]
