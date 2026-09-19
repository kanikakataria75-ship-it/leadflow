"""The book simulation: three sleeves, an explicit funding flow, and rebalancing.

State each day is three sleeve values whose sum is the book:

    V_s   strategy sleeve capital (its cash plus its open positions)
    A     arbitrage sleeve, accruing
    G     gold sleeve, marked to a real INR price

plus two flow variables that do not create or destroy value:

    b     capital currently lent from arbitrage to the strategy
    P     of the strategy's capital, how much is in open positions

The funding flow is modelled rather than assumed. When the positions the
strategy wants exceed its own sleeve, the shortfall is **borrowed from
arbitrage** -- never from gold -- and repaid as positions close.
``arb_redemption_lag_days`` means the request is made on the prior session, so a
same-day spike arbitrage cannot fund in time is recorded as an unfunded
shortfall rather than silently financed.

**The one approximation, stated plainly.** Strategy P&L comes from
``AESPortfolioBacktester`` run on its own capital path. To make it scale with a
book that is also earning arbitrage and gold, daily P&L and position values are
multiplied by ``capital_basis / strategy_equity`` from the prior day. That keeps
sizing proportional to the live book without re-running the backtest inside the
loop. It is exact when book and strategy compound alike and drifts mildly
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .params import BookParams


@dataclass(slots=True)
class BookResult:
    curve: pd.DataFrame
    params: BookParams
    diagnostics: dict[str, Any]

    @property
    def book(self) -> pd.Series:
        return self.curve["book"]

    @property
    def returns(self) -> pd.Series:
        return self.curve["book"].pct_change().fillna(0.0)


def _rebalance_dates(idx: pd.DatetimeIndex, mode: str) -> set[pd.Timestamp]:
    if mode == "never":
        return set()
    freq = "QE" if mode == "quarterly" else "YE"
    out = set()
    for m in pd.Series(1, index=idx).resample(freq).last().index:
        pos = idx.searchsorted(m, side="right") - 1
        if 0 <= pos < len(idx):
            out.add(idx[pos])
    return out


def simulate(
    strat_equity: pd.Series,
    strat_positions: pd.Series,
    gold_close: pd.Series,
    params: BookParams | None = None,
) -> BookResult:
    p = params or BookParams()
    idx = pd.DatetimeIndex(strat_equity.index)
    gold = gold_close.reindex(idx).ffill().bfill()
    se = strat_equity.astype(float)
    sp = strat_positions.reindex(idx).fillna(0.0).astype(float)

    book0 = p.starting_capital
    V_s, A, G = p.w_strategy * book0, p.w_arbitrage * book0, p.w_gold * book0
    b = P = 0.0
    prev_request = 0.0

    rebal = _rebalance_dates(idx, p.rebalance)
    rows = [{
        "date": idx[0], "book": book0, "strategy": V_s, "arb": A, "gold": G,
        "positions": 0.0, "strat_cash": V_s, "borrowed": 0.0,
        "deployment": 0.0, "unfunded": 0.0, "rebalanced": False,
        "strat_ret": 0.0, "arb_ret": 0.0, "gold_ret": 0.0,
    }]
    borrow_days = peak_borrow = unfunded_days = 0
    max_unfunded = 0.0
    unfunded_vals: list[float] = []

    for i in range(1, len(idx)):
        date = idx[i]
        days = max(1, (date - idx[i - 1]).days)

        # 1. passive sleeves. Per-sleeve *investment* returns are recorded
        # separately from sleeve *values*, because rebalancing moves cash
        # between sleeves: a sleeve's value change is return PLUS flow, and
        # only the return belongs in a performance table.
        arb_r = p.arb_annual_rate * days / 365.0
        A *= 1.0 + arb_r
        g0, g1 = float(gold.iloc[i - 1]), float(gold.iloc[i])
        gold_r = (g1 / g0 - 1.0) if g0 > 0 else 0.0
        G *= 1.0 + gold_r

        # 2. strategy P&L, scaled so sizing tracks the live book
        prev_eq, now_eq = float(se.iloc[i - 1]), float(se.iloc[i])
        basis = (V_s + A + G) if p.strategy_capital_basis == "book" else V_s
        scale = (basis / prev_eq) if prev_eq > 0 else 1.0
        pnl = (now_eq - prev_eq) * scale
        strat_r = (pnl / V_s) if V_s > 0 else 0.0
        V_s += pnl

        # 3. financing. Own sleeve is the target weight of the current book.
        book_now = V_s + A + G
        own = p.w_strategy * book_now
        want = float(sp.iloc[i]) * scale

        need = max(0.0, want - own)
        fundable = min(prev_request, max(0.0, A))          # T+1
        target_b = min(need, fundable)
        unfunded = need - target_b
        if unfunded > 1e-6:
            # A shortfall persists across days until arbitrage funds it, so a
            # running SUM double-counts the same rupees every session it lasts.
            # Peak and mean are the meaningful summaries.
            unfunded_days += 1
            max_unfunded = max(max_unfunded, unfunded)
            unfunded_vals.append(unfunded)
        prev_request = need

        transfer = target_b - b
        if transfer > 0:
            transfer = min(transfer, A)                     # cannot lend what it lacks
        A -= transfer
        V_s += transfer
        b += transfer
        if b > 1e-6:
            borrow_days += 1
            peak_borrow = max(peak_borrow, b)

        # 4. positions sit inside strategy capital; the rest is its cash
        P = min(want, max(0.0, V_s))

        # 5. rebalance free cash only -- open positions are never force-closed
        did = False
        if date in rebal:
            book_now = V_s + A + G
            free = max(0.0, V_s - P)
            for tgt, cur, setter in (
                (p.w_gold * book_now, G, "G"),
                (p.w_arbitrage * book_now, A, "A"),
            ):
                delta = tgt - cur
                move = min(delta, free) if delta > 0 else max(delta, -cur)
                if setter == "G":
                    G += move
                else:
                    A += move
                V_s -= move
                free = max(0.0, V_s - P)
            did = True

        book = V_s + A + G
        rows.append({
            "date": date, "book": book, "strategy": V_s, "arb": A, "gold": G,
            "positions": P, "strat_cash": V_s - P, "borrowed": b,
            "deployment": P / book if book > 0 else 0.0,
            "unfunded": unfunded, "rebalanced": did,
            "strat_ret": strat_r, "arb_ret": arb_r, "gold_ret": gold_r,
        })

    curve = pd.DataFrame(rows).set_index("date")
    # Correlation uses the sleeve's own return, not its value change, so that
    # quarterly rebalancing flows cannot manufacture or hide correlation.
    sr = curve["strat_ret"]
    gr = curve["gold_ret"]
    n = max(1, len(curve) - 1)
    return BookResult(
        curve=curve,
        params=p,
        diagnostics={
            "borrow_days": borrow_days,
            "borrow_day_pct": borrow_days / n * 100,
            "peak_borrow": peak_borrow,
            "peak_borrow_pct_of_book": peak_borrow / book0 * 100,
            "unfunded_days": unfunded_days,
            "max_unfunded": max_unfunded,
            "max_unfunded_pct_of_book": max_unfunded / book0 * 100,
            "mean_unfunded_when_short": (
                sum(unfunded_vals) / len(unfunded_vals) if unfunded_vals else 0.0
            ),
            "mean_deployment_pct": float(curve["deployment"].mean() * 100),
            "max_deployment_pct": float(curve["deployment"].max() * 100),
            "rebalances": int(curve["rebalanced"].sum()),
            "strategy_gold_corr": float(sr.corr(gr)),
        },
    )


def correlation_by_year(result: BookResult, gold_close: pd.Series) -> pd.DataFrame:
    """Strategy-sleeve vs gold correlation, per year.

    Reported per year rather than pooled because a diversifier that decorrelates
    on average but co-moves in the years the strategy struggles is not a
    diversifier. The ``strategy_ret_pct`` column is there so the bad years are
    identifiable at a glance.
    """
    c = result.curve
    rows = []

    def _compound(grp: pd.DataFrame, col: str) -> float:
        """Geometric compounding of a daily return column, in percent."""
        return float((1.0 + grp[col]).prod() - 1.0) * 100

    for year, grp in c.groupby(c.index.year):
        if len(grp) < 20:
            continue
        rows.append({
            "year": int(year),
            "corr_strategy_gold": round(float(grp["strat_ret"].corr(grp["gold_ret"])), 3),
            "strategy_ret_pct": round(_compound(grp, "strat_ret"), 2),
            "gold_ret_pct": round(_compound(grp, "gold_ret"), 2),
            "book_ret_pct": round(
                float(grp["book"].iloc[-1] / grp["book"].iloc[0] - 1) * 100, 2),
            "n_days": int(len(grp)),
        })
    return pd.DataFrame(rows)


__all__ = ["BookResult", "correlation_by_year", "simulate"]
