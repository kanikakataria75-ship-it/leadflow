"""Plain-text renderings of ``leadflow_frozen_results.json``.

Every table here obeys the reporting rules the terminal must also obey:

* a headline return never appears without its ex-best-fold twin and the best
  fold's share of the total;
* every window is labelled with its provenance, and as of Phase 16 both are
  ``discovery`` -- nothing in this project is out-of-sample validated;
* variant (c) is labelled a forward-test candidate wherever it wins;
* R is the POSITION-level figure from ``positions_from_rows``, never the
  trade-row mean, which a profit ladder inflates by construction.
"""

from __future__ import annotations

import json
import os
from typing import Any

#: (json key, printed label, decimals, higher-is-better | None if neutral)
RISK_ROWS: tuple[tuple[str, str, int, bool | None], ...] = (
    ("total_return_pct", "Total return %", 2, True),
    ("cagr_pct", "CAGR %", 2, True),
    ("volatility_pct", "Volatility %", 2, False),
    ("downside_deviation_pct", "Downside dev %", 2, False),
    ("sharpe", "Sharpe", 2, True),
    ("sortino", "Sortino", 2, True),
    ("calmar", "Calmar", 2, True),
    ("omega", "Omega", 3, True),
    ("ulcer_index", "Ulcer index", 4, False),
    ("max_drawdown_pct", "Max drawdown %", 2, None),
    ("avg_drawdown_pct", "Avg drawdown %", 2, None),
    ("var_95_pct_daily", "VaR 95% (daily)", 3, None),
    ("cvar_95_pct_daily", "CVaR 95% (daily)", 3, None),
    ("skew", "Skew", 3, None),
    ("kurtosis", "Excess kurtosis", 3, None),
)

LADDERS = ("old_ladder_5_8_20", "variant_c_12_20")
LADDER_LABEL = {"old_ladder_5_8_20": "old 5/8/20", "variant_c_12_20": "variant (c)"}


def _f(v: Any, dp: int) -> str:
    if v is None:
        return "--"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:                       # NaN
        return "--"
    return f"{f:,.{dp}f}"


def _rule(widths: list[int], ch: str = "-") -> str:
    return "  ".join(ch * w for w in widths)


def risk_table(res: dict, window: str, side: str) -> str:
    """Old ladder vs variant (c), one side of the book, one window."""
    w = res["windows"][window]
    cols = [w["ladders"][k][side] for k in LADDERS]
    widths = [22, 16, 16, 12]
    head = ["metric", LADDER_LABEL[LADDERS[0]], LADDER_LABEL[LADDERS[1]], "delta"]

    out = [
        f"{side.upper()}  --  {window}  ({w['provenance']})",
        "  ".join(h.ljust(x) for h, x in zip(head, widths)),
        _rule(widths),
    ]
    for key, label, dp, higher in RISK_ROWS:
        a, b = cols[0].get(key), cols[1].get(key)
        try:
            d = float(b) - float(a)
            ds = f"{d:+,.{dp}f}"
            if higher is not None and abs(d) > 10 ** -dp:
                ds += "  better" if (d > 0) == higher else "  worse"
        except (TypeError, ValueError):
            ds = "--"
        out.append("  ".join([
            label.ljust(widths[0]),
            _f(a, dp).rjust(widths[1]),
            _f(b, dp).rjust(widths[2]),
            ds.ljust(widths[3]),
        ]))
    return "\n".join(out)


def concentration_table(res: dict, window: str) -> str:
    """The headline-with-its-twin table. A headline alone is a lie by omission."""
    w = res["windows"][window]
    widths = [26, 16, 16]
    out = [
        f"CONCENTRATION  --  {window}  ({w['provenance']})",
        "  ".join(h.ljust(x) for h, x in
                  zip(["annual fold return", LADDER_LABEL[LADDERS[0]],
                       LADDER_LABEL[LADDERS[1]]], widths)),
        _rule(widths),
    ]
    for side in ("strategy", "book"):
        cs = [w["ladders"][k][f"{side}_concentration"] for k in LADDERS]
        for key, label in (("mean", "mean %"), ("median", "median %"),
                           ("ex_best_period", "EX-BEST-FOLD %"),
                           ("ex_top2", "ex-top-2 %"),
                           ("best_period_share_pct", "best fold share %"),
                           ("positive", "positive folds")):
            vals = [c.get(key) for c in cs]
            out.append("  ".join([
                f"{side}: {label}".ljust(widths[0]),
                (str(vals[0]) if key == "positive" else _f(vals[0], 2)).rjust(widths[1]),
                (str(vals[1]) if key == "positive" else _f(vals[1], 2)).rjust(widths[2]),
            ]))
        out.append("")
    return "\n".join(out).rstrip()


def trade_table(res: dict, window: str) -> str:
    """Position-level trade statistics. Note the R convention explicitly."""
    w = res["windows"][window]
    widths = [26, 16, 16]
    rows = (
        ("n_positions", "positions", 0),
        ("n_traded_rows", "trade rows (ladder-split)", 0),
        ("win_rate_pct", "win rate %", 2),
        ("profit_factor", "profit factor", 3),
        ("avg_r_multiple", "avg R  (POSITION-level)", 3),
        ("median_r_multiple", "median R (position)", 3),
        ("avg_r_multiple_row_mean_DO_NOT_USE", "  [row-mean R: DO NOT USE]", 3),
        ("payoff_ratio", "payoff ratio", 3),
        ("median_hold_bars", "median hold (bars)", 1),
        ("mean_deployment_pct", "mean deployment %", 2),
        ("cost_pct_of_gross_pnl", "costs % of gross P&L", 2),
    )
    out = [
        f"TRADES  --  {window}  ({w['provenance']})",
        "  ".join(h.ljust(x) for h, x in
                  zip(["metric", LADDER_LABEL[LADDERS[0]],
                       LADDER_LABEL[LADDERS[1]]], widths)),
        _rule(widths),
    ]
    for key, label, dp in rows:
        vals = []
        for k in LADDERS:
            lv = w["ladders"][k]
            vals.append(lv.get(key, lv["strategy"].get(key)))
        out.append("  ".join([
            label.ljust(widths[0]),
            _f(vals[0], dp).rjust(widths[1]),
            _f(vals[1], dp).rjust(widths[2]),
        ]))
    out += [
        "",
        "  R is the POSITION-level figure. The row-mean line above is printed only",
        "  so the two are never confused: a profit ladder splits one position into",
        "  2-4 rows and every ladder row is profitable by construction, so the row",
        "  mean is inflated and must not be quoted.",
    ]
    return "\n".join(out)


def folds_table(res: dict, window: str) -> str:
    w = res["windows"][window]
    widths = [26, 16, 16]
    out = [
        f"ANNUAL FOLDS  --  {window}  ({w['provenance']})",
        "  ".join(h.ljust(x) for h, x in
                  zip(["fold", LADDER_LABEL[LADDERS[0]],
                       LADDER_LABEL[LADDERS[1]]], widths)),
        _rule(widths),
    ]
    a = w["ladders"][LADDERS[0]]
    b = w["ladders"][LADDERS[1]]
    for tag, ykey, vkey in (("strategy", "fold_years", "strategy_folds_pct"),
                            ("book", "book_fold_years", "book_folds_pct")):
        for yr, x, y in zip(a[ykey], a[vkey], b[vkey]):
            out.append("  ".join([
                f"{tag} {yr}".ljust(widths[0]),
                _f(x, 2).rjust(widths[1]),
                _f(y, 2).rjust(widths[2]),
            ]))
        out.append("")
    return "\n".join(out).rstrip()


def render(path: str | None = None) -> str:
    if path is None:
        from ..config import get_config
        path = os.path.join(str(get_config().paths.results_dir),
                            "leadflow_frozen_results.json")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)

    parts = [
        "=" * 74,
        f"LEADFLOW -- FROZEN CONFIG  (frozen {res['frozen_on']}, run {res['generated']})",
        "=" * 74,
        res["config"],
        "",
        "Both windows are DISCOVERY. Nothing here is out-of-sample validated.",
        "Variant (c) is a FORWARD-TEST CANDIDATE, not an established improvement",
        "(best paired t across folds = 1.98).",
        "",
    ]
    for window in res["windows"]:
        for side in ("strategy", "book"):
            parts += [risk_table(res, window, side), ""]
        parts += [concentration_table(res, window), "",
                  trade_table(res, window), "",
                  folds_table(res, window), "", "=" * 74, ""]
    return "\n".join(parts)


__all__ = ["concentration_table", "folds_table", "render", "risk_table", "trade_table"]
