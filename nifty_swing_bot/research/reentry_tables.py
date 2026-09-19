"""Plain-text report for ``reentry_results.json``.

One row per configuration, because 25 cells do not fit as columns. The verdict
applies the bar pre-registered in ``reentry_study``'s docstring and is computed
from the JSON, not written by hand.
"""

from __future__ import annotations

import json
import os
from typing import Any

#: mean-drawdown tolerance, in percentage points, from the pre-registered bar.
DD_TOLERANCE_PP = 2.0


def _f(v: Any, dp: int = 2, dash: str = "--") -> str:
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return dash
    return f"{f:,.{dp}f}"


def _get(cell: dict, *path, default=None):
    cur: Any = cell
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


COLS: tuple[tuple[str, int, tuple], ...] = (
    ("pos/yr",      7, ("fold_summary", "positions_per_year")),
    ("re/yr",       6, ("fold_summary", "reentries_per_year")),
    ("re n",        5, ("n_reentries_total",)),
    ("re win%",     8, ("reentry_outcomes", "win_rate_pct")),
    ("re mean%",    9, ("reentry_outcomes", "mean_ret_pct")),
    ("posEdge%",    9, ("position_edge", "position_edge_pct")),
    ("edge t",      7, ("position_edge", "position_edge_t")),
    ("annRet%",     8, ("fold_summary", "ann_ret")),
    ("EXBEST%",     9, ("fold_summary", "ex_best_period")),
    ("bestShr%",    9, ("fold_summary", "best_period_share_pct")),
    ("meanDD",      7, ("fold_summary", "mean_dd")),
    ("worstDD",     8, ("fold_summary", "worst_dd")),
    ("CAGR%",       7, ("strategy", "cagr_pct")),
    ("Sharpe",      7, ("strategy", "sharpe")),
    ("Sortino",     8, ("strategy", "sortino")),
    ("win%",        6, ("strategy", "win_rate_pct")),
    ("PF",          6, ("strategy", "profit_factor")),
    ("posR",        7, ("strategy", "avg_r_multiple")),
    ("bkCAGR%",     8, ("book", "cagr_pct")),
    ("bkSharpe",    9, ("book", "sharpe")),
)


def window_table(res: dict, window: str) -> str:
    win = res["windows"][window]
    namew = 24
    head = "config".ljust(namew) + "".join(h.rjust(w) for h, w, _ in COLS)
    lines = [
        f"WINDOW {window}   (discovery -- not out-of-sample)",
        f"signals = {win['baseline']['signals']} in every cell "
        f"({win['baseline']['signals_per_year']}/yr) -- re-entry adds no signals",
        "",
        head,
        "-" * len(head),
    ]
    for name, cell in win.items():
        row = name.ljust(namew)
        for _, w, path in COLS:
            dp = 0 if path[-1] in ("n_reentries_total",) else (
                3 if path[-1] in ("profit_factor", "avg_r_multiple") else 2)
            row += _f(_get(cell, *path), dp).rjust(w)
        lines.append(row)
        if name == "baseline":
            lines.append("-" * len(head))
    return "\n".join(lines)


def folds_table(res: dict, window: str, keep: list[str]) -> str:
    win = res["windows"][window]
    years = [f["fold"] for f in win["baseline"].get("folds", [])]
    namew = 24
    head = "config".ljust(namew) + "".join(str(y).rjust(10) for y in years)
    lines = [f"FOLD-BY-FOLD RETURN %  --  {window}", head, "-" * len(head)]
    for name in keep:
        if name not in win:
            continue
        row = name.ljust(namew)
        for y in years:
            f = next((x for x in win[name].get("folds", []) if x["fold"] == y), None)
            row += (_f(f["ret"]) if f else "--").rjust(10)
        lines.append(row)
        if name == "baseline":
            lines.append("-" * len(head))
    return "\n".join(lines)


def verdict(res: dict) -> tuple[str, list[str]]:
    wins = list(res["windows"])
    base = {w: res["windows"][w]["baseline"] for w in wins}
    lines = [
        "VERDICT  (bar pre-registered in reentry_study's docstring)",
        "  improve position edge AND ex-best-fold in BOTH windows,",
        f"  with mean drawdown no more than {DD_TOLERANCE_PP:g}pp worse",
        "",
    ]
    passers: list[str] = []
    for name in res["windows"][wins[0]]:
        if name == "baseline":
            continue
        ok, detail = True, []
        for w in wins:
            a, b = res["windows"][w][name], base[w]
            e = (_get(a, "position_edge", "position_edge_pct", default=float("-inf"))
                 > _get(b, "position_edge", "position_edge_pct", default=float("-inf")))
            x = (_get(a, "fold_summary", "ex_best_period", default=float("-inf"))
                 > _get(b, "fold_summary", "ex_best_period", default=float("-inf")))
            dd = (_get(a, "fold_summary", "mean_dd", default=1e9)
                  <= _get(b, "fold_summary", "mean_dd", default=-1e9) + DD_TOLERANCE_PP)
            ok = ok and e and x and dd
            detail.append(f"{w}: edge{'+' if e else '-'} exbest{'+' if x else '-'} dd{'+' if dd else '-'}")
        if ok:
            passers.append(name)
        lines.append(f"  {name:26} {'PASS' if ok else 'fail'}   " + "   ".join(detail))
    lines += ["", f"  {len(passers)} of {len(res['windows'][wins[0]]) - 1} configs clear the bar."]
    if passers:
        lines.append("  passing: " + ", ".join(passers))
    else:
        lines.append("  No configuration clears the bar.")
    return "\n".join(lines), passers


def render(path: str | None = None) -> str:
    if path is None:
        from ..config import get_config
        path = os.path.join(str(get_config().paths.results_dir), "reentry_results.json")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)

    v, passers = verdict(res)
    # show baseline, every passer, and the extremes of the sweep in the fold table
    keep = ["baseline"] + passers[:6]
    parts = [
        "=" * 200,
        f"RE-ENTRY AFTER STOP-OUT (Phase 23)   run {res['generated']}",
        "=" * 200,
        res["config"],
        "",
        f"bar:  {res['bar_for_adoption']}",
        f"note: {res['note_signals_invariant']}",
        "",
    ]
    for w in res["windows"]:
        parts += [window_table(res, w), "", folds_table(res, w, keep), "", "=" * 200, ""]
    parts.append(v)
    return "\n".join(parts)


__all__ = ["DD_TOLERANCE_PP", "folds_table", "render", "verdict", "window_table"]
