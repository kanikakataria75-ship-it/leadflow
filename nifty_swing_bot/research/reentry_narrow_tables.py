"""Report for ``reentry_narrow_results.json``. Attribution leads, not performance.

The columns are ordered so the three pre-registered conditions are read before
any return figure. That ordering is deliberate: Phase 23 failed because a
config that beat baseline was read as a win before anyone asked where the gain
came from.
"""

from __future__ import annotations

import json
import os
from typing import Any

MAJORITY_PCT = 50.0


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


def _get(c: dict, *path, default=None):
    cur: Any = c
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def window_table(res: dict, window: str) -> str:
    w = res["windows"][window]
    cfgs, null = w["configs"], w["null"]
    namew = 30
    cols = [
        ("re n", 6, ("n_reentries_total",), 0),
        ("re win%", 8, ("reentry_outcomes", "win_rate_pct"), 1),
        ("re mean%", 9, ("reentry_outcomes", "mean_ret_pct"), 2),
        ("reentryP&L", 12, ("attribution", "reentry_pnl"), 0),
        ("equityDelta", 13, ("attribution", "equity_delta"), 0),
        ("SHARE%", 8, ("attribution", "reentry_share_pct"), 1),
        ("annRet%", 8, ("fold_summary", "ann_ret"), 2),
        ("EXBEST%", 9, ("fold_summary", "ex_best_period"), 2),
        ("meanDD", 7, ("fold_summary", "mean_dd"), 2),
        ("CAGR%", 7, ("strategy", "cagr_pct"), 2),
        ("Sharpe", 7, ("strategy", "sharpe"), 2),
        ("posEdge%", 9, ("position_edge", "position_edge_pct"), 2),
    ]
    head = "config".ljust(namew) + "".join(h.rjust(x) for h, x, _, _ in cols)
    lines = [
        f"WINDOW {window}   (discovery -- not out-of-sample)",
        f"population: {res['population']}",
        f"NULL (score-jitter, {null['seeds']} seeds, no positions added): "
        f"mean {_f(null['mean'], 0)}  sd {_f(null['sd'], 0)}  "
        f"p95 {_f(null['p95'], 0)}  max {_f(null['max'], 0)}",
        "",
        head,
        "-" * len(head),
    ]
    for name, c in cfgs.items():
        row = name.ljust(namew)
        for _, x, path, dp in cols:
            row += _f(_get(c, *path), dp).rjust(x)
        lines.append(row)
        if name == "baseline":
            lines.append("-" * len(head))
    return "\n".join(lines)


def verdict(res: dict) -> str:
    wins = list(res["windows"])
    lines = [
        "VERDICT  (pre-registered in reentry_narrow's docstring)",
        "  1. re-entry trades' own net P&L > 0",
        f"  2. that P&L is >= {MAJORITY_PCT:g}% of the total equity gain",
        "  3. the equity gain exceeds the score-jitter null's p95",
        "  all three, in BOTH windows",
        "",
    ]
    names = [n for n in res["windows"][wins[0]]["configs"] if n != "baseline"]
    passers = []
    for name in names:
        ok, detail = True, []
        for wn in wins:
            c = res["windows"][wn]["configs"][name]
            p95 = res["windows"][wn]["null"]["p95"]
            pnl = _get(c, "attribution", "reentry_pnl", default=0.0) or 0.0
            share = _get(c, "attribution", "reentry_share_pct")
            delta = _get(c, "attribution", "equity_delta", default=0.0) or 0.0
            c1 = pnl > 0
            c2 = share is not None and share >= MAJORITY_PCT
            c3 = delta > p95
            ok = ok and c1 and c2 and c3
            detail.append(f"{wn}: pnl{'+' if c1 else '-'} share{'+' if c2 else '-'} null{'+' if c3 else '-'}")
        if ok:
            passers.append(name)
        lines.append(f"  {name:32} {'PASS' if ok else 'fail'}   " + "   ".join(detail))
    lines += ["", f"  {len(passers)} of {len(names)} configs clear the bar."]
    lines.append("  passing: " + ", ".join(passers) if passers
                 else "  No configuration clears the bar.")
    return "\n".join(lines)


def render(path: str | None = None) -> str:
    if path is None:
        from ..config import get_config
        path = os.path.join(str(get_config().paths.results_dir),
                            "reentry_narrow_results.json")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    parts = [
        "=" * 150,
        f"RE-ENTRY, NARROW POPULATION (Phase 23b)   run {res['generated']}",
        "=" * 150,
        f"bar: {res['bar_for_adoption']}",
        "",
    ]
    for wn in res["windows"]:
        parts += [window_table(res, wn), "", "=" * 150, ""]
    parts.append(verdict(res))
    return "\n".join(parts)


__all__ = ["MAJORITY_PCT", "render", "verdict", "window_table"]
