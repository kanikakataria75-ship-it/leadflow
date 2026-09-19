"""Plain-text report for ``gamma_results.json``.

The verdict line at the bottom is computed, not written. The bar was set in
``gamma_study``'s docstring before the run: raise signals/year AND hold
``edge_vs_base`` AND hold ex-best-fold return, in BOTH windows. This module
checks that condition arm by arm and prints the answer it gets, including when
the answer is REJECTED #23.
"""

from __future__ import annotations

import json
import os
from typing import Any

ARM_LABEL = {
    "baseline": "baseline (box only)",
    "a_hhhl_2": "a) HH/HL x2",
    "a_hhhl_3": "a) HH/HL x3",
    "a_hhhl_4": "a) HH/HL x4",
    "b_rising_ma": "b) rising EMA20",
    "c_pullback": "c) shallow pullback",
    "combined": "combined (a3+b+c)",
}


def _f(v: Any, dp: int = 2) -> str:
    if v is None:
        return "--"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "--"
    return f"{f:,.{dp}f}"


def _row(cells: list[str], widths: list[int]) -> str:
    out = [cells[0].ljust(widths[0])]
    out += [c.rjust(w) for c, w in zip(cells[1:], widths[1:])]
    return "  ".join(out)


def recovery_table(res: dict) -> str:
    w = [22, 6, 7, 11, 7]
    lines = [
        "REAL-TRADE RECOVERY  --  39 testable of the 41 (2 have no usable series)",
        "The independent check: does the structure see what is actually being traded?",
        "",
        _row(["arm", "box", "gamma", "RECOVERED", "union"], w),
        "  ".join("-" * x for x in w),
    ]
    for arm, r in res["real_trade_recovery"].items():
        lines.append(_row([
            ARM_LABEL.get(arm, arm), str(r["box_at_entry"]), str(r["gamma_at_entry"]),
            str(r["recovered"]), f"{r['union']}/39",
        ], w))
    lines += [
        "",
        "  'recovered' counts only trades the box detector MISSES. A structure that",
        "  fires on most bars recovers trades trivially, so this column means nothing",
        "  without the signals/year column in the tables below.",
    ]
    return "\n".join(lines)


def arm_table(res: dict, window: str) -> str:
    w = [26] + [13] * len(ARM_LABEL)
    win = res["windows"][window]
    arms = [a for a in ARM_LABEL if a in win]

    def line(label, fn, dp=2):
        return _row([label] + [_f(fn(win[a]), dp) for a in arms], w)

    def fs(a, k):
        return win[a].get("fold_summary", {}).get(k)

    lines = [
        f"WINDOW {window}   (discovery -- not out-of-sample)",
        _row(["metric"] + [ARM_LABEL[a] for a in arms], w),
        "  ".join("-" * x for x in w),
        "-- population --",
        line("signals", lambda d: d.get("signals"), 0),
        line("signals / year", lambda d: d.get("signals_per_year"), 1),
        line("positions / year", lambda d: d.get("fold_summary", {}).get("positions_per_year"), 1),
        line("gamma share of signals %", lambda d: d.get("structures", {}).get("gamma_pct"), 1),
        "",
        "-- per-trade edge --",
        line("edge (abs, net) %", lambda d: d.get("edge", {}).get("edge"), 3),
        line("edge t", lambda d: d.get("edge", {}).get("edge_t"), 2),
        line("EDGE vs base %", lambda d: d.get("edge", {}).get("edge_vs_base"), 3),
        line("edge vs base t", lambda d: d.get("edge", {}).get("edge_vs_base_t"), 2),
        "",
        "-- folds (REJECTED #8/#12/#18 footing) --",
        line("mean fold return %", lambda d: d.get("fold_summary", {}).get("ann_ret")),
        line("median fold %", lambda d: d.get("fold_summary", {}).get("median_fold")),
        line("EX-BEST-FOLD %", lambda d: d.get("fold_summary", {}).get("ex_best_period")),
        line("best fold share %", lambda d: d.get("fold_summary", {}).get("best_period_share_pct"), 1),
        _row(["positive folds"] + [str(fs(a, "pos_folds") or "--") for a in arms], w),
        line("mean drawdown %", lambda d: d.get("fold_summary", {}).get("mean_dd")),
        line("worst drawdown %", lambda d: d.get("fold_summary", {}).get("worst_dd")),
        line("top-1 concentration %", lambda d: d.get("fold_summary", {}).get("top1"), 1),
        line("top-3 concentration %", lambda d: d.get("fold_summary", {}).get("top3"), 1),
        "",
        "-- strategy sleeve (continuous) --",
        line("total return %", lambda d: d.get("strategy", {}).get("total_return_pct")),
        line("CAGR %", lambda d: d.get("strategy", {}).get("cagr_pct")),
        line("Sharpe", lambda d: d.get("strategy", {}).get("sharpe")),
        line("Sortino", lambda d: d.get("strategy", {}).get("sortino")),
        line("Calmar", lambda d: d.get("strategy", {}).get("calmar")),
        line("Ulcer", lambda d: d.get("strategy", {}).get("ulcer_index"), 3),
        line("win rate %", lambda d: d.get("strategy", {}).get("win_rate_pct")),
        line("profit factor", lambda d: d.get("strategy", {}).get("profit_factor"), 3),
        line("position R", lambda d: d.get("strategy", {}).get("avg_r_multiple"), 3),
        "",
        "-- book (30/50/20, annual) --",
        line("total return %", lambda d: d.get("book", {}).get("total_return_pct")),
        line("CAGR %", lambda d: d.get("book", {}).get("cagr_pct")),
        line("Sharpe", lambda d: d.get("book", {}).get("sharpe")),
        line("Sortino", lambda d: d.get("book", {}).get("sortino")),
        line("Calmar", lambda d: d.get("book", {}).get("calmar")),
        line("Ulcer", lambda d: d.get("book", {}).get("ulcer_index"), 3),
        line("max drawdown %", lambda d: d.get("book", {}).get("max_drawdown_pct")),
    ]
    return "\n".join(lines)


def folds_table(res: dict, window: str) -> str:
    win = res["windows"][window]
    arms = [a for a in ARM_LABEL if a in win]
    years = [f["fold"] for f in win["baseline"].get("folds", [])]
    w = [26] + [13] * len(arms)
    lines = [
        f"FOLD-BY-FOLD RETURN %  --  {window}",
        _row(["fold"] + [ARM_LABEL[a] for a in arms], w),
        "  ".join("-" * x for x in w),
    ]
    for y in years:
        cells = [str(y)]
        for a in arms:
            f = next((x for x in win[a].get("folds", []) if x["fold"] == y), None)
            cells.append(_f(f["ret"]) if f else "--")
        lines.append(_row(cells, w))
    return "\n".join(lines)


def verdict(res: dict) -> str:
    """The pre-registered test, applied."""
    lines = ["VERDICT  (bar set before the run, in gamma_study's docstring)",
             "  raise signals/yr AND hold edge_vs_base AND hold ex-best-fold, BOTH windows",
             ""]
    wins = list(res["windows"])
    base = {w: res["windows"][w]["baseline"] for w in wins}
    passed_any = False
    for arm in ARM_LABEL:
        if arm == "baseline" or arm not in res["windows"][wins[0]]:
            continue
        checks, detail = [], []
        for w in wins:
            a, b = res["windows"][w][arm], base[w]
            sig = (a.get("signals_per_year") or 0) > (b.get("signals_per_year") or 0)
            ed = (a.get("edge", {}).get("edge_vs_base") or float("-inf")) >= \
                 (b.get("edge", {}).get("edge_vs_base") or float("-inf"))
            xb = (a.get("fold_summary", {}).get("ex_best_period") or float("-inf")) >= \
                 (b.get("fold_summary", {}).get("ex_best_period") or float("-inf"))
            checks += [sig, ed, xb]
            detail.append(f"{w}: sig{'+' if sig else '-'} edge{'+' if ed else '-'} exbest{'+' if xb else '-'}")
        ok = all(checks)
        passed_any = passed_any or ok
        lines.append(f"  {ARM_LABEL[arm]:22} {'PASS' if ok else 'fail'}   " + "   ".join(detail))
    lines += ["", "  " + ("At least one arm clears the bar -- see the tables before adopting anything."
                         if passed_any else
                         "No arm clears the bar. This is the fifth attempt to raise signal count "
                         "and\n  the fifth to fail it: record as REJECTED #23.")]
    return "\n".join(lines)


def render(path: str | None = None) -> str:
    if path is None:
        from ..config import get_config
        path = os.path.join(str(get_config().paths.results_dir), "gamma_results.json")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    parts = [
        "=" * 118,
        f"GAMMA (Phase 22) -- alternative structures at the box stage   run {res['generated']}",
        "=" * 118,
        res["config"],
        "",
        f"prior: {res['prior']}",
        f"bar:   {res['bar_for_adoption']}",
        "",
        recovery_table(res), "", "=" * 118, "",
    ]
    for w in res["windows"]:
        parts += [arm_table(res, w), "", folds_table(res, w), "", "=" * 118, ""]
    parts.append(verdict(res))
    return "\n".join(parts)


__all__ = ["arm_table", "folds_table", "recovery_table", "render", "verdict"]
