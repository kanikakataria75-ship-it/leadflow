"""Plain-text rendering of ``stop_forensics.json``. Sample size on every line."""

from __future__ import annotations

import json
import os
from typing import Any

from .stop_forensics import HORIZONS, MARGINAL_ATR, THRESHOLDS

#: Below this, a cut is labelled inconclusive rather than read as a result.
MIN_N = 30


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


def cut_table(cut: dict, title: str) -> str:
    n = cut.get("n", 0)
    flag = "" if n >= MIN_N else f"   <-- n={n}, INCONCLUSIVE"
    w = [8, 7, 9, 10, 9, 9, 9, 9]
    head = ["bars", "n", "mean %", "median %", ">+10%", ">+20%", ">+50%", ">+100%"]
    lines = [
        f"{title}   n={n}{flag}",
        "  ".join(h.rjust(x) for h, x in zip(head, w)),
        "  ".join("-" * x for x in w),
    ]
    if n == 0:
        return "\n".join(lines + ["  (empty)"])
    for h in HORIZONS:
        d = cut.get(f"h{h}", {})
        cells = [
            str(h), str(d.get("n", 0)),
            _f(d.get("mean_pct")), _f(d.get("median_pct")),
        ] + [_f(d.get(f"pct_over_{int(t)}"), 1) for t in THRESHOLDS]
        lines.append("  ".join(c.rjust(x) for c, x in zip(cells, w)))
    m = cut.get("max_within_40", {})
    lines += [
        f"  max within 40 bars: median {_f(m.get('median_max_pct'))}%  "
        f"mean {_f(m.get('mean_max_pct'))}%",
        f"  bar of that max:    median {_f(m.get('median_argmax_bar'), 1)}  "
        f"mean {_f(m.get('mean_argmax_bar'), 1)}  "
        f"IQR {_f(m.get('argmax_q25'), 0)}-{_f(m.get('argmax_q75'), 0)}",
        f"  depth below stop:   median {_f(cut.get('median_depth_atr'), 3)} ATR  "
        f"({_f(cut.get('median_depth_pct'))}%)",
    ]
    return "\n".join(lines)


def render(path: str | None = None) -> str:
    if path is None:
        from ..config import get_config
        path = os.path.join(str(get_config().paths.results_dir), "stop_forensics.json")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)

    parts = [
        "=" * 86,
        f"STOP FORENSICS -- what happened after a stop exit   run {res['generated']}",
        "=" * 86,
        res["note"],
        f"marginal rule: {res['marginal_rule']}",
        f"cuts with n < {MIN_N} are labelled INCONCLUSIVE rather than read as findings",
        "",
    ]
    for wname, regimes in res["windows"].items():
        for rname, s in regimes.items():
            parts += ["#" * 86,
                      f"# {wname}   regime={rname}",
                      "#" * 86, ""]
            parts += [cut_table(s["all"], "ALL STOP EXITS"), ""]
            parts += ["--- SPLIT 1: by score at entry ---", ""]
            for k, lbl in (("high_conviction", "HIGH CONVICTION"),
                           ("wait_and_watch", "WAIT & WATCH (the rest)")):
                parts += [cut_table(s["by_score"][k], lbl), ""]
            parts += ["--- SPLIT 2: by depth below the stop ---", ""]
            for k, v in s["by_marginality"].items():
                if k == "unclassified":
                    parts += [f"unclassified (no finite depth): n={v['n']}", ""]
                    continue
                lbl = ("MARGINAL CLOSE" if "marginal" in k else "CLEAN BREAKDOWN")
                parts += [cut_table(v, lbl), ""]
    return "\n".join(parts)


__all__ = ["MIN_N", "cut_table", "render"]
