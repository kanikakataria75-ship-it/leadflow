"""Terminal report: the compact table, with the component values visible.

The score is deliberately never shown on its own. A ranking you cannot argue
with is a ranking you cannot use, so every row carries the four component
values that produced it and the box geometry behind them.
"""

from __future__ import annotations

import pandas as pd

from .pipeline import ScanResult

_RULE = "=" * 118


def _fmt(v, spec="{:.2f}", dash="-"):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return dash
    try:
        return spec.format(v)
    except (TypeError, ValueError):
        return str(v)


def _gate_fail_block(result: ScanResult, top: int = 15) -> list[str]:
    """Names that failed a gate, marked and ranked BELOW the clean ones.

    Shown rather than dropped because a silent rejection cannot be checked, and
    because the nearest misses are where a wrong gate would show itself first.
    Ranked by how close they came: a name with a box that failed one screen is
    listed above one that never reached the detector.
    """
    a = result.audit
    if a is None or a.empty or "verdict" not in a:
        return []
    fails = a[a["verdict"] != "WAIT & WATCH"].copy()
    if "score" in fails:
        fails = fails[fails["score"].notna() | fails.get("box_bars", pd.Series(dtype=float)).notna()]
    if fails.empty:
        return []
    fails["near_box"] = fails.get("box_bars", pd.Series(index=fails.index, dtype=float)).notna()
    fails["near_score"] = pd.to_numeric(fails.get("score"), errors="coerce").fillna(-1.0)
    fails = fails.sort_values(["near_box", "near_score"], ascending=[False, False])
    out = ["", _RULE,
           f"  GATE FAILURES  ({len(fails)})  -- ranked below the clean names, nearest miss first",
           _RULE,
           "  " + f"{'symbol':<13}{'stage':<22}{'score':>7}{'box':>11}  {'reason':<50}"]
    for r in fails.head(top).itertuples():
        box = (f"{int(r.box_bars)}b/{r.box_range_pct:.0f}%"
               if bool(getattr(r, "near_box", False)) and pd.notna(r.box_bars) else "-")
        sc = f"{r.near_score:.3f}" if r.near_score >= 0 else "-"
        out.append("  " + f"{str(r.symbol):<13}{str(r.stage)[:21]:<22}{sc:>7}{box:>11}  "
                          f"{str(r.reason)[:48]:<50}")
    if len(fails) > top:
        out.append(f"    ... {len(fails) - top} more (full list in the audit CSV)")
    return out


def _frozen_block(result: ScanResult) -> str:
    """Config provenance and where the book stands, on every run.

    Printed every day on purpose: the whole point of freezing a configuration
    is that a later reader can tell which configuration produced a given row of
    the forward record.
    """
    from . import live_config as LC

    lines = [LC.describe()]
    eq = getattr(result, "equity", None)
    dep = getattr(result, "deployed_notional", None)
    if eq:
        sleeve = LC.BOOK.w_strategy * eq
        lines.append(f"  BOOK       equity Rs{eq:,.0f}  ->  strategy sleeve "
                     f"Rs{sleeve:,.0f} ({LC.BOOK.w_strategy:.0%})  "
                     f"arbitrage Rs{LC.BOOK.w_arbitrage * eq:,.0f}  "
                     f"gold Rs{LC.BOOK.w_gold * eq:,.0f}")
        if dep is not None:
            pct_book = dep / eq * 100 if eq else 0.0
            pct_sleeve = dep / sleeve * 100 if sleeve else 0.0
            note = "" if dep <= sleeve else "  OVER SLEEVE -- draws on arbitrage (never gold)"
            lines.append(f"  DEPLOYED   Rs{dep:,.0f} = {pct_book:.1f}% of book, "
                         f"{pct_sleeve:.0f}% of the strategy sleeve{note}")
        else:
            lines.append("  DEPLOYED   not supplied -- pass --deployed to see sleeve usage")
    return chr(10).join(lines)


def format_report(result: ScanResult, *, top: int = 25) -> str:
    """Render one scan as text."""
    L: list[str] = ["", _RULE,
                    f"  AES DAILY SCAN  --  {result.scan_date}", _RULE]

    b = result.breadth
    if b is not None:
        L.append(f"  BREADTH   {b.value} distinct names admitted in the last 63 sessions")
        L.append(f"            {b.percentile:.0f}th percentile of history -- {b.label}")
        if b.nearest_year:
            L.append(f"            closest historical analogue: {b.nearest_year} "
                     f"(mean {b.nearest_year_mean:.0f})")
        L.append("            context only -- breadth is not a gate (RESEARCH_AES.md Phase 11/12)")
    L.append(f"  REGIME    benchmark is {'BELOW' if result.regime_weak else 'above'} its SMA(50)"
             + ("  -- section 8 says high-conviction only, sizes reduced" if result.regime_weak else ""))
    L.append(f"  UNIVERSE  {result.universe_size} symbols | watchlist {result.watchlist_size} "
             f"| with a box {len(result.candidates)} | actionable "
             f"{0 if result.candidates.empty else int(result.candidates['actionable'].sum())}"
             f" | {result.duration_s}s")
    L.append(_RULE)
    L.append(_frozen_block(result))
    L.append(_RULE)

    if result.candidates.empty:
        L += ["", "  Nothing on the watchlist has a detectable box today.", ""]
        return "\n".join(L)

    for bucket, title in (("high_conviction", "HIGH CONVICTION"),
                          ("wait_and_watch", "WAIT AND WATCH"),
                          ("reject", "NOTHING YET (below threshold)")):
        g = result.candidates[result.candidates["bucket"] == bucket]
        if g.empty:
            continue
        L += ["", f"  {title}  ({len(g)})", "  " + "-" * 116]
        L.append("  " + f"{'':<2}{'symbol':<13}{'score':>6}{'  fast absb  rs  cyc':<22}"
                        f"{'box':>13}{'pos':>6}{'HL':>4}{'>mid':>6}{'zone':>8}"
                        f"{'resist':>9}{'tf':>4}{'rs':>10}")
        for r in g.head(top).itertuples():
            flag = ">>" if r.actionable else "  "
            comps = (f"{_fmt(r.sc_fast,'{:.2f}'):>5}{_fmt(r.sc_absorbed,'{:.2f}'):>5}"
                     f"{_fmt(r.sc_rs,'{:.2f}'):>5}{_fmt(r.sc_cycles,'{:.2f}'):>5}")
            box = f"{r.box_bars}b/{_fmt(r.box_range_pct,'{:.0f}')}%"
            tf = "".join(x for x, on in (("D", r.tf_daily), ("W", r.tf_weekly), ("M", r.tf_monthly)) if on) or "-"
            L.append("  " + f"{flag:<2}{r.symbol:<13}{_fmt(r.score,'{:.3f}'):>6}  {comps:<20}"
                            f"{box:>13}{_fmt(r.box_position,'{:.2f}'):>6}{r.higher_lows:>4}"
                            f"{_fmt(r.pct_closes_above_mid,'{:.0f}'):>6}{str(r.small_zone):>8}"
                            f"{_fmt(r.resistance_dist_pct,'{:+.1f}%'):>9}{tf:>4}"
                            f"{str(r.rs_class)[:9]:>10}")

    act = result.actionable
    if not act.empty:
        sized = act[act["qty"] > 0]
        unsized = act[act["qty"] <= 0]
        if not sized.empty:
            L += ["", _RULE, f"  TRADE PLANS  ({len(sized)})  "
                             f"-- entry is a reference off today's close, not a fill", _RULE]
            L.append("  " + f"{'symbol':<13}{'mode':>5}{'entry':>10}{'stop':>10}{'basis':>16}"
                            f"{'risk/sh':>9}{'stop%':>7}{'qty':>7}{'notional':>12}{'risk Rs':>10}")
            for r in sized.itertuples():
                L.append("  " + f"{r.symbol:<13}{_fmt(r.entry_mode,'{:.0f}'):>5}"
                                f"{_fmt(r.entry_ref):>10}{_fmt(r.stop):>10}{str(r.stop_basis):>16}"
                                f"{_fmt(r.risk_per_share):>9}{_fmt(r.stop_pct,'{:.1f}'):>7}"
                                f"{_fmt(r.qty,'{:.0f}'):>7}{_fmt(r.notional,'{:,.0f}'):>12}"
                                f"{_fmt(r.risk_amount,'{:,.0f}'):>10}")
                tr = getattr(r, "tranches", ()) or ()
                if tr:
                    parts = "   ".join(f"sell {q} @ {px:,.2f} (+{pct:.0%})" for px, q, pct in tr)
                    L.append("  " + " " * 13 + f"tranches: {parts}   then "
                             f"{getattr(r, 'runner_qty', 0)} trails on the ATR stop")
        if not unsized.empty:
            L += ["", "  TRIGGERED BUT NOT SIZED  -- the state machine fired, the score did not",
                  "  " + "-" * 116]
            for r in unsized.itertuples():
                why = ("below the reject threshold" if r.bucket == "reject"
                       else "weak regime: section 8 takes high-conviction only")
                L.append("  " + f"{r.symbol:<13} mode {_fmt(r.entry_mode,'{:.0f}')}  "
                                f"score {_fmt(r.score,'{:.3f}')}  ({r.bucket}) -- {why}")

    L += _gate_fail_block(result)

    if result.charts:
        L += ["", f"  charts: {len(result.charts)} rendered"]
        for sym, p in list(result.charts.items())[:12]:
            L.append(f"    {sym:<13} {p}")
    L += ["", _RULE,
          "  Ranking is advisory. Component values are shown so it can be overruled.", _RULE, ""]
    return "\n".join(L)


def format_forward_record(df: pd.DataFrame, *, top: int = 30) -> str:
    """The accumulating live record, newest first."""
    if df is None or df.empty:
        return "\n  No forward record yet -- nothing has been flagged actionable.\n"
    L = ["", _RULE, "  AES FORWARD RECORD  (live, not a backtest)", _RULE,
         "  " + f"{'date':<12}{'symbol':<13}{'bucket':<17}{'score':>6}{'bars':>5}"
                f"{'r5':>7}{'r10':>7}{'r20':>7}{'exc20':>7}{'MAE':>7}  {'result':<16}"]
    for r in df.head(top).itertuples():
        L.append("  " + f"{str(r.scan_date):<12}{r.symbol:<13}{str(r.bucket):<17}"
                        f"{_fmt(r.score,'{:.2f}'):>6}{_fmt(r.bars_elapsed,'{:.0f}'):>5}"
                        f"{_fmt(r.ret_5,'{:+.1f}'):>7}{_fmt(r.ret_10,'{:+.1f}'):>7}"
                        f"{_fmt(r.ret_20,'{:+.1f}'):>7}{_fmt(r.excess_20,'{:+.1f}'):>7}"
                        f"{_fmt(r.mae_pct,'{:+.1f}'):>7}  {str(r.exit_reason or 'open'):<16}")
    closed = df[df["realised_pct"].notna()] if "realised_pct" in df else pd.DataFrame()
    if len(closed):
        wins = (closed["realised_pct"] > 0).mean() * 100
        L += ["", f"  closed: {len(closed)}   win rate {wins:.0f}%   "
                  f"mean {closed['realised_pct'].mean():+.2f}%   "
                  f"median {closed['realised_pct'].median():+.2f}%"]
    L += ["", _RULE, ""]
    return "\n".join(L)


def format_funnel(audit: pd.DataFrame) -> str:
    """The full-universe funnel: how many dropped at each stage, and why.

    Printed alongside the candidate table because the candidate table is a
    survivorship view -- it cannot show you a screener that is too tight,
    since everything the screener rejected is already absent from it.
    """
    if audit is None or audit.empty:
        return ""
    from .audit import VERDICT

    order = sorted(audit["stage"].unique(),
                   key=lambda s: audit.loc[audit.stage == s, "stage_rank"].iloc[0],
                   reverse=True)
    total = len(audit)
    L = ["", _RULE, f"  PIPELINE FUNNEL  --  {total} universe symbols", _RULE,
         "  " + f"{'stage':<24}{'verdict':<14}{'n':>6}{'remaining':>11}"]
    remaining = total
    # Walk in pipeline order (entry first) so "remaining" reads downwards.
    for stage in reversed(order):
        g = audit[audit.stage == stage]
        v = VERDICT.get(stage, "")
        if v == "REJECTED":
            remaining -= len(g)
        L.append("  " + f"{stage:<24}{v:<14}{len(g):>6}{remaining:>11}")
    L += ["", "  most common reason per stage:"]
    for stage in reversed(order):
        g = audit[audit.stage == stage]
        top = g["reason"].str.split(" -- ").str[0].value_counts()
        if len(top):
            L.append(f"    {stage:<24}{top.index[0][:74]}")
    L += ["", _RULE, ""]
    return "\n".join(L)


__all__ = ["format_forward_record", "format_funnel", "format_report"]
