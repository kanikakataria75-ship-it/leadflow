"""Phase 3b: can filtering information-events lift the edge above its cost?

The conditioning study established the mechanism: **sector-relative dislocation
reverts, absolute weakness does not**. Ranking the universe by 3-day return minus
sector-median 3-day return and taking the worst name gives +0.405% over three
bars; ranking by absolute weakness gives *minus* 0.193%, because the biggest
absolute decliner on any given day usually has genuine bad news.

That contrast is the whole thesis, and it implies a refinement. If the extreme
absolute decliners are information events contaminating the selection, removing
them should raise the edge of what remains. Each filter below is a specific
prediction from that thesis, not a parameter to be tuned:

``no_orphan_gap``
    The event study found unshared gap-downs continue falling (-0.407%, t>3).
    An overnight move the sector did not share is stock-specific news.
``not_collapsing``
    A decline beyond a few ATRs is a repricing, not an inventory shock.
``sector_calm``
    If the whole sector is falling, a stock falling with it is not dislocated;
    the peer comparison is only meaningful when peers are stable.
``trend_intact``
    Above the 200-day average, the reversion is a pullback in an uptrend. Below
    it, "cheap relative to peers" may just be a failing business.
``liquid_enough``
    A stock too thin to absorb a position cannot be traded at the modelled price.

The test is whether each filter raises edge per trade enough to justify the
trades it removes -- measured against the round-trip cost it has to clear.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.costs import round_trip_cost_pct
from ..config import AppConfig, get_config
from .protocol import DISCOVERY, Window

logger = logging.getLogger(__name__)

#: Filters as (name, predicate, rationale). Each is a prediction of the
#: information-versus-flow thesis, testable independently.
FILTERS: tuple[tuple[str, Callable[[pd.DataFrame], pd.Series], str], ...] = (
    (
        "no_orphan_gap",
        lambda d: ~((d["gap"] < -0.02) & (d["rel_gap"] < -0.015)),
        "unshared gap-down is stock-specific news, and was measured to continue",
    ),
    (
        "not_collapsing",
        lambda d: d["ret_3_atr"] > -3.0,
        "a decline beyond 3 ATRs is a repricing, not an inventory shock",
    ),
    (
        "sector_calm",
        lambda d: d["sector_ret_3"].abs() < 0.04,
        "peer comparison only means something when peers are stable",
    ),
    (
        "trend_intact",
        lambda d: d["above_200"],
        "below the 200-day average, cheap-vs-peers may be a failing business",
    ),
    (
        "liquid_enough",
        lambda d: d["turnover_cr"] >= 5.0,
        "a position must be fillable at the modelled price",
    ),
)


def apply_filters(
    frame: pd.DataFrame, names: Sequence[str]
) -> pd.Series:
    """Combine the named filters into a single boolean mask."""
    mask = pd.Series(True, index=frame.index)
    lookup = {name: fn for name, fn, _ in FILTERS}
    for name in names:
        fn = lookup.get(name)
        if fn is None:
            continue
        mask &= fn(frame).fillna(False)
    return mask


def rank_and_score(
    panel: pd.DataFrame,
    window: Window,
    *,
    signal: str = "rel_sector_3",
    filters: Sequence[str] = (),
    top_n: int = 3,
    horizons: Sequence[int] = (2, 3, 5),
) -> dict[str, Any]:
    """Score a filtered, cross-sectionally ranked selection.

    Filters are applied **before** ranking, so the rank is computed over the
    eligible set rather than being diluted by names that would never be traded.

    Returns:
        Edge, hit rate and t-statistic at each horizon, plus trade frequency.
    """
    scoped = panel[panel["tradeable"] & window.mask(panel["date"])].copy()
    if filters:
        scoped = scoped[apply_filters(scoped, filters)]
    scoped = scoped.dropna(subset=[signal])
    if scoped.empty:
        return {}

    scoped["rank"] = scoped.groupby("date")[signal].rank(method="first", ascending=True)
    picks = scoped[scoped["rank"] <= top_n]

    years = max((window.end - window.start).days / 365.25, 1e-9)
    out: dict[str, Any] = {
        "filters": "+".join(filters) if filters else "(none)",
        "top_n": top_n,
        "events": int(len(picks)),
        "per_year": len(picks) / years,
    }
    for h in horizons:
        values = picks[f"exc_{h}"].dropna()
        if len(values) < 50:
            out[f"exc_{h}"] = np.nan
            out[f"t_{h}"] = np.nan
            continue
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        out[f"exc_{h}"] = mean * 100
        out[f"hit_{h}"] = float((values > 0).mean()) * 100
        out[f"t_{h}"] = mean / (sd / np.sqrt(len(values))) if sd > 0 else np.nan
    return out


def run(
    *,
    top_ns: Sequence[int] = (1, 2, 3, 5, 8),
    horizons: Sequence[int] = (2, 3, 5),
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    """Test filter combinations and selection widths on the discovery window."""
    from .conditioning import CARRY
    from .event_study import build_event_panel, load_panel

    cfg = cfg or get_config()
    prices, industry_of, _ = load_panel(cfg=cfg)
    panel = build_event_panel(prices, industry_of, horizons=(1, 2, 3, 5, 8, 10), carry=CARRY)
    if panel.empty:
        raise RuntimeError("Event panel empty.")

    # Cost to beat, at a realistic position size for a Rs 10 lakh book.
    cost_100k = round_trip_cost_pct(100_000.0, cfg.execution)
    lean = cfg.execution.model_copy(deep=True)
    lean.slippage_bps = 8.0          # mid-caps with >= Rs 5 cr turnover
    lean.brokerage_flat_inr = 0.0    # several brokers charge zero on delivery
    cost_lean = round_trip_cost_pct(100_000.0, lean)

    print("\n" + "=" * 100)
    print(f"  FILTER REFINEMENT -- {DISCOVERY}")
    print("  Signal: 3-day return minus sector median. Long the most dislocated.")
    print(f"  Cost to beat: {cost_100k:.3f}% (modelled) / {cost_lean:.3f}% (lean, liquid names)")
    print("=" * 100)

    # Cumulative filter stacks, each adding one prediction of the thesis.
    stacks: list[tuple[str, ...]] = [
        (),
        ("no_orphan_gap",),
        ("no_orphan_gap", "not_collapsing"),
        ("no_orphan_gap", "not_collapsing", "sector_calm"),
        ("no_orphan_gap", "not_collapsing", "sector_calm", "trend_intact"),
        ("no_orphan_gap", "not_collapsing", "sector_calm", "trend_intact", "liquid_enough"),
    ]

    rows: list[dict[str, Any]] = []
    for stack in stacks:
        for top_n in top_ns:
            record = rank_and_score(
                panel, DISCOVERY, filters=stack, top_n=top_n, horizons=horizons
            )
            if record:
                rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("  No results.")
        return frame

    print(f"\n  {'filters':<52}{'topN':>5}{'/yr':>7}" +
          "".join(f"{f'{h}d exc%':>10}{'t':>6}" for h in horizons) + f"{'net@lean':>10}")
    print("  " + "-" * 96)
    for _, row in frame.iterrows():
        line = f"  {row['filters'][:50]:<52}{int(row['top_n']):>5}{row['per_year']:>7.0f}"
        for h in horizons:
            value, t = row.get(f"exc_{h}"), row.get(f"t_{h}")
            line += f"{value:>10.3f}{t:>6.2f}" if pd.notna(value) else f"{'—':>10}{'—':>6}"
        best = max(
            (row.get(f"exc_{h}", np.nan) for h in horizons),
            default=np.nan,
            key=lambda v: -np.inf if pd.isna(v) else v,
        )
        line += f"{(best - cost_lean):>10.3f}" if pd.notna(best) else f"{'—':>10}"
        print(line)
    print("  " + "-" * 96)
    print("  net@lean = best-horizon edge minus the lean round-trip cost.")
    print("  A positive figure is the necessary condition for the strategy to pay;")
    print("  the portfolio backtest is what decides whether it actually does.")
    print("=" * 100 + "\n")

    out = cfg.paths.results_dir / "refinement_discovery.csv"
    frame.to_csv(out, index=False)
    logger.info("Written to %s", out)
    return frame


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Filter refinement study.")
    parser.add_argument("--top", default="1,2,3,5,8")
    parser.add_argument("--horizons", default="2,3,5")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    run(
        top_ns=tuple(int(x) for x in args.top.split(",")),
        horizons=tuple(int(x) for x in args.horizons.split(",")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
