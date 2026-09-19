"""Event study: does a hypothesis predict anything at short horizons?

Measures, for every hypothesis, the forward return of the event bars against the
base rate of the same universe over the same horizons. Two things make the
measurement honest:

* **Excess over the universe base rate.** Comparing against zero would reward a
  hypothesis for firing during a rising market. The base rate is the average
  forward return of *every* evaluable bar, so what remains is selection skill.
* **Discovery window only.** Nothing here touches validation or holdout. A
  hypothesis that looks good is a candidate, not a result.

Reported horizons are deliberately short (1-10 bars), because the objective is a
short-swing strategy and a hypothesis whose edge only appears at 25 bars has
failed the brief regardless of how profitable it is.

Look-ahead is audited structurally rather than argued: ``find_lookahead``
recomputes each feature on a truncated series and compares.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig, get_config
from . import hypotheses as hyp
from .data_audit import audit_panel, find_lookahead
from .protocol import DATA_START, DISCOVERY, Window, describe

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 8, 10)


def load_panel(
    *, cfg: AppConfig | None = None, exclude_unclean: bool = True
) -> tuple[dict[str, pd.DataFrame], dict[str, str], pd.DataFrame]:
    """Load prices, industry map and benchmark for the mid/small-cap universe.

    Args:
        cfg: Config override.
        exclude_unclean: Drop symbols that fail the structural data audit.

    Returns:
        ``(prices, industry_of, benchmark)``.
    """
    from ..data.fetch_ohlcv import fetch_ohlcv
    from ..data.universe import build_universe

    cfg = cfg or get_config()
    universe = build_universe(cfg=cfg)
    if universe.empty:
        raise RuntimeError("Universe unavailable.")

    symbols = universe["symbol"].tolist()
    prices = fetch_ohlcv(symbols, start=DATA_START, cfg=cfg, show_progress=False)
    prices = {s: f for s, f in prices.items() if f is not None and not f.empty}

    benchmark = fetch_ohlcv(["^CRSMID"], start=DATA_START, cfg=cfg, show_progress=False)["^CRSMID"]

    if exclude_unclean:
        audit = audit_panel(prices, pd.DatetimeIndex(benchmark.index))
        bad = set(audit.unclean())
        if bad:
            logger.info("Excluding %d symbols that failed the data audit.", len(bad))
            prices = {s: f for s, f in prices.items() if s not in bad}

    industry_of = dict(zip(universe["symbol"], universe["industry"].astype(str)))
    return prices, industry_of, benchmark


def build_event_panel(
    prices: Mapping[str, pd.DataFrame],
    industry_of: Mapping[str, str],
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    hypotheses: Sequence[hyp.Hypothesis] = hyp.HYPOTHESES,
    min_turnover_cr: float = 2.0,
    carry: Sequence[str] = (),
) -> pd.DataFrame:
    """Compute features, fire every hypothesis, and attach forward returns.

    Entry is the **next** bar's open, matching how the backtester fills, and the
    exit is the close *h* bars later. Excess return subtracts the equal-weighted
    universe move over the identical window, so market direction is removed.

    Args:
        prices: Symbol -> OHLCV.
        industry_of: Symbol -> industry.
        horizons: Forward horizons in bars.
        hypotheses: Hypotheses to evaluate.
        min_turnover_cr: Liquidity floor in INR crore, applied per bar so that
            a stock is only evaluated on days it was genuinely tradeable.

    Returns:
        Long panel: one row per (symbol, bar) with hypothesis flags and forward
        returns.
    """
    sector_returns, sector_gaps, market_return = hyp.build_sector_aggregates(
        prices, industry_of
    )

    chunks: list[pd.DataFrame] = []
    for symbol, ohlcv in prices.items():
        if ohlcv is None or len(ohlcv) < 260:
            continue
        industry = industry_of.get(symbol, "")
        features = hyp.build_features(
            ohlcv,
            sector_return=sector_returns.get(industry),
            market_return=market_return,
            sector_gap=sector_gaps.get(industry),
        )
        if features.empty:
            continue

        entry = features["open"].shift(-1)
        row = pd.DataFrame(index=features.index)
        row["symbol"] = symbol
        row["industry"] = industry
        row["turnover_cr"] = features["turnover"] / 1e7
        row["tradeable"] = (
            (row["turnover_cr"] >= min_turnover_cr)
            & features["atr_pct"].notna()
            & features["vol_ratio"].notna()
            & features["sma_50"].notna()
        )
        for h in horizons:
            row[f"fwd_{h}"] = features["close"].shift(-h) / entry - 1.0

        # Continuous features kept for conditioning studies. These describe the
        # state at t and never reach forward.
        for column in carry:
            row[column] = features[column].to_numpy() if column in features else np.nan

        for hypothesis in hypotheses:
            try:
                row[hypothesis.name] = hypothesis.fn(features).to_numpy()
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("%s failed on %s: %s", hypothesis.name, symbol, exc)
                row[hypothesis.name] = False

        chunks.append(row)

    if not chunks:
        return pd.DataFrame()

    panel = pd.concat(chunks)
    panel.index.name = "date"
    panel = panel.reset_index()

    # Equal-weighted universe forward return per date: the honest "hold the
    # universe" alternative, and the base rate every hypothesis is judged against.
    for h in horizons:
        universe_mean = (
            panel.loc[panel["tradeable"]].groupby("date")[f"fwd_{h}"].transform("mean")
        )
        panel[f"base_{h}"] = np.nan
        panel.loc[panel["tradeable"], f"base_{h}"] = universe_mean
        panel[f"exc_{h}"] = panel[f"fwd_{h}"] - panel[f"base_{h}"]

    return panel


def evaluate(
    panel: pd.DataFrame,
    window: Window,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    hypotheses: Sequence[hyp.Hypothesis] = hyp.HYPOTHESES,
) -> pd.DataFrame:
    """Score every hypothesis inside one window."""
    scoped = panel[panel["tradeable"] & window.mask(panel["date"])]
    if scoped.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        if hypothesis.name not in scoped.columns:
            continue
        events = scoped[scoped[hypothesis.name]]
        record: dict[str, Any] = {
            "name": hypothesis.name,
            "events": int(len(events)),
            "per_year": len(events) / max((window.end - window.start).days / 365.25, 1e-9),
        }
        for h in horizons:
            values = events[f"exc_{h}"].dropna()
            if len(values) < 30:
                record[f"exc_{h}"] = np.nan
                record[f"t_{h}"] = np.nan
                record[f"hit_{h}"] = np.nan
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            record[f"exc_{h}"] = mean * 100
            record[f"t_{h}"] = mean / (sd / np.sqrt(len(values))) if sd > 0 else np.nan
            record[f"hit_{h}"] = float((values > 0).mean()) * 100
        rows.append(record)
    return pd.DataFrame(rows)


def decay_profile(scores: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """Where each hypothesis peaks, which sets its natural holding period."""
    rows = []
    for _, row in scores.iterrows():
        values = {h: row.get(f"exc_{h}") for h in horizons}
        finite = {h: v for h, v in values.items() if pd.notna(v)}
        if not finite:
            continue
        peak = max(finite, key=lambda h: finite[h])
        rows.append(
            {
                "name": row["name"],
                "peak_bar": peak,
                "peak_exc": finite[peak],
                "exc_at_1": values.get(1),
                "exc_at_10": values.get(10),
                # A hypothesis whose edge is still climbing at the last horizon
                # is not a short-swing mechanism, whatever its size.
                "still_climbing": bool(peak == max(horizons)),
            }
        )
    return pd.DataFrame(rows)


def format_scores(
    scores: pd.DataFrame, window: Window, horizons: Sequence[int]
) -> str:
    """Render one window's hypothesis scores."""
    if scores.empty:
        return f"  No events in {window.name}."

    lines = [
        "",
        "=" * 104,
        f"  HYPOTHESIS EVENT STUDY -- {window}",
        "  Excess forward return over the equal-weighted universe, in %",
        "=" * 104,
        f"  {'hypothesis':<24}{'events':>9}{'/yr':>7}" + "".join(f"{f'{h}d':>11}" for h in horizons),
        "  " + "-" * 100,
    ]
    for _, row in scores.iterrows():
        line = f"  {row['name']:<24}{int(row['events']):>9,}{row['per_year']:>7.0f}"
        for h in horizons:
            value, t = row.get(f"exc_{h}"), row.get(f"t_{h}")
            if pd.isna(value):
                line += f"{'—':>11}"
            else:
                marker = "*" if pd.notna(t) and abs(t) >= 3.0 else " "
                line += f"{value:>10.3f}{marker}"
        lines.append(line)
    lines += [
        "  " + "-" * 100,
        "  * marks |t| >= 3.0. Overlapping windows inflate t, so treat it as a ranking device.",
        "=" * 104,
    ]
    return "\n".join(lines)


def audit_lookahead(prices: Mapping[str, pd.DataFrame], sample: int = 5) -> list[dict[str, Any]]:
    """Run the truncation-based look-ahead audit on the feature builder."""
    findings: list[dict[str, Any]] = []
    for symbol, frame in list(prices.items())[:sample]:
        if frame is None or len(frame) < 400:
            continue
        leaks = find_lookahead(lambda f: hyp.build_features(f), frame)
        for leak in leaks:
            findings.append({"symbol": symbol, **leak})
    return findings


def run(
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Run the discovery-window event study and print the report."""
    print(describe())
    prices, industry_of, _benchmark = load_panel(cfg=cfg)
    logger.info("Loaded %d symbols.", len(prices))

    print("\n  LOOK-AHEAD AUDIT (truncation test on the feature builder)")
    leaks = audit_lookahead(prices)
    if leaks:
        print(f"    FAILED: {len(leaks)} leaking values found")
        for leak in leaks[:10]:
            print(f"      {leak['symbol']} {leak['column']} @ {leak['date']}: "
                  f"full={leak['full']} truncated={leak['truncated']}")
        raise RuntimeError("Look-ahead detected; fix before interpreting any result.")
    print("    PASSED: every feature reproduces exactly on truncated history")

    panel = build_event_panel(prices, industry_of, horizons=horizons)
    if panel.empty:
        raise RuntimeError("Event panel is empty.")

    evaluable = int(panel["tradeable"].sum())
    logger.info("Panel: %d rows, %d tradeable bars.", len(panel), evaluable)

    scores = evaluate(panel, DISCOVERY, horizons=horizons)
    print(format_scores(scores, DISCOVERY, horizons))

    decay = decay_profile(scores, horizons)
    print("\n  SIGNAL DECAY -- where each hypothesis peaks (sets its natural hold)")
    print(f"    {'hypothesis':<24}{'peak bar':>10}{'peak exc%':>12}{'1d':>9}{'10d':>9}   shape")
    print("    " + "-" * 76)
    for _, row in decay.iterrows():
        shape = "still climbing at 10d" if row["still_climbing"] else "peaks and decays"
        print(
            f"    {row['name']:<24}{int(row['peak_bar']):>10}{row['peak_exc']:>12.3f}"
            f"{row['exc_at_1']:>9.3f}{row['exc_at_10']:>9.3f}   {shape}"
        )
    print()

    cfg = cfg or get_config()
    out = cfg.paths.results_dir / "event_study_discovery.csv"
    scores.to_csv(out, index=False)
    logger.info("Scores written to %s", out)
    return {"panel": panel, "scores": scores, "decay": decay}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Short-horizon hypothesis event study.")
    parser.add_argument("--horizons", default="1,2,3,5,8,10")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    run(horizons=tuple(int(h) for h in args.horizons.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
