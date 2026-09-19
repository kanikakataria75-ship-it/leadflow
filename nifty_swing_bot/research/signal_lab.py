"""Hypothesis lab: test candidate entry signals against forward returns.

The diagnostics showed the DAB rule set is *anti*-predictive at swing horizons —
it underperforms a random pick from the same universe. This module exists to
find out what, if anything, in this data actually predicts, without repeating the
mistake of believing a single backtest.

**Method.** Every candidate is a boolean rule over a symbol's feature frame. Each
is scored on **excess forward return** — the stock's return from the next open,
minus the benchmark's over the same window — so market direction is removed and
what remains is the rule's own skill. The comparison that matters is always
against the universe base rate, not against zero.

**Guard against fooling ourselves.** Testing many hypotheses on one dataset
manufactures winners: at 20 candidates, roughly one will clear p<0.05 by chance
alone. So the panel is split by date into a **train** period and a **holdout**
period that is never looked at while ranking. Candidates are ranked on train;
the holdout is then reported for all of them, and only a candidate that survives
*both* deserves any further attention. The report prints a Bonferroni-adjusted
t-threshold for the number of hypotheses actually tested.

Even then: overlapping forward windows inflate t-statistics, and a holdout used
repeatedly stops being a holdout. Treat the output as a shortlist for further
work, never as a validated strategy.

Usage::

    python -m nifty_swing_bot.research.signal_lab
    python -m nifty_swing_bot.research.signal_lab --horizon 10 --min-signals 200
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig, get_config
from ..strategy import indicators as ind

logger = logging.getLogger(__name__)

#: A candidate takes a feature frame and returns a boolean entry mask.
CandidateFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One testable entry hypothesis."""

    name: str
    thesis: str
    fn: CandidateFn


def _safe(mask: pd.Series, frame: pd.DataFrame) -> pd.Series:
    """Coerce a candidate's output into a clean boolean mask."""
    return mask.reindex(frame.index).fillna(False).astype(bool)


# --------------------------------------------------------------------------- #
# Candidate signals
#
# The set is chosen to span the hypothesis space rather than to flatter any one
# idea: momentum-continuation (which the diagnostics say fails), mean-reversion,
# and several ways of using delivery data that do NOT pair it with a volume
# surge, since that pairing is mechanically self-defeating.
# --------------------------------------------------------------------------- #
def _dab_original(f: pd.DataFrame) -> pd.Series:
    return _safe(f["signal"], f)


def _deliv_spike_down_day(f: pd.DataFrame) -> pd.Series:
    """Delivery spike on a DOWN day: someone accumulating into weakness.

    This is the sharpest form of the delivery thesis. A high delivery ratio when
    price is falling means buyers are taking stock off sellers and holding it —
    conviction, not chasing. It is the opposite of the original rule, which
    demanded the spike coincide with an up-day volume surge.
    """
    down = f["close"] < f["open"]
    return _safe(f["is_deliv_spike"] & down, f)


def _deliv_spike_quiet(f: pd.DataFrame) -> pd.Series:
    """Delivery spike while price is going nowhere — quiet absorption."""
    ret5 = ind.pct_return(f["close"], 5).abs()
    return _safe(f["is_deliv_spike"] & (ret5 < 0.03), f)


def _deliv_absolute_high(f: pd.DataFrame) -> pd.Series:
    """Very high absolute delivery, regardless of the spike ratio."""
    return _safe(f["deliv_pct"] > 75.0, f)


def _deliv_rising_trend(f: pd.DataFrame) -> pd.Series:
    """5-day average delivery rising well above its 20-day average."""
    short = f["deliv_pct"].rolling(5, min_periods=5).mean()
    long = f["deliv_pct"].shift(5).rolling(20, min_periods=10).mean()
    return _safe(short > long * 1.25, f)


def _pullback_in_uptrend(f: pd.DataFrame) -> pd.Series:
    """Buy weakness inside strength: above the 50-day, but short-term soft."""
    ma50 = ind.sma(f["close"], 50)
    ret3 = ind.pct_return(f["close"], 3)
    return _safe((f["close"] > ma50) & (ret3 < -0.02), f)


def _deliv_pullback(f: pd.DataFrame) -> pd.Series:
    """The delivery thesis applied to pullbacks rather than breakouts.

    Accumulation happened recently, the stock is in an uptrend, and price has
    just pulled back — buying the dip that informed money is defending.
    """
    ma50 = ind.sma(f["close"], 50)
    ret3 = ind.pct_return(f["close"], 3)
    recent_spike = f["is_deliv_spike"].shift(1).rolling(10, min_periods=1).max().astype(bool)
    return _safe(recent_spike & (f["close"] > ma50) & (ret3 < -0.02), f)


def _oversold_rsi(f: pd.DataFrame) -> pd.Series:
    """Classic short-horizon mean reversion."""
    return _safe(ind.rsi(f["close"], 14) < 30.0, f)


def _oversold_in_uptrend(f: pd.DataFrame) -> pd.Series:
    """Oversold on a slow RSI, but only in names still structurally rising.

    Note this fires almost never: RSI(14) below 45 generally requires enough
    sustained selling to drag price under its own 50-day average, so the two
    conditions are close to disjoint. It is kept as the control that shows why
    short-horizon mean reversion needs a *fast* oscillator instead.
    """
    ma50 = ind.sma(f["close"], 50)
    return _safe((ind.rsi(f["close"], 14) < 45.0) & (f["close"] > ma50), f)


def _rsi2_oversold(f: pd.DataFrame) -> pd.Series:
    """RSI(2) below 10 — the standard short-horizon mean-reversion trigger.

    A 2-period RSI reacts within a couple of bars, so it flags a sharp dip
    without requiring the multi-week decline that RSI(14) needs.
    """
    return _safe(ind.rsi(f["close"], 2) < 10.0, f)


def _rsi2_oversold_uptrend(f: pd.DataFrame) -> pd.Series:
    """RSI(2) < 10 while price holds above its 50-day average.

    Buy a sharp dip inside an intact uptrend: the classic setup that the
    diagnostics point towards, and the direct inverse of what DAB was doing.
    """
    ma50 = ind.sma(f["close"], 50)
    return _safe((ind.rsi(f["close"], 2) < 10.0) & (f["close"] > ma50), f)


def _rsi2_deep_uptrend(f: pd.DataFrame) -> pd.Series:
    """A stricter version: RSI(2) < 5, above the 50- and 200-day averages."""
    ma50 = ind.sma(f["close"], 50)
    ma200 = ind.sma(f["close"], 200)
    return _safe(
        (ind.rsi(f["close"], 2) < 5.0) & (f["close"] > ma50) & (f["close"] > ma200), f
    )


def _pullback_plus_rsi2(f: pd.DataFrame) -> pd.Series:
    """Pullback in an uptrend, confirmed by a fast-oscillator extreme."""
    ma50 = ind.sma(f["close"], 50)
    ret3 = ind.pct_return(f["close"], 3)
    return _safe(
        (f["close"] > ma50) & (ret3 < -0.02) & (ind.rsi(f["close"], 2) < 15.0), f
    )


def _gap_down_recovery_uptrend(f: pd.DataFrame) -> pd.Series:
    """Gap-down reversal, filtered to names still above their 50-day average."""
    ma50 = ind.sma(f["close"], 50)
    gap = f["open"] < f["close"].shift(1) * 0.98
    return _safe(gap & (f["range_position"] > 0.6) & (f["close"] > ma50), f)


def _rsi2_with_delivery(f: pd.DataFrame) -> pd.Series:
    """The mean-reversion setup PLUS recent delivery accumulation.

    The decisive test of whether delivery data adds anything at all: if this
    beats the same setup without the delivery filter, the thesis has some life
    in it; if not, delivery is noise in this universe.
    """
    ma50 = ind.sma(f["close"], 50)
    recent_spike = f["is_deliv_spike"].shift(1).rolling(10, min_periods=1).max().astype(bool)
    return _safe(
        (ind.rsi(f["close"], 2) < 10.0) & (f["close"] > ma50) & recent_spike, f
    )


def _weak_close_reversal(f: pd.DataFrame) -> pd.Series:
    """Close in the BOTTOM of the range — the inverse of DAB rule 3."""
    return _safe(f["range_position"] < 0.2, f)


def _volume_dry_up_near_high(f: pd.DataFrame) -> pd.Series:
    """Volume drying up near the highs — classic quiet-consolidation setup."""
    quiet = f["vol_ratio"] < 0.7
    return _safe(quiet & f["is_near_high"] & f["is_contraction"], f)


def _contraction_only(f: pd.DataFrame) -> pd.Series:
    """Volatility contraction alone, without the breakout requirement."""
    return _safe(f["is_contraction"], f)


def _breakout_only(f: pd.DataFrame) -> pd.Series:
    """Plain 20-day breakout, as a momentum reference point."""
    return _safe(f["is_breakout"], f)


def _gap_down_recovery(f: pd.DataFrame) -> pd.Series:
    """Gapped down but closed strong — an intraday reversal."""
    gap = f["open"] < f["close"].shift(1) * 0.98
    return _safe(gap & (f["range_position"] > 0.6), f)


def _rs_laggard(f: pd.DataFrame) -> pd.Series:
    """Underperformers — the inverse of DAB rule 4."""
    return _safe(f["rs_excess"] < -0.05, f)


def _deliv_spike_oversold(f: pd.DataFrame) -> pd.Series:
    """Accumulation meeting short-term capitulation."""
    recent_spike = f["is_deliv_spike"].shift(1).rolling(10, min_periods=1).max().astype(bool)
    return _safe(recent_spike & (ind.rsi(f["close"], 14) < 40.0), f)


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("dab_original", "The shipped DAB rule set (reference)", _dab_original),
    Candidate("breakout_only", "Plain 20d breakout (momentum reference)", _breakout_only),
    Candidate("contraction_only", "ATR(5) < 0.7 x ATR(20), nothing else", _contraction_only),
    Candidate("deliv_spike_down_day", "Delivery spike on a down day", _deliv_spike_down_day),
    Candidate("deliv_spike_quiet", "Delivery spike while price is flat", _deliv_spike_quiet),
    Candidate("deliv_absolute_high", "Absolute delivery% > 75", _deliv_absolute_high),
    Candidate("deliv_rising_trend", "5d avg delivery > 1.25x its 20d avg", _deliv_rising_trend),
    Candidate("deliv_pullback", "Recent accumulation + pullback in uptrend", _deliv_pullback),
    Candidate("deliv_spike_oversold", "Recent accumulation + RSI < 40", _deliv_spike_oversold),
    Candidate("pullback_in_uptrend", "Above 50d MA, 3d return < -2%", _pullback_in_uptrend),
    Candidate("oversold_rsi", "RSI(14) < 30", _oversold_rsi),
    Candidate("oversold_in_uptrend", "RSI(14) < 35 and above 50d MA", _oversold_in_uptrend),
    Candidate("weak_close_reversal", "Close in bottom 20% of range", _weak_close_reversal),
    Candidate("volume_dry_up_near_high", "Quiet volume near highs, contracting", _volume_dry_up_near_high),
    Candidate("gap_down_recovery", "Gap down, closed strong", _gap_down_recovery),
    Candidate("rs_laggard", "10d return trails the index by >5%", _rs_laggard),
    Candidate("rsi2_oversold", "RSI(2) < 10", _rsi2_oversold),
    Candidate("rsi2_oversold_uptrend", "RSI(2) < 10 and above 50d MA", _rsi2_oversold_uptrend),
    Candidate("rsi2_deep_uptrend", "RSI(2) < 5, above 50d and 200d MA", _rsi2_deep_uptrend),
    Candidate("pullback_plus_rsi2", "Pullback in uptrend + RSI(2) < 15", _pullback_plus_rsi2),
    Candidate("gap_down_recovery_uptrend", "Gap-down reversal above 50d MA", _gap_down_recovery_uptrend),
    Candidate("rsi2_with_delivery", "RSI(2) dip in uptrend + recent delivery spike", _rsi2_with_delivery),
)


def evaluate_candidates(
    features: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    *,
    horizon: int = 10,
    split_frac: float = 0.6,
    candidates: Sequence[Candidate] = CANDIDATES,
    min_signals: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every candidate on train and holdout periods.

    Args:
        features: Symbol -> feature frame.
        benchmark: Benchmark OHLCV.
        horizon: Forward horizon in bars.
        split_frac: Fraction of the date range used for training.
        candidates: Candidates to test.
        min_signals: Candidates producing fewer signals than this in either
            period are reported but flagged as under-powered.

    Returns:
        ``(results_frame, meta)``.
    """
    bench = benchmark["close"].copy()
    bench.index = pd.DatetimeIndex(bench.index).tz_localize(None).normalize()
    bench = bench[~bench.index.duplicated(keep="last")].sort_index()

    rows: list[pd.DataFrame] = []
    for symbol, f in features.items():
        if f is None or f.empty or "signal" not in f or len(f) < 80:
            continue
        entry = f["open"].shift(-1)
        b = bench.reindex(f.index).ffill()
        fwd = f["close"].shift(-horizon) / entry - 1.0
        bfwd = b.shift(-horizon) / b.shift(-1) - 1.0

        frame = pd.DataFrame({"excess": fwd - bfwd}, index=f.index)
        frame["symbol"] = symbol
        for cand in candidates:
            try:
                frame[cand.name] = cand.fn(f).to_numpy()
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Candidate %s failed on %s: %s", cand.name, symbol, exc)
                frame[cand.name] = False
        rows.append(frame)

    if not rows:
        return pd.DataFrame(), {}

    panel = pd.concat(rows)
    panel.index.name = "date"
    panel = panel.reset_index().dropna(subset=["excess"])

    dates = np.sort(panel["date"].unique())
    cut = dates[int(len(dates) * split_frac)]
    train = panel[panel["date"] < cut]
    holdout = panel[panel["date"] >= cut]

    base_train = float(train["excess"].mean())
    base_holdout = float(holdout["excess"].mean())

    def score(subset: pd.DataFrame, mask_col: str, base: float) -> dict[str, Any]:
        values = subset.loc[subset[mask_col], "excess"]
        n = int(len(values))
        if n == 0:
            return {"n": 0, "exc": np.nan, "edge": np.nan, "hit": np.nan, "t": np.nan}
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else np.nan
        return {
            "n": n,
            "exc": mean * 100,
            "edge": (mean - base) * 100,
            "hit": float((values > 0).mean()) * 100,
            # t-stat on the EDGE over the base rate, which is what we care about.
            "t": (mean - base) / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan,
        }

    results = []
    for cand in candidates:
        tr = score(train, cand.name, base_train)
        ho = score(holdout, cand.name, base_holdout)
        results.append(
            {
                "name": cand.name,
                "thesis": cand.thesis,
                "train_n": tr["n"], "train_edge": tr["edge"], "train_hit": tr["hit"], "train_t": tr["t"],
                "hold_n": ho["n"], "hold_edge": ho["edge"], "hold_hit": ho["hit"], "hold_t": ho["t"],
                "underpowered": tr["n"] < min_signals or ho["n"] < min_signals,
                "consistent": (
                    tr["edge"] is not np.nan and ho["edge"] is not np.nan
                    and tr["edge"] > 0 and ho["edge"] > 0
                ),
            }
        )

    frame = pd.DataFrame(results).sort_values("train_edge", ascending=False)
    meta = {
        "horizon": horizon,
        "split_date": pd.Timestamp(cut).strftime("%Y-%m-%d"),
        "train_bars": int(len(train)),
        "holdout_bars": int(len(holdout)),
        "base_train_pct": base_train * 100,
        "base_holdout_pct": base_holdout * 100,
        "n_hypotheses": len(candidates),
        # Bonferroni-adjusted two-sided 5% threshold for this many tests.
        "t_threshold": float(abs(_bonferroni_t(len(candidates)))),
    }
    return frame, meta


def _bonferroni_t(n_tests: int, alpha: float = 0.05) -> float:
    """Two-sided normal critical value adjusted for ``n_tests`` hypotheses."""
    from scipy.stats import norm

    return float(norm.ppf(1 - alpha / (2 * max(n_tests, 1))))


def format_report(frame: pd.DataFrame, meta: Mapping[str, Any]) -> str:
    """Render the lab results."""
    if frame.empty:
        return "No candidates could be evaluated."

    lines = [
        "",
        "=" * 96,
        "  SIGNAL HYPOTHESIS LAB",
        f"  {meta['horizon']}-bar horizon · train/holdout split at {meta['split_date']} · "
        f"{meta['n_hypotheses']} hypotheses tested",
        f"  Base rate: train {meta['base_train_pct']:+.3f}%  holdout {meta['base_holdout_pct']:+.3f}%"
        "   (excess return of a random bar)",
        "=" * 96,
        "",
        f"  {'candidate':<26}{'TRAIN':>26}{'HOLDOUT':>26}",
        f"  {'':<26}{'N':>8}{'edge%':>9}{'t':>9}{'N':>8}{'edge%':>9}{'t':>9}   verdict",
        "  " + "-" * 92,
    ]

    threshold = meta["t_threshold"]
    for _, row in frame.iterrows():
        if row["underpowered"]:
            verdict = "under-powered"
        elif row["train_edge"] > 0 and row["hold_edge"] > 0:
            strong = abs(row["hold_t"]) >= threshold if pd.notna(row["hold_t"]) else False
            verdict = "SURVIVES BOTH" + (" (strong)" if strong else "")
        elif row["train_edge"] > 0:
            verdict = "train only — discard"
        else:
            verdict = "no edge"

        def cell(value: float, width: int, decimals: int = 3) -> str:
            return f"{value:>{width}.{decimals}f}" if pd.notna(value) else f"{'—':>{width}}"

        lines.append(
            f"  {row['name']:<26}{int(row['train_n']):>8,}{cell(row['train_edge'], 9)}"
            f"{cell(row['train_t'], 9, 2)}{int(row['hold_n']):>8,}"
            f"{cell(row['hold_edge'], 9)}{cell(row['hold_t'], 9, 2)}   {verdict}"
        )

    lines += [
        "  " + "-" * 92,
        "",
        "  edge% = excess return over the index, MINUS the universe base rate.",
        "          Positive means the rule beats a random pick from the same universe.",
        f"  t     = t-statistic on that edge. Bonferroni threshold for "
        f"{meta['n_hypotheses']} tests: |t| > {threshold:.2f}.",
        "",
        "  CAVEATS, which matter more than the table:",
        "    · Overlapping forward windows inflate t-statistics. Treat them as a",
        "      ranking device, not a significance test.",
        "    · A holdout consulted repeatedly stops being a holdout.",
        "    · Edge measured on bars is not the same as a tradeable strategy: it",
        "      ignores position sizing, capacity, costs and concurrency. The only",
        "      way to know is to run the full backtester on the survivors.",
        "=" * 96,
        "",
    ]
    return "\n".join(lines)


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    limit: int | None = None,
    horizon: int = 10,
    split_frac: float = 0.6,
    min_signals: int = 100,
    cfg: AppConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load data, evaluate every candidate, print the report."""
    from ..backtest.engine import PortfolioBacktester
    from ..backtest.run_backtest import load_market_data
    from ..data.fetch_delivery import cached_delivery_days

    cfg = cfg or get_config()
    available = cached_delivery_days(cfg)
    if not available:
        raise RuntimeError("No delivery data cached. Run the backfill first.")
    start = start or available[min(35, len(available) - 1)]
    end = end or available[-1]

    _, prices, delivery, benchmark = load_market_data(
        start=start, end=end, limit=limit, cfg=cfg
    )
    features = PortfolioBacktester(cfg).compute_all_features(prices, delivery, benchmark)
    frame, meta = evaluate_candidates(
        features, benchmark, horizon=horizon, split_frac=split_frac, min_signals=min_signals
    )
    print(format_report(frame, meta))

    out = cfg.paths.results_dir / "signal_lab.csv"
    frame.to_csv(out, index=False)
    logger.info("Signal lab results written to %s", out)
    return frame, meta


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Test candidate entry signals.")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--split", type=float, default=0.6)
    parser.add_argument("--min-signals", type=int, default=100)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    run(start=args.start, end=args.end, limit=args.limit, horizon=args.horizon,
        split_frac=args.split, min_signals=args.min_signals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
