"""Phase 3: does the effect concentrate, and do weak signals combine?

The event study found a real but small mean-reversion effect (~0.2% over three
days) that a naive "the stock fell 5%" filter captures almost as well as any
designed hypothesis. At roughly 0.5% round-trip cost, none of it is tradeable as
stated. Two questions decide whether anything here can be:

1. **Does the effect concentrate?** If the most extreme dislocations revert much
   harder than the median one, then selecting cross-sectionally -- taking only
   the worst N in the universe each day -- produces a far larger edge per trade
   than any absolute threshold, at the cost of fewer trades. A monotone
   relationship between signal strength and forward return is also the single
   best evidence that a mechanism is real rather than fitted: noise does not
   produce monotone deciles.

2. **Do conditions combine?** The information-versus-flow framing predicts
   specific interactions. A large decline on thin volume with peers unaffected
   should revert; the same decline with heavy volume and peers falling too is
   information and should not. Testing the interaction is a direct test of the
   mechanism, not a parameter search.

Everything here runs on the DISCOVERY window only.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig, get_config
from .protocol import DISCOVERY, Window

logger = logging.getLogger(__name__)

#: Continuous features the conditioning study needs carried through the panel.
CARRY: tuple[str, ...] = (
    "ret_1", "ret_3", "ret_5", "ret_3_atr", "ret_1_atr",
    "vol_ratio", "atr_pct", "range_position", "rel_sector_3", "sector_ret_3",
    "gap", "rel_gap", "down_streak", "above_50", "above_200", "atr_ratio",
)


def decile_profile(
    panel: pd.DataFrame,
    column: str,
    window: Window,
    *,
    horizon: int = 3,
    bins: int = 10,
    ascending: bool = True,
) -> pd.DataFrame:
    """Forward excess return by decile of a continuous signal.

    A mechanism that is real should show a *monotone* gradient: the further into
    the tail, the stronger the reversion. A flat profile with one good bucket is
    the signature of noise.

    Args:
        panel: Event panel.
        column: Continuous feature to bin.
        window: Period to evaluate in.
        horizon: Forward horizon in bars.
        bins: Number of quantile buckets.
        ascending: Bucket 1 is the lowest value when True.

    Returns:
        One row per bucket with count, mean excess return, hit rate and t-stat.
    """
    scoped = panel[panel["tradeable"] & window.mask(panel["date"])].copy()
    scoped = scoped.dropna(subset=[column, f"exc_{horizon}"])
    if len(scoped) < bins * 100:
        return pd.DataFrame()

    try:
        scoped["bucket"] = pd.qcut(scoped[column], bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for bucket, group in scoped.groupby("bucket"):
        values = group[f"exc_{horizon}"]
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        rows.append(
            {
                "bucket": int(bucket) + 1,
                "n": int(len(values)),
                "lo": float(group[column].min()),
                "hi": float(group[column].max()),
                "exc_pct": mean * 100,
                "hit_pct": float((values > 0).mean()) * 100,
                "t": mean / (sd / np.sqrt(len(values))) if sd > 0 else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    return frame if ascending else frame.iloc[::-1].reset_index(drop=True)


def monotonicity(frame: pd.DataFrame) -> float:
    """Spearman correlation between bucket index and excess return.

    Near -1 means the effect strengthens steadily into the low tail, which is
    what a genuine mean-reversion mechanism looks like. Near 0 means the
    "signal" is not ordering outcomes at all.
    """
    if frame.empty or len(frame) < 3:
        return float("nan")
    return float(frame["bucket"].corr(frame["exc_pct"], method="spearman"))


def cross_sectional_profile(
    panel: pd.DataFrame,
    column: str,
    window: Window,
    *,
    horizon: int = 3,
    ranks: Sequence[int] = (1, 2, 3, 5, 10, 20, 50),
    ascending: bool = True,
) -> pd.DataFrame:
    """Edge when taking only the top-N ranked names in the universe each day.

    This is the practical version of the concentration question: a strategy can
    only hold so many positions, so what matters is not the average bar below a
    threshold but the *worst few in the whole universe today*.

    Args:
        panel: Event panel.
        column: Feature to rank on.
        window: Period to evaluate.
        horizon: Forward horizon.
        ranks: Cut-offs to report.
        ascending: Rank 1 is the lowest value when True.

    Returns:
        One row per cut-off with events per year and mean excess return.
    """
    scoped = panel[panel["tradeable"] & window.mask(panel["date"])].copy()
    scoped = scoped.dropna(subset=[column, f"exc_{horizon}"])
    if scoped.empty:
        return pd.DataFrame()

    scoped["rank"] = scoped.groupby("date")[column].rank(
        method="first", ascending=ascending
    )
    years = max((window.end - window.start).days / 365.25, 1e-9)

    rows: list[dict[str, Any]] = []
    for cut in ranks:
        subset = scoped[scoped["rank"] <= cut]
        values = subset[f"exc_{horizon}"]
        if len(values) < 50:
            continue
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        rows.append(
            {
                "top_n": cut,
                "events": int(len(values)),
                "per_year": len(values) / years,
                "exc_pct": mean * 100,
                "hit_pct": float((values > 0).mean()) * 100,
                "t": mean / (sd / np.sqrt(len(values))) if sd > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def interaction_grid(
    panel: pd.DataFrame,
    row_col: str,
    col_col: str,
    window: Window,
    *,
    horizon: int = 3,
    bins: int = 4,
) -> pd.DataFrame:
    """Mean excess return across a two-way split of the universe.

    Tests the information-versus-flow prediction directly: the same price move
    should revert when participation is thin and persist when it is heavy.

    Returns:
        A grid with ``row_col`` buckets as rows and ``col_col`` buckets as
        columns, holding mean excess return in percent.
    """
    scoped = panel[panel["tradeable"] & window.mask(panel["date"])].copy()
    scoped = scoped.dropna(subset=[row_col, col_col, f"exc_{horizon}"])
    if len(scoped) < bins * bins * 100:
        return pd.DataFrame()

    try:
        scoped["r"] = pd.qcut(scoped[row_col], bins, labels=False, duplicates="drop")
        scoped["c"] = pd.qcut(scoped[col_col], bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    grid = (
        scoped.groupby(["r", "c"])[f"exc_{horizon}"].mean().unstack() * 100
    )
    counts = scoped.groupby(["r", "c"]).size().unstack()
    grid.index = [f"{row_col} q{int(i) + 1}" for i in grid.index]
    grid.columns = [f"{col_col} q{int(c) + 1}" for c in grid.columns]
    grid.attrs["counts"] = counts
    return grid


def format_deciles(frame: pd.DataFrame, column: str, horizon: int) -> str:
    """Render a decile profile."""
    if frame.empty:
        return f"    {column}: insufficient data"
    lines = [
        f"    {column}  ({horizon}-bar forward excess, %)",
        f"      {'bucket':>7}{'n':>9}{'range':>26}{'exc%':>9}{'hit%':>8}{'t':>7}",
    ]
    for _, row in frame.iterrows():
        span = f"[{row['lo']:.3f} .. {row['hi']:.3f}]"
        lines.append(
            f"      {int(row['bucket']):>7}{int(row['n']):>9,}{span:>26}"
            f"{row['exc_pct']:>9.3f}{row['hit_pct']:>8.1f}{row['t']:>7.2f}"
        )
    rho = monotonicity(frame)
    lines.append(f"      monotonicity (Spearman bucket vs excess): {rho:+.2f}")
    return "\n".join(lines)


def run(
    *,
    horizon: int = 3,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Run the conditioning study on the discovery window."""
    from .event_study import build_event_panel, load_panel

    cfg = cfg or get_config()
    prices, industry_of, _ = load_panel(cfg=cfg)
    panel = build_event_panel(prices, industry_of, carry=CARRY)
    if panel.empty:
        raise RuntimeError("Event panel empty.")

    print("\n" + "=" * 96)
    print(f"  CONDITIONING STUDY -- {DISCOVERY}   horizon = {horizon} bars")
    print("=" * 96)

    print("\n  1. DOES THE EFFECT CONCENTRATE? (decile profiles)")
    profiles: dict[str, pd.DataFrame] = {}
    for column in ("ret_3_atr", "rel_sector_3", "ret_3", "vol_ratio"):
        frame = decile_profile(panel, column, DISCOVERY, horizon=horizon)
        profiles[column] = frame
        print()
        print(format_deciles(frame, column, horizon))

    print("\n  2. CROSS-SECTIONAL SELECTION (worst N in the universe each day)")
    for column, label in (
        ("ret_3_atr", "volatility-scaled 3-day return"),
        ("rel_sector_3", "3-day return relative to sector"),
    ):
        frame = cross_sectional_profile(panel, column, DISCOVERY, horizon=horizon)
        if frame.empty:
            continue
        print(f"\n    rank by {label}, ascending (worst first)")
        print(f"      {'top N':>7}{'events':>9}{'/yr':>8}{'exc%':>9}{'hit%':>8}{'t':>7}")
        for _, row in frame.iterrows():
            print(
                f"      {int(row['top_n']):>7}{int(row['events']):>9,}{row['per_year']:>8.0f}"
                f"{row['exc_pct']:>9.3f}{row['hit_pct']:>8.1f}{row['t']:>7.2f}"
            )

    print("\n  3. INFORMATION vs FLOW (does participation change the sign?)")
    grid = interaction_grid(panel, "ret_3_atr", "vol_ratio", DISCOVERY, horizon=horizon)
    if not grid.empty:
        print("     rows = size of the 3-day move (q1 = biggest decline)")
        print("     cols = volume vs its 20-day average (q1 = thinnest)")
        print()
        print(grid.round(3).to_string())
        print()
        print("     Prediction under the flow hypothesis: the biggest declines on the")
        print("     THINNEST volume (top-left) revert hardest; the same declines on heavy")
        print("     volume (top-right) are information and revert least.")

    grid2 = interaction_grid(panel, "ret_3_atr", "rel_sector_3", DISCOVERY, horizon=horizon)
    if not grid2.empty:
        print("\n     rows = size of the 3-day move; cols = move relative to sector")
        print()
        print(grid2.round(3).to_string())

    print("\n" + "=" * 96 + "\n")
    return {"panel": panel, "profiles": profiles}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Conditioning and concentration study.")
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    run(horizon=args.horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
