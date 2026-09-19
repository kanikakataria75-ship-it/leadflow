"""Sector-Relative Dislocation Reversion (SRDR).

**The behaviour.** When a stock falls sharply while its industry peers do not,
the move is far more likely to be *someone selling* than *the market repricing
the business*. Genuine news about a company's economics -- a demand shock, an
input-cost move, a regulatory change -- almost always moves its peers as well,
because they share those economics. A move that leaves peers untouched is an
inventory event: a fund meeting redemptions, a margin call being closed out, a
pledged block being sold. The seller trades on a deadline, not on a view, and
pays whatever the book demands. Once they are done, price recovers.

**Why it should exist here specifically.** Indian mid- and small-caps have thin,
episodic order books and a leveraged retail base. A seller with size and a
deadline moves price several percent without any information changing hands, and
there is not enough dedicated liquidity provision to absorb it immediately. The
compensation for stepping in is the edge this strategy harvests. It is a
liquidity-provision premium, which is why it survives in exactly the tier where
liquidity is scarcest -- and why filtering to the *most* liquid names destroys
it, as was measured.

**The decisive evidence.** Ranking the universe each day by 3-day return minus
the sector median and buying the most dislocated name returns +0.405% over three
bars in excess of the equal-weighted universe. Ranking by *absolute* weakness
instead returns **minus** 0.193%: the biggest absolute decliner on any given day
usually does have real news. Same direction of price, opposite outcome, and the
only difference is whether the peer group moved too. That contrast is the whole
thesis, and it is what makes this different from a generic oversold-bounce rule:
the signal is not that a stock fell, it is that it fell *alone*.

Decile profiles support it further. Sorting on sector-relative return produces a
monotone gradient from +0.262% to -0.354% with a Spearman rank correlation of
-0.96 -- noise does not order outcomes like that -- whereas sorting on raw return
manages only -0.58.

**The filters** each remove a way the "alone" inference can be wrong, and every
one is a prediction of the thesis rather than a tuned parameter:

* an overnight gap the sector did not share is stock-specific news, and was
  measured to keep falling (-0.407%, t > 3), so it is excluded;
* a decline beyond 3 ATRs is a repricing rather than an inventory shock;
* the peer comparison only carries information when the peer group is stable;
* below the 200-day average, "cheap relative to peers" may simply be a failing
  business rather than a dislocated one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..config import AppConfig, SRDRParams, get_config
from ..research import hypotheses as hyp

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume",
    "ret_3", "sector_ret_3", "rel_sector_3", "ret_3_atr", "atr_pct",
    "gap", "rel_gap", "above_200", "turnover_cr", "vol_ratio",
    "eligible", "dislocation", "rank_score", "atr_stop", "signal",
)


def build_signals(
    prices: Mapping[str, pd.DataFrame],
    industry_of: Mapping[str, str],
    *,
    params: SRDRParams | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build SRDR entry signals for every symbol in the universe.

    This is a cross-sectional strategy, so signals cannot be computed one symbol
    at a time: the rank of a stock's dislocation depends on every other stock's
    dislocation on the same date. The panel is therefore assembled first, ranked
    per date, and then split back out per symbol for the portfolio engine.

    **Look-ahead discipline.** Sector aggregates are medians across peers of
    quantities computed from bars up to and including *t*; the daily rank uses
    only that date's cross-section. Nothing reads a later bar. The one
    acknowledged compromise is that industry membership comes from the current
    constituent list rather than a point-in-time one.

    Args:
        prices: Symbol -> OHLCV.
        industry_of: Symbol -> industry label.
        params: Strategy parameters. Defaults to config.
        cfg: Config override.

    Returns:
        Symbol -> feature frame carrying a boolean ``signal`` and a
        ``rank_score`` for the engine's conviction ordering.
    """
    cfg = cfg or get_config()
    params = params or cfg.srdr

    sector_returns, sector_gaps, market_return = hyp.build_sector_aggregates(
        prices, industry_of, lookback=params.dislocation_lookback
    )

    from ..strategy import indicators as ind

    per_symbol: dict[str, pd.DataFrame] = {}
    rows: list[pd.DataFrame] = []

    for symbol, ohlcv in prices.items():
        if ohlcv is None or len(ohlcv) < params.trend_ma + 20:
            continue
        industry = industry_of.get(symbol, "")
        f = hyp.build_features(
            ohlcv,
            sector_return=sector_returns.get(industry),
            market_return=market_return,
            sector_gap=sector_gaps.get(industry),
        )
        if f.empty:
            continue

        f["turnover_cr"] = f["turnover"] / 1e7
        f["above_trend"] = (f["close"] > ind.sma(f["close"], params.trend_ma)).fillna(False)
        f["atr_stop"] = f["atr_14"]

        # -- Eligibility: every filter is a way the "it fell alone" inference
        #    could be wrong, removed.
        orphan_gap = (f["gap"] < -params.gap_threshold) & (
            f["rel_gap"] < -params.orphan_gap_threshold
        )
        f["eligible"] = (
            (f["turnover_cr"] >= params.min_turnover_cr)
            & f["above_trend"]
            & (f["sector_ret_3"].abs() < params.sector_calm_threshold)
            & (f["ret_3_atr"] > -params.collapse_atr)
            & ~orphan_gap.fillna(False)
            & f["rel_sector_3"].notna()
            & (f["rel_sector_3"] < -params.min_dislocation)
        ).fillna(False)

        # Dislocation: how far below its peers the stock has fallen. More
        # negative is more dislocated.
        f["dislocation"] = f["rel_sector_3"]

        per_symbol[symbol] = f
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "dislocation": f["dislocation"],
                "eligible": f["eligible"],
            },
            index=f.index,
        )
        rows.append(frame)

    if not rows:
        return {}

    panel = pd.concat(rows)
    panel.index.name = "date"
    panel = panel.reset_index()

    # Cross-sectional rank per date over the eligible set only, so that names
    # which would never be traded do not dilute the ranking.
    eligible = panel[panel["eligible"] & panel["dislocation"].notna()].copy()
    eligible["rank"] = eligible.groupby("date")["dislocation"].rank(
        method="first", ascending=True
    )
    chosen = eligible[eligible["rank"] <= params.top_n]
    selected: dict[str, set[pd.Timestamp]] = {
        symbol: set(group["date"])
        for symbol, group in chosen.groupby("symbol", sort=False)
    }

    logger.info(
        "SRDR: %d symbols, %d eligible bars, %d selected signals.",
        len(per_symbol), int(len(eligible)), int(len(chosen)),
    )

    out: dict[str, pd.DataFrame] = {}
    for symbol, f in per_symbol.items():
        dates = selected.get(symbol, set())
        f["signal"] = f.index.isin(dates)
        # Conviction ordering for days when more names qualify than there are
        # slots: the more dislocated, the stronger. Negated so larger is better.
        f["rank_score"] = -f["dislocation"] * 100.0
        for column in FEATURE_COLUMNS:
            if column not in f.columns:
                f[column] = np.nan
        out[symbol] = f
    return out


__all__ = ["FEATURE_COLUMNS", "build_signals"]
