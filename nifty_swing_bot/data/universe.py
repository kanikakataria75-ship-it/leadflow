"""Construction of the tradeable universe.

The universe is *not* the raw Nifty 500. It is the mid- and small-cap tier:
Nifty 500 constituents with the Nifty 100 removed, then filtered for liquidity.

Rationale for the exclusion: large caps carry a structurally high baseline
delivery percentage because institutions hold them, so a "1.5x the 20-day
average delivery" spike on a stock already sitting at 65-75% delivery says much
less about fresh accumulation than the same spike on a stock that normally
trades at 20-30%. The signal is sharper where the baseline is lower.

Rationale for the liquidity floor: below roughly Rs 5 crore of average daily
turnover, a position sized off a 1% risk budget becomes a meaningful share of
the day's volume, and backtested fills stop being achievable in practice.
"""

from __future__ import annotations

import io
import logging
from typing import Final

import pandas as pd

from ..config import AppConfig, get_config
from .cache import ParquetCache
from .nse_session import get_nse_session

logger = logging.getLogger(__name__)

# NSE serves index constituent CSVs from two hostnames; the newer
# ``nsearchives`` host is tried first and the legacy one kept as a fallback.
_INDEX_URL_TEMPLATES: Final[tuple[str, ...]] = (
    "https://nsearchives.nseindia.com/content/indices/ind_{index}list.csv",
    "https://archives.nseindia.com/content/indices/ind_{index}list.csv",
)

_COLUMN_MAP: Final[dict[str, str]] = {
    "Company Name": "company",
    "Industry": "industry",
    "Symbol": "symbol",
    "Series": "series",
    "ISIN Code": "isin",
}


def _normalise_constituents(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename NSE's CSV columns to our schema and clean up whitespace."""
    frame = raw.rename(columns={k: v for k, v in _COLUMN_MAP.items() if k in raw.columns})
    keep = [c for c in ("symbol", "company", "industry", "isin", "series") if c in frame.columns]
    frame = frame[keep].copy()
    for col in frame.columns:
        frame[col] = frame[col].astype("string").str.strip()
    frame = frame.dropna(subset=["symbol"])
    frame = frame[frame["symbol"] != ""]
    return frame.drop_duplicates(subset="symbol").reset_index(drop=True)


def fetch_index_constituents(
    index: str = "nifty500",
    *,
    refresh: bool = False,
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    """Download an NSE index constituent list.

    Args:
        index: Index slug as it appears in the archive filename, e.g.
            ``nifty500`` for ``ind_nifty500list.csv``.
        refresh: Bypass the cache.
        cfg: Config override, mainly for tests.

    Returns:
        DataFrame with at least a ``symbol`` column, plus ``company``,
        ``industry`` and ``isin`` when NSE provides them. Empty on failure.
    """
    cfg = cfg or get_config()
    cache = ParquetCache(cfg.paths.universe_dir)
    key = f"constituents_{index}"

    def _load() -> pd.DataFrame | None:
        session = get_nse_session()
        for template in _INDEX_URL_TEMPLATES:
            url = template.format(index=index)
            resp = session.get(url, accept="text/csv,*/*")
            if resp is None:
                continue
            try:
                raw = pd.read_csv(io.StringIO(resp.text))
            except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                logger.warning("Could not parse constituents from %s: %s", url, exc)
                continue
            frame = _normalise_constituents(raw)
            if not frame.empty:
                logger.info("Fetched %d constituents for %s", len(frame), index)
                return frame
        logger.error("Failed to fetch constituent list for %s from NSE.", index)
        return None

    # Constituent lists change only on index review, so a weekly TTL is ample.
    frame = cache.get_or_fetch(key, _load, refresh=refresh, ttl_hours=24 * 7)
    if frame is None:
        stale = cache.get(key, ttl_hours=None)
        if stale is not None:
            logger.warning("Using stale cached constituents for %s.", index)
            return stale
        return pd.DataFrame(columns=["symbol", "company", "industry", "isin"])
    return frame


def build_universe(
    *,
    refresh: bool = False,
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    """Build the mid/small-cap tradeable universe before liquidity filtering.

    Returns:
        DataFrame indexed 0..n with ``symbol``, ``company``, ``industry`` and a
        ``yf_ticker`` column ready for yfinance.
    """
    cfg = cfg or get_config()
    up = cfg.universe

    # Preferred path: union of explicit index lists (Midcap 150 + Smallcap 250).
    if up.include_indices:
        frames = []
        for index in up.include_indices:
            part = fetch_index_constituents(index, refresh=refresh, cfg=cfg)
            if part.empty:
                logger.error("Index %s returned no constituents.", index)
                continue
            part = part.copy()
            part["source_index"] = index
            frames.append(part)
        if not frames:
            logger.error("No constituent lists could be loaded for %s.", up.include_indices)
            return pd.DataFrame(columns=["symbol", "company", "industry", "yf_ticker"])
        base = pd.concat(frames, ignore_index=True)
        before = len(base)
        base = base.drop_duplicates(subset="symbol").reset_index(drop=True)
        logger.info(
            "Universe from %s: %d symbols (%d after de-duplication).",
            " + ".join(up.include_indices), before, len(base),
        )
        base["yf_ticker"] = base["symbol"].astype(str) + cfg.data.yf_suffix
        return base

    base = fetch_index_constituents(up.base_index, refresh=refresh, cfg=cfg)
    if base.empty:
        logger.error("Base index %s returned no constituents.", up.base_index)
        return base

    if up.exclude_index:
        excl = fetch_index_constituents(up.exclude_index, refresh=refresh, cfg=cfg)
        if excl.empty:
            # Silently trading the full 500 would change the strategy's character
            # without the user knowing, so this is a loud warning.
            logger.warning(
                "Exclusion index %s unavailable; universe will include large caps.",
                up.exclude_index,
            )
        else:
            before = len(base)
            base = base[~base["symbol"].isin(set(excl["symbol"]))].reset_index(drop=True)
            logger.info(
                "Excluded %d %s constituents; %d mid/small-cap names remain.",
                before - len(base),
                up.exclude_index,
                len(base),
            )

    base = base.copy()
    base["yf_ticker"] = base["symbol"].astype(str) + cfg.data.yf_suffix
    return base.reset_index(drop=True)


def compute_turnover_stats(
    prices: dict[str, pd.DataFrame],
    *,
    lookback: int = 20,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Average daily traded value per symbol, in INR crore.

    Turnover is approximated as ``close * volume`` averaged over ``lookback``
    bars. That is close enough for a liquidity screen; the exact traded value
    from bhavcopy is used where available in the delivery data.

    Args:
        prices: Mapping of symbol to an OHLCV frame with a DatetimeIndex.
        lookback: Number of trailing bars to average.
        as_of: Only consider bars up to this date. None uses all data.

    Returns:
        DataFrame indexed by symbol with ``avg_turnover_cr``, ``last_close``
        and ``bars`` columns.
    """
    rows: list[dict[str, object]] = []
    for symbol, frame in prices.items():
        if frame is None or frame.empty:
            continue
        window = frame if as_of is None else frame.loc[frame.index <= as_of]
        if window.empty:
            continue
        tail = window.tail(lookback)
        turnover = (tail["close"] * tail["volume"]).mean() / 1e7  # paise-free: INR -> crore
        rows.append(
            {
                "symbol": symbol,
                "avg_turnover_cr": float(turnover) if pd.notna(turnover) else 0.0,
                "last_close": float(window["close"].iloc[-1]),
                "bars": int(len(window)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["avg_turnover_cr", "last_close", "bars"])
    return pd.DataFrame(rows).set_index("symbol")


def apply_liquidity_filter(
    universe: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    cfg: AppConfig | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter a universe down to names that are actually tradeable.

    Applies the turnover floor, the price band and the minimum-history rule from
    :class:`~nifty_swing_bot.config.UniverseParams`.

    Returns:
        The filtered universe with the turnover statistics joined on, sorted by
        descending turnover.
    """
    cfg = cfg or get_config()
    up = cfg.universe

    stats = compute_turnover_stats(prices, lookback=up.turnover_lookback_days, as_of=as_of)
    if stats.empty:
        logger.warning("No price data available; liquidity filter returned nothing.")
        return universe.iloc[0:0].copy()

    merged = universe.merge(stats, left_on="symbol", right_index=True, how="inner")
    start = len(merged)

    merged = merged[merged["bars"] >= up.min_history_days]
    merged = merged[merged["avg_turnover_cr"] >= up.min_avg_turnover_cr]
    merged = merged[merged["last_close"] >= up.min_price]
    if up.max_price is not None:
        merged = merged[merged["last_close"] <= up.max_price]

    # Relative liquidity screen, applied after the absolute floors so the
    # quantile is taken over names that already passed them. Slippage is the
    # biggest single cost component and scales with thinness, so this is the
    # cheapest available lever on total trading cost.
    if up.liquidity_quantile is not None and not merged.empty:
        threshold = merged["avg_turnover_cr"].quantile(up.liquidity_quantile)
        before_q = len(merged)
        merged = merged[merged["avg_turnover_cr"] >= threshold]
        logger.info(
            "Liquidity quantile %.2f: %d -> %d symbols (turnover >= Rs %.1f cr).",
            up.liquidity_quantile, before_q, len(merged), threshold,
        )

    logger.info(
        "Liquidity filter: %d -> %d symbols (turnover >= Rs %.1f cr, price >= Rs %.0f).",
        start,
        len(merged),
        up.min_avg_turnover_cr,
        up.min_price,
    )
    return merged.sort_values("avg_turnover_cr", ascending=False).reset_index(drop=True)


__all__ = [
    "apply_liquidity_filter",
    "build_universe",
    "compute_turnover_stats",
    "fetch_index_constituents",
]
