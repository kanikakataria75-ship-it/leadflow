"""NSE delivery-percentage data from the full security bhavcopy.

Delivery percentage -- deliverable quantity divided by total traded quantity --
is the signal this whole strategy is built around. It separates real
accumulation (shares actually moving into demat accounts) from intraday churn
(the same shares changing hands repeatedly and squaring off by the close). It
is available only from NSE, in the daily ``sec_bhavdata_full`` file, one CSV per
trading day.

Design notes:

* **One cached parquet per trading day.** Historical bhavcopy files never
  change, so entries never expire. This also makes the multi-year backfill
  resumable: an interrupted or blocked run loses only the day in flight.
* **Holidays are cached as empty frames.** NSE returns 404 for non-trading days.
  Without caching that fact, every subsequent run would re-request the same
  missing files and get throttled for its trouble.
* **Only EQ/BE series are kept**, and only the columns the strategy needs, which
  keeps the on-disk footprint to a few MB per year instead of hundreds.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable, Sequence
from datetime import date, timedelta

import pandas as pd

from ..config import AppConfig, get_config
from .cache import ParquetCache
from .nse_session import get_nse_session

logger = logging.getLogger(__name__)

#: Series worth trading. EQ is the rolling-settlement mainstream; BE is the
#: trade-to-trade segment, kept so the data is complete but usually filtered out
#: upstream because it is delivery-only by construction and distorts the signal.
TRADEABLE_SERIES: frozenset[str] = frozenset({"EQ", "BE"})

DELIVERY_COLUMNS: tuple[str, ...] = (
    "symbol",
    "series",
    "prev_close",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "total_qty",
    "turnover_cr",
    "trades",
    "deliv_qty",
    "deliv_pct",
)

_BHAVCOPY_URL_TEMPLATES: tuple[str, ...] = (
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv",
    "https://archives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv",
)

_RAW_COLUMN_MAP: dict[str, str] = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "DATE1": "date",
    "PREV_CLOSE": "prev_close",
    "OPEN_PRICE": "open",
    "HIGH_PRICE": "high",
    "LOW_PRICE": "low",
    "LAST_PRICE": "last",
    "CLOSE_PRICE": "close",
    "AVG_PRICE": "vwap",
    "TTL_TRD_QNTY": "total_qty",
    "TURNOVER_LACS": "turnover_lacs",
    "NO_OF_TRADES": "trades",
    "DELIV_QTY": "deliv_qty",
    "DELIV_PER": "deliv_pct",
}


def _parse_bhavcopy(text: str, day: date) -> pd.DataFrame:
    """Parse one ``sec_bhavdata_full`` CSV into our normalised schema.

    NSE ships this file with leading spaces in every header and in most cell
    values, and uses ``-`` as the null marker for the delivery columns on series
    where delivery is not reported. Both quirks are handled here.
    """
    raw = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    raw.columns = [str(c).strip().upper() for c in raw.columns]

    missing = {"SYMBOL", "SERIES", "DELIV_PER", "TTL_TRD_QNTY"} - set(raw.columns)
    if missing:
        raise ValueError(f"bhavcopy for {day} missing expected columns: {sorted(missing)}")

    frame = raw.rename(columns=_RAW_COLUMN_MAP)
    frame["symbol"] = frame["symbol"].astype("string").str.strip()
    frame["series"] = frame["series"].astype("string").str.strip()
    frame = frame[frame["series"].isin(TRADEABLE_SERIES)].copy()

    numeric = [
        "prev_close", "open", "high", "low", "close", "vwap",
        "total_qty", "turnover_lacs", "trades", "deliv_qty", "deliv_pct",
    ]
    for col in numeric:
        if col in frame.columns:
            # Strip whitespace, then coerce; '-' and '' both become NaN.
            frame[col] = pd.to_numeric(
                frame[col].astype("string").str.strip().replace({"-": None, "": None}),
                errors="coerce",
            )
        else:
            frame[col] = pd.NA

    # NSE reports turnover in lakhs; crore is the unit used everywhere else here.
    frame["turnover_cr"] = frame["turnover_lacs"] / 100.0

    frame["date"] = pd.Timestamp(day)
    frame = frame[["date", *DELIVERY_COLUMNS]]
    frame = frame.dropna(subset=["symbol"])
    frame = frame[frame["symbol"] != ""]
    return frame.drop_duplicates(subset=["symbol", "series"]).reset_index(drop=True)


def fetch_bhavcopy_day(
    day: date,
    *,
    refresh: bool = False,
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    """Fetch (or load from cache) the full bhavcopy for a single trading day.

    Args:
        day: Calendar date to fetch.
        refresh: Re-download even if cached.
        cfg: Config override.

    Returns:
        A normalised delivery frame. An **empty** frame means the day is a
        holiday or the file is not yet published -- both are cached so we do not
        keep asking.
    """
    cfg = cfg or get_config()
    cache = ParquetCache(cfg.paths.delivery_dir, namespace="daily")
    key = day.strftime("%Y-%m-%d")

    def _load() -> pd.DataFrame | None:
        session = get_nse_session()
        ddmmyyyy = day.strftime("%d%m%Y")
        for template in _BHAVCOPY_URL_TEMPLATES:
            url = template.format(ddmmyyyy=ddmmyyyy)
            resp = session.get(
                url,
                accept="text/csv,application/csv,*/*",
                referer=f"{cfg.data.nse_base_url}/all-reports",
            )
            if resp is None:
                continue
            try:
                frame = _parse_bhavcopy(resp.text, day)
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                logger.warning("Bad bhavcopy payload for %s from %s: %s", key, url, exc)
                continue
            logger.debug("Bhavcopy %s: %d rows", key, len(frame))
            return frame

        # Every candidate URL 404'd or failed: treat as a non-trading day.
        logger.debug("No bhavcopy for %s (holiday or not yet published).", key)
        return pd.DataFrame(columns=["date", *DELIVERY_COLUMNS])

    # Historical files are immutable, so cached days never expire. Today's file
    # is the exception: it may have been cached as empty before publication.
    ttl: float | None = None
    if day >= date.today() - timedelta(days=1):
        cached = cache.get(key, ttl_hours=None)
        if cached is not None and cached.empty and not refresh:
            ttl = 3.0  # retry a recent empty day every few hours

    frame = cache.get_or_fetch(key, _load, refresh=refresh, ttl_hours=ttl, store_empty=True)
    return frame if frame is not None else pd.DataFrame(columns=["date", *DELIVERY_COLUMNS])


def fetch_delivery_history(
    start: date,
    end: date,
    *,
    symbols: Iterable[str] | None = None,
    refresh: bool = False,
    cfg: AppConfig | None = None,
    skip_weekends: bool = True,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Backfill delivery data over a date range, one cached day at a time.

    This is the resumable bulk loader. Interrupting it is safe: every completed
    day is already on disk, so a re-run picks up where it left off.

    Args:
        start: Inclusive first date.
        end: Inclusive last date.
        symbols: Restrict the returned rows to these symbols. The cache still
            stores the full bhavcopy, so widening the universe later needs no
            re-download.
        refresh: Re-download days already cached.
        cfg: Config override.
        skip_weekends: Do not even attempt Saturdays and Sundays.
        progress_every: Log a progress line every N days.

    Returns:
        Long-format frame: one row per symbol per day.
    """
    cfg = cfg or get_config()
    wanted = set(symbols) if symbols is not None else None

    days: list[date] = []
    cursor = start
    while cursor <= end:
        if not (skip_weekends and cursor.weekday() >= 5):
            days.append(cursor)
        cursor += timedelta(days=1)

    logger.info("Delivery backfill: %d candidate days (%s to %s).", len(days), start, end)

    chunks: list[pd.DataFrame] = []
    trading_days = 0
    for i, day in enumerate(days, start=1):
        frame = fetch_bhavcopy_day(day, refresh=refresh, cfg=cfg)
        if not frame.empty:
            trading_days += 1
            if wanted is not None:
                frame = frame[frame["symbol"].isin(wanted)]
            if not frame.empty:
                chunks.append(frame)
        if progress_every and i % progress_every == 0:
            logger.info("  ...%d/%d days processed (%d trading days).", i, len(days), trading_days)

    if not chunks:
        logger.warning("Delivery backfill produced no rows for %s to %s.", start, end)
        return pd.DataFrame(columns=["date", *DELIVERY_COLUMNS])

    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(
        "Delivery backfill complete: %d rows, %d symbols, %d trading days.",
        len(out),
        out["symbol"].nunique(),
        trading_days,
    )
    return out


def to_symbol_panel(
    history: pd.DataFrame,
    *,
    series: Sequence[str] = ("EQ",),
) -> dict[str, pd.DataFrame]:
    """Pivot long-format delivery history into per-symbol frames.

    Args:
        history: Output of :func:`fetch_delivery_history`.
        series: Which NSE series to keep. Defaults to EQ only, because BE
            (trade-to-trade) is 100% delivery by construction and would fire the
            delivery-spike rule meaninglessly.

    Returns:
        Mapping of symbol to a DatetimeIndexed frame with ``deliv_pct``,
        ``deliv_qty``, ``total_qty``, ``turnover_cr`` and ``trades``.
    """
    if history.empty:
        return {}

    frame = history
    if series:
        frame = frame[frame["series"].isin(set(series))]
    if frame.empty:
        return {}

    keep = ["deliv_pct", "deliv_qty", "total_qty", "turnover_cr", "trades"]
    panel: dict[str, pd.DataFrame] = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        sub = group.set_index(pd.DatetimeIndex(group["date"]))[keep].copy()
        sub.index.name = "date"
        sub = sub[~sub.index.duplicated(keep="last")].sort_index()
        panel[str(symbol)] = sub
    return panel


def load_delivery_panel(
    symbols: Iterable[str],
    start: date,
    end: date,
    *,
    refresh: bool = False,
    cfg: AppConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper: backfill and pivot in one call."""
    symbols = list(symbols)
    history = fetch_delivery_history(
        start, end, symbols=symbols, refresh=refresh, cfg=cfg
    )
    return to_symbol_panel(history)


def cached_delivery_days(cfg: AppConfig | None = None) -> list[date]:
    """Which days already have a cached bhavcopy, trading days only.

    Useful for reporting backfill progress without hitting the network.
    """
    cfg = cfg or get_config()
    cache = ParquetCache(cfg.paths.delivery_dir, namespace="daily")
    out: list[date] = []
    for key in cache.keys():
        try:
            day = date.fromisoformat(key)
        except ValueError:
            continue
        frame = cache.get(key, ttl_hours=None)
        if frame is not None and not frame.empty:
            out.append(day)
    return sorted(out)


__all__ = [
    "DELIVERY_COLUMNS",
    "TRADEABLE_SERIES",
    "cached_delivery_days",
    "fetch_bhavcopy_day",
    "fetch_delivery_history",
    "load_delivery_panel",
    "to_symbol_panel",
]
