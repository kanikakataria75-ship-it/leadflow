"""OHLCV retrieval from Yahoo Finance, with a local parquet cache.

Prices are fetched with ``auto_adjust=True`` so that splits and bonuses -- which
are frequent in the Indian mid-cap tier -- do not manufacture fake gaps that the
breakout and ATR logic would misread. Volume is split-adjusted by yfinance to
match. Delivery percentage is a ratio and is therefore unaffected by adjustment,
so the two data sources stay consistent when joined.

The cache stores one parquet per symbol covering its full history. Subsequent
runs fetch only the missing tail rather than re-downloading years of data.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import pandas as pd

from ..config import AppConfig, get_config
from .cache import ParquetCache

logger = logging.getLogger(__name__)

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def _empty_ohlcv() -> pd.DataFrame:
    """An empty OHLCV frame that still carries a DatetimeIndex.

    A bare ``pd.DataFrame(columns=...)`` gets a RangeIndex, and comparing that
    against a Timestamp raises. Delisted tickers hit this path routinely, so the
    empty case must be shaped exactly like the populated one.
    """
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS},
        index=pd.DatetimeIndex([], name="date"),
    )


def _to_timestamp(value: str | date | datetime | pd.Timestamp | None) -> pd.Timestamp | None:
    """Coerce the various date inputs we accept into a tz-naive Timestamp."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _normalise_yf_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance output into a tidy lowercase OHLCV frame.

    yfinance returns a MultiIndex column frame when several tickers are
    requested, and a flat one for a single ticker, so both shapes are handled.
    """
    if raw is None or raw.empty:
        return _empty_ohlcv()

    frame = raw
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = set(frame.columns.get_level_values(0))
        if ticker in level0:
            frame = frame.xs(ticker, axis=1, level=0)
        else:
            level1 = set(frame.columns.get_level_values(1))
            if ticker in level1:
                frame = frame.xs(ticker, axis=1, level=1)
            else:
                frame = frame.droplevel(1, axis=1)

    frame = frame.rename(columns={str(c): str(c).lower().replace(" ", "_") for c in frame.columns})
    if "adj_close" in frame.columns and "close" not in frame.columns:
        frame = frame.rename(columns={"adj_close": "close"})

    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        logger.debug("%s missing columns %s from yfinance.", ticker, missing)
        return _empty_ohlcv()

    frame = frame[list(OHLCV_COLUMNS)].copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    frame.index.name = "date"

    for col in OHLCV_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Rows with no price are holidays or bad ticks; a zero-volume row is a
    # non-trading day for that scrip and must not feed the volume average.
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame[frame["close"] > 0]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _fetch_batch(
    tickers: Sequence[str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    """Download a batch of tickers and split the result per ticker."""
    import yfinance as yf

    if not tickers:
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                list(tickers),
                start=start.date() if start is not None else None,
                end=(end + timedelta(days=1)).date() if end is not None else None,
                interval="1d",
                auto_adjust=True,
                actions=False,
                progress=False,
                group_by="ticker",
                threads=False,
            )
    except Exception as exc:  # yfinance raises a wide variety of network errors
        logger.warning("yfinance batch download failed for %d tickers: %s", len(tickers), exc)
        return {}

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            out[ticker] = _normalise_yf_frame(raw, ticker)
        except (KeyError, ValueError) as exc:
            logger.debug("Could not extract %s from batch: %s", ticker, exc)
            out[ticker] = _empty_ohlcv()
    return out


def fetch_ohlcv(
    symbols: Iterable[str],
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    refresh: bool = False,
    cfg: AppConfig | None = None,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for NSE symbols, using and updating the local cache.

    Args:
        symbols: Bare NSE symbols (``RELIANCE``), not yfinance tickers. Tickers
            already carrying the ``.NS`` suffix, and index tickers starting with
            ``^``, are passed through untouched.
        start: First date required. Defaults to the configured backtest start.
        end: Last date required. Defaults to today.
        refresh: Ignore the cache and re-download in full.
        cfg: Config override, mainly for tests.
        show_progress: Log progress every batch.

    Returns:
        Mapping of the input symbol to a DatetimeIndexed frame with columns
        ``open, high, low, close, volume``. Symbols that returned nothing are
        present with an empty frame so callers can distinguish "no data" from
        "not requested".
    """
    cfg = cfg or get_config()
    cache = ParquetCache(cfg.paths.ohlcv_dir)

    start_ts = _to_timestamp(start) or _to_timestamp(cfg.backtest.start_date)
    end_ts = _to_timestamp(end) or _to_timestamp(cfg.backtest.end_date) or pd.Timestamp.today().normalize()

    symbols = list(dict.fromkeys(symbols))  # de-duplicate, preserve order
    results: dict[str, pd.DataFrame] = {}
    to_fetch: dict[str, tuple[str, pd.Timestamp | None]] = {}

    for symbol in symbols:
        ticker = to_yf_ticker(symbol, cfg)
        if refresh:
            to_fetch[symbol] = (ticker, start_ts)
            continue

        cached = cache.get(symbol, ttl_hours=None)
        if cached is None or cached.empty:
            to_fetch[symbol] = (ticker, start_ts)
            continue

        covers_start = cached.index.min() <= start_ts
        last = cached.index.max()
        # Anything older than the previous session needs a top-up. Weekends and
        # holidays are absorbed by the freshness TTL rather than by guessing the
        # exchange calendar here.
        stale = not cache.is_fresh(symbol, cfg.data.cache_ttl_hours)
        if covers_start and not stale:
            results[symbol] = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)]
            continue
        if covers_start and last < end_ts:
            to_fetch[symbol] = (ticker, last - timedelta(days=5))  # small overlap
        else:
            to_fetch[symbol] = (ticker, start_ts)

    if to_fetch:
        logger.info(
            "Fetching OHLCV for %d symbols (%d served from cache).",
            len(to_fetch),
            len(results),
        )
        items = list(to_fetch.items())
        batch_size = cfg.data.yf_batch_size
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        def _run(batch: list[tuple[str, tuple[str, pd.Timestamp | None]]]) -> dict[str, pd.DataFrame]:
            # All symbols in a batch share the earliest required start so a
            # single request covers everyone.
            batch_start = min((s for _, (_, s) in batch if s is not None), default=start_ts)
            tickers = [t for _, (t, _) in batch]
            fetched = _fetch_batch(tickers, batch_start, end_ts)
            return {sym: fetched.get(tick, _empty_ohlcv()) for sym, (tick, _) in batch}

        with ThreadPoolExecutor(max_workers=cfg.data.yf_max_workers) as pool:
            futures = {pool.submit(_run, b): i for i, b in enumerate(batches)}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    fetched = future.result()
                except Exception as exc:
                    logger.warning("OHLCV batch failed: %s", exc)
                    continue
                for symbol, frame in fetched.items():
                    merged = _merge_with_cache(cache, symbol, frame, refresh=refresh)
                    results[symbol] = merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)]
                if show_progress:
                    logger.info("OHLCV batch %d/%d complete.", done, len(batches))

    for symbol in symbols:
        results.setdefault(symbol, _empty_ohlcv())
    return results


def _merge_with_cache(
    cache: ParquetCache,
    symbol: str,
    fresh: pd.DataFrame,
    *,
    refresh: bool,
) -> pd.DataFrame:
    """Combine newly downloaded rows with whatever the cache already holds."""
    existing = None if refresh else cache.get(symbol, ttl_hours=None)
    if existing is None or existing.empty:
        combined = fresh
    elif fresh is None or fresh.empty:
        combined = existing
    else:
        # Fresh rows win on overlap: they carry the latest split adjustments.
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    if combined is not None and not combined.empty:
        cache.put(symbol, combined)
        return combined
    return _empty_ohlcv()


def to_yf_ticker(symbol: str, cfg: AppConfig | None = None) -> str:
    """Map a bare NSE symbol to its yfinance ticker.

    Index tickers (``^CRSLDX``) and already-suffixed tickers pass through.
    """
    cfg = cfg or get_config()
    if symbol.startswith("^") or symbol.endswith(cfg.data.yf_suffix):
        return symbol
    # NSE uses '&' and '-' in some symbols; yfinance expects them as-is.
    return f"{symbol}{cfg.data.yf_suffix}"


def fetch_benchmark(
    ticker: str | None = None,
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    refresh: bool = False,
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    """Fetch the benchmark index series used for the relative-strength rule.

    Falls back through a list of candidate tickers because Yahoo's coverage of
    Indian index symbols is inconsistent: ``^CRSLDX`` (Nifty 500) is the primary,
    with the Nifty 50 as a last resort so the pipeline never hard-fails.
    """
    cfg = cfg or get_config()
    primary = ticker or cfg.backtest.benchmark
    candidates = [primary, "^CRSLDX", "^CNX500", "^NSEI"]
    seen: set[str] = set()

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        frames = fetch_ohlcv(
            [candidate], start=start, end=end, refresh=refresh, cfg=cfg, show_progress=False
        )
        frame = frames.get(candidate)
        if frame is not None and not frame.empty:
            if candidate != primary:
                logger.warning("Benchmark %s unavailable; using %s instead.", primary, candidate)
            return frame

    logger.error("No benchmark data available from any candidate ticker.")
    return _empty_ohlcv()


__all__ = [
    "OHLCV_COLUMNS",
    "fetch_benchmark",
    "fetch_ohlcv",
    "to_yf_ticker",
]
