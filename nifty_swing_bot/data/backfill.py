"""One-shot historical data backfill.

Downloads the NSE bhavcopy archive (delivery data) and the OHLCV history for the
whole universe into the local cache. This is the slow, network-heavy step; every
later stage reads from the cache and runs offline.

The bhavcopy download is **resumable**: each trading day is cached separately, so
interrupting the run (or being throttled by NSE) loses at most the day in flight.
Re-running picks up exactly where it stopped.

Usage::

    python -m nifty_swing_bot.data.backfill --start 2023-01-01
    python -m nifty_swing_bot.data.backfill --start 2023-01-01 --skip-ohlcv
    python -m nifty_swing_bot.data.backfill --status
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta

import pandas as pd

from ..config import get_config
from .fetch_delivery import cached_delivery_days, fetch_delivery_history
from .fetch_ohlcv import fetch_benchmark, fetch_ohlcv
from .universe import build_universe

logger = logging.getLogger(__name__)


def report_status() -> None:
    """Print what is already cached, without touching the network."""
    cfg = get_config()
    days = cached_delivery_days(cfg)
    ohlcv_files = list(cfg.paths.ohlcv_dir.glob("*.parquet"))

    print("\n  CACHE STATUS")
    print(f"    OHLCV symbols cached   : {len(ohlcv_files)}")
    if days:
        gaps = _find_gaps(days)
        print(f"    Delivery trading days  : {len(days)}")
        print(f"    Delivery range         : {days[0]} to {days[-1]}")
        print(f"    Suspicious gaps (>4d)  : {len(gaps)}")
        for a, b in gaps[:5]:
            print(f"      {a} -> {b}  ({(b - a).days} days)")
    else:
        print("    Delivery trading days  : 0  (run the backfill)")
    print()


def _find_gaps(days: list[date], threshold_days: int = 4) -> list[tuple[date, date]]:
    """Consecutive cached days more than ``threshold_days`` apart.

    Long weekends and festival clusters are normal in the Indian market, so this
    flags candidates for inspection rather than definite failures.
    """
    out: list[tuple[date, date]] = []
    for a, b in zip(days, days[1:]):
        if (b - a).days > threshold_days:
            out.append((a, b))
    return out


def backfill(
    start: date,
    end: date | None = None,
    *,
    skip_ohlcv: bool = False,
    skip_delivery: bool = False,
    refresh: bool = False,
    limit: int | None = None,
) -> None:
    """Populate the cache for a date range.

    Args:
        start: First calendar day to fetch.
        end: Last calendar day. Defaults to today.
        skip_ohlcv: Only fetch delivery data.
        skip_delivery: Only fetch price data.
        refresh: Re-download data already cached.
        limit: Restrict to the first N universe symbols (for a quick trial run).
    """
    cfg = get_config()
    end = end or date.today()
    began = time.perf_counter()

    logger.info("Building universe...")
    universe = build_universe(refresh=refresh, cfg=cfg)
    if universe.empty:
        raise RuntimeError("Could not build the universe; NSE constituent lists unavailable.")
    symbols = universe["symbol"].tolist()
    if limit:
        symbols = symbols[:limit]
    logger.info("Universe: %d symbols.", len(symbols))

    if not skip_ohlcv:
        logger.info("Fetching OHLCV %s -> %s for %d symbols...", start, end, len(symbols))
        # A warm-up buffer so rolling windows are seeded at the true start date.
        warmup = pd.Timestamp(start) - pd.Timedelta(days=150)
        frames = fetch_ohlcv(symbols, start=warmup, end=end, refresh=refresh, cfg=cfg)
        got = sum(1 for f in frames.values() if not f.empty)
        logger.info("OHLCV cached for %d/%d symbols.", got, len(symbols))
        fetch_benchmark(start=warmup, end=end, refresh=refresh, cfg=cfg)

    if not skip_delivery:
        total_days = (end - start).days
        logger.info(
            "Fetching NSE bhavcopy %s -> %s (~%d weekdays, roughly %.0f min at %.1fs each).",
            start, end,
            total_days * 5 // 7,
            (total_days * 5 / 7) * cfg.data.nse_request_delay_s / 60,
            cfg.data.nse_request_delay_s,
        )
        fetch_delivery_history(start, end, refresh=refresh, cfg=cfg, progress_every=20)

    logger.info("Backfill finished in %.1f min.", (time.perf_counter() - began) / 60)
    report_status()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Backfill OHLCV and NSE delivery data.")
    parser.add_argument(
        "--start", type=lambda s: date.fromisoformat(s),
        default=date.today() - timedelta(days=365 * 3),
        help="First day to fetch (default: 3 years ago).",
    )
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--skip-ohlcv", action="store_true")
    parser.add_argument("--skip-delivery", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Re-download cached data.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--status", action="store_true", help="Show cache status and exit.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.status:
        report_status()
        return 0

    backfill(
        args.start, args.end,
        skip_ohlcv=args.skip_ohlcv, skip_delivery=args.skip_delivery,
        refresh=args.refresh, limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
