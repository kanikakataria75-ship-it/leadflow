"""Command line entry point: ``python -m nifty_swing_bot.aes_scanner``.

Intended to run once daily after the close, e.g. from Task Scheduler::

    python -m nifty_swing_bot.aes_scanner --equity 1000000
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..config import get_config
from . import live_config as LC
from .pipeline import run_scan, track_outcomes
from .report import format_forward_record, format_funnel, format_report
from .store import AESScanStore

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aes-scan", description="Daily AES scan.")
    ap.add_argument("--equity", type=float, default=None,
                    help="Account equity to size against (default: configured capital).")
    ap.add_argument("--deployed", type=float, default=None,
                    help="Rupees currently in open positions, so the report can "
                         "show sleeve usage against the 30%% strategy target.")
    ap.add_argument("--recent-bars", type=int, default=3,
                    help="An entry trigger this many bars old still counts as actionable.")
    ap.add_argument("--no-render", action="store_true", help="Skip chart rendering.")
    ap.add_argument("--no-persist", action="store_true", help="Do not write to the store.")
    ap.add_argument("--audit", action="store_true",
                    help="Print the full-universe funnel and write the audit CSV.")
    ap.add_argument("--record", action="store_true",
                    help="Print the accumulated forward record and exit.")
    ap.add_argument("--since", default=None, help="With --record: only rows on/after this date.")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = get_config()
    store = AESScanStore(cfg=cfg)

    if a.record:
        print(format_forward_record(store.forward_record(since=a.since)))
        return 0

    try:
        # The FROZEN configuration (aes_scanner/live_config.py) is passed
        # explicitly on every run. Taking dataclass defaults here would silently
        # run the spec ladder and the calibrated scorer -- both rejected.
        result = run_scan(cfg, equity=a.equity, recent_bars=a.recent_bars,
                          render=not a.no_render, persist=not a.no_persist, store=store,
                          screener_params=LC.SCREENER, box_params=LC.BOX,
                          watch_params=LC.WATCHLIST, portfolio_params=LC.PORTFOLIO,
                          score_params=LC.SCORING)
        result.equity = a.equity if a.equity else cfg.risk.starting_capital
        result.deployed_notional = a.deployed
    except Exception:
        logger.exception("Scan failed.")
        return 1

    print(format_report(result))

    if a.audit and len(result.audit):
        from pathlib import Path
        print(format_funnel(result.audit))
        out = Path(cfg.paths.results_dir) / f"aes_audit_{result.scan_date}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        result.audit.to_csv(out, index=False, encoding="utf-8")
        print(f"  full audit ({len(result.audit)} symbols) written to {out}\n")

    if not a.no_persist:
        try:
            from ..data.fetch_ohlcv import fetch_benchmark, fetch_ohlcv
            from ..data.universe import build_universe
            uni = build_universe(cfg=cfg)
            frames = fetch_ohlcv(uni["symbol"].tolist(), cfg=cfg, show_progress=False)
            n = track_outcomes(store, frames, fetch_benchmark(cfg=cfg))
            if n:
                print(f"  forward record updated for {n} prior candidates\n")
        except Exception:
            logger.exception("Outcome tracking failed; the scan itself is still recorded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
