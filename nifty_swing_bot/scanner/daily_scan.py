"""Daily signal scanner and forward-test tracker.

Two jobs, both run by ``main``:

**Scan.** Apply the DAB rules to the current universe using the most recent
bars, and record any signals -- with entry reference, stop, suggested quantity
and the rule values that fired -- into the SQLite ledger.

**Track.** Walk forward every previously recorded signal that is still open,
applying the *same* exit rules the backtester uses, and write the realised
outcome back. This is what makes live performance comparable to the backtest
rather than a separate, incompatible record.

Run it after the close (NSE publishes the bhavcopy in the evening, around
18:00-19:00 IST; before then the delivery file for today does not exist and the
scan will correctly find nothing new).

Usage::

    python -m nifty_swing_bot.scanner.daily_scan
    python -m nifty_swing_bot.scanner.daily_scan --with-llm
    python -m nifty_swing_bot.scanner.daily_scan --date 2026-09-04 --track-only
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.run_backtest import DELIVERY_WARMUP_DAYS, PRICE_WARMUP_DAYS
from ..config import AppConfig, get_config
from ..data.fetch_delivery import fetch_delivery_history, to_symbol_panel
from ..data.fetch_ohlcv import fetch_benchmark, fetch_ohlcv
from ..data.store import SignalStore
from ..data.universe import apply_liquidity_filter, build_universe
from ..strategy.dab_strategy import compute_features, extract_signals, position_size
from ..strategy.indicators import swing_low

logger = logging.getLogger(__name__)


def load_recent_data(
    *,
    as_of: date,
    refresh: bool = True,
    limit: int | None = None,
    cfg: AppConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Load just enough recent history to evaluate the rules as of a date.

    Args:
        as_of: The scan date. Data is loaded up to and including this day.
        refresh: Force a re-fetch of prices so the latest bar is current. This
            matters for a live scan: a stale cache would silently scan yesterday.
        limit: Cap the universe size.
        cfg: Config override.

    Returns:
        ``(universe, prices, delivery, benchmark)``.
    """
    cfg = cfg or get_config()
    price_start = pd.Timestamp(as_of) - pd.Timedelta(days=PRICE_WARMUP_DAYS)
    deliv_start = pd.Timestamp(as_of) - pd.Timedelta(days=DELIVERY_WARMUP_DAYS)

    universe = build_universe(cfg=cfg)
    if universe.empty:
        raise RuntimeError("Universe is empty; NSE constituent lists could not be loaded.")

    symbols = universe["symbol"].tolist()
    prices = fetch_ohlcv(symbols, start=price_start, end=as_of, refresh=refresh, cfg=cfg)
    universe = apply_liquidity_filter(universe, prices, cfg=cfg, as_of=pd.Timestamp(as_of))
    if limit:
        universe = universe.head(limit)

    kept = set(universe["symbol"])
    prices = {s: f for s, f in prices.items() if s in kept and not f.empty}

    delivery = to_symbol_panel(
        fetch_delivery_history(
            pd.Timestamp(deliv_start).date(), as_of, symbols=kept, cfg=cfg, progress_every=0
        )
    )
    benchmark = fetch_benchmark(start=price_start, end=as_of, refresh=refresh, cfg=cfg)
    return universe, prices, delivery, benchmark


def scan(
    *,
    as_of: date | None = None,
    limit: int | None = None,
    refresh: bool = True,
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    """Run the DAB rules across the universe for a single date.

    Args:
        as_of: Date to scan. Defaults to the latest bar available in the data.
        limit: Cap the universe size.
        refresh: Re-fetch prices before scanning.
        cfg: Config override.

    Returns:
        A frame of signals, strongest accumulation first, with ``suggested_qty``
        and ``risk_amount`` filled in from the configured capital.
    """
    cfg = cfg or get_config()
    as_of = as_of or date.today()
    universe, prices, delivery, benchmark = load_recent_data(
        as_of=as_of, refresh=refresh, limit=limit, cfg=cfg
    )
    if not prices:
        logger.warning("No price data available for the scan.")
        return pd.DataFrame()

    # The scan date is the most recent bar we actually have, which on a weekend
    # or holiday is the previous session rather than today.
    latest = max(f.index.max() for f in prices.values() if not f.empty)
    latest = min(latest, pd.Timestamp(as_of))
    logger.info("Scanning %d symbols as of %s.", len(prices), latest.date())

    meta = universe.set_index("symbol")[["company", "industry"]].to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for symbol, ohlcv in prices.items():
        if ohlcv.empty or latest not in ohlcv.index:
            continue
        try:
            features = compute_features(ohlcv, delivery.get(symbol), benchmark, cfg=cfg)
        except (KeyError, ValueError) as exc:
            logger.debug("Features failed for %s: %s", symbol, exc)
            continue
        if features.empty or latest not in features.index:
            continue
        if not bool(features.loc[latest, "signal"]):
            continue

        # entry_on_next_open=False: tomorrow's open is genuinely unknown live, so
        # the close is quoted as the entry reference.
        for sig in extract_signals(
            features.loc[:latest], symbol, cfg=cfg, entry_on_next_open=False
        ):
            if sig.date != latest:
                continue
            row = sig.as_dict()
            qty, risk_amount = position_size(
                cfg.risk.starting_capital, sig.entry_ref, sig.stop, risk=cfg.risk
            )
            row["suggested_qty"] = qty
            row["risk_amount"] = round(risk_amount, 2)
            row["position_value"] = round(qty * sig.entry_ref, 2)
            info = meta.get(symbol, {})
            row["company"] = info.get("company")
            row["industry"] = info.get("industry")
            rows.append(row)

    if not rows:
        logger.info("No signals on %s.", latest.date())
        return pd.DataFrame()

    frame = pd.DataFrame(rows).sort_values(
        "deliv_spike_ratio", ascending=False, na_position="last"
    )
    logger.info("Found %d signals on %s.", len(frame), latest.date())
    return frame.reset_index(drop=True)


def track_open_signals(
    store: SignalStore,
    *,
    cfg: AppConfig | None = None,
    prices: Mapping[str, pd.DataFrame] | None = None,
) -> int:
    """Resolve previously recorded signals against subsequent price action.

    Applies the same exit stack as the backtester -- initial stop, swing-low
    trail past 1R, and the max-hold clock -- so realised forward R multiples are
    measured on the same basis as backtested ones.

    Args:
        store: The signal ledger.
        cfg: Config override.
        prices: Pre-loaded price frames. Fetched if omitted.

    Returns:
        The number of signals whose outcome was written or updated.
    """
    cfg = cfg or get_config()
    risk, execp = cfg.risk, cfg.execution
    open_rows = store.open_signals()
    if open_rows.empty:
        logger.info("No open signals to track.")
        return 0

    symbols = sorted(set(open_rows["symbol"]))
    if prices is None:
        earliest = pd.Timestamp(open_rows["signal_date"].min()) - pd.Timedelta(days=10)
        prices = fetch_ohlcv(symbols, start=earliest, end=date.today(), refresh=True, cfg=cfg)

    updated = 0
    for _, sig in open_rows.iterrows():
        frame = prices.get(sig["symbol"])
        if frame is None or frame.empty:
            continue
        signal_ts = pd.Timestamp(sig["signal_date"])
        forward = frame.loc[frame.index > signal_ts]
        if forward.empty:
            store.upsert_outcome(int(sig["id"]), {"status": "pending_entry"})
            continue

        outcome = _simulate_forward(
            forward,
            signal_low=float(sig["close"]) if pd.isna(sig["stop"]) else None,
            recorded_stop=float(sig["stop"]),
            atr=float(sig["atr14"] or 0.0),
            risk=risk,
            execp=execp,
        )
        store.upsert_outcome(int(sig["id"]), outcome)
        updated += 1

    logger.info("Updated outcomes for %d signals.", updated)
    return updated


def _simulate_forward(
    forward: pd.DataFrame,
    *,
    signal_low: float | None,
    recorded_stop: float,
    atr: float,
    risk: Any,
    execp: Any,
) -> dict[str, Any]:
    """Walk a single signal forward through subsequent bars, applying exit rules.

    The entry is the first available open after the signal, matching the
    backtester. Returns an outcome dict shaped for
    :meth:`~nifty_swing_bot.data.store.SignalStore.upsert_outcome`; the trade is
    reported as ``open`` while it is still running.
    """
    entry_bar = forward.iloc[0]
    entry_price = float(entry_bar["open"]) * (1 + execp.slippage_bps / 10_000)
    stop = recorded_stop
    if not np.isfinite(stop) or stop >= entry_price:
        stop = entry_price - risk.atr_stop_mult * atr if atr > 0 else entry_price * 0.97
    risk_per_share = entry_price - stop
    if risk_per_share <= 0:
        return {"status": "invalid"}

    swings = swing_low(forward["low"], risk.trail_swing_lookback)
    highest, lowest = -np.inf, np.inf
    trail_active = False

    for i, (ts, bar) in enumerate(forward.iterrows()):
        high, low, close, open_ = (
            float(bar["high"]), float(bar["low"]), float(bar["close"]), float(bar["open"])
        )
        highest, lowest = max(highest, high), min(lowest, low)

        exit_price: float | None = None
        reason = ""
        if low <= stop:
            if open_ <= stop:
                exit_price = open_ * (1 - execp.gap_through_extra_slippage_bps / 10_000)
                reason = "gap_through_stop"
            else:
                exit_price = stop * (1 - execp.slippage_bps / 10_000)
                reason = "trailing_stop" if trail_active else "stop"
        elif i >= risk.max_hold_days:
            exit_price = close * (1 - execp.slippage_bps / 10_000)
            reason = "max_hold"

        if exit_price is not None:
            return {
                "status": "closed",
                "entry_date": forward.index[0].strftime("%Y-%m-%d"),
                "entry_price": round(entry_price, 2),
                "exit_date": ts.strftime("%Y-%m-%d"),
                "exit_price": round(exit_price, 2),
                "exit_reason": reason,
                "bars_held": i,
                "return_pct": round((exit_price / entry_price - 1) * 100, 2),
                "r_multiple": round((exit_price - entry_price) / risk_per_share, 3),
                "mae_r": round((lowest - entry_price) / risk_per_share, 3),
                "mfe_r": round((highest - entry_price) / risk_per_share, 3),
            }

        if (close - entry_price) / risk_per_share >= risk.trail_activate_r:
            trail_active = True
        if trail_active:
            level = swings.get(ts, np.nan)
            if np.isfinite(level) and level > stop:
                stop = float(level)

    # Still running.
    last = forward.iloc[-1]
    return {
        "status": "open",
        "entry_date": forward.index[0].strftime("%Y-%m-%d"),
        "entry_price": round(entry_price, 2),
        "bars_held": len(forward) - 1,
        "return_pct": round((float(last["close"]) / entry_price - 1) * 100, 2),
        "r_multiple": round((float(last["close"]) - entry_price) / risk_per_share, 3),
        "mae_r": round((lowest - entry_price) / risk_per_share, 3),
        "mfe_r": round((highest - entry_price) / risk_per_share, 3),
    }


def format_report(signals: pd.DataFrame, forward: Mapping[str, Any]) -> str:
    """Terminal summary of today's signals and the running forward record."""
    rows = ["", "=" * 100, "  DAB DAILY SCAN", "=" * 100]
    if signals.empty:
        rows.append("  No signals today.")
    else:
        rows.append(
            f"  {'SYMBOL':<14}{'CLOSE':>9}{'ENTRY':>9}{'STOP':>9}{'STOP%':>7}"
            f"{'QTY':>7}{'RISK':>9}{'DELIV':>8}{'AGE':>5}{'VOL':>6}{'BO':>4}"
        )
        rows.append("  " + "-" * 96)
        for _, s in signals.iterrows():
            rows.append(
                f"  {s['symbol']:<14}{s['close']:>9.2f}{s['entry_ref']:>9.2f}"
                f"{s['stop']:>9.2f}{s['stop_pct']:>6.1f}%{s['suggested_qty']:>7}"
                f"{s['risk_amount']:>9,.0f}"
                f"{(s['deliv_spike_ratio'] if pd.notna(s['deliv_spike_ratio']) else 0):>7.2f}x"
                f"{(int(s['deliv_spike_age']) if pd.notna(s['deliv_spike_age']) else 0):>5}"
                f"{s['vol_ratio']:>5.1f}x{'Y' if s['is_breakout'] else 'N':>4}"
            )
        rows.append("  " + "-" * 96)
        rows.append("  DELIV = delivery% spike vs baseline   AGE = sessions since that spike")

    rows += ["", "  FORWARD RECORD (live, tracked with backtest exit rules)"]
    if not forward.get("signals_closed"):
        rows.append(
            f"    {forward.get('signals_total', 0)} signals recorded, "
            f"{forward.get('signals_open', 0)} still open, none closed yet."
        )
    else:
        rows += [
            f"    Signals recorded     {forward.get('signals_total', 0)}"
            f"  ({forward.get('signals_open', 0)} open)",
            f"    Closed              {forward.get('signals_closed', 0)}",
            f"    Win rate            {forward.get('win_rate_pct', 0):.1f}%",
            f"    Avg R multiple      {forward.get('avg_r_multiple', 0):+.3f}R",
            f"    Profit factor       {forward.get('profit_factor') or 0:.2f}",
            f"    Total R             {forward.get('total_r', 0):+.2f}R",
        ]
    rows += ["=" * 100, ""]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Scan for DAB signals and track outcomes.")
    parser.add_argument("--date", type=lambda s: date.fromisoformat(s), default=None,
                        help="Scan as of this date (default: today).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-refresh", action="store_true",
                        help="Use cached prices instead of re-fetching.")
    parser.add_argument("--with-llm", action="store_true",
                        help="Generate Anthropic insights for each signal.")
    parser.add_argument("--track-only", action="store_true",
                        help="Only update outcomes for existing signals.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    )
    cfg = get_config()
    store = SignalStore(cfg=cfg)
    started = time.perf_counter()

    if args.track_only:
        track_open_signals(store, cfg=cfg)
        print(format_report(pd.DataFrame(), store.forward_stats()))
        return 0

    scan_date = args.date or date.today()
    signals = scan(as_of=scan_date, limit=args.limit, refresh=not args.no_refresh, cfg=cfg)

    if not signals.empty:
        new = store.add_signals(signals.to_dict(orient="records"), scan_date=scan_date)
        logger.info("Recorded %d new signals (%d already known).", new, len(signals) - new)

    store.record_scan(
        scan_date=scan_date,
        universe_size=0 if signals.empty else int(signals["symbol"].nunique()),
        signal_count=int(len(signals)),
        duration_s=time.perf_counter() - started,
    )
    track_open_signals(store, cfg=cfg)

    if args.with_llm and not signals.empty:
        from ..llm.insight_generator import generate_for_signals

        logger.info("Generating LLM insights for %d signals...", len(signals))
        generate_for_signals(signals.to_dict(orient="records"), store=store, cfg=cfg)

    print(format_report(signals, store.forward_stats()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
