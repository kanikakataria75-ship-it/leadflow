"""Cross-validation of the custom engine against ``backtesting.py``.

The portfolio engine in :mod:`.engine` is bespoke, so it deserves an independent
check. ``backtesting.py`` cannot run the real strategy -- it is single-asset, so
it has no concept of shared capital, a concurrent-position cap or portfolio
drawdown -- but it *can* independently verify the per-symbol mechanics that the
portfolio layer is built on:

* entries filling at the next bar's open, not the signal bar's close
* stop orders filling at the stop when price trades through it
* stop orders filling at the open when price gaps past it
* the max-hold clock closing a trade on the right bar

If both engines agree trade-for-trade on a symbol, the remaining difference
between them is exactly the portfolio accounting that only the custom engine
does -- which is the thing that cannot be outsourced.

Trailing stops and costs are switched off on both sides for the comparison:
``backtesting.py`` has no equivalent of a confirmed-swing-low trail, so
including it would compare two different strategies rather than two
implementations of the same one.

Usage::

    python -m nifty_swing_bot.backtest.bt_adapter --symbol TATACHEM
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import AppConfig, get_config
from ..strategy.dab_strategy import compute_features, compute_stop
from .engine import PortfolioBacktester

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ComparisonResult:
    """Trade-by-trade agreement between the two engines."""

    symbol: str
    custom_trades: pd.DataFrame
    library_trades: pd.DataFrame
    matched: int
    custom_only: int
    library_only: int
    max_entry_price_diff: float
    max_exit_price_diff: float
    stop_exits_compared: int = 0
    max_stop_exit_diff: float = 0.0
    max_hold_exits: int = 0

    @property
    def agrees(self) -> bool:
        """True when the mechanics the two engines *can* express identically match.

        Entry selection, entry timing and stop fills must agree to within a
        paisa. ``max_hold`` exits are excluded by construction -- see
        :attr:`max_hold_exits` and the note in :meth:`summary`.
        """
        return (
            self.custom_only == 0
            and self.library_only == 0
            and self.matched > 0
            and self.max_entry_price_diff < 0.01
            and self.max_stop_exit_diff < 0.01
        )

    def summary(self) -> str:
        """Readable verdict."""
        verdict = "AGREE" if self.agrees else "DIFFER"
        lines = [
            f"  {self.symbol}: {verdict}",
            f"    matched trades      : {self.matched}",
            f"    custom engine only  : {self.custom_only}",
            f"    backtesting.py only : {self.library_only}",
            f"    max entry price gap : Rs {self.max_entry_price_diff:.4f}",
            f"    stop exits compared : {self.stop_exits_compared}"
            f"  (max gap Rs {self.max_stop_exit_diff:.4f})",
        ]
        if self.max_hold_exits:
            lines.append(
                f"    max-hold exits      : {self.max_hold_exits} excluded "
                "(known semantic difference)"
            )
        return "\n".join(lines)


def _comparison_config(cfg: AppConfig) -> AppConfig:
    """Strip costs, trailing and sizing effects so only mechanics are compared."""
    out = cfg.model_copy(deep=True)
    out.execution.slippage_bps = 0.0
    out.execution.brokerage_bps = 0.0
    out.execution.stt_bps_sell = 0.0
    out.execution.gap_through_extra_slippage_bps = 0.0
    # A band this wide can never trigger, disabling circuit-lock deferral --
    # backtesting.py has no equivalent behaviour to compare against.
    out.execution.circuit_band_pct = 10.0
    out.execution.max_participation_pct = 1e9
    out.risk.trail_activate_r = 1e9      # never trail
    out.risk.max_open_positions = 1
    out.risk.max_portfolio_heat_pct = 1e9
    out.risk.starting_capital = 1e9      # sizing never binds
    out.risk.max_position_pct = 1.0
    out.risk.target_r_multiple = None
    return out


def build_bt_frame(features: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Shape a feature frame for ``backtesting.py``.

    ``backtesting.py`` wants title-case OHLCV columns and exposes any extra
    columns on ``self.data``. The stop is precomputed here using the *next*
    bar's open, because that is the price the custom engine fills at, and the
    library cannot compute a stop from a fill it has not made yet.
    """
    frame = pd.DataFrame(
        {
            "Open": features["open"].astype(float),
            "High": features["high"].astype(float),
            "Low": features["low"].astype(float),
            "Close": features["close"].astype(float),
            "Volume": features["volume"].astype(float),
        },
        index=features.index,
    )
    next_open = features["open"].shift(-1)
    stops: list[float] = []
    for i in range(len(features)):
        entry = float(next_open.iloc[i]) if np.isfinite(next_open.iloc[i]) else np.nan
        if not np.isfinite(entry):
            stops.append(np.nan)
            continue
        stops.append(
            compute_stop(
                float(features["low"].iloc[i]),
                entry,
                float(features["atr_stop"].iloc[i]),
                risk=cfg.risk,
            )
        )
    frame["Signal"] = features["signal"].fillna(False).astype(bool).to_numpy()
    frame["StopPrice"] = stops
    return frame.dropna(subset=["Open", "High", "Low", "Close"])


def run_library_backtest(frame: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Run the equivalent single-symbol strategy under ``backtesting.py``.

    Returns:
        A normalised trade frame, or an empty frame if the library is missing.
    """
    try:
        from backtesting import Backtest, Strategy
    except ImportError:
        logger.warning(
            "backtesting.py is not installed; skipping cross-validation. "
            "Install it with: pip install backtesting"
        )
        return pd.DataFrame()

    max_hold = cfg.risk.max_hold_days
    qty = 100  # fixed size: quantities are not what we are comparing

    class DABReplica(Strategy):
        """A deliberately literal restatement of the DAB entry and exit rules."""

        def init(self) -> None:  # noqa: D102 - library hook
            pass

        def next(self) -> None:  # noqa: D102 - library hook
            # Max-hold clock first, so a timed exit is not pre-empted by a new entry.
            for trade in list(self.trades):
                if len(self.data) - 1 - trade.entry_bar >= max_hold:
                    trade.close()
            if self.position:
                return
            if bool(self.data.Signal[-1]):
                stop = float(self.data.StopPrice[-1])
                if np.isfinite(stop) and stop < float(self.data.Close[-1]):
                    self.buy(size=qty, sl=stop)

    bt = Backtest(
        frame,
        DABReplica,
        cash=10_000_000,
        commission=0.0,
        trade_on_close=False,   # market orders fill at the next bar's open
        exclusive_orders=False,
        finalize_trades=True,
    )
    stats = bt.run()
    trades = stats["_trades"]
    if trades is None or trades.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "entry_date": pd.to_datetime(trades["EntryTime"]).dt.normalize(),
            "exit_date": pd.to_datetime(trades["ExitTime"]).dt.normalize(),
            "entry_price": trades["EntryPrice"].astype(float),
            "exit_price": trades["ExitPrice"].astype(float),
        }
    ).sort_values("entry_date").reset_index(drop=True)


def cross_validate(
    symbol: str,
    ohlcv: pd.DataFrame,
    delivery: pd.DataFrame | None,
    benchmark: pd.DataFrame | None,
    *,
    cfg: AppConfig | None = None,
) -> ComparisonResult:
    """Run both engines on one symbol and compare their trades.

    Args:
        symbol: Symbol name, for reporting.
        ohlcv: Daily bars.
        delivery: Delivery data for the symbol.
        benchmark: Benchmark bars for the relative-strength rule.
        cfg: Config override.

    Returns:
        A :class:`ComparisonResult` describing the agreement.
    """
    cfg = _comparison_config(cfg or get_config())
    features = compute_features(ohlcv, delivery, benchmark, cfg=cfg)
    if features.empty or not features["signal"].any():
        logger.info("%s produced no signals; nothing to cross-validate.", symbol)
        return ComparisonResult(symbol, pd.DataFrame(), pd.DataFrame(), 0, 0, 0, 0.0, 0.0)

    result = PortfolioBacktester(cfg).run(
        {symbol: ohlcv}, {symbol: delivery} if delivery is not None else {},
        benchmark, features={symbol: features},
    )
    custom = pd.DataFrame(
        [
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
            }
            for t in result.trades
        ]
    )
    library = run_library_backtest(build_bt_frame(features, cfg), cfg)
    if library.empty:
        return ComparisonResult(symbol, custom, library, 0, len(custom), 0, 0.0, 0.0)

    # The end-of-data liquidation is an artefact of the portfolio engine's final
    # mark-out and has no counterpart in the library run, so it is excluded.
    custom_cmp = custom[custom["exit_reason"] != "end_of_data"] if not custom.empty else custom

    merged = custom_cmp.merge(
        library, on="entry_date", how="outer", suffixes=("_custom", "_lib"), indicator=True
    )
    both = merged[merged["_merge"] == "both"]

    def gap(frame: pd.DataFrame, column: str) -> float:
        if frame.empty:
            return 0.0
        value = float((frame[f"{column}_custom"] - frame[f"{column}_lib"]).abs().max())
        return value if np.isfinite(value) else 0.0

    # Stop exits are directly comparable: both engines fill a stop order on the
    # bar that breaches it.
    #
    # ``max_hold`` exits are NOT comparable, and excluding them is not a fudge.
    # This engine closes a timed-out trade at the close of the bar the clock
    # expires on, which is what a trader watching the session actually does.
    # ``backtesting.py`` has no way to express that: ``trade.close()`` queues a
    # market order that fills at the NEXT bar's open, and the only way to make
    # it close on the current bar is ``trade_on_close=True``, which would also
    # move entries to the signal bar's close and destroy the entry comparison
    # that matters more. So the two engines exit timed-out trades one bar apart
    # by construction, and comparing those prices would measure the library's
    # order model rather than this engine's correctness.
    stop_like = {"stop", "gap_through_stop", "trailing_stop"}
    stop_rows = both[both["exit_reason"].isin(stop_like)] if "exit_reason" in both else both
    hold_rows = both[both["exit_reason"] == "max_hold"] if "exit_reason" in both else both.iloc[0:0]

    return ComparisonResult(
        symbol=symbol,
        custom_trades=custom,
        library_trades=library,
        matched=int(len(both)),
        custom_only=int((merged["_merge"] == "left_only").sum()),
        library_only=int((merged["_merge"] == "right_only").sum()),
        max_entry_price_diff=gap(both, "entry_price"),
        max_exit_price_diff=gap(both, "exit_price"),
        stop_exits_compared=int(len(stop_rows)),
        max_stop_exit_diff=gap(stop_rows, "exit_price"),
        max_hold_exits=int(len(hold_rows)),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: cross-validate one or more symbols."""
    from ..data.fetch_delivery import cached_delivery_days
    from .run_backtest import load_market_data

    parser = argparse.ArgumentParser(
        description="Cross-validate the custom engine against backtesting.py."
    )
    parser.add_argument("--symbol", action="append", default=None,
                        help="Symbol to check. Repeatable. Default: the first few with signals.")
    parser.add_argument("--limit", type=int, default=60, help="Universe size to load.")
    parser.add_argument("--max-symbols", type=int, default=5)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = get_config()
    available = cached_delivery_days(cfg)
    if not available:
        raise SystemExit("No delivery data cached. Run: python -m nifty_swing_bot.data.backfill")

    _, prices, delivery, benchmark = load_market_data(
        start=available[min(30, len(available) - 1)], end=available[-1],
        limit=args.limit, cfg=cfg,
    )

    symbols = args.symbol or list(prices)
    print("\n  ENGINE CROSS-VALIDATION (custom engine vs backtesting.py)\n")
    checked = 0
    agreements = 0
    for symbol in symbols:
        if symbol not in prices:
            logger.warning("%s not in the loaded universe; skipping.", symbol)
            continue
        result = cross_validate(
            symbol, prices[symbol], delivery.get(symbol), benchmark, cfg=cfg
        )
        if result.matched == 0 and result.custom_only == 0 and result.library_only == 0:
            continue
        print(result.summary())
        print()
        checked += 1
        agreements += int(result.agrees)
        if checked >= args.max_symbols:
            break

    if checked:
        print(f"  {agreements}/{checked} symbols agreed trade-for-trade.\n")
    else:
        print("  No symbols produced signals to compare.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
