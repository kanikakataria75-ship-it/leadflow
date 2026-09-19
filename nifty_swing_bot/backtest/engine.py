"""Event-driven portfolio backtester for the DAB strategy.

Why a custom engine rather than ``backtesting.py`` or ``vectorbt``:

* ``backtesting.py`` is strictly single-asset. Running it once per symbol and
  stitching the trade lists together would let every symbol assume the full
  account balance, so the equity curve, the concurrent-position limit and the
  portfolio drawdown would all be fiction. It is still useful as an independent
  check on single-symbol mechanics, which is what ``bt_adapter`` does.
* ``vectorbt`` handles multiple assets, but the exit stack here (trailing swing
  low, activated only past 1R, plus a max-hold clock, plus circuit-aware fills)
  needs custom numba callbacks that are hard to verify.

This engine simulates one shared account across the whole universe, bar by bar:

1. Fill entries queued on the previous bar, at this bar's open.
2. Evaluate exits for every open position, using the stop known at the end of
   the previous bar -- never a stop computed from the current bar.
3. Generate today's signals and queue them for tomorrow's open.
4. Mark the portfolio to market on the close.

**Circuit-filter realism.** Indian mid- and small-caps trade with 5%/10%/20%
price bands. A position cannot be exited into a locked lower circuit: there is
no bid. The engine detects a locked bar (opens at the band and barely trades),
defers the exit to the next bar, and charges an additional adverse move. Bars
that merely gap through the stop fill at the open, worse than the stop, with
extra slippage. Assuming clean stop fills would materially overstate results in
exactly the tier this strategy targets.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..config import AppConfig, get_config
from ..strategy.dab_strategy import compute_features, compute_stop, position_size
from ..strategy.indicators import swing_low
from .costs import entry_charges, exit_charges

logger = logging.getLogger(__name__)

ExitReason = Literal[
    "stop", "gap_through_stop", "circuit_locked_stop", "trailing_stop",
    "max_hold", "target", "end_of_data",
]


def _num(value: object, default: float = 0.0) -> float:
    """Coerce a possibly-missing cell to a finite float.

    Feature frames mix numpy floats, Python floats and ``pd.NA``. The obvious
    ``value or default`` idiom raises on ``pd.NA`` because its truth value is
    ambiguous, so every read of an optional numeric goes through here instead.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


@dataclass(slots=True)
class Position:
    """An open long position."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    qty: int
    stop: float
    initial_stop: float
    initial_risk_per_share: float
    entry_costs: float
    bars_held: int = 0
    highest_high: float = 0.0
    lowest_low: float = float("inf")
    trail_active: bool = False
    deferred_exit: str | None = None
    signal_meta: dict[str, float] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price

    @property
    def initial_risk(self) -> float:
        return self.qty * self.initial_risk_per_share


@dataclass(slots=True)
class Trade:
    """A completed round-trip."""

    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float
    exit_reason: str
    bars_held: int
    initial_stop: float
    initial_risk_per_share: float
    mae_r: float
    mfe_r: float
    signal_meta: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly row for the trade log and the API."""
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date.strftime("%Y-%m-%d"),
            "exit_date": self.exit_date.strftime("%Y-%m-%d"),
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "qty": self.qty,
            "gross_pnl": round(self.gross_pnl, 2),
            "costs": round(self.costs, 2),
            "net_pnl": round(self.net_pnl, 2),
            "return_pct": round((self.exit_price / self.entry_price - 1) * 100, 2),
            "r_multiple": round(self.r_multiple, 3),
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
            "initial_stop": round(self.initial_stop, 2),
            "mae_r": round(self.mae_r, 3),
            "mfe_r": round(self.mfe_r, 3),
            **{k: (round(v, 3) if isinstance(v, float) and np.isfinite(v) else None)
               for k, v in self.signal_meta.items()},
        }


@dataclass(slots=True)
class BacktestResult:
    """Everything a run produces."""

    trades: list[Trade]
    equity_curve: pd.DataFrame
    stats: dict[str, float]
    rejected: dict[str, int]
    params: dict[str, object] = field(default_factory=dict)

    @property
    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.as_dict() for t in self.trades])


@dataclass(slots=True)
class _PendingEntry:
    """A signal awaiting the next bar's open."""

    symbol: str
    signal_date: pd.Timestamp
    signal_low: float
    atr_stop: float
    rank: float
    turnover_cr: float
    meta: dict[str, float]


class PortfolioBacktester:
    """Simulates the DAB strategy over a universe with a single shared account."""

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()

    # ------------------------------------------------------------------ run --
    def run(
        self,
        prices: Mapping[str, pd.DataFrame],
        delivery: Mapping[str, pd.DataFrame],
        benchmark: pd.DataFrame | None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        features: Mapping[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        """Run the simulation.

        Args:
            prices: Symbol -> OHLCV frame.
            delivery: Symbol -> delivery frame with ``deliv_pct``.
            benchmark: Benchmark OHLCV, used for the relative-strength rule and
                for buy-and-hold comparison.
            start: Simulation start. Defaults to the configured backtest start.
            end: Simulation end. Defaults to the last available bar.
            features: Pre-computed feature frames, to avoid recomputation across
                walk-forward folds. Computed here when omitted.

        Returns:
            A :class:`BacktestResult` with trades, equity curve and statistics.
        """
        cfg = self.cfg
        risk, execp = cfg.risk, cfg.execution

        feats = dict(features) if features is not None else self.compute_all_features(
            prices, delivery, benchmark
        )
        if not feats:
            logger.warning("No feature frames; nothing to simulate.")
            return self._empty_result()

        calendar = self._build_calendar(prices, start, end)
        if len(calendar) < 2:
            logger.warning("Calendar has %d bars; nothing to simulate.", len(calendar))
            return self._empty_result()

        # Mark-to-market runs for every open position on every bar, so it must
        # be O(1). Slicing each symbol's full history per position per day was
        # O(bars) and made long backtests quadratic; a date-indexed array plus a
        # row map replaces it with a single integer lookup.
        close_lookup = self._build_close_lookup(prices, calendar)

        signals_by_date = self._index_signals(feats, calendar)
        swings = {s: swing_low(f["low"], risk.trail_swing_lookback) for s, f in feats.items()}
        # A strategy may carry its own exit rule (mean reversion exits when
        # price reverts, not on a fixed clock). When the feature frame
        # provides `exit_signal`, those bars close the position at the close.
        exit_dates: dict[str, set[pd.Timestamp]] = {
            sym: set(f.index[f["exit_signal"].fillna(False).astype(bool)])
            for sym, f in feats.items()
            if "exit_signal" in f
        }

        cash = risk.starting_capital
        positions: dict[str, Position] = {}
        pending: list[_PendingEntry] = []
        trades: list[Trade] = []
        equity_rows: list[dict[str, object]] = []
        rejected = {
            "no_capacity": 0, "no_cash": 0, "zero_qty": 0,
            "already_open": 0, "no_next_bar": 0, "heat_cap": 0,
        }

        for i, today in enumerate(calendar):
            # -- 1. Fill entries queued yesterday, at today's open ---------- #
            equity_now = cash + self._open_value(positions, close_lookup, today)
            for entry in self._prioritise(pending):
                if len(positions) >= risk.max_open_positions:
                    rejected["no_capacity"] += 1
                    continue
                if entry.symbol in positions:
                    rejected["already_open"] += 1
                    continue
                bar = self._bar(prices, entry.symbol, today)
                if bar is None:
                    rejected["no_next_bar"] += 1
                    continue

                fill = float(bar["open"]) * (1 + execp.slippage_bps / 10_000)
                stop = compute_stop(entry.signal_low, fill, entry.atr_stop, risk=risk)

                open_heat = sum(p.initial_risk for p in positions.values())
                heat_room = equity_now * risk.max_portfolio_heat_pct - open_heat
                if heat_room <= 0:
                    rejected["heat_cap"] += 1
                    continue

                # A position may not be a large share of the day's traded value.
                liquidity_cap = (
                    entry.turnover_cr * 1e7 * execp.max_participation_pct
                    if entry.turnover_cr > 0 else None
                )
                qty, rupee_risk = position_size(
                    equity_now, fill, stop, risk=risk,
                    available_cash=cash, max_notional=liquidity_cap,
                )
                if qty <= 0:
                    rejected["zero_qty" if cash > fill else "no_cash"] += 1
                    continue
                if rupee_risk > heat_room:
                    qty = int(heat_room // (fill - stop))
                    if qty <= 0:
                        rejected["heat_cap"] += 1
                        continue

                cost = entry_charges(qty * fill, execp).total
                if qty * fill + cost > cash:
                    rejected["no_cash"] += 1
                    continue
                cash -= qty * fill + cost
                positions[entry.symbol] = Position(
                    symbol=entry.symbol,
                    entry_date=today,
                    entry_price=fill,
                    qty=qty,
                    stop=stop,
                    initial_stop=stop,
                    initial_risk_per_share=fill - stop,
                    entry_costs=cost,
                    highest_high=float(bar["high"]),
                    lowest_low=float(bar["low"]),
                    signal_meta=entry.meta,
                )
            pending = []

            # -- 2. Evaluate exits on today's bar --------------------------- #
            for symbol in list(positions):
                pos = positions[symbol]
                bar = self._bar(prices, symbol, today)
                if bar is None:
                    continue
                if pos.entry_date == today:
                    # Entered at today's open; the bar still counts for stops.
                    pass
                else:
                    pos.bars_held += 1

                pos.highest_high = max(pos.highest_high, float(bar["high"]))
                pos.lowest_low = min(pos.lowest_low, float(bar["low"]))

                exit_price, reason = self._evaluate_exit(
                    pos, bar, prices, symbol, today,
                    exit_now=today in exit_dates.get(symbol, ()),
                )
                if exit_price is not None:
                    proceeds, costs = self._sell(exit_price, pos.qty)
                    cash += proceeds
                    trades.append(self._close(pos, today, exit_price, costs, reason=reason))
                    del positions[symbol]
                    continue

                # -- 3. Update the trailing stop for TOMORROW --------------- #
                self._update_trail(pos, bar, swings.get(symbol), today)

            # -- 4. Queue tomorrow's entries from today's signals ----------- #
            if i + 1 < len(calendar):
                for sig in signals_by_date.get(today, []):
                    if sig.symbol not in positions:
                        pending.append(sig)

            # -- 5. Mark to market ------------------------------------------ #
            open_val = self._open_value(positions, close_lookup, today)
            equity_rows.append(
                {
                    "date": today,
                    "equity": cash + open_val,
                    "cash": cash,
                    "positions_value": open_val,
                    "open_positions": len(positions),
                    "exposure": (open_val / (cash + open_val)) if (cash + open_val) > 0 else 0.0,
                }
            )

        # -- Liquidate anything still open at the final close ---------------- #
        last = calendar[-1]
        for symbol, pos in list(positions.items()):
            bar = self._bar(prices, symbol, last)
            price = float(bar["close"]) if bar is not None else pos.entry_price
            proceeds, costs = self._sell(price, pos.qty)
            cash += proceeds
            trades.append(self._close(pos, last, price, costs, reason="end_of_data"))
            del positions[symbol]
        if equity_rows:
            equity_rows[-1]["equity"] = cash
            equity_rows[-1]["cash"] = cash
            equity_rows[-1]["positions_value"] = 0.0

        equity = pd.DataFrame(equity_rows).set_index("date")
        from .metrics import compute_stats  # local import avoids a cycle

        stats = compute_stats(trades, equity, cfg=cfg, benchmark=benchmark)
        logger.info(
            "Backtest complete: %d trades, final equity Rs %.0f (%.1f%%).",
            len(trades),
            stats.get("final_equity", 0.0),
            stats.get("total_return_pct", 0.0),
        )
        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            stats=stats,
            rejected=rejected,
            params=cfg.dab.model_dump(),
        )

    # -------------------------------------------------------------- helpers --
    def compute_all_features(
        self,
        prices: Mapping[str, pd.DataFrame],
        delivery: Mapping[str, pd.DataFrame],
        benchmark: pd.DataFrame | None,
    ) -> dict[str, pd.DataFrame]:
        """Compute DAB features for every symbol that has enough data."""
        out: dict[str, pd.DataFrame] = {}
        for symbol, frame in prices.items():
            if frame is None or len(frame) < self.cfg.universe.min_history_days:
                continue
            try:
                feats = compute_features(
                    frame, delivery.get(symbol), benchmark, cfg=self.cfg
                )
            except (KeyError, ValueError) as exc:
                logger.debug("Features failed for %s: %s", symbol, exc)
                continue
            if not feats.empty:
                out[symbol] = feats
        logger.info("Computed features for %d/%d symbols.", len(out), len(prices))
        return out

    @staticmethod
    def _build_calendar(
        prices: Mapping[str, pd.DataFrame],
        start: pd.Timestamp | str | None,
        end: pd.Timestamp | str | None,
    ) -> list[pd.Timestamp]:
        """Union of all symbols' trading days, clipped to the window."""
        idx: pd.DatetimeIndex | None = None
        for frame in prices.values():
            if frame is None or frame.empty:
                continue
            idx = frame.index if idx is None else idx.union(frame.index)
        if idx is None:
            return []
        if start is not None:
            idx = idx[idx >= pd.Timestamp(start)]
        if end is not None:
            idx = idx[idx <= pd.Timestamp(end)]
        return list(idx)

    def _index_signals(
        self,
        features: Mapping[str, pd.DataFrame],
        calendar: Sequence[pd.Timestamp],
    ) -> dict[pd.Timestamp, list[_PendingEntry]]:
        """Bucket every historical signal by its signal date."""
        wanted = set(calendar)
        out: dict[pd.Timestamp, list[_PendingEntry]] = {}
        for symbol, feats in features.items():
            if "signal" not in feats:
                continue
            hits = feats.index[feats["signal"].fillna(False).astype(bool)]
            for ts in hits:
                if ts not in wanted:
                    continue
                row = feats.loc[ts]
                meta = {
                    "deliv_pct": _num(row.get("deliv_pct"), float("nan")),
                    "deliv_ratio": _num(row.get("deliv_ratio"), float("nan")),
                    "deliv_spike_ratio": _num(row.get("deliv_spike_ratio"), float("nan")),
                    "deliv_spike_age": _num(row.get("deliv_spike_age"), float("nan")),
                    "vol_ratio": _num(row.get("vol_ratio"), float("nan")),
                    "range_position": _num(row.get("range_position"), float("nan")),
                    "rs_excess": _num(row.get("rs_excess"), float("nan")),
                    "atr_ratio": _num(row.get("atr_ratio"), float("nan")),
                }
                out.setdefault(ts, []).append(
                    _PendingEntry(
                        symbol=symbol,
                        signal_date=ts,
                        signal_low=float(row["low"]),
                        atr_stop=_num(row.get("atr_stop"), float("nan")),
                        rank=_num(
                            row.get("rank_score"),
                            _num(row.get("deliv_spike_ratio"), _num(row.get("deliv_ratio"))),
                        ),
                        turnover_cr=_num(row.get("turnover_cr")),
                        meta=meta,
                    )
                )
        return out

    @staticmethod
    def _prioritise(pending: Sequence[_PendingEntry]) -> list[_PendingEntry]:
        """Rank same-day candidates when capacity is scarce.

        Highest conviction first, where conviction is whatever the strategy says
        it is: a feature frame may publish an explicit ``rank_score`` column, and
        otherwise the delivery spike ratio is used.

        This matters more than it looks. The strategies here generate several
        times more signals than the position limit can take -- on some days
        hundreds -- so the ranking, not the entry rule, decides most of what is
        actually traded. A missing rank column silently degrades to sorting by
        symbol name, which quietly turns the whole backtest into a test of the
        alphabet.
        """
        return sorted(pending, key=lambda e: (-_num(e.rank), e.symbol))

    @staticmethod
    def _bar(
        prices: Mapping[str, pd.DataFrame], symbol: str, day: pd.Timestamp
    ) -> pd.Series | None:
        """The OHLCV row for a symbol on a day, or None if it did not trade."""
        frame = prices.get(symbol)
        if frame is None or frame.empty:
            return None
        try:
            return frame.loc[day]
        except KeyError:
            return None

    @staticmethod
    def _build_close_lookup(
        prices: Mapping[str, pd.DataFrame],
        calendar: Sequence[pd.Timestamp],
    ) -> dict[str, Any]:
        """Pre-compute an O(1) last-known-close lookup keyed by calendar row.

        Each symbol's closes are reindexed onto the shared calendar and
        forward-filled, so a symbol that did not trade on a given day carries its
        last traded price rather than a gap. Returns the row map alongside the
        arrays so callers can turn a date into an index once per bar.
        """
        index = pd.DatetimeIndex(calendar)
        row_of = {stamp: i for i, stamp in enumerate(index)}
        arrays: dict[str, np.ndarray] = {}
        for symbol, frame in prices.items():
            if frame is None or frame.empty:
                continue
            series = frame["close"].reindex(index).ffill()
            arrays[symbol] = series.to_numpy(dtype=float)
        return {"rows": row_of, "closes": arrays}

    @staticmethod
    def _open_value(
        positions: Mapping[str, Position],
        close_lookup: Mapping[str, Any],
        day: pd.Timestamp,
    ) -> float:
        """Mark-to-market value of all open positions on a given day."""
        row = close_lookup["rows"].get(day)
        closes = close_lookup["closes"]
        total = 0.0
        for symbol, pos in positions.items():
            array = closes.get(symbol)
            if array is None or row is None:
                total += pos.notional
                continue
            price = array[row]
            total += pos.qty * (price if np.isfinite(price) else pos.entry_price)
        return total

    def _evaluate_exit(
        self,
        pos: Position,
        bar: pd.Series,
        prices: Mapping[str, pd.DataFrame],
        symbol: str,
        today: pd.Timestamp,
        *,
        exit_now: bool = False,
    ) -> tuple[float | None, str]:
        """Decide whether and at what price a position exits on this bar.

        The stop used is the one set at the end of the previous bar, so no
        decision on this bar can depend on this bar's own trailing update.

        Returns:
            ``(exit_price, reason)``, or ``(None, "")`` to stay in the trade.
        """
        cfg = self.cfg
        risk, execp = cfg.risk, cfg.execution
        open_, high, low, close = (
            float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        )

        # A circuit-locked exit deferred from yesterday takes priority: we are
        # trapped and get out at whatever this bar offers.
        if pos.deferred_exit is not None:
            reason = pos.deferred_exit
            pos.deferred_exit = None
            fill = open_ * (1 - execp.locked_circuit_penalty_pct)
            return fill, reason

        prev_close = self._prev_close(prices, symbol, today, fallback=open_)

        if low <= pos.stop:
            band_floor = prev_close * (1 - execp.circuit_band_pct)
            bar_range = (high - low) / prev_close if prev_close > 0 else 1.0
            locked = open_ <= band_floor * 1.002 and bar_range < 0.005

            if locked:
                # No bid on the other side: cannot exit today at any price.
                pos.deferred_exit = "circuit_locked_stop"
                return None, ""
            if open_ <= pos.stop:
                # Gapped straight through: the fill is the open, not the stop.
                fill = open_ * (1 - execp.gap_through_extra_slippage_bps / 10_000)
                return fill, "gap_through_stop"
            fill = pos.stop * (1 - execp.slippage_bps / 10_000)
            return fill, "trailing_stop" if pos.trail_active else "stop"

        if risk.target_r_multiple is not None:
            target = pos.entry_price + risk.target_r_multiple * pos.initial_risk_per_share
            if high >= target:
                return target * (1 - execp.slippage_bps / 10_000), "target"

        # The strategy's own exit rule, checked after the protective stop but
        # before the max-hold clock, so a reversion exit is not pre-empted.
        if exit_now and pos.bars_held >= risk.min_hold_days:
            return close * (1 - execp.slippage_bps / 10_000), "signal_exit"

        if pos.bars_held >= risk.max_hold_days:
            return close * (1 - execp.slippage_bps / 10_000), "max_hold"

        return None, ""

    @staticmethod
    def _prev_close(
        prices: Mapping[str, pd.DataFrame],
        symbol: str,
        day: pd.Timestamp,
        *,
        fallback: float,
    ) -> float:
        """Previous session's close, used as the circuit-band reference."""
        frame = prices.get(symbol)
        if frame is None or frame.empty:
            return fallback
        window = frame.loc[frame.index < day]
        return float(window["close"].iloc[-1]) if not window.empty else fallback

    def _update_trail(
        self,
        pos: Position,
        bar: pd.Series,
        swings: pd.Series | None,
        today: pd.Timestamp,
    ) -> None:
        """Ratchet the stop up to the latest confirmed swing low.

        Trailing only activates once the trade is up ``trail_activate_r``, so a
        trade is given room to work before the stop starts tightening. The stop
        only ever moves up.
        """
        risk = self.cfg.risk
        if pos.initial_risk_per_share <= 0:
            return
        gain_r = (float(bar["close"]) - pos.entry_price) / pos.initial_risk_per_share
        if gain_r >= risk.trail_activate_r:
            pos.trail_active = True
        if not pos.trail_active or swings is None:
            return
        try:
            level = float(swings.loc[today])
        except (KeyError, TypeError, ValueError):
            return
        if np.isfinite(level) and level > pos.stop:
            pos.stop = level

    def _sell(self, price: float, qty: int) -> tuple[float, float]:
        """Proceeds and charges for a sale.

        Charges come from the configured cost model, so a delivery trade is
        debited STT, stamp-free sell-side duty, exchange and SEBI fees, GST and
        the flat depository charge -- not just a notional brokerage percentage.
        """
        execp = self.cfg.execution
        gross = price * qty
        costs = exit_charges(gross, execp).total
        return gross - costs, costs

    @staticmethod
    def _close(
        pos: Position,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_costs: float,
        *,
        reason: str | None = None,
    ) -> Trade:
        """Build the completed :class:`Trade` record for a position."""
        gross = (exit_price - pos.entry_price) * pos.qty
        costs = pos.entry_costs + exit_costs
        net = gross - costs
        risk_amt = pos.initial_risk
        r_mult = net / risk_amt if risk_amt > 0 else 0.0
        rps = pos.initial_risk_per_share
        mae_r = (pos.lowest_low - pos.entry_price) / rps if rps > 0 else 0.0
        mfe_r = (pos.highest_high - pos.entry_price) / rps if rps > 0 else 0.0
        return Trade(
            symbol=pos.symbol,
            entry_date=pos.entry_date,
            exit_date=exit_date,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.qty,
            gross_pnl=gross,
            costs=costs,
            net_pnl=net,
            r_multiple=r_mult,
            exit_reason=reason or "end_of_data",
            bars_held=pos.bars_held,
            initial_stop=pos.initial_stop,
            initial_risk_per_share=rps,
            mae_r=mae_r,
            mfe_r=mfe_r,
            signal_meta=pos.signal_meta,
        )

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            trades=[],
            equity_curve=pd.DataFrame(columns=["equity", "cash", "positions_value"]),
            stats={},
            rejected={},
        )


__all__ = ["BacktestResult", "PortfolioBacktester", "Position", "Trade"]
