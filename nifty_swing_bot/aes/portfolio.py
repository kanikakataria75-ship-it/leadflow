"""AES portfolio backtest: exits (spec section 6), sizing (section 7), regime (section 8).

Phase 4 measured bar-level edge on a *fixed* forward horizon (5/10/15/20
bars from entry) -- deliberately simple, so a signal's own quality could be
judged without the exit machinery also being in play. This module is the
opposite: given the same signals, it simulates what actually happens to a
position under AES's own exit rules, with a shared account, a hard cap on
concurrent positions, and every rupee cost the ``backtest.costs`` module
already prices for a delivery trade.

**Why this is a new engine rather than the existing ``PortfolioBacktester``.**
That engine's exit check is intraday by construction (``if low <= pos.stop``)
and closes a position in one shot -- neither is compatible with what section
6 asks for: a closing-basis stop ("intraday wicks must not trigger") and a
staged ladder that sells part of a position at a time while the remainder
keeps running. Reusing its bar-by-bar architecture, its cost model, and its
circuit-aware fill logic verbatim would have meant silently reinterpreting
"closing basis" as "the same intraday check with a different number," which
is exactly the kind of drift this project has caught itself in before. What
*is* reused directly: ``backtest.costs`` for every charge, and
``backtest.metrics.compute_stats``/``alpha_beta`` for every statistic --
``AESTrade`` is shaped to satisfy the same duck-typed ``Trade`` interface
``compute_stats`` already expects, rather than a second copy of win-rate,
Sharpe, and drawdown arithmetic existing alongside the first.

**Two ranges the spec gives, not numbers -- resolved once, in
``PortfolioParams``, not here.** The profit ladder ("50% at 5-8%") and the
time stop ("15-20 sessions") are ranges in the spec; ``PortfolioParams``
fixes each at its lower bound, documented there. Nothing in this module
re-decides that.
"""

from __future__ import annotations

import warnings

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.costs import entry_charges, exit_charges
from ..strategy.indicators import atr
from ..backtest.metrics import alpha_beta, compute_stats
from ..config import ExecutionParams, get_config
from .params import PortfolioParams
from .reentry import STOP_REASONS, EntryOrder, ReentryParams, ReentryWatch, volume_ok
from .signals import Signal

logger = logging.getLogger(__name__)

BUCKET_ORDER = {"reject": 0, "wait_and_watch": 1, "high_conviction": 2}


@dataclass(slots=True)
class AESPosition:
    """An open AES position, partially exited as the profit ladder fires."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    initial_qty: int
    remaining_qty: int
    big_box_bottom: float
    entry_mode: int
    score: float
    bucket: str
    entry_costs: float
    bars_held: int = 0
    ladder_stage: int = 0     # how many of the 3 profit-ladder stages have fired
    peak_close: float = 0.0   # highest close seen since entry, for the ATR trailing stop
    signal_meta: dict[str, float] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.remaining_qty * self.entry_price

    @property
    def initial_risk_per_share(self) -> float:
        """Structural stop distance -- entry to the big-box floor -- used
        only to express P&L as an R-multiple for reporting, matching the
        existing engine's Trade shape. AES does not risk-size off this
        distance the way the ATR-stop engine does (see ``PortfolioParams``)."""
        return max(self.entry_price - self.big_box_bottom, 0.01)


@dataclass(slots=True)
class AESTrade:
    """One partial or full exit, shaped to satisfy ``compute_stats``'s Trade duck-type."""

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
    entry_mode: int
    score: float
    bucket: str

    def as_dict(self) -> dict[str, object]:
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
            "r_multiple": round(self.r_multiple, 3),
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
            "entry_mode": self.entry_mode,
            "score": round(self.score, 3),
            "bucket": self.bucket,
        }


@dataclass(slots=True)
class AESBacktestResult:
    trades: list[AESTrade]
    equity_curve: pd.DataFrame
    stats: dict[str, object]
    rejected: dict[str, int]
    #: Phase 23. One row per re-entry actually filled, so a study can match
    #: re-entries back to their trades and report their hit rate separately
    #: from the population they sit inside. Empty unless re-entry is enabled.
    reentries: list[dict] = field(default_factory=list)

    @property
    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.as_dict() for t in self.trades])


def _bucket_size_mult(bucket: str, p: PortfolioParams) -> float:
    return {
        "reject": p.size_mult_reject,
        "wait_and_watch": p.size_mult_wait_and_watch,
        "high_conviction": p.size_mult_high_conviction,
    }.get(bucket, 0.0)


def _risk_appetite(trailing_pnl_pct: float, p: PortfolioParams) -> float:
    """Linear map from trailing realised P&L (% of starting capital) to a
    size multiplier, saturating at ``risk_appetite_saturation_pct`` either way."""
    span = p.risk_appetite_max - p.risk_appetite_min
    frac = (trailing_pnl_pct + p.risk_appetite_saturation_pct) / (2 * p.risk_appetite_saturation_pct)
    frac = min(1.0, max(0.0, frac))
    return p.risk_appetite_min + span * frac


class AESPortfolioBacktester:
    """Simulates AES entries (already found by ``watchlist.run_watchlist_for_symbol``)
    under section 6/7/8's exit, sizing and regime rules, one shared account."""

    def __init__(
        self,
        params: PortfolioParams | None = None,
        execp: ExecutionParams | None = None,
        reentry: ReentryParams | None = None,
    ) -> None:
        self.params = params or PortfolioParams()
        self.execp = execp or ExecutionParams()
        #: Phase 23. Disabled by default, so an unmodified call site is
        #: bit-identical to the pre-Phase-23 engine.
        self.reentry = reentry or ReentryParams()

    def run(
        self,
        frames: dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
        signals: list[Signal],
        *,
        min_bucket: str = "wait_and_watch",
    ) -> AESBacktestResult:
        """Run the simulation.

        Args:
            frames: symbol -> OHLCV frame (must include every symbol any
                signal references).
            index_df: Benchmark OHLCV, for the section 8 regime series and
                the alpha/beta regression.
            signals: Already-scored ``Signal`` objects (``signal.score`` /
                ``signal.bucket`` set -- see ``scoring.score_signal``).
            min_bucket: Signals scoring below this are not queued at all --
                "Reject -> hard-delete or below threshold" (section 4).
                Defaults to excluding ``reject`` only.

        Returns:
            Trades, equity curve, and the full ``compute_stats`` bundle plus
            the alpha/beta regression and time-in-market.
        """
        p = self.params
        if p.atr_stop_mult is None or p.max_hold_bars is None:
            # The silent trap in aes.locked: these defaults are the spec
            # SMA(10) stop, uncapped, and they reproduce §9.2's *spec* column
            # rather than §9.1's walk-forward. Legitimate variant work passes
            # them deliberately, so this warns rather than raises -- but it is
            # never silent.
            warnings.warn(
                "running the SPEC stop, not the locked exit: "
                f"atr_stop_mult={p.atr_stop_mult!r}, max_hold_bars={p.max_hold_bars!r}. "
                "The locked configuration is ATR(14) x 2.5 with a hard 25-bar cap "
                "(aes.locked.LOCKED_PORTFOLIO_PARAMS, PROVEN #1). Pass it explicitly "
                "if you meant the locked config; this reproduces RESEARCH_AES.md "
                "§9.2's spec column otherwise.",
                RuntimeWarning,
                stacklevel=2,
            )
        min_rank = BUCKET_ORDER[min_bucket]
        eligible = [s for s in signals if BUCKET_ORDER.get(s.bucket, 0) >= min_rank]
        eligible.sort(key=lambda s: s.entry_date)

        if not eligible:
            return self._empty_result()

        calendar = self._build_calendar(frames, eligible)
        if len(calendar) < 2:
            return self._empty_result()

        by_entry_date: dict[pd.Timestamp, list[Signal]] = {}
        for s in eligible:
            by_entry_date.setdefault(s.entry_date, []).append(s)

        sma_lookup = {
            sym: frames[sym]["close"].rolling(p.sma_stop_period).mean()
            for sym in {s.symbol for s in eligible} if sym in frames
        }
        atr_lookup: dict[str, pd.Series] = {}
        if p.atr_stop_mult is not None:
            atr_lookup = {
                sym: atr(frames[sym]["high"], frames[sym]["low"], frames[sym]["close"], p.atr_stop_period)
                for sym in {s.symbol for s in eligible} if sym in frames
            }
        # Phase 23: volume relative to its own 20-day average, precomputed per
        # symbol. Built only when re-entry is enabled so a baseline run does no
        # extra work and cannot be perturbed by it.
        rp = self.reentry
        vol_ratio_lookup: dict[str, pd.Series] = {}
        if rp.enabled and rp.vol_mult is not None:
            vol_ratio_lookup = {
                sym: frames[sym]["volume"]
                / frames[sym]["volume"].rolling(rp.vol_period).mean()
                for sym in {s.symbol for s in eligible} if sym in frames
            }
        reentry_watch: dict[str, ReentryWatch] = {}
        pending_reentries: list[EntryOrder] = []
        reentry_log: list[dict] = []

        regime_weak = self._regime_series(index_df, calendar, p.regime_sma_period)

        cash = p.starting_capital
        positions: dict[str, AESPosition] = {}
        trades: list[AESTrade] = []
        equity_rows: list[dict[str, object]] = []
        realised: list[tuple[pd.Timestamp, float]] = []
        rejected = {"no_capacity": 0, "already_open": 0, "no_bar": 0, "zero_qty": 0, "no_cash": 0, "regime_or_appetite_restricted": 0}

        close_lookup = self._build_close_lookup(frames, calendar)

        for day_idx, today in enumerate(calendar):
            equity_now = cash + self._open_value(positions, close_lookup, today)
            weak_today = bool(regime_weak.get(today, False))
            trailing_pnl_pct = self._trailing_realised_pct(realised, today, p)
            appetite = _risk_appetite(trailing_pnl_pct, p)
            restrict_to_high_conviction = weak_today or appetite < p.risk_appetite_restrict_below

            # -- 1. fill today's entries, at today's open ------------------- #
            # Normal signals and any re-entries queued by yesterday's close go
            # through ONE list and one code path, so the slot, regime,
            # appetite, sizing and cash rules cannot drift apart between them.
            # Re-entries are appended rather than prepended and the whole list
            # is sorted by score, so a re-entry never jumps a queue ahead of a
            # fresh signal that scored higher.
            candidates: list[EntryOrder] = [
                EntryOrder(
                    symbol=s.symbol, score=s.score, bucket=s.bucket,
                    big_box_bottom=s.nested.big.bottom, entry_mode=s.entry_mode,
                )
                for s in by_entry_date.get(today, [])
            ]
            candidates.extend(pending_reentries)
            pending_reentries = []
            candidates.sort(key=lambda o: -o.score)
            for sig in candidates:
                if len(positions) >= p.max_open_positions:
                    rejected["no_capacity"] += 1
                    continue
                if sig.symbol in positions:
                    rejected["already_open"] += 1
                    continue
                if restrict_to_high_conviction and sig.bucket != "high_conviction":
                    rejected["regime_or_appetite_restricted"] += 1
                    continue
                bar = self._bar(frames, sig.symbol, today)
                if bar is None:
                    rejected["no_bar"] += 1
                    continue

                fill = float(bar["open"]) * (1 + self.execp.slippage_bps / 10_000)
                size_mult = _bucket_size_mult(sig.bucket, p) * appetite * (p.regime_weak_size_mult if weak_today else 1.0)
                base_pct = p.base_position_pct_of_equity or (1.0 / p.target_concurrent_positions)
                cap_notional = equity_now * base_pct * size_mult

                if p.size_by_risk:
                    sma = sma_lookup.get(sig.symbol)
                    # Yesterday's SMA(10) -- known before today's open, so
                    # sizing a fill at that open never reaches into today's
                    # own close. Matches the exit rule's "whichever binds
                    # first" logic: the stop that actually fires first is
                    # the *higher* of the two levels as price falls.
                    prior_sma = None
                    if sma is not None:
                        prior = sma.shift(1)
                        if today in prior.index and np.isfinite(prior.loc[today]):
                            prior_sma = float(prior.loc[today])
                    if p.atr_stop_mult is not None:
                        a = atr_lookup.get(sig.symbol)
                        prior_atr = None
                        if a is not None:
                            pa = a.shift(1)
                            if today in pa.index and np.isfinite(pa.loc[today]):
                                prior_atr = float(pa.loc[today])
                        atr_level = fill - p.atr_stop_mult * prior_atr if prior_atr else None
                        stop_ref = max(v for v in (atr_level, sig.big_box_bottom) if v is not None)
                    else:
                        buffered = prior_sma * (1 - p.sma_stop_buffer_pct) if prior_sma is not None else None
                        stop_ref = max(v for v in (buffered, sig.big_box_bottom) if v is not None)
                    risk_per_share = max(fill - stop_ref, fill * p.min_risk_per_share_pct)
                    risk_budget = equity_now * p.risk_per_trade_pct * size_mult
                    notional = min(risk_budget / risk_per_share * fill, cap_notional)
                else:
                    notional = cap_notional
                qty = int(notional // fill)
                if qty <= 0:
                    rejected["zero_qty"] += 1
                    continue
                cost = entry_charges(qty * fill, self.execp).total
                if qty * fill + cost > cash:
                    qty = int(cash // fill)
                    cost = entry_charges(qty * fill, self.execp).total if qty > 0 else 0.0
                if qty <= 0 or qty * fill + cost > cash:
                    rejected["no_cash"] += 1
                    continue

                cash -= qty * fill + cost
                positions[sig.symbol] = AESPosition(
                    symbol=sig.symbol, entry_date=today, entry_price=fill,
                    initial_qty=qty, remaining_qty=qty,
                    big_box_bottom=sig.big_box_bottom, entry_mode=sig.entry_mode,
                    score=sig.score, bucket=sig.bucket, entry_costs=cost,
                )
                if sig.is_reentry:
                    w = reentry_watch.get(sig.symbol)
                    if w is not None:
                        w.used += 1
                    reentry_log.append({
                        "symbol": sig.symbol, "entry_date": today,
                        "stop_level": sig.stop_level, "entry_price": fill,
                        "n_in_cycle": w.used if w is not None else 1,
                    })

            # -- 2. evaluate exits on today's bar (closing basis only) ------ #
            for symbol in list(positions):
                pos = positions[symbol]
                bar = self._bar(frames, symbol, today)
                if bar is None:
                    continue
                if pos.entry_date != today:
                    pos.bars_held += 1
                close = float(bar["close"])

                sma = sma_lookup.get(symbol)
                sma_today = float(sma.loc[today]) if sma is not None and today in sma.index and np.isfinite(sma.loc[today]) else None

                pos.peak_close = max(pos.peak_close, close)
                if p.atr_stop_mult is not None:
                    a = atr_lookup.get(symbol)
                    atr_today = (
                        float(a.loc[today])
                        if a is not None and today in a.index and np.isfinite(a.loc[today])
                        else None
                    )
                    trail_level = (
                        pos.peak_close - p.atr_stop_mult * atr_today
                        if atr_today is not None else None
                    )
                    trail_hit = trail_level is not None and close < trail_level
                    stopped = trail_hit or (close < pos.big_box_bottom)
                    reason = "atr_trail_stop" if trail_hit else "big_box_bottom_stop"
                    breached_level = trail_level if trail_hit else pos.big_box_bottom
                else:
                    sma_level = sma_today * (1 - p.sma_stop_buffer_pct) if sma_today is not None else None
                    sma_hit = sma_level is not None and close < sma_level
                    stopped = sma_hit or (close < pos.big_box_bottom)
                    reason = "sma10_close_stop" if sma_hit else "big_box_bottom_stop"
                    breached_level = sma_level if sma_hit else pos.big_box_bottom
                if stopped:
                    trade = self._exit_all(pos, today, close, reason)
                    trades.append(trade)
                    cash += trade.qty * trade.exit_price - trade.costs
                    realised.append((today, trade.net_pnl))
                    # Phase 23: arm the re-entry watch on the level that
                    # actually fired, not on the entry price. The thesis is
                    # "the stop was wrong", so the level to reclaim is the one
                    # that stopped it out.
                    if (rp.enabled and reason in STOP_REASONS
                            and breached_level is not None
                            and rp.eligible(pos.bucket, pos.bars_held)):
                        prior = reentry_watch.get(symbol)
                        used = prior.used if prior is not None else 0
                        if used < rp.max_reentries:
                            reentry_watch[symbol] = ReentryWatch(
                                symbol=symbol,
                                stop_level=float(breached_level),
                                expires_idx=day_idx + rp.watch_bars,
                                big_box_bottom=pos.big_box_bottom,
                                entry_mode=pos.entry_mode,
                                score=pos.score,
                                bucket=pos.bucket,
                                used=used,
                            )
                        else:
                            reentry_watch.pop(symbol, None)
                    del positions[symbol]
                    continue

                ret = close / pos.entry_price - 1.0
                for stage, (trigger, frac) in enumerate(
                    (
                        (p.ladder1_trigger_pct, p.ladder1_fraction),
                        (p.ladder2_trigger_pct, p.ladder2_fraction),
                        (p.ladder3_trigger_pct, p.ladder3_fraction),
                    ),
                    start=1,
                ):
                    if frac <= 0:
                        # A disabled stage sells nothing. Without this the
                        # max(1, ...) below would sell a single token share.
                        continue
                    if pos.ladder_stage < stage and ret >= trigger and pos.remaining_qty > 0:
                        sell_qty = min(pos.remaining_qty, max(1, int(round(pos.initial_qty * frac))))
                        trade = self._exit_partial(pos, today, close, sell_qty, f"ladder{stage}")
                        trades.append(trade)
                        cash += trade.qty * trade.exit_price - trade.costs
                        realised.append((today, trade.net_pnl))
                        pos.ladder_stage = stage

                if pos.remaining_qty <= 0:
                    del positions[symbol]
                    continue

                if pos.bars_held >= p.time_stop_bars and 0.0 < ret < p.time_stop_profit_ceiling:
                    trade = self._exit_all(pos, today, close, "time_stop")
                    trades.append(trade)
                    cash += trade.qty * trade.exit_price - trade.costs
                    realised.append((today, trade.net_pnl))
                    del positions[symbol]
                    continue

                # Hard maximum hold: unconditional, unlike the time stop
                # above, which only fires on a stagnant *modest-profit*
                # position and so lets a strong trend run without limit.
                if p.max_hold_bars is not None and pos.bars_held >= p.max_hold_bars:
                    trade = self._exit_all(pos, today, close, "max_hold")
                    trades.append(trade)
                    cash += trade.qty * trade.exit_price - trade.costs
                    realised.append((today, trade.net_pnl))
                    del positions[symbol]

            # -- 3. Phase 23: evaluate re-entry watches on today's CLOSE, and
            # queue any that trigger for tomorrow's OPEN. Evaluating and
            # filling on the same bar would be lookahead; this matches the
            # engine's existing trigger-then-fill-next-open convention.
            if rp.enabled and reentry_watch:
                for symbol in list(reentry_watch):
                    w = reentry_watch[symbol]
                    if day_idx > w.expires_idx or w.used >= rp.max_reentries:
                        del reentry_watch[symbol]
                        continue
                    if symbol in positions:
                        continue          # already back in; nothing to re-enter
                    bar = self._bar(frames, symbol, today)
                    if bar is None:
                        continue
                    if float(bar["close"]) <= w.stop_level:
                        continue
                    if not volume_ok(vol_ratio_lookup.get(symbol), today, rp):
                        continue
                    pending_reentries.append(EntryOrder(
                        symbol=symbol, score=w.score, bucket=w.bucket,
                        big_box_bottom=w.big_box_bottom, entry_mode=w.entry_mode,
                        is_reentry=True, stop_level=w.stop_level,
                    ))

            open_val = self._open_value(positions, close_lookup, today)
            equity_rows.append({
                "date": today, "equity": cash + open_val, "cash": cash,
                "positions_value": open_val, "open_positions": len(positions),
                "exposure": (open_val / (cash + open_val)) if (cash + open_val) > 0 else 0.0,
            })

        # -- liquidate anything left open at the final close ----------------- #
        last = calendar[-1]
        for symbol, pos in list(positions.items()):
            bar = self._bar(frames, symbol, last)
            price = float(bar["close"]) if bar is not None else pos.entry_price
            trade = self._exit_all(pos, last, price, "end_of_data")
            trades.append(trade)
            cash += trade.qty * trade.exit_price - trade.costs
            del positions[symbol]
        if equity_rows:
            equity_rows[-1].update(equity=cash, cash=cash, positions_value=0.0, exposure=0.0)

        equity = pd.DataFrame(equity_rows).set_index("date")
        # compute_stats reads its starting-capital reference from cfg.risk,
        # not from a parameter -- align it with this run's own capital so
        # total_return_pct is not silently computed against a different base.
        cfg = get_config().model_copy(
            update={"risk": get_config().risk.model_copy(update={"starting_capital": p.starting_capital})}
        )
        stats = compute_stats(trades, equity, cfg=cfg, benchmark=index_df)
        stats.update(self._extra_stats(equity, index_df))

        return AESBacktestResult(
            trades=trades, equity_curve=equity, stats=stats,
            rejected=rejected, reentries=reentry_log,
        )

    # -------------------------------------------------------------- helpers --
    def _exit_all(self, pos: AESPosition, date: pd.Timestamp, price: float, reason: str) -> AESTrade:
        return self._exit_partial(pos, date, price, pos.remaining_qty, reason)

    def _exit_partial(
        self, pos: AESPosition, date: pd.Timestamp, price: float, qty: int, reason: str
    ) -> AESTrade:
        fill = price * (1 - self.execp.slippage_bps / 10_000)
        gross_notional = qty * fill
        costs = exit_charges(gross_notional, self.execp).total
        entry_cost_share = pos.entry_costs * (qty / pos.initial_qty) if pos.initial_qty else 0.0
        gross = (fill - pos.entry_price) * qty
        net = gross - costs - entry_cost_share
        r_mult = net / (pos.initial_risk_per_share * qty) if pos.initial_risk_per_share > 0 else 0.0
        pos.remaining_qty -= qty
        return AESTrade(
            symbol=pos.symbol, entry_date=pos.entry_date, exit_date=date,
            entry_price=pos.entry_price, exit_price=fill, qty=qty,
            gross_pnl=gross, costs=costs + entry_cost_share, net_pnl=net,
            r_multiple=r_mult, exit_reason=reason, bars_held=pos.bars_held,
            entry_mode=pos.entry_mode, score=pos.score, bucket=pos.bucket,
        )

    @staticmethod
    def _trailing_realised_pct(
        realised: list[tuple[pd.Timestamp, float]], today: pd.Timestamp, p: PortfolioParams
    ) -> float:
        if not realised:
            return 0.0
        cutoff = today - pd.Timedelta(days=int(p.risk_appetite_lookback_days * 1.5))  # calendar-day buffer for trading days
        window_sum = sum(pnl for d, pnl in realised if d > cutoff and d <= today)
        return window_sum / p.starting_capital if p.starting_capital else 0.0

    @staticmethod
    def _regime_series(
        index_df: pd.DataFrame, calendar: list[pd.Timestamp], sma_period: int
    ) -> dict[pd.Timestamp, bool]:
        """True on days the index closes below its own SMA -- a "weak" regime."""
        close = index_df["close"]
        sma = close.rolling(sma_period).mean()
        weak = (close < sma).reindex(pd.DatetimeIndex(calendar)).ffill().fillna(False)
        return weak.to_dict()

    @staticmethod
    def _build_calendar(frames: dict[str, pd.DataFrame], signals: list[Signal]) -> list[pd.Timestamp]:
        symbols = {s.symbol for s in signals}
        idx: pd.DatetimeIndex | None = None
        for sym in symbols:
            frame = frames.get(sym)
            if frame is None or frame.empty:
                continue
            idx = frame.index if idx is None else idx.union(frame.index)
        if idx is None:
            return []
        start = min(s.entry_date for s in signals)
        idx = idx[idx >= start]
        return list(idx)

    @staticmethod
    def _bar(frames: dict[str, pd.DataFrame], symbol: str, day: pd.Timestamp) -> pd.Series | None:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            return None
        try:
            return frame.loc[day]
        except KeyError:
            return None

    @staticmethod
    def _build_close_lookup(frames: dict[str, pd.DataFrame], calendar: list[pd.Timestamp]) -> dict:
        index = pd.DatetimeIndex(calendar)
        row_of = {stamp: i for i, stamp in enumerate(index)}
        arrays = {}
        symbols = set()
        for sym, frame in frames.items():
            if frame is None or frame.empty:
                continue
            symbols.add(sym)
        for sym in symbols:
            arrays[sym] = frames[sym]["close"].reindex(index).ffill().to_numpy(dtype=float)
        return {"rows": row_of, "closes": arrays}

    @staticmethod
    def _open_value(positions: dict[str, AESPosition], close_lookup: dict, day: pd.Timestamp) -> float:
        row = close_lookup["rows"].get(day)
        closes = close_lookup["closes"]
        total = 0.0
        for symbol, pos in positions.items():
            array = closes.get(symbol)
            if array is None or row is None:
                total += pos.notional
                continue
            price = array[row]
            total += pos.remaining_qty * (price if np.isfinite(price) else pos.entry_price)
        return total

    def _extra_stats(self, equity: pd.DataFrame, index_df: pd.DataFrame) -> dict[str, object]:
        returns = equity["equity"].astype(float).pct_change().dropna()
        bench_close = index_df["close"].reindex(equity.index).ffill()
        bench_returns = bench_close.pct_change().dropna()
        annual_alpha, beta, n = alpha_beta(returns, bench_returns, periods=252)
        time_in_market_pct = float(equity.get("exposure", pd.Series(dtype=float)).gt(0).mean() * 100) if "exposure" in equity else 0.0
        return {
            "alpha_annual_pct": round(annual_alpha * 100, 2),
            "beta": round(beta, 3),
            "alpha_beta_n_days": n,
            "time_in_market_pct": round(time_in_market_pct, 2),
        }

    def _empty_result(self) -> AESBacktestResult:
        return AESBacktestResult(
            trades=[], equity_curve=pd.DataFrame(columns=["equity", "cash", "positions_value"]),
            stats={}, rejected={},
        )


__all__ = ["AESBacktestResult", "AESPortfolioBacktester", "AESPosition", "AESTrade"]
