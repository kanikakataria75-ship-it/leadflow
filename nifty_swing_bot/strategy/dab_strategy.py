"""Delivery Accumulation Breakout (DAB) signal engine.

The thesis: a stock that suddenly trades far more shares than usual, *and*
where an unusually large fraction of those shares are actually taken to
delivery, is being accumulated by someone with a multi-day horizon rather than
churned by intraday traders. Layer a strong close, relative strength and a
breakout/volatility-contraction setup on top and you have a short-horizon swing
entry with a defined invalidation point.

A signal fires on bar T only when **all five** conditions are true on that bar:

===  ===========================================================================
1    Delivery% spike:  delivery% > ``delivery_spike_mult`` x its 20-bar baseline
2    Volume surge:     volume > ``volume_surge_mult`` x its 20-bar baseline
3    Strong close:     close in the top ``close_range_top_pct`` of the day range
4    Relative strength: stock 10-bar return > benchmark 10-bar return
5    Breakout / VCP:   close > prior 20-bar high, OR within
                       ``breakout_proximity_pct`` of it AND ATR(5) <
                       ``vcp_contraction_ratio`` x ATR(20)
===  ===========================================================================

Every intermediate value is kept as a column on the output frame, not just the
final boolean. The backtester needs them for sizing, and the LLM insight layer
needs them to explain *why* a signal fired in plain English.

Entries are assumed to fill on the **next** bar's open: the rules are evaluated
on the close of bar T, so bar T's close is the earliest a human could act, and
bar T+1's open is the earliest they could actually transact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import AppConfig, DABParams, RiskParams, get_config
from . import indicators as ind

logger = logging.getLogger(__name__)

#: Columns produced by :func:`compute_features`, in display order.
FEATURE_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume",
    "deliv_pct", "deliv_baseline", "deliv_ratio", "is_deliv_spike",
    "deliv_spike_ratio", "deliv_spike_age", "cond_delivery",
    "vol_baseline", "vol_ratio", "cond_volume",
    "range_position", "cond_close",
    "ret_stock", "ret_bench", "rs_excess", "cond_rs",
    "high_n", "dist_from_high", "atr_fast", "atr_slow", "atr_ratio",
    "is_breakout", "is_near_high", "is_contraction", "cond_setup",
    "atr_stop", "signal",
)

#: Human-readable labels for each rule, reused by the API and the LLM prompt.
RULE_LABELS: dict[str, str] = {
    "cond_delivery": "Delivery% spike",
    "cond_volume": "Volume surge",
    "cond_close": "Strong close",
    "cond_rs": "Relative strength",
    "cond_setup": "Breakout / VCP setup",
}


@dataclass(frozen=True, slots=True)
class SignalRow:
    """A single actionable signal with everything needed to place the trade."""

    date: pd.Timestamp
    symbol: str
    close: float
    entry_ref: float
    stop: float
    risk_per_share: float
    atr14: float
    deliv_pct: float
    deliv_ratio: float
    deliv_spike_ratio: float
    deliv_spike_age: float
    vol_ratio: float
    range_position: float
    rs_excess: float
    dist_from_high: float
    atr_ratio: float
    is_breakout: bool

    def as_dict(self) -> dict[str, object]:
        """Plain dict, JSON-friendly, for the API and the signal log."""
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "symbol": self.symbol,
            "close": round(self.close, 2),
            "entry_ref": round(self.entry_ref, 2),
            "stop": round(self.stop, 2),
            "risk_per_share": round(self.risk_per_share, 2),
            "stop_pct": round(self.risk_per_share / self.entry_ref * 100, 2)
            if self.entry_ref
            else None,
            "atr14": round(self.atr14, 2),
            "deliv_pct": round(self.deliv_pct, 2),
            "deliv_ratio": round(self.deliv_ratio, 2),
            "deliv_spike_ratio": round(self.deliv_spike_ratio, 2)
            if np.isfinite(self.deliv_spike_ratio) else None,
            "deliv_spike_age": int(self.deliv_spike_age)
            if np.isfinite(self.deliv_spike_age) else None,
            "vol_ratio": round(self.vol_ratio, 2),
            "range_position": round(self.range_position, 3),
            "rs_excess": round(self.rs_excess * 100, 2),
            "dist_from_high": round(self.dist_from_high * 100, 2),
            "atr_ratio": round(self.atr_ratio, 3),
            "is_breakout": bool(self.is_breakout),
        }


def prepare_input(
    ohlcv: pd.DataFrame,
    delivery: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join OHLCV with delivery data on date, keeping the price index intact.

    Delivery data comes from NSE bhavcopy and price data from Yahoo. The two
    agree on trading days but can disagree at the edges (a newly listed scrip,
    a suspended day). The price index wins, and missing delivery becomes NaN,
    which makes rule 1 evaluate False rather than silently passing.
    """
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()

    frame = ohlcv.copy()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    if delivery is not None and not delivery.empty:
        deliv = delivery.copy()
        deliv.index = pd.DatetimeIndex(deliv.index).tz_localize(None).normalize()
        deliv = deliv[~deliv.index.duplicated(keep="last")].sort_index()
        cols = [c for c in ("deliv_pct", "deliv_qty", "total_qty", "turnover_cr", "trades") if c in deliv.columns]
        frame = frame.join(deliv[cols], how="left")
    else:
        for col in ("deliv_pct", "deliv_qty", "total_qty", "turnover_cr", "trades"):
            frame[col] = np.nan

    return frame


def compute_features(
    ohlcv: pd.DataFrame,
    delivery: pd.DataFrame | None,
    benchmark: pd.DataFrame | pd.Series | None,
    *,
    params: DABParams | None = None,
    risk: RiskParams | None = None,
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    """Compute every DAB rule component for one symbol.

    Args:
        ohlcv: Daily bars with ``open/high/low/close/volume`` on a DatetimeIndex.
        delivery: Per-symbol delivery frame with a ``deliv_pct`` column, or None.
        benchmark: Benchmark price frame (uses ``close``) or a close Series.
            None disables the relative-strength rule by leaving it False.
        params: DAB parameters. Defaults to config.
        risk: Risk parameters, needed for the ATR stop column. Defaults to config.
        cfg: Config override.

    Returns:
        A frame indexed like ``ohlcv`` carrying :data:`FEATURE_COLUMNS`. Bars
        with insufficient history have NaN components and ``signal == False``.
    """
    cfg = cfg or get_config()
    params = params or cfg.dab
    risk = risk or cfg.risk

    frame = prepare_input(ohlcv, delivery)
    if frame.empty:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))

    excl = params.baseline_excludes_today
    high, low, close, volume = frame["high"], frame["low"], frame["close"], frame["volume"]

    # -- Rule 1: delivery accumulation -------------------------------------- #
    # Delivery% is deliverable qty / TOTAL traded qty. Demanding the spike on
    # the same bar as the volume surge is self-defeating: the surge inflates the
    # denominator, so delivery% mechanically falls on exactly the breakout bars
    # the other rules select for (measured correlation of log ratios: -0.26).
    # ``prior_window`` mode therefore looks for the accumulation BEFORE the
    # breakout, which is both tradeable and truer to the strategy's thesis.
    deliv = frame.get("deliv_pct", pd.Series(np.nan, index=frame.index)).astype(float)
    frame["deliv_pct"] = deliv
    frame["deliv_baseline"] = ind.rolling_baseline(
        deliv, params.delivery_lookback, exclude_today=excl
    )
    frame["deliv_ratio"] = ind.safe_ratio(deliv, frame["deliv_baseline"])

    is_spike = frame["deliv_ratio"] > params.delivery_spike_mult
    if params.min_delivery_pct > 0:
        is_spike &= deliv >= params.min_delivery_pct
    is_spike = is_spike.fillna(False)
    frame["is_deliv_spike"] = is_spike

    if params.delivery_mode == "same_day":
        frame["deliv_spike_ratio"] = frame["deliv_ratio"]
        frame["deliv_spike_age"] = 0.0
        frame["cond_delivery"] = is_spike
    else:
        window, min_bars = params.delivery_spike_window, params.delivery_spike_min_bars
        # Shift by one so the signal bar itself is never the accumulation bar.
        prior = frame["deliv_ratio"].shift(1)
        frame["deliv_spike_ratio"] = prior.rolling(window, min_periods=min_bars).max()
        frame["cond_delivery"] = (
            is_spike.shift(1).fillna(False).rolling(window, min_periods=1).max().astype(bool)
        )
        # How many bars ago the strongest accumulation happened. The LLM layer
        # uses this to say "delivery spiked to 2.1x four sessions ago".
        frame["deliv_spike_age"] = _bars_since_spike(is_spike, window)

    # -- End rule 1 --------------------------------------------------------- #

    # -- Rule 2: volume surge ----------------------------------------------- #
    frame["vol_baseline"] = ind.rolling_baseline(
        volume.astype(float), params.volume_lookback, exclude_today=excl
    )
    frame["vol_ratio"] = ind.safe_ratio(volume.astype(float), frame["vol_baseline"])
    frame["cond_volume"] = (frame["vol_ratio"] > params.volume_surge_mult).fillna(False)

    # -- Rule 3: strong close within the day's range ------------------------ #
    frame["range_position"] = ind.close_range_position(high, low, close)
    frame["cond_close"] = (frame["range_position"] >= 1.0 - params.close_range_top_pct).fillna(False)

    # -- Rule 4: relative strength versus the benchmark --------------------- #
    frame["ret_stock"] = ind.pct_return(close, params.rs_lookback)
    bench_close = _benchmark_close(benchmark)
    if bench_close is None:
        frame["ret_bench"] = np.nan
        frame["rs_excess"] = np.nan
        frame["cond_rs"] = False
        logger.debug("No benchmark supplied; relative-strength rule disabled.")
    else:
        # Reindex onto the stock's calendar, carrying the last known index level
        # over any day the stock traded but the index print is missing.
        aligned = bench_close.reindex(frame.index).ffill()
        frame["ret_bench"] = ind.pct_return(aligned, params.rs_lookback)
        frame["rs_excess"] = frame["ret_stock"] - frame["ret_bench"]
        frame["cond_rs"] = (frame["rs_excess"] > params.rs_min_excess).fillna(False)

    # -- Rule 5: breakout or volatility-contraction setup ------------------- #
    frame["high_n"] = ind.rolling_high(high, params.breakout_lookback, exclude_today=excl)
    frame["is_breakout"] = (close > frame["high_n"]).fillna(False)
    frame["dist_from_high"] = ind.safe_ratio(frame["high_n"] - close, frame["high_n"])
    frame["is_near_high"] = (
        (frame["dist_from_high"] >= 0) & (frame["dist_from_high"] <= params.breakout_proximity_pct)
    ).fillna(False)

    frame["atr_fast"] = ind.atr(high, low, close, params.vcp_atr_fast)
    frame["atr_slow"] = ind.atr(high, low, close, params.vcp_atr_slow)
    frame["atr_ratio"] = ind.safe_ratio(frame["atr_fast"], frame["atr_slow"])
    frame["is_contraction"] = (frame["atr_ratio"] < params.vcp_contraction_ratio).fillna(False)

    frame["cond_setup"] = (
        frame["is_breakout"] | (frame["is_near_high"] & frame["is_contraction"])
    ).fillna(False)

    # -- Stop-loss reference ------------------------------------------------ #
    frame["atr_stop"] = ind.atr(high, low, close, risk.atr_stop_period)

    # -- Composite signal --------------------------------------------------- #
    frame["signal"] = (
        frame["cond_delivery"]
        & frame["cond_volume"]
        & frame["cond_close"]
        & frame["cond_rs"]
        & frame["cond_setup"]
    )

    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame


def _bars_since_spike(is_spike: pd.Series, window: int) -> pd.Series:
    """Bars elapsed since the most recent delivery spike strictly before each bar.

    Returns NaN when no spike occurred within ``window`` bars, so the value is
    only meaningful on bars where rule 1 actually passes.
    """
    positions = pd.Series(np.arange(len(is_spike), dtype=float), index=is_spike.index)
    last_spike = positions.where(is_spike).ffill().shift(1)
    age = positions - last_spike
    return age.where(age <= window)


def _benchmark_close(benchmark: pd.DataFrame | pd.Series | None) -> pd.Series | None:
    """Extract a tz-naive close Series from whatever benchmark shape we're given."""
    if benchmark is None:
        return None
    if isinstance(benchmark, pd.Series):
        series = benchmark
    elif "close" in benchmark.columns:
        series = benchmark["close"]
    else:
        return None
    if series.empty:
        return None
    series = series.copy()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    return series[~series.index.duplicated(keep="last")].sort_index()


def compute_stop(
    signal_low: float,
    entry_price: float,
    atr_value: float,
    *,
    risk: RiskParams | None = None,
    cfg: AppConfig | None = None,
) -> float:
    """Initial stop: ``min(signal-day low, entry - atr_mult * ATR)``.

    Taking the minimum of the two means the stop sits below *both* the
    structural invalidation level (the signal bar's low) and a volatility-scaled
    buffer, so ordinary noise cannot take the trade out.

    Returns:
        The stop price. Guaranteed strictly below ``entry_price``; if the inputs
        are degenerate (zero ATR and an entry at the signal low) it falls back to
        a 0.5% stop so the caller never divides by zero when sizing.
    """
    risk = risk or (cfg or get_config()).risk
    candidates = [signal_low]
    if np.isfinite(atr_value) and atr_value > 0:
        candidates.append(entry_price - risk.atr_stop_mult * atr_value)
    stop = float(min(candidates))
    if not np.isfinite(stop) or stop >= entry_price:
        stop = entry_price * 0.995
    return stop


def position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    *,
    risk: RiskParams | None = None,
    cfg: AppConfig | None = None,
    available_cash: float | None = None,
    max_notional: float | None = None,
) -> tuple[int, float]:
    """Size a position to risk exactly ``risk_per_trade_pct`` of equity.

    Three caps are applied in order, and the tightest wins:

    1. **Risk budget** -- ``equity * risk_pct / (entry - stop)`` shares. This is
       the primary rule: every trade risks the same rupee amount regardless of
       how volatile the stock is.
    2. **Max position size** -- ``equity * max_position_pct`` of notional. A very
       tight stop would otherwise produce an enormous position; on a small cap
       that concentration is more dangerous than the stop is protective.
    3. **Available cash**, and optionally a participation cap on the day's
       traded value.

    Args:
        equity: Current portfolio equity.
        entry_price: Assumed fill price.
        stop_price: Initial stop.
        risk: Risk params. Defaults to config.
        cfg: Config override.
        available_cash: Cash on hand; caps the notional if lower than the size cap.
        max_notional: Optional externally computed cap (e.g. liquidity based).

    Returns:
        ``(quantity, rupee_risk)``. Quantity is 0 when no valid size exists.
    """
    risk = risk or (cfg or get_config()).risk
    per_share_risk = entry_price - stop_price
    if per_share_risk <= 0 or entry_price <= 0 or equity <= 0:
        return 0, 0.0

    risk_budget = equity * risk.risk_per_trade_pct
    qty = int(risk_budget // per_share_risk)

    caps = [equity * risk.max_position_pct]
    if available_cash is not None:
        caps.append(max(available_cash, 0.0))
    if max_notional is not None:
        caps.append(max(max_notional, 0.0))
    qty_cap = int(min(caps) // entry_price)

    qty = max(0, min(qty, qty_cap))
    return qty, qty * per_share_risk


def extract_signals(
    features: pd.DataFrame,
    symbol: str,
    *,
    risk: RiskParams | None = None,
    cfg: AppConfig | None = None,
    entry_on_next_open: bool = True,
) -> list[SignalRow]:
    """Turn a feature frame into concrete, tradeable signal rows.

    Args:
        features: Output of :func:`compute_features`.
        symbol: Symbol the frame belongs to.
        risk: Risk params. Defaults to config.
        cfg: Config override.
        entry_on_next_open: Use bar T+1's open as the entry reference. The final
            bar of the frame has no next bar, so it is reported with its close as
            the entry reference -- correct for the live scanner, where tomorrow's
            open is genuinely unknown.

    Returns:
        Signals in chronological order.
    """
    cfg = cfg or get_config()
    risk = risk or cfg.risk
    if features is None or features.empty or "signal" not in features:
        return []

    hits = features.index[features["signal"].fillna(False).astype(bool)]
    out: list[SignalRow] = []
    positions = {ts: i for i, ts in enumerate(features.index)}

    for ts in hits:
        row = features.loc[ts]
        i = positions[ts]
        if entry_on_next_open and i + 1 < len(features):
            entry_ref = float(features["open"].iloc[i + 1])
        else:
            entry_ref = float(row["close"])
        if not np.isfinite(entry_ref) or entry_ref <= 0:
            continue

        atr_val = float(row.get("atr_stop", np.nan))
        stop = compute_stop(float(row["low"]), entry_ref, atr_val, risk=risk)
        out.append(
            SignalRow(
                date=ts,
                symbol=symbol,
                close=float(row["close"]),
                entry_ref=entry_ref,
                stop=stop,
                risk_per_share=entry_ref - stop,
                atr14=atr_val if np.isfinite(atr_val) else 0.0,
                deliv_pct=float(row.get("deliv_pct", np.nan)),
                deliv_ratio=float(row.get("deliv_ratio", np.nan)),
                deliv_spike_ratio=float(row.get("deliv_spike_ratio", np.nan)),
                deliv_spike_age=float(row.get("deliv_spike_age", np.nan)),
                vol_ratio=float(row.get("vol_ratio", np.nan)),
                range_position=float(row.get("range_position", np.nan)),
                rs_excess=float(row.get("rs_excess", np.nan)),
                dist_from_high=float(row.get("dist_from_high", np.nan)),
                atr_ratio=float(row.get("atr_ratio", np.nan)),
                is_breakout=bool(row.get("is_breakout", False)),
            )
        )
    return out


def scan_universe(
    prices: dict[str, pd.DataFrame],
    delivery: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame | pd.Series | None,
    *,
    cfg: AppConfig | None = None,
    on_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Run the DAB rules across a whole universe.

    Args:
        prices: Symbol -> OHLCV frame.
        delivery: Symbol -> delivery frame.
        benchmark: Benchmark prices.
        cfg: Config override.
        on_date: If given, only return signals on that exact date (the live
            scanner's use case). None returns every historical signal.

    Returns:
        A frame of signals with one row per symbol per signal date, sorted by
        date then delivery ratio (strongest accumulation first).
    """
    cfg = cfg or get_config()
    rows: list[dict[str, object]] = []

    for symbol, ohlcv in prices.items():
        if ohlcv is None or ohlcv.empty:
            continue
        try:
            features = compute_features(
                ohlcv, delivery.get(symbol), benchmark, cfg=cfg
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Feature computation failed for %s: %s", symbol, exc)
            continue
        if features.empty:
            continue
        if on_date is not None:
            stamp = pd.Timestamp(on_date).normalize()
            if stamp not in features.index:
                continue
            features = features.loc[[stamp]]
        for sig in extract_signals(features, symbol, cfg=cfg):
            rows.append(sig.as_dict())

    if not rows:
        return pd.DataFrame(columns=["date", "symbol"])
    out = pd.DataFrame(rows)
    return out.sort_values(["date", "deliv_ratio"], ascending=[True, False]).reset_index(drop=True)


__all__ = [
    "FEATURE_COLUMNS",
    "RULE_LABELS",
    "SignalRow",
    "compute_features",
    "compute_stop",
    "extract_signals",
    "position_size",
    "prepare_input",
    "scan_universe",
]
