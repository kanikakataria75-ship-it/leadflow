"""Central configuration for the Nifty Swing Terminal.

All tunable strategy, risk, universe and execution parameters live here as
pydantic models so that they can be (a) validated, (b) serialised to JSON for
the Settings page of the web terminal, and (c) overridden from ``config.yaml``
or environment variables without touching code.

Load order (later wins):
    1. Defaults defined in this module
    2. ``config.yaml`` in the repository root, if present
    3. Environment variables / ``.env`` (secrets only)
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PROJECT_ROOT.parent


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
#: A Windows drive-letter path, with either separator.
_WINDOWS_ABS = re.compile("^[A-Za-z]:[" + re.escape(chr(92)) + "/]")


class Paths(BaseModel):
    """Filesystem layout. Every directory is created by :meth:`ensure`.

    Paths written into ``config.yaml`` are made portable on the way in, because
    the same ``config.yaml`` is read by a Windows research machine, a Linux web
    host and a Linux CI runner:

    * a **relative** path is resolved against :data:`REPO_ROOT`, so
      ``nifty_swing_bot/cache`` means the same directory everywhere. This is the
      form ``config.yaml`` should use.
    * an **absolute path belonging to another platform** is discarded in favour
      of this module's default. A drive-letter path is a single *legal
      filename* on POSIX, so without this the deploy silently creates one
      directory whose whole name is that Windows path, under the process CWD;
      committed artefacts in ``nifty_swing_bot/results`` are never found, and
      the Backtest and Book pages return 503 with nothing in the log.
    * an absolute path that the running platform can use is left alone, so a
      deliberate local override (or a mounted disk) still works.

    This only relocates directories. No strategy, risk or execution parameter
    is touched.
    """

    cache_dir: Path = PROJECT_ROOT / "cache"
    ohlcv_dir: Path = PROJECT_ROOT / "cache" / "ohlcv"
    delivery_dir: Path = PROJECT_ROOT / "cache" / "delivery"
    llm_dir: Path = PROJECT_ROOT / "cache" / "llm"
    universe_dir: Path = PROJECT_ROOT / "cache" / "universe"
    results_dir: Path = PROJECT_ROOT / "results"
    db_path: Path = PROJECT_ROOT / "cache" / "terminal.db"

    @field_validator("*", mode="before")
    @classmethod
    def _portable(cls, v: Any, info: Any) -> Any:
        """Resolve relative paths, and drop paths from a foreign platform."""
        if v is None:
            return v
        text = str(v)
        if not text:
            return v

        on_windows = os.name == "nt"
        foreign = (
            bool(_WINDOWS_ABS.match(text)) and not on_windows
        ) or (
            on_windows and text.startswith("/") and not text.startswith("//")
        )
        if foreign:
            default = cls.model_fields[info.field_name].default
            logger.warning(
                "config.yaml paths.%s is %r, which is not a usable absolute path on "
                "this platform; using the repository default %s instead.",
                info.field_name, text, default,
            )
            return default

        path = Path(text)
        if not path.is_absolute():
            return REPO_ROOT / path
        return path

    def ensure(self) -> "Paths":
        """Create every directory in the layout if it does not yet exist."""
        for field_name, value in self.model_dump().items():
            if field_name == "db_path":
                Path(value).parent.mkdir(parents=True, exist_ok=True)
            else:
                Path(value).mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
class UniverseParams(BaseModel):
    """Defines the tradeable universe.

    The strategy deliberately targets the *mid- and small-cap tier* rather than
    the raw Nifty 500. Large caps carry a structurally high baseline delivery
    percentage because of FII/DII/mutual-fund holding, which dilutes the
    delivery-spike signal: a 1.5x spike on a stock that already sits at 70%
    delivery is far less informative than the same spike on one that normally
    trades at 25%. Excluding the Nifty 100 keeps the signal clean.
    """

    include_indices: list[str] | None = Field(
        default=["niftymidcap150", "niftysmallcap250"],
        description=(
            "Union of these NSE index constituent lists forms the universe. "
            "Takes precedence over base_index/exclude_index when set. "
            "Midcap 150 + Smallcap 250 is the intended tier: large caps are "
            "excluded because their structurally high institutional ownership "
            "makes their participation statistics behave differently, and "
            "because the short-horizon inefficiencies this research targets are "
            "compensation for providing liquidity where liquidity is scarce."
        ),
    )
    base_index: str = Field(
        default="nifty500",
        description="Fallback constituent list when include_indices is unset.",
    )
    exclude_index: str | None = Field(
        default="nifty100",
        description=(
            "Constituents of this index are removed from the universe. Set to "
            "null to trade the raw base index."
        ),
    )
    min_avg_turnover_cr: float = Field(
        default=5.0,
        gt=0,
        description=(
            "Liquidity floor: minimum N-day average daily traded value in INR "
            "crore. Filters out names where a realistic position could not be "
            "filled without moving the market."
        ),
    )
    turnover_lookback_days: int = Field(default=20, gt=0)
    liquidity_quantile: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description=(
            "Keep only symbols above this quantile of average daily turnover, "
            "evaluated after the absolute floors. 0.75 keeps the most liquid "
            "quartile. Slippage is the largest single cost component and scales "
            "with thinness, so restricting to liquid names is the cheapest way "
            "to reduce it -- at the price of a smaller universe and fewer signals."
        ),
    )
    min_price: float = Field(
        default=20.0,
        gt=0,
        description="Drops penny stocks, where percentage moves are mostly noise.",
    )
    max_price: float | None = Field(default=None)
    min_history_days: int = Field(
        default=120,
        gt=0,
        description="Minimum number of bars required before a symbol is considered.",
    )


# --------------------------------------------------------------------------- #
# Strategy - Delivery Accumulation Breakout (DAB)
# --------------------------------------------------------------------------- #
class DABParams(BaseModel):
    """Entry-signal parameters for the Delivery Accumulation Breakout rule set.

    A signal fires only when *every* condition is true on the same bar.
    """

    baseline_excludes_today: bool = Field(
        default=True,
        description=(
            "Compute rolling baselines (delivery average, volume average, "
            "N-day high) over the PRIOR N bars rather than the window ending "
            "today. A spike must be measured against a baseline that does not "
            "already contain the spike, and 'close > 20-day high' is only a "
            "meaningful test when today's own high is excluded from that high."
        ),
    )

    # 1. Delivery% spike -- the core "real accumulation" signal.
    delivery_mode: Literal["prior_window", "same_day"] = Field(
        default="prior_window",
        description=(
            "How the delivery spike is required to relate to the breakout bar.\n\n"
            "``prior_window`` (default): a delivery% spike must have occurred in "
            "the ``delivery_spike_window`` bars BEFORE the signal bar. This "
            "treats Delivery Accumulation Breakout as the sequence its name "
            "describes -- quiet accumulation first, breakout after.\n\n"
            "``same_day``: the original formulation, requiring the spike on the "
            "breakout bar itself. Retained for comparison, but note that it "
            "conflicts mechanically with the volume-surge rule: delivery% is "
            "deliverable qty over TOTAL traded qty, so a 2x volume surge doubles "
            "the denominator and demands deliverable quantity more than triple "
            "for delivery% to rise 1.5x. Measured over 150 mid-caps across 17 "
            "months, the two rules co-fired on 0.88% of otherwise-qualifying "
            "bars -- 9 signals in total, which is not a tradeable sample."
        ),
    )
    delivery_lookback: int = Field(default=20, gt=1)
    delivery_spike_mult: float = Field(
        default=1.5,
        gt=0,
        description="delivery% > mult * its rolling mean delivery% counts as a spike",
    )
    delivery_spike_window: int = Field(
        default=10,
        gt=0,
        description=(
            "In ``prior_window`` mode, how many bars back to look for the "
            "accumulation spike. Should be comparable to the intended holding "
            "period so the accumulation is still recent when price breaks out."
        ),
    )
    delivery_spike_min_bars: int = Field(
        default=5,
        gt=0,
        description="Minimum bars of delivery history needed inside the spike window.",
    )
    min_delivery_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Optional absolute floor on today delivery%. 0 disables it. Useful "
            "to avoid counting spikes off a very low base."
        ),
    )

    # 2. Volume surge.
    volume_lookback: int = Field(default=20, gt=1)
    volume_surge_mult: float = Field(default=2.0, gt=0)

    # 3. Strong close within the day range.
    close_range_top_pct: float = Field(
        default=0.30,
        gt=0,
        le=1.0,
        description=(
            "Close must sit in the top X of the high-low range: "
            "close >= high - X * (high - low)."
        ),
    )

    # 4. Relative strength versus the benchmark.
    rs_lookback: int = Field(default=10, gt=0)
    rs_benchmark: str = Field(
        default="^CRSLDX",
        description="yfinance ticker for the Nifty 500 index.",
    )
    rs_min_excess: float = Field(
        default=0.0,
        description="Stock return minus index return must exceed this (as a fraction).",
    )

    # 5. Breakout / volatility-contraction setup.
    breakout_lookback: int = Field(default=20, gt=1)
    breakout_proximity_pct: float = Field(
        default=0.03,
        ge=0,
        description="Within this fraction of the N-day high counts as a setup.",
    )
    vcp_atr_fast: int = Field(default=5, gt=0)
    vcp_atr_slow: int = Field(default=20, gt=0)
    vcp_contraction_ratio: float = Field(
        default=0.70,
        gt=0,
        description="ATR(fast) < ratio * ATR(slow) qualifies as contraction.",
    )

    @field_validator("vcp_atr_slow")
    @classmethod
    def _slow_gt_fast(cls, v: int, info: Any) -> int:
        fast = info.data.get("vcp_atr_fast")
        if fast is not None and v <= fast:
            raise ValueError("vcp_atr_slow must be greater than vcp_atr_fast")
        return v


# --------------------------------------------------------------------------- #
# Strategy - Pullback Reversion (PBR)
# --------------------------------------------------------------------------- #
class MRParams(BaseModel):
    """Parameters for the mean-reversion strategy.

    These defaults come from the hypothesis lab
    (``python -m nifty_swing_bot.research.signal_lab``), which measured every
    candidate's excess forward return against the universe base rate on a
    date-separated train/holdout split. They are a starting point supported by
    measurement, not tuned optima -- tuning them on the same data is how the
    edge gets fitted away.
    """

    trend_ma: int = Field(
        default=50,
        gt=1,
        description="Close must be above this moving average for the trend filter.",
    )
    require_long_trend: bool = Field(
        default=False,
        description="Also require close above ``long_trend_ma``. Stricter, far fewer signals.",
    )
    long_trend_ma: int = Field(default=200, gt=1)

    rsi_period: int = Field(
        default=2,
        gt=0,
        description=(
            "RSI lookback for the dip trigger. Must be fast: on RSI(14) the "
            "oversold threshold and the trend filter are near-disjoint, because "
            "reaching it requires enough decline to break the moving average."
        ),
    )
    rsi_entry: float = Field(
        default=10.0, gt=0, le=100, description="Enter when RSI(period) falls below this."
    )
    rsi_exit: float = Field(
        default=70.0, gt=0, le=100, description="Exit once RSI(period) recovers above this."
    )

    pullback_lookback: int = Field(default=3, gt=0)
    min_pullback_pct: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Minimum decline over ``pullback_lookback`` bars, as a fraction. "
            "0 disables the depth requirement and relies on the RSI trigger alone."
        ),
    )
    min_volume_ratio: float = Field(
        default=0.3,
        ge=0,
        description=(
            "Volume floor relative to the 20-day average. A dip on almost no "
            "volume is a stale print, not real selling."
        ),
    )
    use_reversion_exit: bool = Field(
        default=False,
        description=(
            "Exit as soon as price reverts (close back above ``exit_ma``, or RSI "
            "above ``rsi_exit``) instead of holding for ``max_hold_days``. "
            "Defaults to OFF because it was measured to be strongly negative. "
            "The reversion exit fires almost immediately after the bounce -- mean "
            "holding period 2.7 bars versus 9.1 without it -- so it banks a tiny "
            "gain while still carrying the full 2.5x ATR stop on the downside. "
            "Measured over the same universe and window: win rate rises from "
            "47.7% to 55.7%, but the average win/average loss ratio collapses "
            "from 1.16 to 0.57 and the strategy goes from +7.95% to -41.50%. "
            "It is the textbook way to be right often and lose anyway. "
            "The deeper reason: the entry edge was measured over a 10-bar "
            "horizon, so an exit at bar 2.7 collects only a fraction of it while "
            "paying the full round-trip cost."
        ),
    )
    exit_ma: int = Field(
        default=5,
        gt=0,
        description=(
            "Short moving average used by the reversion exit, when "
            "``use_reversion_exit`` is enabled."
        ),
    )


# --------------------------------------------------------------------------- #
# Strategy - Sector-Relative Dislocation Reversion (SRDR)
# --------------------------------------------------------------------------- #
class SRDRParams(BaseModel):
    """Parameters for the sector-relative dislocation strategy.

    Every value below is either (a) a direct consequence of the measured
    mechanism, or (b) a deliberately round number chosen to sit in the middle of
    a measured plateau rather than at its optimum. The refinement study showed
    edge is stable across top_n from 1 to 8 and horizons from 2 to 5 bars, so
    none of these sits on a cliff.
    """

    dislocation_lookback: int = Field(
        default=3,
        gt=0,
        description=(
            "Bars over which the stock is compared with its sector. Three days "
            "is long enough for an inventory event to play out and short enough "
            "that the comparison is not contaminated by genuine divergence."
        ),
    )
    top_n: int = Field(
        default=3,
        gt=0,
        description=(
            "Names bought per day, taken from the most dislocated end of the "
            "cross-section. Measured edge runs 0.41% at top_n=1 through 0.32% at "
            "top_n=8, so 3 sits inside a plateau rather than on a peak."
        ),
    )
    min_dislocation: float = Field(
        default=0.02,
        ge=0,
        description=(
            "Minimum shortfall versus the sector before a name is eligible, as a "
            "fraction. Stops the strategy from buying the least-bad name on a day "
            "when nothing is actually dislocated."
        ),
    )
    trend_ma: int = Field(
        default=200,
        gt=1,
        description=(
            "Close must be above this average. Below it, 'cheap relative to "
            "peers' may be a failing business rather than a dislocated one."
        ),
    )
    sector_calm_threshold: float = Field(
        default=0.04,
        gt=0,
        description=(
            "The peer comparison only carries information when the peer group "
            "is stable. If the whole sector moved this much over the lookback, "
            "a stock moving with it is not dislocated."
        ),
    )
    collapse_atr: float = Field(
        default=3.0,
        gt=0,
        description=(
            "Declines beyond this many ATRs are excluded as repricings rather "
            "than inventory shocks. Ranking on absolute weakness rather than "
            "relative weakness produced a NEGATIVE edge, which is what this "
            "filter removes."
        ),
    )
    gap_threshold: float = Field(
        default=0.02,
        ge=0,
        description="Overnight gap size that triggers the orphan-gap check.",
    )
    orphan_gap_threshold: float = Field(
        default=0.015,
        ge=0,
        description=(
            "How far a gap must exceed the sector's own gap to count as "
            "stock-specific news. Unshared gap-downs were measured to keep "
            "falling (-0.407%), so they are excluded rather than bought."
        ),
    )
    min_turnover_cr: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Liquidity floor in INR crore. Deliberately low: restricting to the "
            "most liquid quartile was measured to REMOVE the edge, because the "
            "premium being harvested is compensation for providing liquidity."
        ),
    )
    hold_bars: int = Field(
        default=5,
        gt=0,
        description=(
            "Fixed holding period. The measured edge rises from 0.32% at 2 bars "
            "to 0.54% at 5 while costs are paid once, so the longer hold is "
            "better; 5 keeps the strategy inside the short-swing framework."
        ),
    )
    stop_atr: float = Field(
        default=2.5,
        gt=0,
        description=(
            "Protective stop in ATR(14) units. Wide, because the thesis is that "
            "the position is being entered into ongoing selling pressure and "
            "needs room before the reversion arrives."
        ),
    )


# --------------------------------------------------------------------------- #
# Risk and position sizing
# --------------------------------------------------------------------------- #
class RiskParams(BaseModel):
    """Position sizing, stops and exits."""

    starting_capital: float = Field(default=1_000_000.0, gt=0)
    risk_per_trade_pct: float = Field(
        default=0.01,
        gt=0,
        le=0.1,
        description="Fraction of current equity risked per trade (1% default).",
    )
    max_position_pct: float = Field(
        default=0.10,
        gt=0,
        le=1.0,
        description=(
            "Hard cap on notional exposure to a single stock as a fraction of "
            "equity, applied AFTER the 1%-risk formula. Without this, a tight "
            "stop on a quiet small cap produces an absurdly large position, and "
            "small-cap volatility makes that dangerous."
        ),
    )
    max_open_positions: int = Field(
        default=10,
        gt=0,
        description="Concurrent position cap; also bounds portfolio heat.",
    )
    max_portfolio_heat_pct: float = Field(
        default=0.06,
        gt=0,
        description=(
            "Total open risk (sum of per-trade risk still at stake) capped as a "
            "fraction of equity."
        ),
    )

    # Stop loss
    atr_stop_period: int = Field(default=14, gt=0)
    atr_stop_mult: float = Field(default=1.5, gt=0)

    # Exits
    max_hold_days: int = Field(default=10, gt=0)
    min_hold_days: int = Field(default=0, ge=0)
    trail_swing_lookback: int = Field(
        default=3,
        gt=0,
        description="Bars used to identify the most recent swing low to trail behind.",
    )
    trail_activate_r: float = Field(
        default=1.0,
        ge=0,
        description="Trailing only begins once the trade is up this many R.",
    )
    target_r_multiple: float | None = Field(
        default=None,
        description="Optional fixed profit target in R. null = no hard target.",
    )


# --------------------------------------------------------------------------- #
# Execution realism
# --------------------------------------------------------------------------- #
class ExecutionParams(BaseModel):
    """Fill assumptions, costs, and circuit-filter modelling.

    Mid- and small-cap Indian equities carry 5%/10%/20% circuit bands. When a
    stock is locked at a circuit there is no liquidity on the other side, so a
    stop simply cannot be filled at the stop price. The backtester models this
    explicitly rather than assuming clean fills.
    """

    entry_timing: Literal["next_open", "signal_close"] = Field(
        default="next_open",
        description=(
            "Signals are computed on the close of day T; 'next_open' fills at "
            "the open of T+1, which is the only assumption a real trader can act on."
        ),
    )
    cost_model: Literal["simple_bps", "india_delivery"] = Field(
        default="india_delivery",
        description=(
            "Which charge stack to apply. ``india_delivery`` models a real "
            "Indian discount-broker delivery (CNC) trade: flat brokerage, STT on "
            "BOTH sides, stamp duty, exchange and SEBI charges, GST, and a flat "
            "depository charge per sell. ``simple_bps`` is the older "
            "percentage-only approximation, kept for comparison and for tests "
            "that need exact fill arithmetic."
        ),
    )
    slippage_bps: float = Field(
        default=15.0,
        ge=0,
        description=(
            "Per-side slippage in basis points. This is the component that "
            "genuinely varies with liquidity: 10-15bps is fair for thin mid-caps, "
            "3-6bps for the most liquid quartile."
        ),
    )
    brokerage_bps: float = Field(
        default=3.0, ge=0,
        description="Per-side brokerage in bps. Used only by ``simple_bps``.",
    )
    brokerage_flat_inr: float = Field(
        default=20.0,
        ge=0,
        description=(
            "Flat brokerage per order, in rupees. Rs 20 is the common discount "
            "rate; several brokers charge Rs 0 on delivery, so 0 is a valid and "
            "realistic setting."
        ),
    )
    brokerage_pct_cap: float = Field(
        default=2.5,
        ge=0,
        description=(
            "Cap brokerage at this percent of turnover, as brokers quote it "
            "('Rs 20 or 2.5%, whichever is lower'). 0 disables the cap."
        ),
    )
    stt_bps_buy: float = Field(
        default=10.0,
        ge=0,
        description=(
            "Securities transaction tax on the BUY side. For delivery equity "
            "this is 0.1%, the same as the sell side -- a charge the earlier "
            "simple model omitted entirely."
        ),
    )
    stt_bps_sell: float = Field(
        default=10.0,
        ge=0,
        description="Securities transaction tax on the sell side (delivery 0.1%).",
    )
    stamp_duty_bps_buy: float = Field(
        default=1.5, ge=0, description="Stamp duty, buy side only (0.015%)."
    )
    exchange_txn_bps: float = Field(
        default=0.297, ge=0, description="NSE transaction charges per side (~0.00297%)."
    )
    sebi_bps: float = Field(
        default=0.01, ge=0, description="SEBI turnover fee per side (Rs 10 per crore)."
    )
    gst_pct: float = Field(
        default=18.0, ge=0, description="GST on brokerage and exchange charges."
    )
    dp_charge_inr: float = Field(
        default=16.0,
        ge=0,
        description=(
            "Depository charge per scrip per sell, in rupees, inclusive of GST. "
            "Flat, so it weighs heavily on small positions."
        ),
    )
    circuit_band_pct: float = Field(
        default=0.10,
        gt=0,
        description=(
            "Assumed circuit band for the tier. A bar that opens beyond this "
            "distance from the previous close and never trades back is treated "
            "as circuit-locked, where a stop cannot be filled at all."
        ),
    )
    gap_through_extra_slippage_bps: float = Field(
        default=50.0,
        ge=0,
        description=(
            "Extra slippage charged when price gaps through the stop, on top of "
            "filling at the (already worse) open price."
        ),
    )
    locked_circuit_penalty_pct: float = Field(
        default=0.02,
        ge=0,
        description=(
            "When a stock is circuit-locked against the position the exit is "
            "deferred to the next bar and this additional adverse move is "
            "charged, modelling the further damage taken while trapped."
        ),
    )
    max_participation_pct: float = Field(
        default=0.02,
        gt=0,
        description=(
            "Position notional may not exceed this fraction of the signal day "
            "traded value; prevents unfillable small-cap sizes."
        ),
    )


# --------------------------------------------------------------------------- #
# Backtest / walk-forward
# --------------------------------------------------------------------------- #
class BacktestParams(BaseModel):
    """Backtest window and walk-forward split geometry."""

    start_date: str = Field(default="2021-01-01")
    end_date: str | None = Field(default=None, description="null = today.")
    benchmark: str = Field(default="^CRSLDX")
    risk_free_rate: float = Field(
        default=0.065, description="Annualised, used for Sharpe. Indian 10y ~6.5%."
    )
    trading_days_per_year: int = Field(default=252, gt=0)

    # Walk-forward
    wf_train_months: int = Field(default=12, gt=0)
    wf_test_months: int = Field(default=3, gt=0)
    wf_step_months: int = Field(default=3, gt=0)
    wf_optimise: bool = Field(
        default=True,
        description="Grid-search a small parameter set on each training window.",
    )


# --------------------------------------------------------------------------- #
# LLM layer
# --------------------------------------------------------------------------- #
class LLMParams(BaseModel):
    """Anthropic insight-layer settings."""

    model: str = Field(default="claude-sonnet-5")
    max_tokens: int = Field(default=1600, gt=0)
    temperature: float = Field(default=0.3, ge=0, le=1)
    enabled: bool = Field(default=True)
    cache_ttl_hours: int = Field(
        default=24, gt=0, description="Insights are cached per symbol per day."
    )


# --------------------------------------------------------------------------- #
# Data fetching
# --------------------------------------------------------------------------- #
class DataParams(BaseModel):
    """Network/data-source behaviour."""

    ohlcv_source: Literal["yfinance"] = "yfinance"
    yf_suffix: str = Field(default=".NS")
    yf_batch_size: int = Field(default=40, gt=0)
    yf_max_workers: int = Field(default=4, gt=0)

    nse_base_url: str = "https://www.nseindia.com"
    nse_archive_url: str = "https://archives.nseindia.com"
    nse_request_delay_s: float = Field(
        default=0.8, ge=0, description="Politeness delay between bhavcopy requests."
    )
    nse_max_retries: int = Field(default=3, ge=0)
    nse_timeout_s: float = Field(default=30.0, gt=0)
    cache_ttl_hours: int = Field(
        default=12, gt=0, description="How long cached OHLCV stays fresh."
    )


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #
def _relativise(value: Any) -> str:
    """``value`` as a POSIX path relative to :data:`REPO_ROOT` when it is inside it."""
    p = Path(str(value))
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


class AppConfig(BaseModel):
    """Root configuration object, serialisable to and from JSON/YAML."""

    paths: Paths = Field(default_factory=Paths)
    universe: UniverseParams = Field(default_factory=UniverseParams)
    dab: DABParams = Field(default_factory=DABParams)
    mr: MRParams = Field(default_factory=MRParams)
    srdr: SRDRParams = Field(default_factory=SRDRParams)
    risk: RiskParams = Field(default_factory=RiskParams)
    execution: ExecutionParams = Field(default_factory=ExecutionParams)
    backtest: BacktestParams = Field(default_factory=BacktestParams)
    llm: LLMParams = Field(default_factory=LLMParams)
    data: DataParams = Field(default_factory=DataParams)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        """Load config from YAML, falling back to the defaults in this module."""
        path = path or (REPO_ROOT / "config.yaml")
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            cfg = cls.model_validate(raw)
        else:
            cfg = cls()
        cfg.paths.ensure()
        return cfg

    def save(self, path: Path | None = None) -> Path:
        """Persist the current config to YAML.

        Paths inside the repository are written back **relative** to it. The
        loaded values are absolute (``Paths`` resolves them), and round-tripping
        them verbatim would bake this machine's layout into a file that a Linux
        host also reads -- which is the problem ``Paths._portable`` exists to
        undo. Nothing outside the ``paths`` block is altered.
        """
        path = path or (REPO_ROOT / "config.yaml")
        data = self.model_dump(mode="json")
        data["paths"] = {
            name: _relativise(value) for name, value in data.get("paths", {}).items()
        }
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path


class Secrets(BaseSettings):
    """Environment-sourced settings. Never commit these."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str | None = Field(default=None, alias="ANTHROPIC_MODEL")


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide singleton config."""
    return AppConfig.load()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """Process-wide singleton secrets."""
    return Secrets()  # type: ignore[call-arg]


def reload_config() -> AppConfig:
    """Clear the cache and re-read config from disk (used by the Settings API)."""
    get_config.cache_clear()
    return get_config()


__all__ = [
    "PROJECT_ROOT",
    "REPO_ROOT",
    "AppConfig",
    "BacktestParams",
    "DABParams",
    "DataParams",
    "ExecutionParams",
    "LLMParams",
    "MRParams",
    "Paths",
    "RiskParams",
    "SRDRParams",
    "Secrets",
    "UniverseParams",
    "get_config",
    "get_secrets",
    "reload_config",
]
