"""Tunable parameters for the Accumulation-Expansion Swing strategy (AES).

Every number the AES rules depend on lives here, in one place, so that a
calibration sweep changes a dataclass field rather than editing logic. Nothing
in this module reads data or makes a decision.

Two conventions worth stating, because they shape the whole implementation:

**Detection is generous; scoring is opinionated.** The spec (§2.3) says a box
longer than about a month is stale, and (§4) that an over-long box is a
*negative score factor*. Those are different statements. If the detector
refused to return a 30-bar box, the scorer could never penalise one. So the
search windows here are deliberately wider than the spec's "healthy" ranges,
and the healthy ranges appear separately as ``norm_min_bars`` /
``norm_max_bars`` for the scorer to grade against.

**Box edges have two different jobs.** The big box bottom is a *stop level*
(§6.1), so it is the hard minimum low of the window. The small box is a
description of where closing bodies sit, and the spec explicitly allows wicks
to poke outside it (§2.1), so its edges are quantiles of high/low rather than
extremes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScreenerParams:
    """Watchlist admission (§1.1) and hard deletes (§1.2).

    Admission is not entry. These thresholds decide who gets *watched*; the box
    detector and the scorer decide who gets bought, days or weeks later.
    """

    # --- §1.1 screener ---
    move_pct: float = 0.20               # 20% up ...
    move_lookback: int = 10              # ... in 10 sessions
    high_lookback: int = 252             # 52 weeks
    #: "At or near" the 52-week high. Exactly at it would make the condition
    #: fire on single bars only; the spec says at/near.
    high_proximity: float = 0.05
    min_price: float = 50.0
    turnover_lookback: int = 20
    min_turnover_inr: float = 1e7        # Rs 1 Cr average daily turnover
    #: §1.1 also asks for market cap > 500 Cr. No point-in-time market-cap
    #: history exists in this cache, so it is not applied; the turnover floor
    #: is a partial proxy. Recorded here so the gap stays visible.
    market_cap_applied: bool = False

    # --- §1.2 hard deletes ---
    delete_below_price: float = 100.0
    delete_above_price: float = 5000.0
    #: Bars of sustained sub-floor turnover before the name is deleted.
    illiquid_sustained_bars: int = 20
    #: Circuit-mover threshold. The spec says "calibrate N; start at 3" --
    #: this is the Phase 3 sweep target, not a believed value.
    max_circuit_hits: int = 3
    circuit_lookback: int = 60
    #: Circuit days are inferred, not read from exchange data: a locked bar has
    #: almost no range and a large move.
    circuit_max_range: float = 0.005
    circuit_min_move: float = 0.045


@dataclass(frozen=True, slots=True)
class BoxParams:
    """Geometry and structure thresholds for nested accumulation boxes (spec §2)."""

    # --- big box: the outer consolidation, 15-25% tall (§2.1) ----------------
    big_min_bars: int = 10
    big_max_bars: int = 45
    big_range_min: float = 0.15
    big_range_max: float = 0.25
    #: The big box is *anchored*, not swept. Its top is the swing high price
    #: stalled under (§2.2), found as the running maximum over each of these
    #: lookbacks; the box then runs from that bar to today. Sweeping every
    #: window length instead was tried and rejected: nearly every component of
    #: box quality (edge touches, midpoint crossings) grows with window length,
    #: so a sweep picks the longest window almost regardless of shape, and 76%
    #: of all bars came back as "a box". Anchoring lets price structure set the
    #: duration, which is also what makes the §2.3 duration norm meaningful.
    anchor_lookbacks: tuple[int, ...] = (20, 30, 45, 60, 90)

    # --- small box: the tight recent range nested inside, 5-10% tall (§2.1) --
    small_min_bars: int = 5
    small_max_bars: int = 22
    small_range_min: float = 0.05
    small_range_max: float = 0.10
    #: Small-box edges are trimmed extremes, not raw ones, so an isolated wick
    #: does not define the box -- but a *cluster* of wicks does, at the level
    #: they cluster at.
    #:
    #: Was a fixed quantile trim (90/10, then loosened to 80/20 to let boxes
    #: extend further in duration -- see the §1.3 note in RESEARCH_AES.md).
    #: Both were replaced on direct feedback after chart review: a quantile
    #: discards a fixed fraction of bars regardless of whether they cluster,
    #: so it can throw away a genuine multi-bar level along with real noise,
    #: or keep a lone spike just because the sample was too small for it to
    #: fall in the trimmed tail. The rule now is the literal one given: an
    #: outlying wick is dropped only if nothing else is near it; the moment a
    #: neighbour sits within ``isolated_wick_gap_pct``, trimming stops there
    #: and that level -- covering the whole cluster -- becomes the edge. See
    #: ``_trim_isolated_extreme``.
    isolated_wick_gap_pct: float = 0.04
    #: Ceiling on how many points a single edge may discard this way, so a
    #: window with no clustering at all cannot be trimmed down to nearly
    #: nothing.
    max_isolated_trim: int = 3
    #: How far outside the big box the small box may poke, as a fraction of
    #: the big box's height, before nesting is rejected.
    small_wick_tolerance: float = 0.02

    # --- internal structure (§2.4) ------------------------------------------
    #: Bars either side required for a local low to count as a swing low.
    #:
    #: Set to 1, not the more conventional 2, for a measurement reason. Boxes
    #: that form *after* screener admission have a median duration of 11 bars,
    #: and at k=2 an 11-bar window yields a median of **one** confirmed swing
    #: low -- so §2.4's "minimum 2 higher swing lows" is satisfiable by only 2%
    #: of real boxes and the feature is a near-constant zero. At k=1 the median
    #: is 2 swing lows and the criterion becomes a live measurement (14%).
    #:
    #: This is a validity choice, not a tuning one, and it cuts against the
    #: factor: on a crude break-up proxy, k=1 boxes meeting "2 higher lows"
    #: resolved upward *less* often than those that did not (23.7% vs 31.1%).
    #: Whether higher lows carry any weight at all is a Phase 3 question.
    swing_k: int = 1
    #: A bar counts as "touching" an edge if it comes within this fraction of
    #: the box height of it.
    edge_touch_tol: float = 0.15
    #: Two touches closer together than this count as one visit.
    min_touch_separation: int = 3
    #: A window whose high is the last bar or two is a breakout in progress,
    #: not an accumulation box. Refuse to call it one.
    min_bars_since_top: int = 2
    #: What defines the big box's ceiling and floor: ``"wick"`` uses the
    #: extremes of high and low, ``"body"`` uses the extremes of the candle
    #: bodies (max/min of open and close) so that tails poke outside.
    #:
    #: This is an open interpretive question, not a settled one. §2.1 says of
    #: the *small* box that "price wicks may exceed the small box; closing
    #: bodies should mostly stay inside", and §6.1 makes the stop a closing-
    #: basis rule -- both of which argue for bodies. Against that, §6.1 also
    #: says box-bottom entries stop at "the low of the big-box bottom", which
    #: reads like a wick. Renders under both are in
    #: ``results/aes_phase1_edge_basis/``.
    edge_basis: str = "wick"
    #: ``"close"`` was added in Phase 15 as a third option, on the argument
    #: that a wick basis lets a single spike break an otherwise valid base:
    #: the box edges are taken from closes alone, so intrabar tails neither
    #: set the ceiling nor the floor. Unlike ``"body"`` it ignores the open
    #: as well, which makes the big box a pure closing-range measurement and
    #: puts it on the same basis as the section 6.1 stop.
    #: The big box's floor is trimmed the same way the small box's edges are
    #: (see ``isolated_wick_gap_pct`` / ``max_isolated_trim`` below): an
    #: isolated single-bar low is dropped, a cluster of low bars is kept and
    #: becomes the floor.
    #:
    #: Found by looking at renders. With the floor defined as the plain
    #: minimum low, a single deep intraday wick on the closing bar set the box
    #: bottom: NH on 07-Apr-2025 consolidated in a 6.5% range, then printed one
    #: bar with a long tail, and the detector drew an 18.8% "box" whose floor
    #: no other bar had been near. That also inflated its health -- dragging
    #: the midpoint down put 100% of closes above it and scored the box a
    #: perfect 1.00.
    #:
    #: The first attempt at a fix required the floor to be *touched twice*.
    #: That was wrong, and measurably so: demanding two visits to the low
    #: forces price to have gone back down there, which pushes the nested
    #: small box into the lower half by construction. Upper-zone boxes -- the
    #: §2.1 positive case -- fell from 1.7% of detections to exactly zero. A
    #: detection gate that cannot produce the spec's best setup is not a gate,
    #: it is a bug.
    #:
    #: The floor is deliberately not trimmed on the *top* side. The big box
    #: top is anchor-defined (it is, by construction, the running-maximum bar
    #: that set the box's start), so there is nothing to trim there -- the
    #: anchor bar's own high always is the window's maximum. Only the floor is
    #: a free extreme that a single bad wick can distort.
    #:
    #: How well the floor is held is left to the scorer, where §4 says it
    #: belongs. A consequence worth stating: on the bar that breaks a box
    #: downward, no box is detected -- correct, since the watchlist carries
    #: the box frozen from the prior session and marks it invalidated rather
    #: than redrawing it bigger.

    # --- Phase 15 geometry relaxations, all default-off ----------------------
    #: Detect *sloped* channels as well as horizontal boxes (§7.2 group 3:
    #: PB Fintech / CAMS style staircases of higher lows into the high).
    #:
    #: A horizontal box cannot express these for two independent reasons, and
    #: both have to be lifted together: the range test sees the whole rise as
    #: box height, and the running-maximum anchor puts the top on the last bar
    #: so ``min_bars_since_top`` rejects it. In sloped mode the window start is
    #: the lookback itself rather than the running-maximum bar, a Theil-Sen
    #: line is fitted through the closes, and the band is measured on the
    #: *residuals* around that line. The reported ``top``/``bottom`` are the
    #: channel's value at the evaluation bar -- the operative level for entry
    #: and for the §6.1 stop -- not the whole sloped region.
    allow_sloped: bool = False
    #: Bound on how much of a sloped candidate is trend rather than base:
    #: (trend rise over the window) / (channel height). Without a bound, any
    #: quiet uptrend measures as a tight rising channel, which is the failure
    #: mode the anchored horizontal box was built to avoid.
    sloped_max_drift: float = 1.5
    #: Sloped channels must rise, not fall; a falling channel is distribution.
    sloped_min_drift: float = 0.0
    #: Exclude the prior expansion leg from the range measurement (§7.2 group
    #: 5: Balu Forge, a real shelf sitting on a +167% vertical run).
    #:
    #: The box is measured from the first bar that closes inside the shelf
    #: zone below the anchor top, rather than from the anchor bar itself, so
    #: the run-up that *created* the top does not also set the floor. The leg's
    #: magnitude is still recorded -- ``NestedBox.prior_leg_pct`` is exactly
    #: that measurement -- so this moves the leg from the gate to the context,
    #: which is where §2.2 already says it belongs.
    exclude_prior_leg: bool = False

    # --- duration norms: graded by the scorer, never gated here (§2.3) -------
    norm_min_bars: int = 5
    norm_max_bars: int = 22

    # --- context used to describe the box, not to select it -----------------
    #: Lookback before the box start used to measure the expansion leg that
    #: created it, and the volume baseline it dried up from (§2.2, §2.5).
    prior_leg_lookback: int = 30
    vol_ma_bars: int = 20

    # --- historical episodes, for the per-stock adaptive norm (§2.3) --------
    #: Two detections sharing at least this fraction of their bars are treated
    #: as the same accumulation episode.
    episode_overlap: float = 0.50
    #: A past episode counts as a completed acc->expansion cycle if price
    #: cleared its top by ``resolve_pct`` within ``resolve_window`` bars.
    resolve_window: int = 20
    resolve_pct: float = 0.03
    #: Bars of history required before the detector will look at a bar at all.
    min_history_bars: int = 120

    # --- how a candidate box is graded as box-shaped ------------------------
    #: These four are the spec's own §2.4/§2.1 health criteria, not invented
    #: ones. ``drift_ratio`` and ``oscillation`` are still computed and
    #: reported, but deliberately carry no weight: with a structurally anchored
    #: start bar, drift is mechanically negative (the box begins at its own
    #: high) and midpoint crossings scale with length, so scoring either would
    #: measure the anchoring convention rather than the stock.
    w_above_mid: float = 0.30
    w_higher_lows: float = 0.25
    w_containment: float = 0.25
    w_touch: float = 0.20
    #: Normalisation ceilings: this many higher lows or edge touches is full
    #: marks, more adds nothing.
    higher_lows_cap: int = 3
    touch_cap: int = 3
    #: Mild preference for the longer of two equally tight *small* boxes -- a
    #: range that has held longer is more established. Applies to the small-box
    #: sweep only; the big box is anchored and never swept.
    duration_pref: float = 0.15

    # --- where the small box sits inside the big box (§2.1) -----------------
    zone_bottom_max: float = 0.35
    zone_upper_min: float = 0.65

    def __post_init__(self) -> None:
        if self.big_min_bars < 2 * self.swing_k + 1:
            raise ValueError("big_min_bars too small for the swing-low window")
        if self.edge_basis not in ("wick", "body", "close"):
            raise ValueError(f"edge_basis must be wick, body or close, got {self.edge_basis!r}")
        if self.big_range_min >= self.big_range_max:
            raise ValueError("big_range_min must be below big_range_max")
        if self.small_range_min >= self.small_range_max:
            raise ValueError("small_range_min must be below small_range_max")
        if not 0.0 <= self.zone_bottom_max <= self.zone_upper_min <= 1.0:
            raise ValueError("zone thresholds must satisfy 0 <= bottom <= upper <= 1")


@dataclass(frozen=True, slots=True)
class ResistanceParams:
    """Age-weighted resistance detection and prior-visit reaction (§3.2/§3.3).

    A resistance "level" here is a cluster of swing highs at similar prices.
    Age is measured from the *oldest* touch in the cluster, because that is
    what §3.2 means by "how far back it formed" -- a level first made three
    years ago and retested last month is old, not recent.
    """

    #: Bars either side required for a local high to count as a swing high.
    #: Kept separate from ``BoxParams.swing_k`` -- resistance is found over a
    #: much longer history than one box, so a slightly wider fractal avoids
    #: treating every minor wiggle as its own level.
    swing_k: int = 3
    #: How far back to search for resistance levels at all.
    lookback_bars: int = 750     # roughly 3 years
    #: Swing highs within this fraction of each other's price are the same
    #: level, not two different ones.
    cluster_tol_pct: float = 0.025
    #: Only levels within this far above the reference price are "next
    #: resistance" -- a level 80% away is not what §3.2 is asking about.
    max_headroom_pct: float = 0.60
    #: §3.2's number, verbatim: 12-15% is enough headroom for a box-bottom
    #: entry. Used as the near-gating threshold in the scorer, not enforced
    #: here -- this module measures headroom, it does not reject on it.
    min_headroom_pct: float = 0.12
    #: Age is expressed on a log scale and capped here (in bars) before
    #: normalising to a 0-1 strength score, so one ancient level does not
    #: dominate every comparison.
    age_saturation_bars: int = 500

    # --- prior-visit reaction (§3.3) ---
    #: A touch's volume is "elevated" above this multiple of the trailing
    #: volume average, evaluated as of the touch itself (excludes the touch
    #: bar's own volume from its own baseline).
    touch_vol_mult: float = 1.3
    touch_vol_lookback: int = 20
    #: Bars after a touch used to classify the reaction.
    reaction_window: int = 10
    #: A touch is "rejected" if price falls at least this far within the
    #: reaction window; anything milder is "absorbed" (ground through rather
    #: than sharply turned away).
    rejection_drop_pct: float = 0.05
    #: Two touches within this many bars of each other count as one visit,
    #: matching the intent of ``BoxGeometry``'s own touch counting.
    min_touch_separation: int = 5


@dataclass(frozen=True, slots=True)
class TimeframeParams:
    """Multi-timeframe breakout classification (§3.1).

    Weekly and monthly bars are built from the daily frame *truncated to the
    evaluation bar*, not from a resample of the whole history -- otherwise the
    in-progress week or month would quietly include days that have not
    happened yet. This is the same look-ahead discipline as everywhere else in
    ``aes/``, just applied across a resample instead of a rolling window.
    """

    weekly_rule: str = "W-FRI"
    monthly_rule: str = "ME"
    #: A timeframe's own "N-period high" definition. Chosen so daily, weekly
    #: and monthly each look back a comparable amount of calendar time
    #: (roughly a year, three years, six years) rather than a comparable bar
    #: count, since a bar means something different on each timeframe.
    daily_lookback: int = 252
    weekly_lookback: int = 156
    monthly_lookback: int = 72
    #: A fresh N-period high must clear the prior high by this margin, so a
    #: fractional new high by a few paise does not count as a breakout.
    breakout_margin_pct: float = 0.001


@dataclass(frozen=True, slots=True)
class RelativeStrengthParams:
    """Drawdown-conditional relative strength (§3.4).

    Deliberately not "stock return minus index return" over some window --
    that is ordinary relative strength, and it rewards stocks that merely went
    up more, regardless of what happened when the market was under pressure.
    This isolates index down-days and measures capture on those days only, so
    a stock that hugs the tape when things are fine but barely moves when the
    index sells off scores well, and one that beta-chases down along with the
    market does not, however strong the headline chart looks.
    """

    #: Trailing window the down-day decomposition is measured over.
    lookback_bars: int = 60
    #: An index day counts as "down" if its return is at or below this
    #: (negative) threshold -- small daily noise is not a drawdown day.
    index_down_threshold: float = -0.003
    #: Down days below this count are too few to measure a ratio from.
    min_down_days: int = 5
    #: Downside-capture ratio bands: stock decline / index decline over the
    #: same down days, both expressed as negative sums. Ratio well below 1
    #: means the stock fell less than the index on the index's bad days.
    capture_resilient_max: float = 0.70
    capture_fragile_min: float = 1.30


@dataclass(frozen=True, slots=True)
class WatchlistParams:
    """The persistent watchlist state machine (§1.3) and all three entry modes (§5).

    Replaces a per-admission search window with an actual watchlist: a name
    is checked *every session* for a forming box for as long as it stays on
    the list, and can produce more than one signal per admission if a box
    resolves without an entry and a fresh one later forms -- the behaviour
    §1.3 describes and the first implementation (one search per admission,
    "results/aes_discovery_signals.parquet"'s original pass) fell short of.
    """

    #: A name with no box ever detected for this many bars since it (re)joined
    #: the watchlist stops being actively scanned, until a fresh admission
    #: re-adds it. Not in the spec -- a practical bound, since §1.3 places no
    #: ceiling on how long a name may sit on the list without a box forming.
    #: Chosen generously (roughly 4.5 months) so it binds only on names that
    #: have plainly stopped being an accumulation story, not on slow formers.
    max_watch_without_box: int = 90
    #: Bars a tracked box may run with no entry before it is marked stale and
    #: dropped -- taken directly from §2.3's own "beyond [~1 month] the
    #: setup is stale" line, i.e. ``BoxParams.norm_max_bars``.
    box_stale_bars: int = 22
    #: Bars a box stays *armed as an entry candidate* after it would
    #: otherwise have gone stale (Phase 8). The state machine holds a box
    #: fixed from the bar it is first seen, so "still a box" was never the
    #: entry requirement -- but ``box_stale_bars`` dropped the episode 22
    #: bars in, and the ``last_box_start_idx`` guard then refused to re-arm
    #: on the same structure, so a thrust out of a shelf the detector had
    #: already found could not fire. Phase 7 measured the cost of that
    #: directly: on 41 real trades a nested box existed a median 16 bars
    #: before entry and on the entry bar itself for only 6 of 39, while the
    #: entry hit rate was zero. While carried, only the breakout modes (1
    #: and 2) may fire -- a box-bottom entry into a structure that has
    #: already gone stale is a different trade, not this one. A close below
    #: the big-box floor still kills the episode outright at any time.
    #: 0 reproduces the original behaviour exactly.
    box_carry_bars: int = 0
    #: A breakout must clear the operative top by at least this margin.
    breakout_margin_pct: float = 0.001
    #: Breakout-day volume baseline window (excludes the breakout bar itself).
    vol_baseline_bars: int = 20
    #: The volume ratio that routes a breakout into mode 1 (immediate entry)
    #: versus mode 2 (wait for a retest). This is the spec's own starting
    #: guess (§5), used here only to *classify which entry style applies* --
    #: Phase 3 swept 1.5x-3.0x as a possible hard entry gate and found no
    #: reliable threshold to gate entry on at all; using the spec's own
    #: number to route between two entry mechanics that both still trade is
    #: a materially different, much weaker claim than gating on it.
    mode1_vol_threshold: float = 2.0
    #: Bars a weak breakout is allowed to wait for a qualifying retest before
    #: the attempt is abandoned.
    retest_window: int = 15
    #: The retest may come within this far *below* the broken level and still
    #: count as holding it (wicks undershooting, not a clean re-break).
    retest_hold_tol: float = 0.01
    #: A close this far below the broken level after a weak breakout counts
    #: as the retest failing, not merely being in progress.
    retest_fail_pct: float = 0.02
    #: Box-bottom entries (§5 mode 3) fire within this far above the big-box
    #: bottom, matching the spec's own "±1-2%".
    box_bottom_zone_pct: float = 0.02
    #: Box-bottom entries additionally require price to be "respecting
    #: SMA(10)" -- i.e. above it on a closing basis.
    sma_period: int = 10
    #: Forward horizons (bars from entry) at which return is measured.
    horizons: tuple[int, ...] = (5, 10, 15, 20)


@dataclass(frozen=True, slots=True)
class ScoreParams:
    """The section 4 weighted confidence score -- calibrated, not guessed.

    The spec's own §4 table assigns qualitative weights (High/Medium/Bonus/
    Negative) to eleven factors and says outright that they are "to be
    calibrated by backtest." Phase 3 did that calibration on 787 discovery-
    period signals, and the honest result is that almost none of those
    eleven individually clear conventional significance. Four inputs here,
    not eleven, and every one earned its place by surviving measurement --
    the rest are computed and reported elsewhere (a signal's flat row still
    carries them) but are not weighted into this score. That is the literal
    answer to "report which factors carry predictive weight and which are
    noise": most of them are noise at this sample size, so they carry none.

    ``fast_resolution`` (weight 0.40, the largest) is not one of the spec's
    named §4 factors at all -- it is a *timing* characteristic discovered
    while investigating why widening the watchlist's search window kept
    diluting edge (RESEARCH_AES.md §3.5/§3.6). It reproduced under three
    independent constructions without being searched for each time, which is
    why it carries the most weight here despite not being anticipated by the
    spec's own table.
    """

    #: A breakout resolving within this many *calendar* days of the screener
    #: admission that led to it scores the maximum on this factor. Not a
    #: swept-and-chosen value -- it is the exploratory bucket boundary
    #: reported in RESEARCH_AES.md §3.6 before this score existed, kept
    #: exactly rather than re-optimised now that it is a scoring input.
    fast_resolution_days: int = 20
    w_fast_resolution: float = 0.40

    #: §3.3 resistance-absorption count, capped and normalised: 0 touches
    #: scores 0, this many or more scores the maximum on this factor.
    absorbed_count_cap: int = 4
    w_absorbed: float = 0.25

    #: §3.4 downside-capture ratio, reusing the exact resilient/fragile bands
    #: already defined in ``RelativeStrengthParams`` rather than introducing
    #: a second pair of thresholds for the same measurement. Capture at or
    #: below ``capture_resilient_max`` scores the maximum; at or above
    #: ``capture_fragile_min`` it scores zero; linear in between.
    w_rs_capture: float = 0.20

    #: §0's own central thesis -- a stock that has completed an
    #: accumulation-to-expansion cycle before is more trustworthy. Weighted
    #: lower than the strength of the spec's own emphasis would suggest,
    #: because the evidence for it is genuinely mixed across runs (positive
    #: and significant, p=0.0014, in the larger but admission-attribution-
    #: buggy 3,119-signal pass; positive but not significant, p=0.75, in the
    #: corrected 787-signal pass) -- reported as such rather than rounded up.
    prior_cycles_cap: int = 2
    w_prior_cycles: float = 0.15

    #: Bucket boundaries on the resulting [0, 1] weighted average. Fixed,
    #: principled cut points on the score's own natural scale -- not
    #: tercile-fit to this sample's own distribution, which would be one
    #: more data-mined threshold layered on top of the ones already spent.
    reject_below: float = 0.33
    high_conviction_at_or_above: float = 0.66


@dataclass(frozen=True, slots=True)
class PortfolioParams:
    """Exits (§6), position sizing (§7) and regime handling (§8).

    Two sections of the spec give *ranges* rather than numbers -- the profit
    ladder ("50% at 5-8%") and the time stop ("15-20 sessions"). A backtest
    needs a deterministic trigger, so each range's **lower bound** is used
    throughout: the earliest point a discretionary trader following the
    described process would act, and the conservative choice (locks in
    profit sooner, exits stagnant trades sooner) rather than a middle value
    picked to flatter the result.
    """

    starting_capital: float = 1_000_000.0
    #: §7's "2-3 concurrent positions". The cap the position-count limit
    #: enforces; ``target_concurrent_positions`` below sets sizing, which is
    #: a separate number from the hard cap.
    max_open_positions: int = 3
    target_concurrent_positions: float = 2.5
    #: Base position notional before any conviction/risk-appetite/regime
    #: scaling: capital divided across the target slot count. At the default
    #: ₹10L / 2.5 that is ₹4L -- deliberately not the spec's illustrative
    #: "~₹1L each", which undersizes if taken literally (₹10L / 2-3 is
    #: ₹3.3-5L, not ₹1L): the "~₹1L" reads as a discretionary trader's typical
    #: ticket on a smaller live account, not a formula, and sizing every
    #: position to a fixed 10% of capital regardless of account size would be
    #: a second, uncalibrated free parameter. Capital-proportional sizing is
    #: the one that scales. Used only as a *cap* on notional when
    #: ``size_by_risk`` is on (below); acts as the sizing rule outright when
    #: it is off.
    base_position_pct_of_equity: float | None = None   # None -> derived as 1/target_concurrent_positions

    #: Size by risk-per-share (entry to the anticipated stop) rather than a
    #: flat fraction of equity. Added after the first full run: flat-equity
    #: sizing gave every position the same rupee size regardless of how far
    #: its stop sat, and section 6.1's stop is a *closing* basis with no
    #: intraday circuit breaker -- on a handful of names that gapped down
    #: hard, SMA(10) lagged far enough behind the close for one trade to lose
    #: 10-15% before the stop caught it, at the same notional as trades whose
    #: stop was 2% away. That produced a 62% win rate with a sub-1 profit
    #: factor -- winning often, losing rarely but large, exactly the
    #: signature of risk that was never sized for. This is a completion of
    #: ordinary position construction, not a tuned fix: no competent trader,
    #: discretionary or systematic, sizes a position without reference to
    #: where its own stop sits, and the spec's "size scales with conviction"
    #: describes a multiplier on top of that, not a replacement for it.
    size_by_risk: bool = True
    #: Rupee risk per trade, as a fraction of current equity, at baseline
    #: (score-neutral) conviction -- conviction/appetite/regime multipliers
    #: scale this up or down exactly as they scaled flat-equity sizing.
    risk_per_trade_pct: float = 0.01
    #: The anticipated stop distance for sizing is entry price minus
    #: whichever of SMA(10) or the big-box bottom sits *higher* as of the
    #: bar before entry -- the one that will actually bind first, matching
    #: the exit rule's own "either condition" logic exactly rather than
    #: sizing against a stop tighter or looser than the one that can fire.
    #: A stop distance narrower than this floor (as a fraction of price) is
    #: clamped to it, so a coincidentally razor-thin gap does not imply an
    #: oversized position.
    min_risk_per_share_pct: float = 0.02

    # --- §4 bucket -> size multiplier ---
    #: "Reject -> hard-delete or below threshold": excluded from trading
    #: entirely, not merely downsized.
    size_mult_reject: float = 0.0
    size_mult_wait_and_watch: float = 0.7
    size_mult_high_conviction: float = 1.2

    # --- §6.1 stop loss ---
    sma_stop_period: int = 10
    #: Slack below the SMA before the closing-basis stop fires: the stop
    #: triggers on ``close < sma * (1 - buffer)`` rather than ``close < sma``.
    #: 0.0 is the spec's rule exactly. A buffer widens the stop without
    #: lengthening the average it is measured against, which is a different
    #: lever from raising ``sma_stop_period`` -- the average keeps reacting at
    #: the same speed, the trade just gets more room around it.
    sma_stop_buffer_pct: float = 0.0
    #: Chandelier-style alternative: stop on ``close < peak_close_since_entry
    #: - mult * ATR``, replacing the SMA condition entirely (the big-box-bottom
    #: condition still applies alongside it, as in the spec). ``None`` keeps
    #: the SMA rule. Position sizing follows whichever stop is active, so a
    #: wider stop takes a proportionally smaller position rather than the same
    #: notional at more risk -- the sizing bug §5 already paid for once.
    atr_stop_mult: float | None = None
    atr_stop_period: int = 14

    # --- §6.2 profit ladder: (trigger_pct, fraction_of_ORIGINAL_qty_sold) ---
    #: Lower bound of "5-8%".
    ladder1_trigger_pct: float = 0.05
    ladder1_fraction: float = 0.50
    #: Lower bound of "8-15%".
    ladder2_trigger_pct: float = 0.08
    ladder2_fraction: float = 0.20
    #: Lower bound of "20%+"; the spec's own range for this stage ("20-30%")
    #: is a *fraction sold*, not a trigger -- its lower bound (20%) is used
    #: for the same reason as the others.
    ladder3_trigger_pct: float = 0.20
    ladder3_fraction: float = 0.20
    #: Whatever remains after all three stages (10% at the defaults) trails
    #: on the §6.1 stop until it triggers.
    #: A stage whose fraction is <= 0 is skipped entirely rather than selling
    #: a token share -- this is what lets a variant run with no ladder at all
    #: (all three fractions zero) instead of quietly dribbling out 1 share
    #: per stage, which ``max(1, ...)`` on the sell quantity would otherwise
    #: do.

    # --- §6.3 time stop ---
    #: Lower bound of "15-20 sessions".
    time_stop_bars: int = 15
    #: "Modest profit": positive, but below the first ladder trigger -- a
    #: trade already through ladder 1 is working, not stagnant, and the spec
    #: does not intend the clock to override a position that is succeeding.
    time_stop_profit_ceiling: float = 0.05

    # --- hard maximum hold (not in §6; a guard, not a spec rule) ---
    #: Absolute ceiling on how many bars a position may stay open, regardless
    #: of profit. ``None`` reproduces the original §6 behaviour exactly, where
    #: the only ways out are the closing-basis stop, the ladder, and the
    #: "stagnant, modest profit" time stop -- a position that keeps closing
    #: above SMA(10) while running past the time stop's profit ceiling can
    #: stay open indefinitely. §5.1 traced the best-looking stop-widening
    #: configuration to exactly that: two positions held 86 and 156 bars
    #: through the 2020-21 recovery, contributing 32% of total P&L. This cap
    #: makes "the trade rode a multi-month rally" structurally impossible, so
    #: that any improvement a ladder variant shows has to come from the
    #: 15-20 bar window where Phase 4 actually measured the edge.
    max_hold_bars: int | None = None

    # --- §7 risk-appetite multiplier, driven by trailing realised P&L ---
    risk_appetite_lookback_days: int = 20
    #: Trailing realised P&L as a fraction of starting capital maps linearly
    #: onto the multiplier between these bounds.
    risk_appetite_min: float = 0.5
    risk_appetite_max: float = 1.5
    #: The trailing-P&L fraction (of starting capital) that saturates the
    #: multiplier at each bound.
    risk_appetite_saturation_pct: float = 0.03
    #: Below this multiplier, wait_and_watch entries are skipped entirely --
    #: "when it isn't [a good month], only clean setups" -- not merely
    #: downsized.
    risk_appetite_restrict_below: float = 0.8

    # --- §8 regime handling ---
    #: The regime reference index's own SMA period; below it on a closing
    #: basis is "weak".
    regime_sma_period: int = 50
    #: Additional size multiplier applied in a weak regime, on top of
    #: conviction and risk-appetite scaling -- "reduce quantity per trade".
    regime_weak_size_mult: float = 0.6
    #: In a weak regime, only high-conviction entries are taken at all --
    #: "take only high-conviction trades" -- matching the risk-appetite
    #: restriction rather than introducing a second, differently-shaped rule.


__all__ = [
    "BoxParams",
    "PortfolioParams",
    "RelativeStrengthParams",
    "ResistanceParams",
    "ScoreParams",
    "ScreenerParams",
    "TimeframeParams",
    "WatchlistParams",
]
