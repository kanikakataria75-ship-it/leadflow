# AES research log — Accumulation-Expansion Swing

Implementation and validation of `STRATEGY_SPEC.md`. This document records what
was built, what the data said, and — importantly — where the spec and the data
disagree.

Costs are the ones established in `RESEARCH.md` §8: realistic Indian delivery
round trip is **0.55–0.65%** (STT 0.10% on *both* sides). No result in this
document is reported gross.

---

## Phase 1 — Box detection engine

Modules: `aes/params.py`, `aes/boxes.py`, `aes/screener.py`, `aes/render.py`.
Tests: `tests/test_aes_boxes.py` (35 passing).

### 1.1 What was built

**Screener (§1.1).** 20% in 10 sessions, at/near the 52-week high, price > ₹50,
₹1 Cr average turnover. Over 382 symbols and 2015–2026 this admits **9,324
events across 327 names**, a median of 3 per session. That is a plausible
watchlist feed rate rather than a trickle or a flood.

*Gap:* §1.1 also asks for market cap > 500 Cr. There is no point-in-time
market-cap history in the cache, so it is not applied and the turnover floor is
a partial proxy. The reconstructed universe is therefore slightly wider than the
one actually traded.

**Box detection (§2).** The big box is *anchored*, not swept: its top is the
running-maximum high over each of several lookbacks — "the recent high where
price stalled" (§2.2) — and the box runs from that bar to today. Among valid
candidates the one with the **highest top** wins.

The first implementation swept every window length from 10 to 45 bars and kept
the best-scoring one. That was wrong in a way worth recording, because it is not
obvious: **almost every component of box quality grows with window length.** A
45-bar window has more swing lows, more chances to touch an edge, and more
midpoint crossings than a 12-bar one. Measured directly, `corr(quality, bars)`
was **0.60** — the detector was picking the longest candidate more than the
best-formed one, and 76% of all bars came back as "a box". Anchoring plus
rate-based structure metrics brought that correlation to **−0.01** and the
detection rate to 34%. There is a regression test pinning it.

### 1.2 Two bugs that only the renders exposed

**A single wick was defining the box floor.** NH on 07-Apr-2025 consolidated in
a 6.5% range, then printed one bar with a long tail. With the floor defined as
the plain minimum low, the detector drew an **18.8% "box"** whose bottom no
other bar had been near — and because dragging the midpoint down put 100% of
closes above it, the box scored a **perfect 1.00**. The worst-drawn box in the
sample was also the highest-rated one. No summary statistic would have shown
this.

**The first fix was worse than the bug.** Requiring the floor to be *touched
twice* removes the spike, but it also forces price to have traded back down to
the low — which pushes the nested small box into the lower half by construction.
Upper-zone boxes, the §2.1 positive case, fell from 1.7% of detections to
**exactly zero**. A detection gate that cannot produce the spec's best setup is
a bug, not a gate. Replaced with a test for the *isolated* floor (gap between
lowest and second-lowest low), leaving "how well is the floor held" to the
scorer where §4 says it belongs. There is a smoke test asserting upper-zone
boxes remain reachable.

### 1.3 Small box: length vs height

First-pass review: boxes matched well, but the small box was cutting itself
short — it should extend to cover more candles without getting taller.

Diagnosis: for every example checked, the small-box search was already picking
the *longest* window that stayed under the 10% range ceiling. It wasn't a
scoring artifact — going one bar further genuinely pushed the measured range
past 10%, every time. But the jump was often sharp (7.3% → 15.3% in one extra
bar), which meant a single bar's wick was doing the damage, not a real
widening of the typical range.

Fix: loosened the small-box edge quantiles from 90th/10th percentile to
80th/20th. This trims more of each window's outliers before measuring range, so
one bar's wick can no longer single-handedly end the search. Effect on the
three examples checked by hand: MINDACORP 7→10 bars, HEG 6→11 bars (now spans
the whole big box), CANFINHOME unchanged at 8 (the wider window didn't score
better even once it was admissible). Renders regenerated; all 35 tests still
pass.

One side effect worth flagging: extending the small box backward can pull in
earlier, lower-priced bars, which drags its midpoint down and can flip its zone
classification (MINDACORP's small box moved from "upper" to "middle" as it grew
from 7 to 10 bars). That's not a bug — a genuinely wider coil *is* positioned
lower — but it means zone and length aren't independent, and a future version
scoring both should account for that coupling rather than treating them as
orthogonal signals.

### 1.4 Where the spec and the data disagree

**"Minimum 2 higher swing lows" (§2.4) is close to unmeasurable.** Boxes that
form after screener admission have a **median duration of 11 bars**. With a
conventional 2-bar fractal, an 11-bar window yields a median of **one** confirmed
swing low, so the criterion is satisfiable by 2% of real boxes — a near-constant
zero rather than a feature.

Loosening to a 1-bar fractal makes it live (median 2 swing lows, 14% meet the
criterion). But on a crude forward proxy — did price clear the box top by 3%
within 20 sessions — it does not separate:

| §2.4 criterion | met | break-up rate | not met |
|---|---|---|---|
| ≥2 higher lows | 14% | **23.7%** | 31.1% |
| lows tilt > 0 | 31% | 30.8% | 29.7% |
| **closes above midpoint > 50%** | 32% | **35.1%** | 27.7% |

Higher lows point the *wrong way*; time spent in the upper half separates. Both
are §2.4 criteria and they are not equally useful. This is a provisional read on
a raw break-up proxy, not forward return net of cost — it is a Phase 3 weight
calibration question, and the factor stays in until then.

**"Box top is where price stalled" (§2.2) is not typical.** In **69%** of
detected boxes the top is touched exactly once: the leg peaks, price pulls back
and bases, and the high is never retested before resolution. Retest is a real
discriminator but it cannot be a requirement.

**The 25% ceiling silently substitutes a lower top.** When the genuine
consolidation is wider than 25%, no valid box exists at the real high, so the
detector anchors on a lower, more recent one. CANFINHOME 21-May-2015 is the
clean example: the box top is drawn at 139.0 while the actual overhead supply
sits at ~152, only 9% higher. Clearing 139 there is not a breakout. This is not
a detector fix — it is exactly what §3.2 resistance headroom exists to catch,
and Phase 2 must surface it to the scorer.

**A "stale" box resolved perfectly.** APARINDS 14-Mar-2024 ran 25 bars — past
§2.3's one-month staleness line — with the top touched 4 times, volume drying to
0.56x, and then a clean, sustained breakout. One case proves nothing, but the
staleness penalty is a calibration target rather than a given.

### 1.5 Base rates worth carrying forward

Recomputed after the §1.3 quantile fix (all figures below reflect it).

| Measure | Value |
|---|---|
| Admissions producing a box within 30 sessions | 53% |
| **Median gap, screener day → box detection** | **15 sessions** |
| Median big-box duration / range | 11 bars / 19.1% |
| Small box present | **86%** of detections (was 66% before §1.3) |
| Small-box zone: bottom / middle / upper | 33% / 51% / **2.5%** |
| Closes above midpoint > 50% | 32% |
| Cleared box top by 3% within 20 sessions | 30% |
| Closed below box bottom within 20 sessions | 50% |

The 15-session median gap is the point of the whole design: **entry is a fortnight
after the screener fires**, not on it. Whatever else happens, this study is not
repeating the previous one's mistake of buying maximum short-term stretch.

Raw break-up rate by small-box zone, which is the §2.1 claim:

| Zone | Break-up rate | n |
|---|---|---|
| bottom (spec: red flag) | 21.4% | 523 |
| middle | 31.5% | 812 |
| upper (spec: strong) | **42.5%** | **40** |

Still directionally what §2.1 says, and letting the small box run longer (§1.3)
roughly tripled the upper-zone sample (12 → 40) and pulled its rate down from
58% to 43% — the earlier number was an artifact of a tiny, favourably-selected
sample, not a more accurate read. n=40 is still too small to lean on. Worth
noting *why* upper-zone stays rare even after the fix: a 15–25% big box whose
coil sits in the top third requires a deep pullback followed by a full
recovery, and §1.3 showed that letting the small box extend can itself pull its
midpoint down into "middle" — so length and zone partly trade off against each
other. If the upper zone still carries the edge once Phase 3 measures net of
cost, the strategy is more selective than "2–3 concurrent positions" implies.

### 1.6 Look-ahead audit

Structural, not statistical, and carried over from the earlier studies:

- detection at bar `i` is identical whether the frame ends at `i` or runs years further
- overwriting every bar after `i` with garbage (×7.5, volume 9.9e9) changes nothing at `i`
- the prior-cycle count cannot see a future breakout
- historical episode boundaries do not shift when later data arrives

### 1.7 Open decision

**Box edges: wicks or bodies?** §2.1 says wicks may exceed the *small* box and
§6.1 makes the stop closing-basis — both argue for bodies. But §6.1 also says
box-bottom entries stop at "the low of the big-box bottom", which reads like a
wick, and the 15–25% figure was presumably read off a chart, which shows wicks.

| Basis | Boxes | Median range | Upper-zone |
|---|---|---|---|
| wick (current default) | 1,073 (54% of admissions) | 19.1% | 0.8% |
| body | 881 (44%) | 18.4% | 0.9% |

Bodies lose ~18% of detections, mostly by falling under the 15% floor, and
change little else. Default stays `wick` pending a decision; renders of both are
in `results/aes_phase1_edge_basis/`.

---

## Phase 2 — Context features

Modules: `aes/resistance.py` (§3.2/§3.3), `aes/timeframes.py` (§3.1),
`aes/relative_strength.py` (§3.4). Tests: `tests/test_aes_context.py` (26
passing, 139 total). All three follow the same look-ahead discipline as
Phase 1: every function reads only `[0 .. i]`, proven with truncation and
future-mutation invariance tests, not just asserted.

### 2.1 Resistance, age-weighted (§3.2) and with a visit history (§3.3)

A resistance "level" is a cluster of swing highs at similar prices (tolerance
2.5%). Age is measured from the level's **first** touch, not its most recent
one — matching §3.2's "how far back it formed" literally, since a level made
three years ago and retested last month is old, not recent. Each touch also
carries a volume ratio against its own trailing baseline and a reaction label
(`rejected` if price fell 5%+ within 10 bars afterward, `absorbed` otherwise),
which is §3.3's "was it rejected hard, or absorbed and moved through".

**This directly catches the Phase 1 CANFINHOME problem.** Phase 1 flagged that
the 25% box-height ceiling forced the detector to anchor on a top at 139 when
the real overhead supply sat near 152 — a false breakout waiting to happen.
Run against a box-bottom entry (reference price 111.4), the resistance module
finds a level at **117.4, only 5.4% away** — well short of the 12-15% minimum
headroom §3.2 asks for — before ever reaching the 139 top or the real 152
ceiling. This is exactly what headroom is supposed to prevent, and it fires
correctly without having seen the Phase 1 write-up: it is reading the same
price history the box detector already had.

### 2.2 Multi-timeframe breakout (§3.1) — an honest limitation

Daily/weekly/monthly are each tested for a fresh closing high over a roughly
1-year / 3-year / 6-year lookback, built from the frame truncated to the
evaluation bar *before* resampling — a plain whole-history resample would let
Thursday's bar see Friday's, which the truncate-then-resample order rules out
by construction (tested by constructing a week where a naive resample and the
causal version disagree).

The limitation the spec's own build order predicted needs saying plainly.
Section 3.1 wants three tiers of significance — "daily good, weekly strong,
monthly strongest" — implying they're usually distinguishable events. On this
universe they mostly aren't: of screener-admission days where **any**
timeframe confirms a fresh high, **86% confirm on all three simultaneously**.
The reason is structural, not a bug: the screener itself selects for stocks at
or near their 52-week high, so a stock clearing a 1-year closing high on this
universe has usually not printed a higher close in 3 or 6 years either — the
three conditions are correlated by construction of who gets on the watchlist
in the first place. The three booleans are kept (a genuine daily-only or
weekly-only breakout is real when it happens, and the 14% of cases where they
diverge may still be informative), but Phase 3 should not expect this factor
to carry three independent tiers of signal on this universe.

### 2.3 Drawdown-conditional relative strength (§3.4)

Downside-capture ratio: cumulative stock return on the index's down days
(index daily return ≤ −0.3%, 60-bar trailing window), divided by the index's
own cumulative return on those same days. Built to reject a specific failure
mode the spec called out: this is **not** stock-return-minus-index-return,
which would reward a stock that simply went up more regardless of what it did
under pressure. A unit test constructs a stock that is calm on the index's
down days but rips on the index's up days — plain relative strength would call
that strong; downside capture correctly reports it as resilient *only* on the
down-day behaviour, which is the distinction §3.4 is asking for.

Across 35 real names, capture ratios split into a plausible three-way spread —
resilient 35% / inline 32% / fragile 33% — median 0.97 (essentially in-line
with the index, as a population should be) with a 10th-to-90th percentile
range of 0.08 to 1.88. No calibration claim yet; this is a distribution check,
not a backtest.

---

## Phase 3 & 4 — Calibration and bar-level edge

Protocol: `aes/protocol.py`. `AES_DISCOVERY` (2015-2021) only; `AES_HOLDOUT`
(2022-today) is not opened by any code in this section. Reproduce with:

```bash
python -m nifty_swing_bot.research.aes_calibration
```

### 3.0 A structural bug found before any calibration number could be trusted

The screener re-admits a name on **every** consecutive day it satisfies "20%
up in 10 days and near its 52-week high" -- which, for a genuine multi-week
move, is many sessions in a row. The first pass through this pipeline treated
each admission as an independent trial. Each one walks forward and finds the
*same* box and the *same* breakout, so **94% of the first-pass "signals" were
the same underlying event counted up to six times** -- and the duplication
count itself correlates with the outcome, since a stronger, longer-lasting
move satisfies the screener on more consecutive days. Left unfixed, this
silently overweights exactly the cases most likely to look good, in the
strategy's favour, without anyone tuning anything.

Fixed structurally, not by filtering after the fact: `aes.signals.build_signals_for_symbol`
walks each symbol's admissions in order and only starts a new search once the
*previous* one has actually resolved (its entry bar, if a signal fired; the
full search horizon, if it never did). A second bug in the same family:
`load_universe`'s glob over the OHLCV cache silently includes the benchmark
index itself (`CRSLDX.parquet`) as if it were a 382nd stock. Its impact was
negligible (an index compared against itself has zero excess by construction,
so it only diluted the base-rate panel with ~0.4% of rows that contribute
exactly zero) but it was methodologically wrong and is now excluded explicitly.

After both fixes: **4,661 raw admissions collapse to 342 genuine episodes**
across 169 symbols, 2015-2021, using the screener's own admission floor (₹1 Cr
turnover -- no additional liquidity restriction; that was tested and rejected
in the prior study and is not reapplied here).

### 3.1 Breakout volume multiplier -- no reliable relationship found

```
  bucket      n   mean_net%  median_net%    t      win%
  <1.0x      71      4.24         1.56     2.35    54.9
  1.0-1.5x   46      2.18        -2.26     0.91    45.7
  1.5-2.0x   43      3.40        -1.22     1.29    48.8
  2.0-2.5x   35      1.27        -0.76     0.52    45.7
  2.5-3.0x   30      1.48         0.17     0.56    50.0
  3.0x+     117      1.05        -1.71     0.72    45.3
```

This does not look like the spec's assumption. If anything, the **weakest**
volume (below the stock's own 20-day average) shows the strongest bucket
(t=2.35, the only one nominally significant on its own, though this is one of
six comparisons and would not survive a Bonferroni correction). There is no
monotonic climb toward 2-3x, and the buckets above 2x are indistinguishable
from noise (t between 0.5 and 1.3, n=30-117 each). The starting-guess default
of 2.0x sits in the middle of a flat, noisy relationship, not at a peak.

**Two honest limitations, not a rejection.** First, sample size: 30-117 per
bucket over 7 years is not enough to resolve a moderate effect if one exists
-- this sweep is underpowered, not necessarily null. Second, and more
important: this measures volume ratio at whatever bar happened to be the
*first* close above the top, which is a fixed, already-realised event. It
does not simulate what would happen if the strategy *waited* for a bigger
volume day before entering (a materially different mechanism -- delaying
entry changes which bar becomes the trigger, not just which of the same bars
gets kept). That harder simulation is future work if this factor is worth
revisiting; on the data available now, **no defensible multiplier can be
calibrated**, and the honest recommendation is not to hard-gate on one.

### 3.2 Circuit-mover threshold -- sample too thin to calibrate at all

```
  N>0: excludes  17 signals (5.0% of 342), mean_net =  0.54%
  N>1: excludes  11 signals,               mean_net = -1.63%
  N>2: excludes   6 signals,               mean_net = -5.83%
  N>3: excludes   3 signals,               mean_net = -0.25%
```

Only 17 of 342 signals come from names with *any* circuit-like day in the
trailing 60 sessions at admission, and only 6 have more than 2. The `N>2`
excluded group's mean (-5.83%) is suggestive of the spec's intuition, but n=6
proves nothing on its own. This is not a null result on the filter's
*usefulness* -- it is a null result on being able to *calibrate* the exact
threshold from this sample. The existing turnover/price floors already
exclude most true circuit-prone names before AES ever sees them, which is
probably why the spec's own instinct (start at 3, expect to tune it) barely
has anything left to bind on. Recommendation: keep the exclusion as a
cheap safety rail at the spec's starting value (3 hits / 60 sessions) rather
than tuning it -- it costs almost nothing on this universe (5% of signals at
most) and there is no data-driven reason to move it.

### 3.3 Scoring-factor calibration -- almost nothing survives individually

Every section 2-4 factor, tested for a univariate relationship with net
20-day forward return, n=342 (n=238 where resistance headroom exists):

```
  factor                      n    pearson r     p
  above_mid                  342     +0.009    0.86
  big_quality                342     -0.004    0.94
  higher_lows                342     +0.021    0.70
  duration_vs_norm           342     +0.006    0.91
  prior_boxes                342     +0.057    0.29
  prior_cycles               342     +0.069    0.21
  vol_dryup_ratio            342     +0.077    0.16
  headroom_pct               238     +0.006    0.93
  resistance_age_strength    238     +0.052    0.42
  resistance_rejected_count  342     +0.004    0.94
  resistance_absorbed_count  342     -0.013    0.81
  tf_score                   342     +0.029    0.60
  rs_capture                 342     -0.079    0.15
  vol_ratio                  342     -0.003    0.96

  factor (categorical)        n    ANOVA F     p
  zone                       342      1.15    0.33
  used_small_top             342      0.18    0.67
  clears_min_headroom        238      0.00    0.97
  tf_monthly                 342      0.51    0.48
  rs_classification          342      0.91    0.40
```

Stated as bluntly as asked for: **not one factor clears conventional
significance at this sample size.** The best of them --
drawdown-conditional relative strength (`rs_capture`, §3.4) -- comes closest
(p=0.15, and its direction is right: lower capture, meaning the stock fell
*less* than the index on the index's bad days, associates with better
forward return) but does not clear p<0.05, let alone survive correction for
testing 19 factors at once. Box quality, higher lows, zone, duration-vs-own-
norm, prior cycles, resistance headroom and age, timeframe confluence -- all
of them are statistically indistinguishable from zero here.

**This is a power problem as much as a content problem, and both matter.**
n=342 spread across 7 years is a small sample for resolving a moderate
per-factor effect -- Phase 1/2's multi-stage filter (screener admission ->
box within 30 bars -> breakout within 40 bars) is far more selective than the
DAB/MR study's single-bar rules, which mechanically shrinks the usable sample
by roughly an order of magnitude for the same history. That is a structural
consequence of the watchlist design, not a bug, but it means this sample
cannot currently support fine-grained weight calibration, and assigning the
spec's suggested weights (§4's table) as anything more than provisional
would be pretending to a precision the data does not have.

**One concrete, checkable finding regardless of significance:** resistance
headroom as a **near-gating** filter, as §4's weight table suggests, is not
practically viable on this data. Only **9 of 238** signals with a measurable
resistance level (3.8%) clear the 12-15% minimum headroom the spec asks for.
Gating on it would eliminate the large majority of otherwise-valid signals.
Whatever weight it carries, it cannot be a near-gate as specified.

### 3.4 Bar-level edge vs. a random entry in the same universe (Phase 4)

Same yardstick the DAB/MR study used (`research/signal_lab.py`): excess
return over the index, compared against the same excess computed on **every**
evaluable bar of the universe (464,361 bars, 2015-2021) -- the random-entry
baseline -- net of the realistic 0.585% round-trip cost.

```
  horizon   n_sig   AES(idx-rel)%   base(idx-rel)%   gross_excess%   net_excess%    t
     5       342        0.09             -0.02           0.11          -0.47      0.25
    10       342        0.74              0.21           0.53          -0.05      0.92
    15       342        1.44              0.44           1.00          +0.41      1.48
    20       342        1.72              0.68           1.04          +0.46      1.30
```

AES entries beat the random-entry baseline at every horizon, and the gap
widens with horizon -- the same shape the DAB/MR mean-reversion study found
at its optimum (net edge crossing zero between 15 and 20 bars). Net-of-cost
edge is **positive at 15 and 20 bars** (+0.41%, +0.46%).

**It is a much weaker result than that comparison invites, and I am not
going to round it up.** The DAB/MR finding had t=4.65-4.94 on 3,600+
signals. This has t=1.30-1.48 on 342. A t-stat around 1.3-1.5 corresponds to
roughly an 85-90% one-sided confidence the true edge is positive -- better
than a coin flip, nowhere near the bar anything should be committed to
capital on. The point estimate is positive; the confidence behind it is not.

I am stopping here rather than proceeding into Phase 5's portfolio backtest,
per the standing instruction to tell you plainly rather than let a portfolio
simulation make a marginal number look more decisive than it is. This is
not a "the strategy failed" stop -- the edge is directionally there, at the
horizon the spec's own profit ladder targets, in a sample built specifically
to preserve the screener/entry gap and to exclude the biases the prior
studies were burned by. It is a "the sample cannot yet support the next,
much more expensive step with any confidence" stop.

### 3.5 Relaxing the search windows -- tested directly, and it made things worse

Asked which lever to pull, the choice was to widen `max_box_wait` /
`max_breakout_wait` (30/40 bars) to catch more episodes from the same seven
years. Tested directly, symmetric window sweep, `fwd_net_20`:

```
  wait(box/brk)    n   syms  gap_med  gap_p90    net20%    t20
     8/8            7     7      17      21       4.63    0.76
    10/10          51    47      19      28       2.33    1.17
    12/12          77    63      21      31       2.08    1.30
    15/15         116    85      24      35       1.76    1.53
    18/18         159   109      28      41       2.09    2.02
    20/20         181   115      30      42       1.30    1.53
    25/25         250   139      36      52       0.51    1.13
    30/30         317   165      39      62       0.66    1.46
```

**Widening did not help.** Sample size climbs steadily (7 -> 317), but net
edge does not climb with it -- past roughly 18-20 bars it *falls* as the
window widens further (0.66% at 30/30, versus 2.09% at 18/18), and the
admission-to-breakout gap stretches out to a median of 39 days, p90 62 days
at the widest setting -- two months from screener admission to entry, which
strains the "days to weeks" framing the whole design is built around. The
extra episodes a wider window catches are, on average, *weaker* ones. This
directly contradicts the hope that more data would also mean more
confidence, and it is worth having tested rather than assumed.

There is a visible peak around 15-20 bars for both windows (18/18 reaches
t=2.02 at 20 days, the single best t-stat anywhere in this document). **I am
not adopting that as a calibrated value, and want to be explicit about
why not, since it is the exact failure mode I was told to avoid.** That
number is the best of eight configurations tried specifically because they
were tried -- searching a free parameter for the best-looking t-stat and then
reporting it as a finding is p-hacking with extra steps, regardless of intent.
What the sweep legitimately shows is *shape*, not a *value*: edge is
concentrated in the faster cycles and dilutes as the window widens, which is
independently consistent with the spec's own §2.3 language ("beyond [~22
sessions] the setup is stale -- deprioritise"). That is a reason to prefer a
spec-grounded window (something in the 20-25 bar range, matching §2.3's own
duration ceiling) over an empirically-mined one, not a reason to declare 18
special.

**Net assessment of the three original options, now that one has been
tried:** relaxing the windows is answered -- it does not resolve the
confidence problem, it trades it for a different, weaker one. What remains:

1. **Extend history earlier than 2015**, if point-in-time constituent data
   allows it -- the one option that adds genuinely independent information
   rather than diluting what is already there.
2. **Proceed to Phase 5 anyway**, on a spec-grounded (not swept-and-chosen)
   window near the original 20-30 bar default, with the low confidence at
   this bar-level stage stated plainly on every number Phase 5 produces, and
   Phase 6's walk-forward treated as the real test.

---

## Phase 3.6 — The persistent watchlist and all three entry modes

Reassessment, not more parameter search. Two real gaps in scope, not sample
size: only entry mode 1 (breakout momentum) of the spec's three (§5) existed
in code, and `build_signals_for_symbol` searched a fixed window from each
*admission*, not a genuinely persistent watchlist that stays open on a name
-- checking it every session -- until hard-delete, invalidation, or
staleness (§1.3). A name that formed a box, failed to break out, and later
formed a *different* box without a fresh admission was invisible to the
first pass. New modules: `aes/watchlist.py` (the state machine, all three
entry modes), tests in `tests/test_aes_watchlist.py` (15 passing, 175 total).

### Two real bugs the implementation surfaced

**Box noise re-triggering itself.** The first version resolved a mode-3
entry and immediately re-detected the *same* oscillating box on the very
next down-tick, firing again on ordinary internal wiggle. Fixed by requiring
a freshly detected box to start strictly later than the one an episode just
ended on -- a genuinely new structure, not the same one seen again.

**An 819-day median admission-to-breakout gap.** The watchlist's "drop after
N bars with no box" clock was being reset by *any* episode ending, including
one that formed and was invalidated in days. Since ordinary volatility means
some box or other recurs often enough on most active stocks, that clock
almost never accumulated to the limit, and a name admitted once, years
earlier, stayed "on watch" for effectively its entire subsequent history --
with every signal it ever produced misattributed to that one ancient
admission. First full run: 3,126 signals, median gap 819 days, net edge at
20 bars **-0.28%** (t=1.29) across every entry mode. Both numbers were a
symptom, not a result. Fixed by tying the eligibility clock to admission
*recency* specifically -- refreshed by a fresh admission, unaffected by
episode churn in between -- rather than resetting it on any episode ending.
Median gap dropped to a sane 76 days; both fixes are now regression tests.

### The corrected numbers

787 signals (was 342 under the old per-admission, mode-1-only pass), 218
symbols. Entry mode counts: 480 breakout / 289 retest / 18 box-bottom.

```
  horizon   n    net_excess%    t
     5     786      -0.69      -0.40
    10     786      -0.22       0.96
    15     786      +0.02       1.44
    20     786      +0.01       1.26

  by entry mode, h=20:
  1 breakout   480    -0.42    0.27
  2 retest     289    +0.48    1.40
  3 box-bottom  18    +3.92    1.30   (n too small to trust the point estimate)
```

**The complete, correctly-implemented mechanics do not resolve the
confidence problem -- if anything the pooled result is now flatter than the
narrower 342-signal measurement**, not stronger. Net edge at 15-20 bars is
essentially zero (t~1.3-1.4), not the +0.41-0.46% the mode-1-only pass
showed. One new, real piece of structure: mode 2 (retest) looks like it
may carry more of what edge exists than mode 1 (breakout momentum) does --
opposite of the spec's own "most used" framing for mode 1 -- though at
n=289 that is a lead to weigh, not a settled result.

### A third independent confirmation of the same pattern

Phase 3.5's window sweep found edge concentrating in fast admission-to-entry
cycles and diluting in slow ones, with the caveat that mining the sweep for
its best-looking configuration would be self-fooling. This dataset was not
built by searching for that pattern -- the watchlist's eligibility window is
now the same 90 bars for every signal -- and it reproduces it anyway:

```
  gap bucket   n     net%    t     win%
  <20 days     41    7.12   2.77   63.4
  20-40        115   1.70   1.21   43.5
  40-60        110   2.00   1.65   52.7
  60-90        185   1.05   0.99   49.2
  90+          311   1.45   1.79   50.5
```

The under-20-day bucket is the single most convincing number in this
document (t=2.77, n=41, 63% win rate) -- and it is the *smallest* bucket,
found without being searched for. Three separate analyses now, under three
different constructions, agree: whatever edge AES has concentrates in fast
resolutions and is diluted by slow ones. That is worth real weight.

### Updated factor calibration and calibration targets

`resistance_absorbed_count` (§3.3) is newly significant at this sample size
(p=0.037, n=787) and cleanly monotonic: signals with more than two absorbed
touches at a nearby resistance zone average 3.86% net at 20 bars versus
1.50% for the rest -- exactly the spec's own claim ("strong prior volume at
the zone with eventual resolution upward = positive"). Every other Phase 3
factor remains statistically indistinguishable from noise at this n.

The circuit-hit sample is still thin (25 of 787 signals have any hits) but
now shows a clean, monotonically worsening pattern as the threshold
tightens -- N>0 excludes 25 signals averaging -1.66%, N>1 excludes 18
averaging -6.79%, N>2 excludes 9 averaging -10.75% -- more suggestive than
before, though n=9-25 still is not a number to calibrate a precise
threshold from. The recommendation stands: keep the exclusion at the spec's
own starting value as a safety rail.

### Where this leaves Phase 5

The honest picture, combining everything from Phase 3 onward: bar-level net
edge on the complete, correctly-implemented mechanics is close to zero, not
clearly positive and not clearly negative, at conventional confidence. Three
things now point the same direction on what to do about it: fast cycles
show real edge; slow ones do not; and a couple of factors
(`resistance_absorbed_count`, and directionally `rs_capture` and
`prior_cycles` from the earlier, larger-but-buggy run) look like genuine,
spec-consistent structure rather than noise. The open question is whether a
scored combination of "fast resolution" plus these surviving factors carves
a genuinely positive-edge subset out of this population -- which is what
§4's scoring system is *for* -- versus the pooled, unfiltered population
simply not having a reliably positive edge to find. That is a Phase 4/5
question, not a reason to keep re-deriving Phase 3 inputs.

---

## Phase 4 — The section 4 scoring system

Modules: `aes/scoring.py`, `ScoreParams` in `aes/params.py`. Tests:
`tests/test_aes_scoring.py` (10 passing, 185 total).

### Weights, calibrated rather than assumed

The spec's own §4 table lists eleven factors with qualitative weights
(High/Medium/Bonus/Negative) and says outright they are "to be calibrated by
backtest." Phase 3 did that calibration on 787 discovery-period signals.
Four factors are weighted into the score; the other seven the spec names
are still measured and reported on every signal, but carry **zero** weight,
because none showed a measurable relationship with net forward return at
this sample size:

| Factor | Weight | Evidence |
|---|---|---|
| Fast resolution (admission→breakout ≤ 20 days) | 0.40 | Strongest signal in the project; reproduced under three independent constructions without being searched for |
| Resistance absorption (§3.3) | 0.25 | p=0.037, n=787, cleanly monotonic |
| Drawdown-conditional relative strength (§3.4) | 0.20 | Directionally consistent every run; significance is sample-size dependent |
| Prior completed cycles (§0's own thesis) | 0.15 | Positive both runs; significant (p=0.0014) only in the larger, buggier one |

Box quality, higher lows, zone, duration-vs-own-norm, timeframe confluence,
resistance headroom as a near-gate: zero weight. That is the blunt answer
the brief asked for -- most of the spec's named factors are noise on the
data measured so far.

### Does the score actually separate edge? Yes, cleanly.

Score correlates with net 20-day return directly, not just at bucket
boundaries: **r=+0.093, p=0.0087, n=787**. Small, but real and in the right
direction. Bucketed:

```
                    n     net10%   t10    net15%   t15    net20%   t20    win%(20d)
  reject           412    -0.47   -0.84    0.20    0.31    0.63    0.88    46.1
  wait_and_watch   331    +1.36    2.38    2.56    3.92    2.92    3.92    54.7
  high_conviction   44    +2.66    1.24    3.29    1.54    4.82    2.05    54.5
```

Monotonic at every horizon. `wait_and_watch` (42% of the sample) carries the
statistical weight (t=3.92 at both 15 and 20 bars); `high_conviction` (6% of
the sample) carries the largest point estimate with a still-solid t=2.05
despite n=44. `reject` (52% of the sample) is flat to slightly positive and
never significant -- the score is successfully finding a no-edge subgroup
and separating it out, not just re-labelling the whole population.

**A caveat worth stating rather than skipping past.** Three of the four
weighted factors were *selected* by calibrating against this same 787-signal
sample, which is a real, if modest, in-sample-selection risk -- the score
is not guaranteed to separate this cleanly on data it has not seen. The
strongest factor (fast resolution) is the one exception: it was found on an
earlier, structurally different (and buggy) dataset before this score
existed, and reproduced here independently. The other three should be read
as "survived calibration on the only data available," not "proven." This is
exactly what `AES_HOLDOUT` exists to settle, once, at the end.

### Where this leaves the phase-4 stop condition

The standing instruction was explicit: measure bar-level edge net of cost,
and stop rather than proceed to a portfolio backtest if it is not positive.
On the *pooled* 787-signal population that measurement was flat (§3.6). On
the scored population it is not: `wait_and_watch` + `high_conviction`
together are 48% of the sample (**n=375**) and show **net 20-day edge
+3.15%, t=4.41, p=0.00001** pooled -- a materially different, more
confident result than anything the pooled measurement produced, and the
strongest t-stat anywhere in this document on a sample this size. That
clears the bar this checkpoint was built to enforce, on the population the
strategy would actually trade -- conviction-scaled entries per §7, not
every technically valid breakout indiscriminately.

---

## Phase 5 — Portfolio backtest

New module: `aes/portfolio.py` (`AESPortfolioBacktester`). Tests:
`tests/test_aes_portfolio.py` (15 passing, 200 total). Driver:
`research/aes_portfolio_run.py`.

### Why a new engine, not the existing `PortfolioBacktester`

The existing engine's exit check is intraday by construction
(`if low <= pos.stop`) and closes a position in one shot. Neither is
compatible with §6: a **closing-basis** stop ("intraday wicks must not
trigger") and a **staged ladder** that sells part of a position while the
remainder keeps running. Reusing the existing engine's exit method verbatim
would have meant silently reinterpreting "closing basis" as the same
intraday check with a different number. What *is* reused directly:
`backtest.costs` for every charge, and `backtest.metrics.compute_stats` /
the new `alpha_beta` regression -- `AESTrade` is shaped to satisfy
`compute_stats`'s existing duck-typed interface rather than a second copy of
win-rate/Sharpe/drawdown arithmetic living alongside the first.

Two ranges the spec gives, not numbers, needed resolving once: the profit
ladder ("50% at 5-8%") and the time stop ("15-20 sessions"). Both are fixed
at their **lower bound** in `PortfolioParams` -- the earliest point a
discretionary trader following the process would act, and the conservative
choice, not a middle value picked to flatter the result.

### A sizing bug the first run's numbers exposed

First pass sized every position as a flat fraction of equity, scaled only
by conviction/appetite/regime -- with no reference to how far its own stop
sat. Result: **62% win rate, profit factor 0.963, total return -11.45%**.
That combination -- winning often, losing rarely but large -- is the
textbook signature of risk that was never sized for, and the trade log
confirmed it directly: GRANULES lost 15.3% before its SMA(10) stop caught
it, PFOCUS 13.4%, IDBI 12.5%, each carrying the *same* notional as trades
whose stop sat 2% away. Section 6.1's stop is closing-basis with no intraday
circuit-breaker, so a name that gaps down and keeps closing lower for a few
days can run well past where a 10-day average would ordinarily catch it --
and nothing in the first pass's sizing accounted for that.

Fixed by sizing on risk-per-share (entry to whichever of SMA(10) or the
big-box bottom sits higher -- the one that actually binds first, matching
the exit rule's own logic) rather than a flat notional. This is a
completion of ordinary position construction, not a tuned fix: no
competent trader sizes a position without reference to where its own stop
sits, and §7's "size scales with conviction" describes a multiplier on top
of that, not a replacement for it. After the fix: **65% win rate, profit
factor 1.043, total return -0.8%** -- both runs are reported here rather
than only the second, since silently replacing a bad number with a better
one without showing the delta would be exactly the kind of thing this
project has tried not to do.

### The headline result: roughly flat, and a mechanical reason why

```
  total_return_pct     -0.80%   (7 years, ₹10L starting capital)
  cagr_pct              -0.07%
  alpha_annual_pct      -0.36%
  beta                   0.031
  time_in_market_pct    28.22%
  max_drawdown_pct      13.45%
  sharpe                -1.54
  win_rate_pct          65.14%
  profit_factor          1.043
  avg_bars_held           5.95
  trade_count             327
```

**Beta is clean.** 0.031 is negligible market exposure -- whatever this
number is, it is not the previous DAB/MR study's problem of apparent
outperformance being beta from sitting 85% invested. Time-in-market at 28%
confirms it independently: AES, run this way, is mostly in cash.

**The result does not match Phase 4's bar-level measurement, and the reason
is mechanical, not mysterious.** Phase 4 measured net edge on a *fixed*
20-bar horizon: +3.15% (t=4.41) on this exact signal population. Broken
down by horizon, that edge is heavily back-loaded --

```
  h= 5:  +0.34%  t=0.82   (not distinguishable from noise)
  h=10:  +1.51%  t=2.69
  h=15:  +2.64%  t=4.22
  h=20:  +3.15%  t=4.41
```

-- while the portfolio's average hold is **5.95 bars**. Section 6's own
exit discipline -- a 10-day closing-basis stop, and 70% of every winning
position sold by the +5%/+8% ladder stages -- resolves most trades well
before bar 10, let alone bar 20. The strategy is being asked to hold long
enough to realise an edge that, on this data, does not show up until
roughly the third week, while its own stop and profit-taking rules are
tuned to release capital in the first week. Both of those rules are
faithful to §6 as specified, not implementation choices made here -- this
is a genuine tension *within the spec*, not a bug to quietly patch by
loosening the stop or delaying the ladder to chase a better number.

### What this does and does not say

This is not "AES has no edge" -- Phase 4 showed a real, reasonably
confident one on the scored population. It is "AES's specified exit
discipline, implemented exactly as written, does not hold positions long
enough to capture most of that edge on this data." Two honest ways to read
that, and the choice between them is not mine to make unilaterally:

1. **The exit discipline is the strategy, deliberately.** §6 was traded
   discretionarily for four years and reflects real risk management (an
   early, aggressive ladder that locks in gains and a tight stop that
   limits how long a wrong idea is funded). If so, a near-flat, low-beta,
   low-drawdown result over a genuinely difficult mid/small-cap period
   (2015-2021, including the 2018-19 small-cap bear and the COVID crash) is
   itself not a bad outcome, and the honest report is "roughly breakeven
   net of realistic cost, cleanly beta-neutral."
2. **The mismatch is real and worth resolving before the holdout.** If the
   measured edge lives at 15-20 bars, an exit discipline that releases
   capital in 5-6 does not fit it, and the ladder/stop parameters -- fixed
   here at the spec's own lower bounds, not calibrated -- are a legitimate
   Phase 3-style calibration target Phase 3 never actually reached, because
   it did not yet have a portfolio engine to measure against.

### 5.1 Loosening the exit rules to fit the edge's timeframe — tested, and rejected

Asked directly: does widening the closing-basis stop (so positions can
survive long enough to reach the 15-20 bar horizon where Phase 4's edge
actually lives) produce a real, robust improvement? Swept the SMA stop
period and time-stop bars, holding the profit ladder at its spec-stated
lower bounds:

```
  config                        n    avg_hold   total_ret%  alpha%   beta    PF     maxDD%
  sma10/time15 (baseline)      327     5.95       -0.80     -0.36   0.031  1.043    13.45
  sma15/time15                 296     7.95      +13.09     +0.73   0.037  1.250    11.28
  sma20/time15                 282    10.04       -2.78     -0.61   0.035  0.995    14.95
  sma20/time20                 286    10.65       +2.68     -0.13   0.036  1.096    12.05
  sma25/time20                 259    12.37      +23.13     +1.49   0.036  1.575    11.89
```

**This is not a calibration curve -- it is noise, and the swing is large
enough that mining it for the best-looking value would be the exact
p-hacking failure mode flagged in §3.5's window sweep.** sma20 with time15
is *worse* than baseline; sma20 with time20 is mildly better; sma15 and
sma25 both look dramatically better, with no smooth trend connecting any of
these five points.

Traced to source rather than left as a mystery: under `sma25/time20`,
**two trades -- SAREGAMA (156 bars, ~7 months) and LAURUSLABS (86 bars, ~4
months) -- contribute ₹82,000 of the configuration's ₹254,000 total P&L**,
32% of it from two trades out of 259, both held through the 2020-2021
COVID-recovery small/mid-cap rally. That is not "held long enough to
realise the measured 15-20 bar edge" -- 15-20 bars is three to four weeks;
these positions ran three to six times longer than that, becoming de facto
multi-month directional bets on an exceptional, one-off market regime
rather than AES trades in any recognisable sense. Loosening the stop did
not make the strategy capture more of its own edge; it occasionally let a
position stop being a swing trade and start being a lucky buy-and-hold
through a historic rally. That is the same beta-versus-alpha confusion this
project has flagged twice already (the prior DAB/MR study's 85%-invested
result; this project's own alpha/beta split), showing up again one level
down, at the level of individual trades rather than the whole portfolio.

**Recommendation: none of the swept values are adopted.** The honest
result remains the baseline (spec's own lower-bound exit parameters,
§5's headline table above). Whether a *robust* improvement exists at some
exit-rule configuration is a question for Phase 6's walk-forward -- which
tests stability across time folds rather than an in-sample sweep over the
whole discovery period at once -- not for picking whichever of five points
happened to land highest.

### 5.2 The profit ladder as the lever — tested, and rejected for a reason worth keeping

§5.1 widened the stop and was rejected. The follow-up hypothesis was that the
wrong lever had been tested: that §6.2's ladder, which sells 70% of every
winner by +8%, is what truncates an edge that does not materialise until bar
20, and that the stop should be left alone precisely because it is what
prevents unbounded holds. Four exit variants, SMA(10) closing-basis stop
unchanged in all of them, plus a **hard 25-bar maximum hold** added to every
variant so no trade can structurally ride a multi-month rally:

```
  variant                          ret%   alpha%   beta   TIM%  medBars  maxBars  pos  exits    PF   win%   maxDD%
  a  no ladder                    10.88    +0.46  0.057  27.69      6.0       25  182    182  1.206  39.6    18.71
  b  ladder at +12% (50%)         15.22    +0.83  0.050  27.62      6.0       25  184    224  1.247  39.7    15.71
  c  ladder at +15% (30%)         18.07    +1.03  0.055  27.44      6.0       25  181    209  1.291  40.3    16.49
  d  current ladder + 25-bar cap  -0.77    -0.35  0.031  28.22      6.0       25  187    327  1.044  41.7    13.45
  d0 current ladder, no cap       -0.80    -0.36  0.031  28.22      6.0       26  187    327  1.043  41.7    13.45
```

`d0` reproduces §5's headline exactly (-0.80%, alpha -0.36%, beta 0.031),
which is the check that the variant harness is measuring the same thing §5
measured. **Trade count is reported two ways on purpose**: the engine writes
one record per *exit*, so the ladder variants are not comparable on `exits`
(327 for the control, 182 for no-ladder) — `pos` is the count of actual
positions, and every rate above is computed on positions, not exit records.

**The hypothesis is falsified, and the falsifying number is the survival
curve, not the return column.** If the ladder were what truncates holds,
removing it entirely would extend them. It does not:

```
  % of positions still open at bar...     5      10      15      20      25
  a  no ladder                         69.8    28.0     9.9     3.3     0.5
  d  current ladder                    68.4    27.3     8.6     3.2     0.5
```

Deleting the profit ladder outright moves median hold from 6 bars to 6 bars
and bar-20 survival from 3.2% to 3.3%. Under variant (a), **181 of 182
positions exit on `sma10_close_stop` and exactly one reaches the 25-bar
cap.** The ladder was never what ended these trades — the closing-basis
stop ends ~99% of them, at a median of six bars. What the ladder changes is
how much of a position is still owned when the stop finally fires, not how
long the position lives. No ladder configuration can capture a 20-bar edge
when only 3% of positions survive to bar 20 under *any* of them.

**The remaining return spread is the same 2020-21 concentration §5.1
rejected, arriving through a different door.** Net P&L by entry year:

```
  year      a no ladder   b +12%    c +15%   d control    n_pos
  2015          -22,534  -10,008   -23,462      +6,190       18
  2016          -11,407  -25,270   -17,843     -29,819       17
  2017          +86,899  +81,117   +91,647     +66,773       31
  2018          -30,271  -25,845   -28,827     -29,892       14
  2019          -35,881  -35,820   -35,875     -27,012        8
  2020          +76,252 +102,491   +75,075     +83,042       40
  2021          +90,324 +111,771  +165,589     -34,547       43
  2022           -5,275   -5,360    -5,142      -4,870       10
  ------------------------------------------------------------
  2015-2019     -13,194  -15,827   -14,361     -13,759       92
  2020-2021    +161,301 +208,902  +235,521     +43,625       90
```

Over 2015-2019 — five years, 92 positions — **every variant loses money and
they lose within ₹2,600 of each other.** The ladder change is worth
essentially nothing outside 2020-21, where all of the spread lives. Stated
as a share: 2020-21 entries account for 106-109% of each variant's total
P&L, i.e. ex-2020-21 every configuration is net negative (-18,469 /
-21,187 / -19,503 / -18,629 for a/b/c/d).

**Concentration, asked for explicitly and answered explicitly.** Position
level, as a share of each variant's *total* net P&L:

```
  variant                 net P&L    top-1    top-2    top-5   P&L after removing top 3
  a  no ladder            148,107    79.8%   119.4%   196.5%                    -71,395
  b  ladder at +12%       193,076    47.4%    83.9%   157.1%                    -22,587
  c  ladder at +15%       221,161    46.2%    79.7%   143.9%                    -25,994
  d  current ladder        29,866   200.6%   349.8%   625.8%                   -111,184
```

The best-looking variant (c) has a **max single-position profit share of
46.2%** — ITI, entered 2017-04-13, ₹102,174 of ₹221,161, 10.2% of starting
capital from one trade. Its top two positions are 79.7% of total P&L, and
**removing the top three of 181 positions turns the configuration net
negative.** A top-2 share above 100% (variants a and d) means everything
outside the top two is collectively loss-making. This is the same
two-trade concentration flagged in §5.1, at a similar magnitude, reached
without widening the stop at all.

Paired against the control on matched positions, none of the variants is
distinguishable from it: a t=0.52, b t=1.04, c t=0.78, every bootstrap 95%
interval spanning zero by a wide margin (c: [-89,696, +265,312] on a point
estimate of +71,207). **Recommendation: no variant is adopted.** The 25-bar
cap is kept in the parameter set as `max_hold_bars` because it is a cheap
structural guard that costs nothing when inert (it changed the control by
0.03pp), but it is not a fix for anything.

**What this leaves standing.** The exit-mechanism mismatch §5 identified is
real, but the binding constraint is the *stop*, not the ladder — and §5.1
already established that widening the stop buys 20-bar holds at the price of
multi-month COVID-recovery rides. Both levers have now been tested and both
land in the same place: the only way to hold these positions to the horizon
where the measured edge lives is to stop exiting them on the rule that makes
them swing trades.

---


### 5.3 Wide stop *with* the hard cap — the combination §5.1 could not test

A correction to §5.2's closing claim that "both exit levers have now been
tested". They had not been tested *together*. §5.1's stop-widening ran before
`max_hold_bars` existed, which is precisely why it produced 86- and 156-bar
holds and caught the COVID recovery as beta. With the cap in place the
combination is a different experiment, and the §5.2 survival curve makes it
the only remaining mechanism that can get positions to bar 20.

Scoring here is **equal-weight** (0.25 x 4), not the §4 calibrated weights,
since Phase 6 found those overfit and they are not being rescued. That
changes the traded population: 443 `wait_and_watch` + 70 `high_conviction`
against 331 + 44 under the calibrated weights.

```
  stop                 cap    ret%  alpha%   beta   TIM%  pos  med  maxBars   DD%
  sma10 (spec)          25    14.40   +0.77  0.048  30.16  211    6      25  18.22
  sma15                 25    -1.12   -0.57  0.047  34.00  184    9      25  13.25
  sma20                 25    -8.34   -1.24  0.046  38.20  176   10      25  16.23
  sma25                 25     8.89   +0.28  0.047  38.66  162   14      25  13.67
  sma10 +3% buffer      25    14.80   +0.82  0.040  40.11  162   13      25   9.14
  sma10 +5% buffer      25    17.75   +1.06  0.038  41.55  147   16      25   6.20
  sma10 +8% buffer      25    26.77   +1.72  0.038  46.95  144   20      25   5.62
  ATR 2.5x trail        25    37.84   +2.39  0.047  44.62  147   17      25   5.81
  ATR 3.5x trail        25    28.54   +1.82  0.040  47.34  143   21      25   4.12
```

**The cap is doing exactly the job it was added for.** Run the same stops
with `max_hold_bars=None` and the longest hold goes to 962 bars (sma10 +8%),
339 (+5%), 160 (ATR 2.5x) — §5.1's failure mode reproducing on cue. Capped,
the maximum is 25 by construction, and the return survives: ATR 2.5x goes
37.84 capped vs 24.25 uncapped, so the improvement is *not* coming from the
long tail this time.

**The survival curve finally moves**, which is the thing §5.2 could not make
happen from the ladder side:

```
  % of positions still open at bar      5     10     15     20     25
  sma10 (spec)                       68.2   29.4   11.8    4.3    1.4
  sma10 +5% buffer                   91.8   81.0   68.0   40.8   29.3
  ATR 2.5x trail                     95.9   89.1   76.2   42.2   28.6
  ATR 3.5x trail                     95.1   93.7   88.1   55.9   38.5
```

4.3% of positions reached bar 20 under the spec stop; 42.2% do under the ATR
trail. Median hold goes 6 -> 17.

**Concentration and regime, the two checks §5.2's variants failed:**

```
  stop               netPnL   top1%  top2%  top3%   2015-19   2020-21
  sma10 (spec)      191,865    32.8   60.0   82.4   -15,614   229,304
  sma10 +3% buffer  173,911    23.1   42.2   59.7  +149,882    57,868
  sma10 +5% buffer  198,698    35.7   51.2   66.7  +106,274   109,190
  sma10 +8% buffer  286,929    15.5   25.3   34.5   +55,125   257,644
  ATR 2.5x trail    402,246    15.9   24.1   31.6  +120,364   315,494
  ATR 3.5x trail    303,842    19.7   28.9   37.7   +95,078   236,048
```

ATR 2.5x: removing the top three of 147 positions leaves +274,943 of
+402,246 standing, and removing the top *ten* still leaves +108,046. Compare
§5.2's best variant, which went net negative on removing three of 181. The
two-trade dependence is gone.

**And 2015-2019 — five years with no COVID recovery in them — is positive
for the first time in this project.** ATR 2.5x over 2015-19 standalone:
+120,364 on 83 positions, 62.7% win rate, PF 1.51, positive in 4 of 5 years,
bootstrap P(P&L>0) = 0.93. The spec stop over the same 83-position window is
-15,614, 41.1% win, PF 0.97, positive in 2 of 5 years, P(>0) = 0.43.

**What stops this being another §5.1.** Three things, and they matter
because this is the best cell of a nine-configuration sweep, which is
exactly the situation §5.1 warned about:

1. It is a **family**, not a cell. All five widened-stop configurations
   (3/5/8% buffer, ATR 2.5x, ATR 3.5x) are positive over 2015-19, with
   bootstrap P(>0) of 0.78-0.94. The spec stop is the outlier at 0.43.
2. The **mechanism separates cleanly from the one that does not work**.
   Widening by *lengthening the average* (sma15/20/25) is the same noise
   §5.1 found: -1.12 / -8.34 / +8.89, no trend. Widening by *keeping a fast
   average and adding room around it* (buffer, ATR) is monotone in the
   buffer size. Those are different operations and only the second one works.
3. **Bootstrap on the full sample excludes zero** for the first time: ATR
   2.5x 95% CI [+124,955, +685,709], P(>0) = 1.00. Every §5.2 variant
   spanned zero comfortably.

**What still argues caution.** 2020-21 remains 78% of ATR 2.5x's total P&L,
and 2019 and 2022 are negative in every configuration. The result is a
nine-config sweep maximum and has not been walk-forwarded. **This is not an
adoption recommendation — it is a finding that the exit line of enquiry is
not closed**, which is the opposite of §5.2's conclusion, and the correction
is owed to the observation that the cap changed what the stop test means.

### 5.4 The deployment problem, which is larger than the exit problem

Time-in-market was the wrong statistic to worry about. It counts *days with
any position open*; what determines compounding is the fraction of capital
actually deployed, and those are very different numbers here:

```
  config                 TIM%   mean exposure%   positions   median bars
  sma10 (spec)          30.16              5.3         211             6
  ATR 2.5x trail        44.62              6.2         147            17
  ATR 2.5x, 5 slots     44.76              8.2         210            17
  ATR 2.5x, 8 slots     45.29             10.7         271            17
```

**Mean capital deployment is 6.2%, not 44.6%.** Risk-sizing at 1% of equity
per trade across at most three concurrent slots cannot put more than a small
fraction of the book to work, whatever the exit rule does.

The edge does not permit faster turnover as a remedy, because it does not
exist at short horizons. Like-for-like (vs the universe base rate, net of
cost): h=5 -0.00%, h=10 +0.70%, h=15 **+1.24%**, h=20 +1.21%. Holding period
has a floor of roughly 15 bars.

```
  annual return  ~=  edge(H) x (252/H) x utilisation

  utilisation required for a target annual return
  target      H=10   H=15   H=20   H=25
    5%/yr      28%    24%    33%    41%
    8%/yr      45%    39%    53%    66%
   10%/yr      57%    48%    66%    82%
   12%/yr      68%    58%    79%    99%
   15%/yr      85%    72%    99%      -
```

At the current 6.2% deployment and a 17-bar hold, the edge implies roughly
**1.1% a year**. Reaching 10%/yr needs ~48% deployment: an eight-fold
increase. Raising the slot count was tested directly and does not get there
— 5 slots lifts deployment to 8.2% and 8 slots to 10.7%, but total return
*falls* (37.8 -> 18.6 -> 21.3), because the marginal signals admitted by the
extra capacity are worse than the ones score-ranking was already taking.

**So the binding constraint is neither holding period nor the exit rule: it
is signal supply and position size.** The screener produces ~110 signals a
year, of which ~19 become positions. Any path to a worthwhile annual return
runs through admitting more, or sizing bigger, not through holding longer.

---

---

## Phase 6 — Walk-forward, and a correction to what "+3.15%" measures

### 6.1 The headline was an absolute return, not an edge over a random entry

Before any fold-level result can be read, one thing has to be corrected,
because it changes the size of the problem Phase 5 was trying to solve.

The number that cleared Phase 4's stop condition and authorised the
portfolio backtest — "net 20-day edge **+3.15%, t=4.41**, p=0.00001" on the
375-signal scored subset — reproduces exactly. It is `fwd_net_20`: the
**absolute** forward return, net of the 0.585% round trip. It is not
measured against the index, and it is not measured against §3.4's
random-entry baseline. Decomposed on the same 375 signals:

```
  gross absolute 20-bar forward return                      +3.73%   t=5.23
  less round-trip cost                                      -0.58%
  = fwd_net_20   <-- the documented +3.15%                  +3.15%   t=4.41
      of which index drift over the same 20 bars            +1.26%
      of which universe base rate (any entry, index-rel)    +0.68%
  = edge over a random entry in the same universe, net      +1.21%   t=2.76
```

§3.4 had already established the right yardstick for this project and
applied it honestly — "excess return over the index, compared against the
same excess computed on every evaluable bar of the universe... net of the
realistic 0.585% round-trip cost" — and reported +0.46%, t=1.30 on the
pre-watchlist sample, then **stopped rather than proceed to Phase 5**. The
watchlist rebuild plus scoring genuinely improves that: **+1.21%, t=2.76**
is roughly 2.6x the point estimate and clears significance where §3.4 did
not. That is a real result. But it is a third of the headline, and the two
figures are not interchangeable.

**This matters directly for Phase 5's puzzle.** A book running beta=0.031
and 28% time-in-market cannot earn the +1.26pp of index drift inside the
+3.15%, and a strategy trading this universe cannot claim the +0.68pp that
any random entry in it collected. Roughly 1.94pp of the 3.15pp gap between
"bar-level edge" and "flat portfolio" was never available to a
beta-neutral portfolio in the first place — it is not an exit-mechanism
loss, and no exit rule was ever going to recover it. The genuine
exit-mechanism question is why a +1.21% 20-bar edge does not show up in a
book whose positions live six bars, and §5.2 answers that: the stop, not
the ladder.

### 6.2 Do the scoring factors hold fold by fold? Mostly not.

§4 flagged that three of the four weighted factors were selected by
calibrating on the same 787-signal sample they are scored against. Tested
per-year (Pearson r of each factor subscore against 20-bar index-relative
excess; p in brackets):

```
  fold    n    fast_resolution  resistance_abs     rs_capture    prior_cycles          score
  2015   88     +0.197(0.07)     -0.018(0.87)    -0.017(0.88)    -0.019(0.86)   +0.100(0.35)
  2016   52     -0.117(0.41)     -0.105(0.46)    +0.004(0.98)    +0.040(0.78)   -0.121(0.39)
  2017  120     +0.177(0.05)     +0.032(0.73)    -0.057(0.53)    +0.036(0.70)   +0.124(0.18)
  2018   72     -0.083(0.49)     -0.054(0.65)    +0.190(0.11)    +0.189(0.11)   +0.086(0.47)
  2019   22     -0.257(0.25)     +0.463(0.03)    -0.122(0.59)           const   +0.028(0.90)
  2020   90     +0.262(0.01)     +0.208(0.05)    +0.070(0.51)    -0.244(0.02)   +0.291(0.01)
  2021  282     -0.047(0.44)     +0.055(0.36)    +0.075(0.21)    +0.032(0.59)   +0.036(0.55)
  ---------------------------------------------------------------------------------------
  POOLED 786    +0.066(0.07)     +0.042(0.24)    +0.048(0.18)    -0.016(0.65)   +0.089(0.01)
```

Blunt reading, factor by factor:

- **Fast resolution (weight 0.40)** — the one factor §4 explicitly exempted
  from the in-sample caveat, on the grounds that it was found on an earlier,
  structurally different dataset and reproduced independently. It holds the
  pooled sign in **3 of 7 folds** and ranges from -0.257 to +0.262. It is
  the strongest factor pooled and the least stable one fold-to-fold. The
  exemption does not survive: reproducing on a second *pooled* sample is not
  the same as being stable in time, and it is not.
- **Resistance absorption (0.25)** — 5/8 folds, range -0.105 to +0.463. Its
  single significant fold (2019, p=0.03) has n=22.
- **rs_capture (0.20)** — 5/8 folds, range -0.122 to +0.190, not significant
  in any fold.
- **Prior cycles (0.15)** — pooled r is **negative** (-0.016), holds that
  sign in 2/6 folds, and its one significant fold (2020, p=0.02) is
  significant in the **wrong direction** (-0.244). This factor is carrying
  15% of the weight on the strength of §0's thesis and a p=0.0014 result
  from the larger, acknowledged-buggy earlier run. On this data it is noise
  with a negative point estimate.

True walk-forward — weights fitted on a 3-year training window only, applied
unchanged to the next year, against the production weights:

```
  fold               weights        nTr  nTe   r_train   r_test  p_test   hi-lo%
  2015-2017 -> 2018  production     260   72    +0.076   +0.086    0.47    +3.00
  2015-2017 -> 2018  train-fitted   260   72    +0.144   -0.049    0.68       n/a
  2016-2018 -> 2019  production     244   22    +0.094   +0.028    0.90    -2.15
  2016-2018 -> 2019  train-fitted   244   22    +0.125   -0.260    0.24    -0.57
  2017-2019 -> 2020  production     214   90    +0.110   +0.291    0.01    +7.63
  2017-2019 -> 2020  train-fitted   214   90    +0.158   +0.245    0.02    +6.15
  2018-2020 -> 2021  production     184  282    +0.278   +0.036    0.55    +2.65
  2018-2020 -> 2021  train-fitted   184  282    +0.299   +0.056    0.35    +2.27
```

The score separates out of sample in **one fold of four** (2020, p=0.01).
In the other three it is indistinguishable from zero, and in 2019 it
separates the wrong way. Weights refitted per fold are unstable in exactly
the way the per-factor table predicts — fast resolution draws 0.83 of the
weight in the first fold and 0.38 in the last; rs_capture draws 0.0 in three
folds and 0.345 in the fourth; prior cycles draws 0.0 in the fold where it
turns negative.

Edge on the traded population (score ≥ wait_and_watch), fold by fold, on
§3.4 yardstick, discovery entries only:

```
  year     n   net_excess%      t    absolute%
  2015    30        -0.12    0.27       -1.84
  2016    25        +1.11    0.62       +2.23
  2017    55        +2.80    2.15       +5.20
  2018    28        -4.66   -2.49       -4.31
  2019    14        -4.86   -1.61       -2.72
  2020    67        +0.69    0.82       +2.59
  2021   134        +2.70    2.78       +6.42
```

Two folds significantly positive (2017, 2021), one significantly negative
(2018), four indistinguishable from zero. **The pooled +1.21% t=2.76 is
carried by 2017 and 2021**, which together hold 189 of 353 discovery-window
scored signals.

### 6.3 Verdict, and one protocol note

**The scoring system should be treated as overfit**, on the terms §4 itself
set for this test. Three of four factors were flagged as in-sample-selected
and none of the three holds its sign in a majority of folds with any
consistency; the fourth — the one that carried the exemption — is the least
stable of all. The score pooled separation (r=+0.089, p=0.01) is real as a
pooled statement and does not reproduce out of sample in three folds of four.

This is not a statement that AES has no edge. The scored subset beats a
random entry in the same universe by +1.21% net at 20 bars (t=2.76), and
that survives the correction in §6.1. It is a statement that the *weights*
are not carrying the information they appear to, and that a large part of
what looks like score quality is two good years.

**A protocol note that should not be buried.** Screener *admissions* are
filtered to AES_DISCOVERY, but the watchlist walk can produce an entry up to
~70 bars after admission, so **60 of 787 signals have entry dates in 2022**
(2022-01-04 to 2022-07-04), inside AES_HOLDOUT. Ten of them become positions
in the Phase 5 backtest (-₹4,870 to -₹5,360 depending on variant). This is
pre-existing — it is in §5's headline run too, not introduced here — and it
is immaterial to every conclusion above (§6.2's fold tables and the
discovery-only edge, +1.10% t=2.53 on n=353, exclude it). But it means
AES_HOLDOUT is not strictly untouched, and the one-shot holdout consultation
should be run on 2022-07 onward, or with these entries explicitly excluded,
rather than on 2022-01-01 as written.

**AES_HOLDOUT has not been consulted.** Nothing above reads a 2022+ result
other than the incidental spill just described, which was found rather than
sought.

---

## Phase 7 — 41 real trades as a detector test

41 discretionary trades taken Jul 2024 - Jan 2025, supplied as ground truth
for the detector. Exit data was not supplied, so every outcome below is
reconstructed from price rather than recalled.

**Data resolution.** 39 of 41 mapped to a verified price series; each mapping
was checked by requiring the stated entry zone to actually price on the
stated date. Four names (CAMS, Jash Engineering, RIR Power, Mazda) initially
failed that check at ratios of 4.87-9.36 and were confirmed as post-trade
stock splits (5:1, 5:1, 10:1, 5:1) rather than bad tickers; Akzo Nobel India
resolved to JSWDULUX after the rename. **RMC Switchgears and Dhani Services
have no usable series** (NSE SME / delisted) and are excluded throughout.

### 7.1 Detector overlap: 0 of 39

```
  BOX ONLY (structure found, no entry fired)    11    28.2%
  WATCHED, NO BOX (admitted, no structure)      18    46.2%
  COMPLETE MISS (never admitted, no structure)  10    25.6%

  HIT (entry signal within +/-1 session)         0/39 =  0.0%
  + NEAR-MISS (within +/-5 sessions)             0/39 =  0.0%
  box present around the entry date             11/39 = 28.2%
  screener-admitted within 60 sessions          25/39 = 64.1%
```

**The detector and the trader do not overlap at all.** Not one of the 39
produced an entry signal within a week of the actual trade.

### 7.2 The reason is timing, not blindness

The single most useful number in this phase:

```
  a nested box was detectable at some bar in the 60 before entry:  30/39
  distribution of that gap:   p25 = 3    median = 16    p75 = 31 bars
  a box was present ON the entry bar itself:                        6/39
```

**The detector sees the structure. It sees it about three weeks before the
trade is taken.** By entry the box has resolved and the measurement window
now contains the breakout thrust, so the same consolidation no longer
measures as one. Median 30-bar range at entry across the 39 is **41.2%**,
against a `big_range_min/max` band of **15-25%**.

Rendering the ten complete misses as they looked on the entry date, the
structures present fall into five recognisable groups, none of which the
current geometry admits:

1. **Entry on the breakout thrust itself** (Jubilant Pharmova, Akzo Nobel,
   Gland Pharma) — a genuine tight shelf (6-8%) exists 10-25 bars back, then
   a 10-15% thrust out of it, and the buy is on that thrust. The 30-bar
   window straddles shelf plus thrust and measures 23-25%.
2. **Bases longer than the detector allows** (Orient Technologies ~90 bars,
   Modisons ~70) — `big_max_bars=45` cannot express them.
3. **Rising consolidations rather than horizontal boxes** (PB Fintech, CAMS,
   Akzo Nobel) — a staircase of higher lows into the high. These fail on
   containment/drift, which are built to reject exactly this shape.
4. **Consolidations tighter than the floor** (Colgate Palmolive, 8.8% over
   30 bars) — below `big_range_min=0.15`, so the cleanest, quietest base in
   the whole set is rejected for being too quiet.
5. **Post-parabolic shelves** (Balu Forge, +167% over the prior 60 sessions,
   115% 30-bar range) — a real shelf sits on top of a vertical run that
   dominates any window containing it.

Group 1 is the largest and the cheapest to fix: it is an *alignment*
problem. The detector evaluates the box in a window ending at the evaluation
bar; the trader buys after it breaks. Evaluating the box as of the breakout
bar's *pre-thrust* window — or carrying a resolved box forward for N bars as
a live entry candidate — would capture it without loosening any geometry.
Groups 2 and 4 are one-line parameter questions (`big_max_bars`,
`big_range_min`). Group 3 is a real design decision, because horizontality
is currently load-bearing.

The screener is the smaller problem: 64% were admitted within 60 sessions,
and 28 of 39 had a >=20% 10-session move somewhere in the prior 60 bars.
Only 10 of 39 clear that gate *on the entry bar* (median 10-session gain at
entry: +7.9%), which is consistent with buying the pause after the move
rather than the move.

### 7.3 Reconstructed outcomes

Entry at the entry-zone midpoint (split-adjusted), net of 0.585% round trip:

```
  exit rule                   total%   mean%  median%   win%  medBars   best%  worst%
  (a) spec as written           82.4   +2.11    +0.28   53.8        5   +17.3    -8.8
  (b) ATR 2.5x + 25-bar cap     54.8   +1.41    +3.25   66.7       17   +18.8   -19.2
  (c) fixed 10-bar hold        193.4   +4.96    +0.93   51.3       10   +69.7   -16.1
  (c) fixed 20-bar hold         61.2   +1.57    -1.68   43.6       20   +54.2   -19.4
  (c) fixed 25-bar hold         95.0   +2.44    +0.11   51.3       25   +49.4   -26.1
```

The recalled claim that most were profitable is roughly right under the
better exits (66.7% win under rule b) and marginal under the others. Rule
(b) has the best win rate and the best *median* but the lowest total, i.e.
it trades tail for consistency. The fixed 10-bar hold has the best total on
a +69.7% single outlier and a +0.93% median — the same concentration
signature this project keeps finding.

**Entries, not exits, are the stronger half here.** Against Nifty 500 over
each trade's own holding period, mean excess is positive under every exit
rule tested (+1.95% to +4.85%), which says the entry timing carried the
result rather than any particular exit.

### 7.4 Regime split — the opposite of the expected finding

Split at the 2024-09-24 Nifty Smallcap 250 peak. Benchmarks over each half:

```
                          pre-peak (01 Jul -> 24 Sep)   post-peak (24 Sep -> 01 Mar)
  Nifty 500                          +7.31%                      -18.49%
  Nifty Smallcap 250                 +7.00%                      -25.50%
```

```
  rule (b)      n    raw mean%   bench mean%   excess%   win vs index
  pre-peak     25       +1.65        +0.86      +0.79           68%
  post-peak    14       +0.97        -3.06      +4.03           57%
```

**Entries were not strongly profitable pre-peak and weak after.** Raw
returns were mildly positive in both halves (+1.65% / +0.97%), and *relative
to the benchmark the post-peak half was the stronger one* — +4.03% excess
while the smallcap index fell 25.5%. The same ordering holds under all five
exit rules.

Stated plainly, since the brief pre-committed to accepting a regime-
dependence finding: **the data does not support one in the expected
direction.** These entries held up through a severe smallcap drawdown. What
the data will not support either is confidence — mean excess under rule (b)
is +1.95% at t=1.21, p=0.23 on n=39. A regime filter is not indicated by
this evidence; nothing here is significant at n=39, and that cuts both ways.

### 7.5 Discretionary overrides versus the stated rules

```
  below the stated Rs 100 floor      1 testable (Radhika Jeweltech; Dhani
                                     Services also violates but has no data)
  below the coded Rs 50 floor        0
  tagged Risky / Extremely Risky     4
  rejected by the stated rules       5 of 39

                n   mean%   win%   total%      (rule b)
  rejected      5   +2.30   80.0    +11.5
  kept         34   +1.27   64.7    +43.3
```

The five overridden trades returned roughly twice the mean of the
rule-compliant ones and won 80% of the time: Windsor Machines +6.7%, Himadri
+0.6%, KPR Mill -9.3%, Zaggle +2.9%, Radhika Jeweltech +10.6%. **The
overrides outperformed the stated rules** — on n=5, which is far too small
to act on, but it is the answer to the question asked, and it points the
same way as §7.2: the written rules are a narrower description of the
process than the process actually is.

### 7.6 Protocol

The holdout boundary is moved to **2022-07-05** (`aes/protocol.py`),
following §6.3's finding that discovery-window signals spilled entries to
2022-07-04. These 41 trades fall inside the holdout window by date, and were
run under the explicit instruction that they are a detector-validation
exercise rather than a strategy evaluation. That is a real consumption of
2024-25 information about *detector geometry*, in the same category as the
Phase 1/2 visual validation the protocol already discloses, and it is
recorded here rather than left implicit. **No strategy parameter has been
fitted to this window, and AES_HOLDOUT has not been consulted as a
performance test.**

## Phase 8 — The carry-forward fix, and what it revealed instead

### 8.1 Implemented and swept as specified

`WatchlistParams.box_carry_bars` (default 0, so the original behaviour is
unchanged) holds a box armed as an entry candidate for N bars after it would
otherwise have gone stale. While carried, only the breakout modes fire — a
box-bottom entry into an already-stale structure is a different trade — and a
close below the big-box floor still kills the episode outright.

Before implementing, the state machine was instrumented (a `trace` hook,
also default-off) so it could say which path was actually blocking entry
rather than the one assumed. That mattered: the assumed cause was wrong.

```
  box_carry_bars    n   HIT(+/-1)   near(+/-5)   hit%   hit+near%
       0           39       0            0       0.0%      0.0%
      10           39       0            1       0.0%      2.6%
      20           39       0            1       0.0%      2.6%
      30           39       0            1       0.0%      2.6%
      45           39       0            1       0.0%      2.6%
```

**Carry-forward alone moves the hit rate from 0 to 1 of 39.** The trace says
why. The last state the machine was in at each of the 39 entries:

```
  watch_dropped_stale              9   the name was not being watched at all
  box_suppressed_same_structure    4   (3 of them at 0 bars from entry)
  breakout_weak_vol_to_retest      3   breakout fired, retest never confirmed
  box_armed                        2
  silent / never admitted         20
```

The dominant blocker was not box memory but **watchlist** memory:
`max_watch_without_box=90` had dropped the name, with gaps to the last
admission as large as 659 bars. Extending that is the same class of fix —
memory, not geometry — so it was swept alongside:

```
  carry   maxWatch   HIT   near   hit+near%
      0         90     0      0       0.0%
      0        750     0      1       2.6%
     45         90     0      1       2.6%
     45        750     0      2       5.1%
```

**Both memory fixes together: 0 -> 2 of 39, and still zero exact hits.**

### 8.2 The ±5-session yardstick was measuring the wrong thing

Widening the window changes the picture completely:

```
  overlap within...        baseline      carry45 + watch750
       +/- 5 bars          0/39   0.0%        2/39    5.1%
       +/-10 bars          1/39   2.6%        4/39   10.3%
       +/-15 bars          4/39  10.3%        7/39   17.9%
       +/-20 bars          6/39  15.4%       10/39   25.6%
       +/-30 bars         10/39  25.6%       14/39   35.9%
       +/-60 bars         18/39  46.2%       24/39   61.5%

  median offset of the nearest signal:  -13 bars (detector fires EARLIER)
```

The memory fix roughly **doubles overlap at every window** — it did work. It
does not show up at ±5 because the detector and the trader are not out of
phase by noise, they are out of phase by a systematic **13 bars, with the
detector earlier**.

The causal box measurement confirms the mechanism. Taking only boxes
detectable at or before the entry bar, at the moment of entry price sits a
median **+6.5% above** the last detectable box top, and **20 of 29 entries
are already above it**. The detector buys the cross out of the box; the
trader buys the continuation roughly three weeks and 6.5% later. Phase 7's
"box present a median 16 bars before entry" was the same fact seen from the
other side, and was misread here as lateness in the detector when it is
earliness.

**So the detector is not blind to these setups.** It sees 62% of them within
±60 bars once memory is fixed, and trades them sooner.

### 8.3 Is the trader's later entry better? No evidence that it is

The whole point of aligning the detector to the trader's entry is the
assumption that the trader's timing is the better one. Tested directly on
the 24 names where both fire, same exit rule (ATR 2.5x + 25-bar cap), net of
cost:

```
                           trader    detector    diff
  mean net %                 1.57        6.53   +4.97
  median net %              -6.15       -3.91
  win rate %                 37.5        41.7
  paired difference                          t=0.92  p=0.369
```

The detector's earlier entry is **not worse** — point estimate materially
better, not significant at n=24, and both medians are negative with the
means carried by a few large winners (AWHCL +59%, RIR +75%, Refex +46%).
There is no evidence that moving the detector's entry later would improve
anything, and some weak evidence it would make things worse.

### 8.4 Where total portfolio return peaks: at no change

Run on the discovery universe, equal-weight scoring, ATR 2.5x + 25-bar cap
as the working exit:

```
  config                    sig/yr  traded/yr  edge%    t    ret%  alpha%   beta  deploy%   DD%
  baseline (carry 0/w 90)    120.3       79.0   2.42  5.19  38.17    2.42  0.047     6.09  5.81
  carry 45                   133.1       87.1   2.62  5.61  31.57    1.95  0.048     6.26  4.79
  carry 45 + watch 750       534.9      376.1   1.70  8.85  22.50    0.76  0.098    15.29 14.60
```

The expectation was that per-trade edge would fall as signal count rose, and
that total return would peak somewhere past the baseline. **It peaks at the
baseline.** Carry-forward alone slightly *raises* per-trade edge (2.42 ->
2.62) yet lowers total return; the full memory fix quadruples signal supply
and finally moves deployment (6.09% -> 15.29%, the only thing that has ever
moved it), but per-trade edge falls to 1.70%, alpha falls 2.42 -> 0.76, beta
doubles, and drawdown goes 5.81% -> 14.60%.

This is §5.4's finding again, from the other direction: more signals do
relieve the deployment constraint, but the marginal signals are not good
enough to pay for themselves — exactly as adding portfolio slots was not.

### 8.5 Why the remaining four miss groups were not swept

The four geometry changes queued after the carry-forward fix —
`big_max_bars`, sloped channels, the 15% range floor, post-parabolic
shelves — were all premised on Phase 7's 0/39 result meaning the detector
cannot see these setups. **That premise no longer holds**: it sees 62% of
them within ±60 bars, it trades them earlier, and its earlier entry is not
measurably worse. Each of those four changes loosens geometry to admit more
signals, and §8.4 measures what admitting more signals does on the only
sample large enough to test it: total return falls monotonically while
drawdown rises.

Running four more sweeps to maximise hit rate against 39 trades, toward an
entry point with no demonstrated edge advantage, against a portfolio curve
already declining, would be fitting to the smallest sample in the project.
Stopped and reported rather than run.

`box_carry_bars` is left in the codebase at its default of 0 — it is
implemented, tested and available, but not adopted, because on the discovery
universe it costs 6.6pp of total return for 13 extra signals a year.

## Phase 9 — Walk-forward on the locked configuration

Locked before the run and unchanged by it: baseline detector geometry
(`BoxParams()` / `WatchlistParams()` defaults, `box_carry_bars=0`),
equal-weight scoring, ATR 2.5x trailing stop + 25-bar hard cap with the spec
ladder, default sizing and regime handling. Nothing is fitted per fold, so
this is not a re-optimisation walk-forward: it runs one fixed configuration
over disjoint annual windows and asks whether the result is stable or
carried by a few of them.

Each fold truncates the price frames at its own end plus a 40-bar tail —
long enough for a 25-bar-capped position to close naturally, short enough
that no fold can earn anything from a later period. As a check on that
isolation, the compounded fold returns come to **37.47%** against the
single-pass figure of 38.17%.

### 9.1 Fold by fold

```
  fold  sigs  traded  pos    ret%  alpha%   beta  maxDD%   win%   edge%      t   top1%  top3%
  2015    91      50   15   -0.17    1.66  0.099    2.44   66.7   -1.94  -0.92  3095.5 7159.3
  2016    58      42   17    1.50    0.24  0.067    2.87   64.7   +2.05   1.41    86.3  203.8
  2017   133      74   28   10.33    5.15  0.147    5.82   60.7   +5.07   4.07    24.9   69.6
  2018    80      49   15    1.86    2.14  0.047    2.40   66.7   -5.76  -3.33   137.3  219.0
  2019    23      19   12   -1.30   -1.58  0.048    2.14   66.7   -2.46  -0.79       -      -
  2020    98      88   27    8.97    7.52  0.016    2.03   77.8   +2.24   1.66    18.9   48.7
  2021   297     194   32   12.24    8.99  0.083    4.03   78.1   +5.61   6.30    18.1   44.0
```

(The absurd `top1%` figures in 2015 and 2018 are an artefact of dividing by a
near-zero fold total, not a concentration signal; they are reported rather
than hidden. In the folds that actually make money, top-1 share is 18-25%.)

```
  mean annual fold return 4.78%   sd 5.55   t=2.28   p=0.063   positive in 5/7
  share of summed fold return from the best fold          37%
  ...from the best two folds                              68%
  ...from the best three folds (2017, 2020, 2021)         94%
  summed return of the other four folds combined        +1.89%
```

**Four of the seven years produce essentially nothing** (-1.30%, -0.17%,
+1.50%, +1.86%), and the mean fold return does not clear conventional
significance. The per-trade edge — the quantity actually being validated —
is positive in **4 of 7** folds, significantly positive in two (2017
t=4.07, 2021 t=6.30) and significantly **negative** in one (2018 t=-3.33).
The pooled +2.42% at t=5.19 is carried by 2017 and 2021.

Those are the same two years Phase 6 found carrying the scoring factors,
arrived at through a completely independent analysis. This dataset appears
to contain two good years and five ordinary ones, and most pooled
significance in this project traces back to them.

Coarser 2-year folds, run as a guard against reading annual noise:

```
  fold        pos    ret%  alpha%  maxDD%   win%   edge%      t   top1%
  2015-2016    32    1.33    0.15    2.87   65.6   -0.12   0.39    84.4
  2017-2018    42    9.18    3.08    5.82   57.1   +0.75   1.17    31.0
  2019-2020    39    7.51    3.48    3.77   74.4   +1.41   1.36    21.8
  2021         32   12.24    8.99    4.03   78.1   +5.61   6.30    18.1
```

All four positive, best fold 40% of the summed return — a kinder picture
than the annual view, but the per-trade edge is still not significant in
three of the four, and 2021 still carries the significance.

### 9.2 Was best-of-nine selection or signal? Both, on different axes

This was the sharpest question the walk-forward was asked to settle. Same
ladder, same cap, only the stop differs:

```
  fold   ATR ret%  spec ret%     diff   ATR dd%  spec dd%   ATR win%  spec win%
  2015      -0.17       1.16    -1.33      2.44      5.99       66.7       45.0
  2016       1.50       3.19    -1.69      2.87      3.06       64.7       48.0
  2017      10.33       8.37    +1.96      5.82      7.18       60.7       50.0
  2018       1.86      -2.67    +4.53      2.40      5.11       66.7       47.4
  2019      -1.30      -5.36    +4.06      2.14      6.24       66.7       38.5
  2020       8.97      18.28    -9.31      2.03      2.71       77.8       53.5
  2021      12.24      -1.29   +13.53      4.03      9.38       78.1       41.7
```

```
  return     ATR beats spec 4/7 folds (3/5 excluding 2020-21)
             mean difference +1.68pp   t=0.63   p=0.550
  drawdown   ATR lower in 7/7 folds    3.10% vs 5.67%   t=-3.56  p=0.0119
  win rate   ATR higher in 7/7 folds  68.8% vs 46.3%   t= 7.17  p=0.0004
```

**The return advantage that made ATR the best of nine does not replicate.**
It wins four folds of seven, loses 2020 by 9.3pp, and the paired difference
is indistinguishable from zero (p=0.55). That part was selection.

**The risk advantage does replicate, and strongly.** Lower drawdown in
every fold and a higher win rate in every fold, both with real p-values on
n=7. A trailing volatility stop resolving losers earlier than a 10-day
moving average is a mechanical property, not a fitted one, and it behaves
like one. That part is signal.

So the honest description of ATR 2.5x + 25-bar cap is: **a materially better
risk profile at an unproven return advantage** — which is a real result, and
a different one from "the best configuration found."

### 9.3 The holdout was not consulted, and why

The standing rule was set before the run: hold up fold-by-fold and consult
AES_HOLDOUT once; carried by one or two folds and do not spend it.

Two folds carry 68% of the summed return and three carry 94%; the remaining
four years produce +1.89% between them. The mean fold return sits at
p=0.063. The per-trade edge is negative in three folds and significantly
negative in one. On the criterion as written, this is a result carried by a
minority of folds, and **the holdout has not been consulted.**

That is the conservative side of a genuinely close call — 5/7 annual and
4/4 coarse folds are positive, which is not nothing — and the asymmetry
decided it: the holdout is a one-shot resource, declining to spend it now
costs nothing that cannot be recovered, and spending it on a result whose
central quantity fails in three folds of seven cannot be undone. If the
judgement is that 5/7 positive folds and a robust risk profile are worth the
one consultation, that is a reasonable reading of the same numbers and the
configuration is locked and ready — but it is a decision to take
deliberately, not one to arrive at by default.

**What survives Phase 9 as a finding rather than a candidate strategy:** a
closing-basis ATR trailing stop with a hard hold cap materially improves the
risk profile of this system — lower drawdown and higher win rate in every
fold tested — without a demonstrated return advantage. The return figure
that headlined it (38.17%, alpha +2.42%) is real as a single-pass number and
is 94% attributable to three of seven years.

## Phase 10 — Is 2015-2021 the right period to calibrate against?

The 41 trades sit inside the holdout window but are the user's own entries,
not detector output, so analysing them tests the *period*, not the strategy.
Three angles, because the obvious comparison is confounded — the 41 are one
person's picks and the discovery signals are the detector's, so a raw
difference mixes "different market" with "different selector".

**Disclosed:** angle B uses 2024-25 universe price data descriptively
(index statistics and screener admission *counts*). No strategy return,
signal forward return, or P&L is computed on post-2022 data anywhere in this
phase.

### 10.1 The setups really are different (confounded, but large)

```
  measure                  2024-25 med   2015-21 med   ratio    Mann-Whitney p
  30-bar range %                  41.2          25.3    1.63          <0.0001
  % below 52w high                 2.3           5.1    0.45           0.3199
  prior 60-bar leg %              28.5          17.1    1.66           0.0036
  annualised vol %                53.3          40.1    1.33          <0.0001
  price Rs                       606.7         247.0    2.46           0.0003
  20-day turnover Rs          61.6 cr       15.7 cr     3.93           0.0001
```

Only one measure is *not* different: distance below the 52-week high
(p=0.32). Both populations buy within a few percent of the high. Everything
else is: the 2024-25 trades are in **bigger, higher-priced, 4x more liquid,
one-third more volatile names, after runs two-thirds larger, in
consolidations two-thirds wider.**

**One tension worth flagging explicitly.** The 4x turnover difference points
straight at `RESEARCH.md` §8.2, where restricting to the liquid quartile cut
edge by 68% and collapsed t from 4.65 to 0.74 — edge and cost were
positively correlated because the premium *was* illiquidity. That finding
was on a mean-reversion signal, not an accumulation-breakout one, so it does
not automatically transfer. But the calibrated population and the traded
population differ by a factor of four on precisely the axis that destroyed
the prior study's edge, and that should not be discovered later.

### 10.2 Market structure: 2024 is not unusual — it is 2021

```
  year   Nifty500 ret%   ann vol%   maxDD%   up-days%   admissions   names
  2015            -0.9       16.2    -13.9       52.0          556      79
  2016             3.4       15.5    -13.9       54.5          314      63
  2017            35.5        9.7     -4.8       60.5          885     107
  2018            -2.8       13.4    -15.8       51.2          265      52
  2019             7.3       13.8    -12.1       51.7          159      45
  2020            16.5       29.6    -38.3       62.0          689     113
  2021            28.3       15.2    -10.0       56.3         1793     181
  ---------------------------------------------------------------------------
  2024            15.2       15.2    -10.9       61.1         1707     214
  2025             4.6       13.3    -12.9       49.4          370      70
```

**2024 sits above every discovery year except 2021 on breadth** (1707
admissions across 214 names vs 2021's 1793/181), with comparable volatility
and drawdown. It is not a market this project has never seen — it is the
market of the single best fold.

### 10.3 The finding that reframes Phase 9

Fold performance tracks breadth and trend, and nothing else:

```
  driver                        vs fold return       vs per-trade edge
  distinct names admitted       r=+0.89  p=0.008     r=+0.79  p=0.036
  Nifty 500 annual return       r=+0.88  p=0.009     r=+0.86  p=0.012
  screener admissions           r=+0.84  p=0.017     r=+0.75  p=0.053
  index annual volatility       r=+0.18  p=0.699     r=+0.08  p=0.871
  index max drawdown            r=-0.08  p=0.866     r=+0.10  p=0.826
```

```
  high-breadth folds (>=600 admissions: 2017, 2020, 2021)
      mean fold return  10.51%      mean per-trade edge  +4.31%
  low-breadth folds  (<600: 2015, 2016, 2018, 2019)
      mean fold return   0.47%      mean per-trade edge  -2.03%
```

**Phase 9's "carried by three folds" is not random concentration — it is a
regime dependency with a mechanism.** An accumulation-to-expansion strategy
needs expansions to exist; when few names are breaking out to 52-week highs,
there is nothing for it to trade and what it does trade fails. The three
folds that carry the result are the three with breadth; the four that fail
are the four without it.

Two things make this more than a restatement. First, it is **alpha** that
moves with breadth, not just return: fold alpha runs +5.15 / +7.52 / +8.99
in the broad years against +1.66 / +0.24 / +2.14 / −1.58 in the narrow ones,
on a book whose beta never exceeds 0.147. Second, volatility and drawdown
show **no relationship at all** — this is specifically breadth and trend,
not risk appetite or market calm.

### 10.4 Return shape: the user's entries front-load, the detector's back-load

Index-relative excess by horizon, net of cost:

```
  horizon     2024-25 mean%      t        2015-21 mean%      t
      5                3.18   2.38                 0.16   0.58
     10                4.04   1.76                 0.69   1.77
     15                3.18   1.38                 1.31   2.92
     20                1.57   0.68                 1.41   2.76
     25                2.14   0.87                 1.86   3.11
     30                3.38   1.26                 1.86   2.77
     40                4.30   1.25                 2.72   3.30
```

The discovery population earns nothing in the first week and builds
steadily. The 2024-25 entries deliver **+3.18% in five bars (t=2.38)** and
then go flat for a fortnight. Same destination by h=40 (4.30 vs 2.72), very
different path.

This is Phase 8's finding from the other side: the detector enters a median
13 bars *earlier* than the trader on the same structures. It therefore sits
through the slow accumulation phase; the trader skips it and enters at the
inflection. Neither is obviously better — by h=40 they converge, and only
the h=5 point is significant on n=39, which is one horizon out of seven
tested. But it does mean **the 15-bar holding floor established on discovery
data is a property of the detector's entry timing, not of the setups
themselves.**

### 10.5 Answer

**Is 2015-2021 the wrong period to calibrate against?** No — but it is the
wrong period to *average over*. It contains two distinct regimes: three
broad, trending years in which this strategy works (mean fold return 10.51%,
edge +4.31%) and four narrow years in which it does not (0.47%, −2.03%).
Pooling them produces a number that describes neither, and that pooled
number is what every headline in this project has reported.

**Does the market the user trades look structurally different?** On setup
geometry, yes and substantially — bigger, more liquid, more volatile names
after larger runs. On market structure, no: 2024 is a high-breadth trending
year closely resembling 2021, the best fold in the walk-forward. 2025 so far
is a narrow year (370 admissions, 70 names) and the user's post-peak trades
in it returned +0.97% raw — low, exactly as the breadth relationship
predicts, while still beating a −25% smallcap index.

**The natural next hypothesis, stated but not tested:** a breadth or
participation filter — trade only when the count of names newly admitted is
above some threshold — is the first thing this project has found that is
motivated by a mechanism rather than by a sweep. It would also directly
attack the deployment constraint from §5.4, since it concentrates capital
into the periods where the edge exists rather than spreading it thin across
periods where it does not. It is a new free parameter and needs its own
validation; n=7 folds with collinear drivers is suggestive, not settled.

## Phase 11 — The breadth filter, and the turnover-transfer check

### 11.1 Turnover: the transfer concern is not confirmed

The worry: the 41 real trades have 3.9x the turnover of the discovery signal
population, which is the axis that killed the liquid-quartile restriction in
`RESEARCH.md` §8.2 (edge −68%, t 4.65 → 0.74). If the AES edge also lives in
illiquid names, nothing calibrated here transfers to what is actually traded.

Measured properly — **each quartile against its own universe base rate**, so
that a quartile which simply outperformed over this window is not credited
as signal quality:

```
  quartile     n   med turnover Rs   signal exc%   base rate%   NET EDGE%      t
  Q1          69        14,221,304          3.00         1.17        1.24   1.26
  Q2         144        58,158,446          1.32         0.91       -0.18   0.45
  Q3         189       196,244,947          1.35         0.66        0.11   0.74
  Q4         151       724,484,470          1.64         0.03        1.02   1.65
  ALL        553       156,716,198          1.63         0.62        0.43
```

**The edge is not concentrated in low-turnover names.** It is flat to
U-shaped: Q1 +1.24% and Q4 +1.02%, with the middle two quartiles at
approximately zero. Q4 — where **24 of the user's 39 trades sit** — carries
net edge comparable to the pooled figure.

The base-rate column is why this had to be done carefully. Low-turnover
names *did* outperform over discovery (Q1 base rate +1.17% vs Q4's +0.03%),
so the naive comparison of raw signal excess (Q1 3.00% vs Q4 1.64%) would
have shown exactly the false positive feared — an apparent illiquidity
concentration that is really a size/liquidity factor in the universe, not
signal quality.

**Plain answer: the calibrated and traded populations differ in level (the
41 sit at the 83rd percentile of discovery universe bars, the discovery
signals at the 59th) but not in a way that destroys transfer.** The prior
study's liquidity finding does not appear to carry over to this strategy.
Caveats, stated rather than buried: no quartile is individually significant
(max t=1.65), and Q4 by fold is as unstable as everything else in this
project — significantly negative in 2015 (−11.26%, t=−2.23), positive in
2017/2020/2021, near zero elsewhere.

### 11.2 The breadth filter, specified in advance

Stated before running and not swept:

```
  breadth(t)  = distinct symbols with >=1 screener admission in the trailing
                63 sessions, using only admissions on or before t
  threshold   = expanding-window median of breadth up to and including t,
                accumulated from 2014-01-01 (a year of warm-up)
  gate        = a new entry may open on day t only if breadth(t) >= threshold(t)
```

Causal by construction; no future admissions, no year labels, no return
information. The median was chosen because the brief named it and because
"participation above its own typical level" is a mechanism statement rather
than a return-maximising one.

**Result: it fails the prediction the mechanism made.**

```
  fold   ret% un   ret% gt   alpha un   alpha gt   dd un   dd gt   edge un   edge gt
  2015     -0.17     -0.17       1.66       1.66    2.44    2.44     -1.94     -1.94
  2016      1.50      1.18       0.24       0.06    2.87    2.57      2.05      2.13
  2017     10.33     10.33       5.15       5.15    5.82    5.82      5.07      5.07
  2018      1.86      3.82       2.14       3.77    2.40    1.44     -5.76     -3.71
  2019     -1.30     -3.30      -1.58      -4.67    2.14    4.23     -2.46     -8.36
  2020      8.97      6.71       7.52       5.65    2.03    3.37      2.24      1.40
  2021     12.24     12.24       8.99       8.99    4.03    4.03      5.61      5.61

  LOW-breadth folds (2015,16,18,19)  summed return  1.89% -> 1.53%   edge -2.03% -> -2.97%
  HIGH-breadth folds (2017,20,21)    summed return 31.54% -> 29.28%  edge +4.31% -> +4.03%
```

The prediction was that gating turns the four low-breadth folds from +1.89%
into flat-but-not-negative while preserving the high-breadth folds. It does
neither: the low folds get slightly *worse* on both return and edge, and the
high folds lose a little. The gate retains 92% of signals and binds in only
two folds (2018, 29% of days open; 2019, 23%) — helping 2018 (+1.86 → +3.82)
and hurting 2019 (−1.30 → −3.30). A wash, from a mechanism that predicted
something specific.

The construction also has a flaw worth naming: an expanding *median* is a
relative threshold that re-centres on whatever breadth has been running at.
In 2021, with breadth at 89-100 names, the threshold was still 23 and the
gate was open 100% of the time. A rule meant to say "enough breakouts exist
to trade" cannot be built from a statistic that redefines "enough" upward
whenever conditions improve.

### 11.3 Why no better threshold is worth trying: breadth is a year label

Rather than sweep thresholds, the mechanism was tested directly at signal
level. That measurement first looks encouraging:

```
  breadth(t) vs the signal's own 20-bar index-relative excess, n=553
      Pearson r=+0.113  p=0.008        Spearman r=+0.087  p=0.040

  breadth quartile     n   med breadth   net edge%      t
  Q1                 147            19       -0.25    0.34
  Q2                 130            37        0.59    1.41
  Q3                 138            59        0.60    1.21
  Q4                 138           100        3.28    3.23
```

Monotone, significant, with the effect concentrated in the top quartile — a
much better result than the median threshold produced, and the obvious next
move would be to re-set the threshold at the top quartile.

**That move would be invalid, and the cross-tabulation says why.**

```
  signals per breadth quartile x year
  q\year  2015  2016  2017  2018  2019  2020  2021  2022
  Q1        22    33     8    29    19    36     0     0
  Q2        28     9    52    13     0    20     0     8
  Q3         0     0    14     7     0    32    56    29
  Q4         0     0     0     0     0     0   138     0
```

**Every one of the 138 top-quartile signals is from 2021.** Breadth's top
quartile and the year 2021 are the same set of rows; they cannot be told
apart on this data. And once that year is removed the relationship is gone:

```
  breadth vs excess, excluding 2021        r=+0.040  p=0.448  (n=359)
  breadth vs excess, excluding 2020-21     r=+0.117  p=0.054  (n=271)
  WITHIN-year (between-year variation removed)
                                           r=+0.020  p=0.647  (n=553)
```

The last line is the one that matters, because a same-day gate can only
trade within-year variation — it cannot know which year it is in. **There is
none: r=+0.020, p=0.647.**

Phase 10's r=+0.89 on seven annual points was an aggregation artefact.
Collapsing 553 signals into 7 year-means discards exactly the variation a
tradeable rule would need and leaves a correlation driven by the fact that
2021 was both the best year and the highest-breadth year — one observation,
not a relationship.

### 11.4 Verdict

**The breadth filter is rejected.** The pre-specified causal version fails
the prediction; the version that looks like it works is 100% collinear with
a single year; and within-year, where a gate would have to operate, the
effect is zero. This is precisely the "curve-fitting with extra steps"
outcome the brief asked to have flagged, and it is being flagged.

The mechanism — an accumulation-to-expansion system needs expansions to
exist — may well be true. It is not *identifiable* on seven years of data
in which the high-breadth regime occurred essentially once.

**The holdout was not consulted.** The stated condition was that the breadth
filter hold fold-by-fold and the turnover check not invalidate transfer.
The turnover check passed; the breadth filter did not. One of two is not the
condition, and the configuration that would have been locked is unchanged
from Phase 9 — which Phase 9 already declined to spend the holdout on.

## Phase 12 — Extended history (2010→), and the final answer on breadth

Data re-fetched from 2010-01-01 into a **separate** cache
(`cache/ohlcv_2010`), deliberately not overwriting `cache/ohlcv`, since every
result through Phase 11 was produced against the 2014-start cache. 397 of 402
symbols returned data; 372 clear the 400-bar minimum. Eleven annual folds
(2011-2021) instead of seven, after a year of warm-up for the 252-bar
52-week-high lookback.

### 12.1 Data integrity — both checks passed, one better than expected

**Survivorship is material and is the real limitation.** These are *today's*
midcap150 + smallcap250 constituents, so a name present in 2010 is a survivor
by construction — it stayed listed and stayed mid/small for fifteen years:

```
  year   listed   % of universe        year   listed   % of universe
  2010      211           53.3%        2017      259           65.4%
  2011      217           54.8%        2019      282           71.2%
  2013      226           57.1%        2021      317           80.1%
  2015      234           59.1%        2025      396          100.0%
```

The pre-2015 folds see barely half the universe. The bias direction is not
simply optimistic — a name that did spectacularly would have graduated to
large-cap and left the index — but it is a different, smaller, differently
selected population, and the early folds must be read with that in mind.

**Corporate-action adjustment is not worse pre-2015 — it is better.**

```
  era          |r|>35% days   per 10k bars   split-ratio-like moves
  2010-2014               4           0.15                        0
  2015-2021              19           0.42                        3
  2022-2026               5           0.12                        2
```

Targeted check against known split dates: of 20 sampled names, 11 had splits
before 2015 (Ajanta 2:1 and 1:x, Balkrishna 5:1, City Union 10:1, Lupin 5:1,
Supreme 5:1, Voltas 10:1, and others). **Maximum single-day move within ±4
days of any of those split dates: 13.8%, and zero flagged suspect.** The
adjustment is applied correctly.

**The re-fetch reproduces the validated cache exactly.** Over the 2015-2021
overlap, 296 symbols compared: median relative close difference **0.0000%**,
zero symbols with a median difference above 1%, zero with any single day
above 5%.

### 12.2 The locked configuration on eleven folds

```
  fold  sigs  traded  pos    ret%  alpha%   beta  maxDD%   win%  depl%   edge%      t
  2011    19      15    5   -2.19   -2.90  0.023    3.35   20.0   2.17   -3.86  -1.28
  2012    46      39   19   -0.30   -0.23  0.053    4.78   73.7   6.66   +1.87   1.83
  2013    32      26   18    7.34    6.64  0.023    1.92   77.8  11.85   +4.62   2.12
  2014   158     101   28   -1.85   -5.63  0.108    5.08   53.6  10.52   +3.59   2.69
  2015   138      94   24   -5.17   -3.01  0.134    7.16   37.5  10.35   -0.30   0.26
  2016    56      43   18    2.27    1.22  0.078    2.85   66.7   6.49   +1.53   1.17
  2017   132      73   28   10.33    5.15  0.147    5.82   60.7  12.61   +5.11   4.05
  2018    76      47   14    1.05    1.41  0.044    3.17   64.3   4.62   -6.03  -3.45
  2019    23      19   12   -1.30   -1.58  0.048    2.14   66.7   5.60   -2.46  -0.79
  2020    96      87   26    7.73    6.50  0.013    2.23   73.1   9.03   +2.29   1.68
  2021   291     189   32   15.39   11.63  0.086    3.48   81.2  14.51   +5.88   6.48
```

2011 is **unusable** (5 positions); 2018 and 2019 are thin (14, 12). Every
fold subset that could be defended is non-significant:

```
  fold set                          n   mean    t      p     pos     best3   ex-best3
  2011-2021 (all)                  11   3.03  1.59  0.143   6/11     100%     -0.15%
  2012-2021 (drop unusable 2011)   10   3.55  1.75  0.114   6/10      94%     +2.04%
  2014-2021 (drop thin 2011-13)     8   3.56  1.44  0.192    5/8     118%     -5.00%
  2015-2021 (the Phase 9 window)    7   4.33  1.60  0.160    5/7     110%     -3.15%
```

Per-trade edge across eleven folds: positive in 7/11, mean +1.11%, **t=0.95,
p=0.363**.

**More data made the result weaker on every axis**, which is the direction a
partly-noise result moves when the sample grows.

**A correction that has to be recorded.** The last row above is the Phase 9
window, and it no longer reads t=2.28, p=0.063 — it reads **t=1.60,
p=0.160**. The difference is not a change of configuration; it is that the
old cache began 2014-01-01, so the 252-bar 52-week-high lookback was still
filling through early 2015 and suppressed admissions (2015: 91 signals then,
138 now). Phase 9's 2015 fold was computed on a truncated sample.
**Phase 9's headline was inflated by a warm-up artefact, and its p=0.063 —
already short of significance — should be read as p=0.160.**

### 12.3 Breadth, re-tested exactly as specified

Only the two checks that failed before, no threshold sweep. The extension
did what it was meant to: a second high-breadth year now exists.

```
  signals per breadth quartile x year
  q\year  2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022
  Q1        15   38   26   10   11   29    7   25   15   20    0    0
  Q2         0    1    0    5   74   14   51   12    4   35    0    3
  Q3         0    0    0   55    9    0   15   10    0   32   28   34
  Q4         0    0    0   31    0    0    0    0    0    0  161    0
```

Q4 now spans two years instead of one — but **161 of 192 (84%) are still
2021**, with 31 from 2014. And the checks:

```
  pooled                             r=+0.074  p=0.041   n=770
  excluding 2021                     r=+0.016  p=0.694   n=581
  excluding 2020-21                  r=+0.042  p=0.354   n=494
  excluding 2014 and 2021            r=+0.026  p=0.575   n=480
  WITHIN-year (the tradeable part)   r=+0.010  p=0.773   n=770
```

**The within-year correlation — the only variation a same-day gate could
act on — is +0.010 at p=0.773, on 770 signals instead of 553.** Adding four
years and 217 signals moved it from +0.020 to +0.010.

**And the one genuinely independent high-breadth year contradicts the
hypothesis.** 2014 has mean breadth 61 (second-highest of eleven years), 101
traded signals, and a *positive* mean signal excess of +1.94% — yet the fold
returned **−1.85% with alpha −5.63%**. The prediction was that an
independent high-breadth year would reproduce 2021's portfolio result. It
does the opposite.

### 12.4 Final answer

The condition set in advance was: if breadth still cannot be separated from a
single year, the honest conclusion is that this strategy is regime-dependent
in a way that seven-to-twelve years of Indian mid/smallcap data cannot
validate.

**That condition is met.** Breadth's top quartile remains 84% one year; the
within-year effect is zero; and the one independent high-breadth year points
the wrong way. Twelve years of data, 1,129 signals and eleven folds do not
separate the hypothesis from a single exceptional year, and the underlying
configuration is not significant on any defensible fold set.

**The holdout has not been consulted and should not be.** There is no
candidate whose validation it would settle — the configuration that would be
locked is the Phase 9 one, whose central quantity is now p=0.160 on its own
window and p=0.143 on the full eleven folds.

**What remains true after twelve years of data** is what Phase 9 established
on the risk axis and Phase 11 on transfer: an ATR trailing stop with a hard
hold cap lowers drawdown and raises win rate in every fold tested, and the
edge is not concentrated in illiquid names the user cannot trade. Those are
components, not a strategy. The return edge is not established, and this
dataset cannot establish it.

## Phase 13 — Size up instead of adding slots

Adding slots was tested three times (3/5/8) and failed each time, because the
marginal *signal* is worse than the average one. Position size is a different
lever: the signal set is byte-identical across every row below, slots stay at
3, and only the rupee size changes. Any difference is sizing, not selection.
Ten annual folds, 2012-2021 (2011 excluded: 5 positions).

```
  config                  deploy%    ret%  alpha%  meanDD  worstDD  ret/DD  pos folds
  risk 1% (current)           9.2    3.55    2.21    3.86     7.16    0.92       6/10
  risk 2%                    16.6    5.87    3.30    7.82    12.36    0.75       6/10
  risk 3%                    24.5    8.41    4.88   11.50    17.85    0.73       7/10
  risk 4%                    28.5    9.50    5.04   12.82    18.72    0.74       5/10
  cap 15% @1% risk            8.5    3.23    1.97    3.73     7.04    0.87       6/10
  cap 25% @1% risk            8.9    3.30    2.03    3.77     7.29    0.88       7/10
  cap 40% @1% risk            9.2    3.55    2.21    3.86     7.16    0.92       6/10
  cap 60% @1% risk            9.4    3.61    2.25    3.91     7.16    0.92       6/10
  risk 2% + cap 60%          16.9    6.03    3.44    7.75    12.71    0.78       6/10
  risk 3% + cap 60%          25.2   10.59    6.57   11.03    18.20    0.96       7/10
  risk 4% + cap 60%          29.5   11.96    7.03   13.92    23.79    0.86       6/10
  risk 4% + cap 100%         29.3   12.15    7.44   14.65    24.42    0.83       6/10
```

**The notional cap is not the binding constraint and never was.** Sweeping it
from 15% to 60% of equity moves deployment from 8.5% to 9.4% -- it barely
binds at all, because the 1% risk budget exhausts first. Anyone reaching for
the cap to raise deployment is pulling a lever that is not connected.

**The answer to "what does 6% to 20-25% deployment cost in drawdown":**

```
  deployment    9.2%  ->  24.5%      (risk 1% -> 3%)
  mean fold DD  3.86% ->  11.50%     roughly triples
  worst fold DD 7.16% ->  17.85%
  mean return   3.55% ->   8.41%
  mean alpha    2.21% ->   4.88%
```

Return and drawdown both scale close to linearly, which is what pure leverage
on an unchanged signal set should do -- and is itself a useful negative
result: there is no hidden convexity, good or bad, in sizing this book up.
`risk 3% + cap 60%` is the best cell on return-per-unit-drawdown (0.96, above
even the 1% baseline's 0.92) and lands exactly in the 20-25% deployment band.

**Concentration does not deteriorate with size**, which was the live worry:
median top-1 share of fold P&L is 26.1% at 1% risk and 29.9% at 3% + cap 60%,
with two folds of seven above 100% either way. Sizing up magnifies the same
distribution rather than making it more lottery-like.

**None of this fixes significance.** Every configuration sits at p=0.09-0.19
on ten folds, and the fold pattern is the one Phase 9/12 already described:
2017 and 2021 carry it, 2015 and 2019 lose money at every size. Sizing up
multiplies an unproven edge; it does not make it proven. At 3% risk the 2015
fold loses 10.1% and 2019 loses 4.8%.

---

## Phase 14 — The admission bottleneck

The Phase 12 audit changed the diagnosis. Of 400 universe names on a live
scan, only 17 reach the scoring stage at all: 205 time out on the watch
clock, 112 never pass the screener, 40 are price-deleted, 26 have no box.
Softening the *score* thresholds cannot help when almost nothing arrives at
them. So the question moved upstream: is the admission gate itself too tight?

**Market context first, because it decides how to read today's empty scan.**
On the live universe, the best 10-session move is **+18.5%**, the median is
**-3.85%**, and 25 of 381 names sit within 5% of a 52-week high. **Zero names
pass both conditions.** Today's zero actionable is a correct reading of a
narrow market, not a broken filter, and nothing below should be adopted if it
would populate a day like this one.

### 14.1 Is the +20% / within-5% cutoff arbitrary? No -- it is the best cell

Measured at the admission level: every discovery bar clearing price and
liquidity (334,523 of them), bucketed by its 10-session move and its distance
below the 52-week high, against its own forward 20-bar index-relative return
net of the universe base rate and cost.

```
  cell                     n    net edge%       t
  PASSES 20/5           4660        1.83     9.32
  15-20% / <=5%         4494        0.57     5.98
  10-15% / <=5%         9625       -0.15     3.90
  >=20% / 5-10%          758        0.16     1.17
  >=20% / 10-15%         406        0.35     1.21
  15-20% / 5-10%         878        0.42     2.21
  everything else     329863       -0.62    -1.90
```

**Both conditions are load-bearing and the AND is real.** Relaxing the move
from 20% to 15% while holding the high condition costs 69% of the edge
(1.83 -> 0.57). Relaxing the high condition from 5% to 10% while holding the
move costs 91% (1.83 -> 0.16). The gate is not a preference expressed too
strongly; it is the one cell in the grid where the edge lives.

That also kills the specific construction proposed. An additive "admission
strength" that averages a move score and a proximity score gives the same
0.5 to a name at (move 1.0, proximity 0.0) as to one at (0.5, 0.5) -- but the
data says the first is worthless and only the joint condition pays. A
gradient built by averaging destroys exactly the interaction that carries the
result. Decile analysis confirms it: Spearman between that strength score and
forward excess is **+0.0234** on 334k bars -- significant only because n is
enormous, and economically nothing.

Blended edge over the whole admitted set, for each widening:

```
  admission rule        bars   vs 20/5   net edge%
  20% / 5% (current)    4660      1.0x        1.83
  18% / 5%              6080      1.3x        1.60
  15% / 5%              9154      2.0x        1.21
  20% / 10%             5418      1.2x        1.60
  15% / 10%            10790      2.3x        1.07
  10% / 15%            25539      5.5x        0.34
```

### 14.2 End to end: the fourth attempt to raise signal count, failing the same way

Each variant changes only who gets watched. Detector, entry modes, scoring
and the ATR exit are untouched.

```
  variant               sig/yr  traded   edge%      t   fold ret%  fold p   meanDD%
  V0  baseline 20/5      120.0     551    2.39   5.14        4.78   0.063      3.10
  V1b move only 15/5     187.7     952    1.82   5.50        3.08   0.127      3.92
  V1  widened 15/10      205.7    1060    1.79   5.81        1.19   0.424      4.89
  V3  readmit on 52wH    259.4    1418    1.53   6.28        2.40   0.334      4.60
  V13 widened + readmit  280.4    1551    1.59   6.78        1.21   0.561      5.75
```

Stated plainly, as asked: **every one of these raises signal count and lowers
edge.** Signals more than double, per-trade edge falls a third, mean fold
return falls from 4.78% to as low as 1.19%, drawdown rises by half, and the
fold p-value degrades from 0.063 to 0.56. This is the fourth independent
attempt to raise signal supply (after portfolio slots, carry-forward and
watchlist memory) and the fourth to fail in the same direction.

The rising t-statistics (5.14 -> 6.78) are not a counter-argument: they rise
because n rises. There is more *total* edge in the wider net and less in each
unit of it, and with three concurrent slots the book can only ever take the
few it ranks highest -- so a worse average is the only thing that reaches it.

### 14.3 The finding that matters more than any of the above

Coverage of the 41 real trades, under each admission rule:

```
  variant               on watchlist at entry   box at entry   median gap
  V0  baseline 20/5             25/39                3/39         8 sessions
  V1b move only 15/5            31/39                3/39         4
  V1  widened 15/10             31/39                3/39         3
  V3  readmit on 52wH           33/39                3/39         3
  V13 widened + readmit         33/39                3/39         1
```

Relaxing admission genuinely does reach more of the trades actually taken --
25 of 39 becomes 33 of 39, and the median gap from last qualification to
entry collapses from 8 sessions to 1. **But "box at entry" does not move at
all: 3 of 39 under every rule.** Getting a name onto the watchlist is not the
binding constraint on matching these entries; the box geometry is, exactly as
Phases 7 and 8 measured. Admission relaxation buys coverage that the detector
then throws away.

### 14.4 Price deletion: a smaller question than it looked

**`hard_delete_mask` is never called in the research pipeline.** It appears in
`aes/screener.py`, in the Phase 12 scanner audit, and nowhere else -- not in
`run_watchlist_for_symbol`, not in any calibration path. The Rs 100 / Rs 5,000
band has therefore never influenced a single discovery or walk-forward number
in this project. It affects only what the live scanner displays.

It is also already re-evaluated daily rather than permanently: the scanner
reads `hard_delete_mask(df, SP).iloc[i]`, today's value, so a name that falls
back under Rs 5,000 returns on its own. "Permanent deletion" was a
misreading of the audit output, and the audit's own wording ("removed from
the watchlist for good") caused it -- that phrasing is wrong and is corrected.

What the band costs on a live scan:

```
  variant                              watchlist   with a box   recovered
  current (floor 100, ceiling 5000)           40           17   --
  no ceiling                                  43           17   APARINDS, NEULANDLAB, PTCIL
  no floor                                    42           18   IFCI, NSLNISP
  neither                                     45           18   all five
```

**Removing the ceiling is free and is recommended.** It was stated as a
personal preference rather than a finding, it excludes three names today, and
none of them produce a box, so it changes no recommendation while removing an
unjustified rule. The floor is the one with evidence *against* it -- Radhika
Jeweltech at Rs 95 returned +10.6% in the real-trade set -- and removing it
recovers two names and one extra box today. Neither change can be validated
on discovery, because the rule was never in that path.

### 14.5 Verdict

- **Admission gradient: rejected.** The 20/5 cutoff is the best cell in the
  grid, both conditions are load-bearing, and every widening lowers edge and
  fold return while raising drawdown.
- **Weak re-admission: rejected.** Largest signal increase (+116%), largest
  edge decrease (2.39 -> 1.53), fold return halved.
- **Price band: ceiling removed, floor kept under review.** No discovery
  impact either way; the ceiling was never justified and costs three live
  names, the floor has one real counter-example against it.
- **Sizing: the one lever that works as intended.** 3% risk with a 60% cap
  puts deployment at 25.2% and return at 10.59% with alpha 6.57%, at the cost
  of mean drawdown rising 3.86% -> 11.03% and worst-fold drawdown 7.16% ->
  18.20%. Still p=0.09 on ten folds; sizing multiplies an unproven edge.

The bottleneck is now located precisely. It is not admission and it is not
scoring -- it is that the box detector recognises the structure in **3 of 39**
real entries regardless of what is admitted. Every change tested since
Phase 7 has moved the constraint around without moving that number.

## Phase 15 — the geometry sweeps REJECTED #14 deferred

### 15.1 Why they were run after being rejected

REJECTED #14 deferred four geometry changes on the premise that the detector
"already sees these setups" — it finds a box for 62% of the 41 real trades
somewhere within ±60 bars, trades them ~13 bars earlier, and its earlier entry
is not measurably worse (p=0.37).

That premise is falsified on the axis that decides whether a signal exists.
Five admission rules have now been run, and `box_at_entry` is **3 of 39 under
every one of them** while watchlist coverage moves 25→33 (§14, §4b). Seeing a
box sixteen bars before the trade and seeing one *now* are different
properties, and only the second can fire. The deferral reasoning was sound
against the question it asked; it was answering a different question from the
one the bottleneck table poses.

Admission is held at V0 20/5 throughout — §14.1 established both conditions
are load-bearing — and only `BoxParams` changes.

**Reproduction first.** The harness (`research/geometry_sweep.py`) reproduces
the V0 row of `admission_variants.json` exactly: 840 signals, 551 traded,
edge +2.3918 at t=5.1356, fold return +4.78% (t=2.28, p=0.063, 5/7), mean
drawdown 3.10%, worst 5.82%, and all seven fold returns and drawdowns to the
decimal. Getting there required naming two things the dataclass defaults get
wrong — see STATE.md §5a.

### 15.2 What actually rejects the 39, measured rather than assumed

Before sweeping anything, every one of the 39 real trades was diagnosed at its
own entry bar, recording which gate refused it
(`results/geometry_rejection_reasons.csv`):

```
  gate that rejected the box at the entry bar        n
  big_min_bars  (candidate 1-9 bars long)           27
  a box formed                                       6
  range too tight / too wide                         2 / 2
  min_history_bars (44 and 88 bars of history)       2
```

**27 of 39 die on `big_min_bars`, and their range is never evaluated.** The
box top is the running maximum over the anchor lookbacks; at entry these names
sit at their running 90-bar maximum, so the anchor lands on roughly the
evaluation bar and the candidate box is 1-9 bars long. This is §7.2's "group 1
alignment problem" showing up as the dominant cause, not one of five.

Two consequences for the briefed sweeps, both fatal before any backtest:

- **`big_max_bars` cannot matter.** Every duration failure is at the *low*
  bound. Coverage is flat at 3/39 for every value from 45 to 120.
- **Its two named exemplars do not exist as long bases.** Orient Technologies
  has **44 bars** of history at its entry (it listed ~2 months earlier) and
  Balu Forge has **88**; both are refused by `min_history_bars=120` before
  geometry is consulted. A ~90-bar base was never available on either series.
  The same fact makes `exclude_prior_leg` untestable on Balu Forge, the name
  that motivated it.

`results/geometry_coverage_grid.csv` holds the 47-setting coverage sweep.

### 15.3 The seven folds, per arm

Full results in `results/geometry_variants.json`. `ret_abs` is the
`admission_variants.json` definition; `edge/base` is the PROVEN 6 definition.

```
  arm                          box@entry   sig/yr   ret_abs   edge/base   fold ret      p   meanDD  worstDD
  V0 baseline                     3/39        120     +2.39      +0.48       +4.78  0.063     3.10     5.82
  (a) ceiling 25->35%             5/39        132     +2.24      +0.47       +3.87  0.152     3.52     5.55
  (a) floor 15->5%                3/39        179     +2.41      +0.45       +2.79  0.334     4.65     7.42
  (a) both ends 5->35%            5/39        191     +2.09      +0.21       +2.27  0.282     5.15     6.05
  (b) big_max_bars 90             3/39        112     +2.35      +0.41       +3.41  0.132     3.46     6.30
  (b) big_max_bars 120            3/39        111     +2.39      +0.41       +2.86  0.177     3.48     6.26
  (c) sloped, drift 1.5          10/39        282     +1.97      +0.29       +2.65  0.326     6.13     8.73
  (c) sloped, drift 2.5          12/39        290     +2.13      +0.50       +2.08  0.231     5.77     8.59
  (d) edge_basis close            2/39         64     +1.88      +0.20       +1.03  0.648     3.24     6.34
  (e) exclude prior leg           3/39        132     +2.04      +0.17       +3.27  0.087     4.17     6.17
  (f) big_min_bars 6              7/39        121     +2.19      +0.31       +3.84  0.039     3.40     6.17
  COMBO a+c                      20/39        384     +1.47      -0.17       +0.24  0.800     5.90     7.13
  COMBO a+c+f                    22/39        370     +1.56      -0.10       +1.49  0.460     5.38     7.92
  COMBO all five as briefed      23/39        319     +1.27      -0.37       -1.50  0.457     9.54    12.78
```

**(a) splits.** The ceiling is wrong and the floor is right. 25→35% buys
3→5 coverage for +12 signals/yr; 15→5% buys **nothing** for +59 signals/yr and
costs 2pp of fold return and two positive folds. The floor's named
counter-example fails: Colgate Palmolive at 8.8% is rejected on *duration* (a
3-bar candidate), so no floor setting recovers it. The four names the range
arm does recover are exactly the four the diagnosis predicted — PB Fintech and
Akzo Nobel (too tight), LIC Housing and Radhika Jeweltech (too wide).

**(b) and (e) are no-ops**, for the reasons in §15.2.

**(d) is backwards.** Closing-basis edges make every box *narrower*, so more
candidates fall below the range floor rather than fewer breaking containment.
Signals halve, and it is the only arm that **loses** already-detected names
(HBL Power Systems and Zaggle).

**(c) is the one real mover, and the one genuine anomaly.** Sloped channels
take coverage 3→12 and recover precisely the shapes §7.2 group 3 named (PB
Fintech, Akzo Nobel), plus Modisons and Marine Electricals — which were filed
under "bases longer than the detector allows" and turn out to be *sloped*, not
long. Its per-trade edge over the universe base rate **holds**: +0.50% at
t=3.17 on n=1,532, against +0.48% at t=2.01 on n=551. Its portfolio result
does not: fold return 4.78→2.08, mean drawdown 3.10→5.77. PROVEN 5 is the
explanation — at three slots, 2.4x the signals means trading a differently and
worse-ranked subset, not the same trades more often.

**Selectivity, as a guard.** The share of sampled bars that register as a box:
V0 33.6%, `big_min_bars=6` 35.6%, sloped 60.6%, range 5-35% 67.5%. The
codebase records 76% as the pathology that killed window-sweeping (§1.3), and
both wide arms approach it. `big_min_bars` does not.

**A caveat on the sloped construction.** Horizontal boxes are *anchored* —
the start bar is the running maximum, so duration comes out of price structure
and the §2.3 duration norm means something. A rising channel has no such
anchor (its high is the last bar by construction), so the sloped arm uses the
anchor lookbacks as fixed window lengths instead. That keeps the candidate
count identical to the horizontal path rather than sweeping every length, but
it does mean a sloped box's duration is chosen by the lookback set rather than
by the stock, and `duration_vs_own_norm` is correspondingly less meaningful
for them. Sloped candidates are also evaluated only when no horizontal box is
found, so the arm is purely additive and nothing already detected is redrawn.

### 15.4 What this settles

The bottleneck was correctly located. Geometry *is* what discards the
coverage admission buys, and the coverage is reachable: all five arms together
reach 23 of 39, against a ceiling of 25 set by admission. It costs the sign
of the fold return (−1.50%, positive in 2/7) and triples mean drawdown to
9.54%.

This is the fifth attempt to raise signal count and it fails the same way as
the previous four — with two qualifications that are recorded rather than
rounded off:

1. The sloped arm is the first change in this project to quadruple real-trade
   coverage **without** lowering the per-trade edge over the universe base
   rate. It still loses on the portfolio curve.
2. `big_min_bars` 10→6 is the first lever found that raises coverage
   (3→7) at **flat signal count** (120.0→120.7), and it improves fold
   consistency (p 0.063→0.039, 6/7 positive folds). It still lowers mean fold
   return and per-trade edge, so it is not adopted — but it is the only place
   the coverage/signal-count coupling breaks, and the only geometry lever that
   is not a flooding change.

Nothing here is adopted. Every option added is default-off. **AES_HOLDOUT
remains unspent**, and sizing was not touched.

## Phase 16 — outcome-first: what precedes a +30% run

Every prior phase started from a stated rule and asked whether it worked. This
starts from the outcome. Code: `research/runner_study.py`.

**Protocol cost, stated first.** This spends AES_HOLDOUT as discovery data —
see STATE.md §5. What replaces it was defined before any result was computed:
a 25% symbol holdout by salted hash, and forward bars from 2026-09-17.

### 16.1 Design

- **Runners:** 2022-2026, 402-name universe, every bar from which close gains
  **+30% within 25 sessions**. The recorded bar is the last bar *before* the
  run, so all features are causal. Episodes de-duplicated by skipping the run
  window, or one 40% move contributes twenty overlapping "runners".
  **1,215 runners** on the 75% discovery slice, 229 distinct symbols, median
  gain 32.4%, spread evenly across years (227/316/299/184/189).
- **Controls:** for each runner, two names on the **same date**, in the same
  **turnover quintile**, that explicitly did *not* gain 30% in the next 25
  sessions. Same-day matching holds market regime constant, which otherwise
  dominates everything.
- **A second control arm matched on turnover *and* realised volatility.** A
  +30% move in 25 sessions is mechanically easier for a volatile name, so
  without this the study cannot separate "this setup precedes runs" from
  "this stock moves a lot".
- Point-in-time market cap does not exist in this cache, so turnover is the
  size proxy. It conflates size with activity.

### 16.2 Most of the raw signal is volatility

Turnover-matched only, then turnover+volatility-matched (AUC, runners vs
controls):

```
  feature              turnover-matched   +vol-matched
  atr_pct_40                 0.686            0.589
  range_pct_40               0.673            0.584
  realized_vol_20            0.650            0.513   <- the matching variable
  bb_width_20                0.634            0.541
  dist_52w_low               0.600            0.547
  days_since_20pct_move      0.341            0.430
  max_dd_40                  0.342            0.449
```

Volatility matching removes roughly half of every raw effect. What survives is
the real question, and it is a **pullback-within-uptrend** profile, not a
coiled-base one.

### 16.3 What separates runners from controls (vol-matched)

Runner median vs control median, ranked by |AUC − 0.5|. Full table in
`results/runner_features_vol_matched.csv`.

```
  feature                 runner    control    AUC    reading
  atr_pct_40               0.042      0.039   0.589   structurally wider bars
  range_pct_40             0.296      0.262   0.584   WIDER 40-bar range, not tighter
  vol_ratio_20_60          0.971      1.035   0.428   volatility CONTRACTING
  days_since_20pct_move   75 bars   117 bars  0.430   had a big move ~3 months ago
  ema20_dist              -0.024     -0.007   0.427   BELOW the 20-EMA
  dd_from_40b_peak        -0.105     -0.084   0.440   further below the 40-bar peak
  close_position_40        0.416      0.475   0.454   LOW in its own 40-bar range
  rs_20_vs_index          -0.021     -0.003   0.453   WEAK 20-day relative strength
  rs_20_vs_sector         -0.018      0.001   0.451   weak vs its own sector too
  dist_52w_high           -0.205     -0.192   0.464   FURTHER from the 52-week high
  ret_252                  0.340      0.244   0.539   12-month uptrend intact
  vol_ratio_10_40          0.884      0.933   0.468   volume drying up
```

**The profile:** a structurally volatile name in a twelve-month uptrend, which
had a big move about three months ago, has since pulled back below its 20-EMA
into the lower half of its recent range on contracting volume and volatility,
and is nowhere near its 52-week high.

That is a **pullback**, bought weak. Not a breakout, bought strong.

### 16.4 It is directional, not just "big move"

The obvious failure mode is a volatility detector wearing a direction label.
Tested against **-30% in 25 sessions** crashers with their own matched
controls:

```
  AUC(+30% runners  vs controls)  0.599
  AUC(-30% crashers vs controls)  0.444
  directional edge                +0.154
```

Per-feature it is cleanly two-sided: `ema20_dist` is 0.427 for runs and
**0.681** for crashes; `close_position_40` 0.454 / 0.635; `rs_20_vs_index`
0.453 / 0.631; `dist_52w_high` 0.464 / 0.605. Extended names crash, pulled-back
names run. Same feature, opposite ends.

### 16.5 The validation slice, spent once

Composite: equal-weight average of the 21 vol-matched survivors
(|AUC−0.5| ≥ 0.03), each signed by its discovery direction, standardised on
discovery. **Equal weight deliberately, not fitted** — UNPROVEN #3 is what
fitted weights cost last time.

```
  discovery AUC (75% of symbols)   0.599
  VALIDATION AUC (held-out 25%)    0.612
  shrinkage                       -0.014
  features keeping their sign      21 / 21
```

**Nothing else in this project has held out of sample.** Every prior finding
weakened when the sample grew (UNPROVEN #1: eleven folds made every statistic
worse). This did not weaken. A logistic fit gives 0.640 in 5-fold CV, so 0.61
is close to what these features can do, not a floor.

What it does **not** establish: tradeability, profitability, or survival
outside this regime. AUC 0.61 is modest, and the holdout shares the calendar.

### 16.6 The detector, checked against it — two true facts that look opposed

**Fact one: AES's admission gate is a genuinely good runner filter.**

```
  +30%-in-25-session rate, all eligible bars   12,429 / 283,526 =  4.38%
  +30%-in-25-session rate, AES admission bars       392 /   3,523 = 11.13%
  lift                                                              2.54x
```

No composite involved. The screener's "+20% in 10 sessions and within 5% of
the 52-week high" picks a pool that runs **2.5 times** as often as the
universe. That is a real, large, and previously unmeasured property of the
gate, and it is the opposite of the conclusion the composite alone suggests.

**Fact two: within that pool, AES measures almost none of what further
separates runners, and reads six things backwards.**

```
  composite score      n      median    AUC vs controls
  +30% runners       1215     +0.058         0.599
  matched controls   2409     -0.118         0.500
  the 41 real trades   39     -0.325         0.413
  AES admissions     3523     -0.588         0.191
```

Six of the 21 features *are* AES criteria, so 0.191 is partly circular.
Re-run on only the **12 features AES does not measure at all**:

```
  AUC(runners        vs controls)  0.598   <- essentially the whole effect
  AUC(AES admissions vs controls)  0.259   <- non-circular, still far off
  AUC(real trades    vs controls)  0.488   <- neutral
```

**All of the residual discriminating power (0.598 of 0.599) lives in features
the detector does not compute.** EMA distance, volatility contraction,
pullback depth, sector-relative strength, band width, 12-month return,
distance from the 52-week low, gap frequency.

```
  status      features
  MEASURES    atr_pct_40 (exit sizing only, not entry), vol_ratio_10_40
  BACKWARDS   dist_52w_high, close_position_40, rs_20_vs_index,
              range_pct_40, range_pct_10, turnover_ratio_10_40
  IGNORES     ema20_dist, ema50_dist, vol_ratio_20_60, dd_from_40b_peak,
              max_dd_40, mean_dd_40, rs_20_vs_sector, dist_52w_low,
              bb_width_20, ret_252, gap_freq_40, up_gap_freq_40
```

"Backwards" is literal: the screener requires **within 5%** of the 52-week
high while runners sit a median **20.5% below** it; the entry fires on a
breakout to the top of the range while runners sit at **0.42** of their own
40-bar range; the box wants a **tight** 15-25% range while runners run
**29.6%** wide.

**Reconciling the two facts.** They answer different questions. AES selects a
pool — volatile, liquid, in-motion names — that unconditionally runs 2.5x more
than the universe. Conditional on being in that pool, it has no instrument for
telling which members will run, because volatility and turnover matching
removes exactly the axis it selects on, and the residual axes are ones it never
computes. The gate is good; the discrimination *inside* the gate is absent.

### 16.7 The 41 real trades, third reference point

On the full composite the real trades score **-0.325** against a control
median of -0.118 (AUC 0.413) — further from the runner profile than an average
control. On the 12 ignored features alone they are **neutral** (0.488). So the
gap comes entirely from the AES-criteria features: the trader is selecting on
the same things the detector does — near the 52-week high (median **-5.4%**
vs runners' -20.5%), above the 20-EMA (**+7.3%** vs -2.4%), high in the range
(**0.82** vs 0.42), strong 20-day RS (**+19.3%** vs -2.1%), and a 20% move
**5 sessions** ago rather than 75.

**This does not make the trader wrong.** §7.3 measured those 41 trades at
+1.95% to +4.85% mean excess over Nifty 500 under every exit rule. They target
a different animal: a 17-bar continuation of 5-8%, not a 25-session 30%
expansion. The two objectives are genuinely different, and the runner profile
is evidence about the second, not the first.

What it does say is that **the detector and the trader agree with each other
and disagree with the runner data on the same six axes** — which is why five
phases of tuning the detector toward the trader never moved the outcome. They
were converging on each other, not on the thing that produces runs.

### 16.8 Status

Nothing here is adopted, and nothing is tradeable yet. The composite predicts
*direction of a 30% expansion*, not return net of cost, and says nothing about
what the position risks while waiting. The next question is not "does this beat
the detector" but "does buying the pullback profile produce a positive net
edge after cost", which is a backtest that has not been run and which must be
judged on forward bars (`FORWARD_CHECK_FROM`), not on 2022-2026 again.

## Phase 17 — does any of Phase 16 survive cost?

All of this is on 2022-2026, which is **discovery** (STATE.md §5). Nothing here
is validated. `FORWARD_CHECK_FROM` (2026-09-17) is the only clean evidence left.

### 17.1 The composite alone fails at every swing horizon

Top decile of the Phase 16 composite, ranked cross-sectionally per day (so the
decile is not secretly a date filter), entered at the **next open**, excess over
the universe base rate, net of the 0.585% round trip. n ≈ 39k bar-signals over
1,167 dates.

```
   H        n   top excess%   base%   gross%    NET%   t(obs)   t(day)
  10   39,662          0.65    0.42     0.23   -0.35     5.19     2.29
  15   39,473          1.05    0.69     0.36   -0.22     6.68     3.06
  20   39,287          1.43    0.96     0.47   -0.12     7.51     3.40
  25   39,102          1.80    1.24     0.56   -0.02     8.11     3.63
  40   38,534          3.15    2.13     1.01   +0.43    11.14     5.01
```

`t(obs)` treats every overlapping hold as independent and is inflated by
roughly the square root of names held per day; `t(day)` is the t across daily
means and is the honest one.

**Net edge is negative at 10, 15 and 20 bars, zero at 25, and only positive at
40** — where it is +0.43%, i.e. the edge is smaller than the cost that produced
it. AUC 0.61 buys 1.01% gross over eight weeks.

### 17.2 What the AUC was hiding: max adverse excursion

Worst low between entry and horizon, top decile against all bars:

```
   H   MAE med   MAE p25   MAE p10   < -10%   < -20%   |  all-bar MAE med
  10      -5.1      -9.1     -13.6    20.8%     2.3%   |            -4.2
  15      -6.3     -11.0     -16.3    29.4%     4.6%   |            -5.2
  25      -8.1     -13.9     -19.9    40.3%     9.9%   |            -6.7
  40     -10.0     -16.8     -23.5    49.8%    16.9%   |            -8.2
```

At the only horizon that clears cost, the median position is **10% underwater**
at its worst, **half** go below -10%, and one in six below -20%. A stop tight
enough to be a swing stop takes you out of most of them; a stop wide enough to
survive makes the risk-per-share so large that §7 sizing gives a position too
small for +0.43% to matter. This is the cost the AUC could not see, and it is
larger than the edge.

**Verdict: the composite is not tradeable as a standalone entry.** Buying the
pullback profile off the open universe fails in-sample, before any question of
whether it survives forward.

### 17.3 The combination: same gate, different trigger

The one construct the two Phase 16 findings jointly imply, and which had never
been run. Both arms share the AES admission gate (+20%/10d, within 5% of the
52-week high), the same window, the same cost, the same simulator.

- **Arm A — breakout.** The real AES watchlist: box detection then §5 modes
  1/2/3. **782 signals, 166/yr.**
- **Arm B — pullback.** After admission, wait up to 90 bars for
  `close < EMA20` **and** lower half of the 40-bar range **and**
  `vol20/vol60 < 1`. **789 signals, 168/yr.**

Near-identical signal counts, so this is a like-for-like timing comparison
rather than a volume change.

```
  arm                    H     n   gross%    NET%      t
  A breakout (V0)       10   776     0.76   +0.17   2.26
                        15   773     0.96   +0.37   2.42
                        20   771     1.19   +0.60   2.43
                        25   766     1.16   +0.58   2.20
                        40   758     1.62   +1.04   2.37
  B pullback (new)      10   785     0.38   -0.21   1.26
                        15   782     0.88   +0.29   2.29
                        20   779     0.97   +0.38   2.14
                        25   776     1.26   +0.67   2.41
                        40   766     1.76   +1.17   2.66
```

**The breakout trigger wins at every short horizon.** At 10 bars it is +0.17%
against the pullback's **-0.21%**; at 15 bars +0.37% against +0.29%; at 20 bars
+0.60% against +0.38%. Only at 25 bars and beyond does the pullback pull ahead
(+0.67 vs +0.58, +1.17 vs +1.04).

Portfolio, 3 slots, equal weight, ATR 2.5x trailing stop, cost charged. **These
are not comparable to §9.1's fold table** — no risk sizing, no ladder, no
regime, so each position is ~33% of equity against the locked config's ~6-9%
deployment. Only the A-vs-B contrast is meaningful; the absolute levels are an
artefact of the simplification, and 2023 is plainly outlier-driven.

```
  hard cap 10 bars        total%   maxDD%     2022     2023     2024     2025    2026
  A breakout              272.08   -42.76    11.83   188.10    54.13   -25.55   -7.69
  B pullback              -39.27   -52.06   -20.11    28.52   -25.59   -25.54    0.54

  hard cap 15 bars
  A breakout              213.87   -43.58    18.47    93.67    39.61    -1.75   -2.80
  B pullback              167.23   -36.52    -4.89    68.34    79.35   -10.12    1.08

  hard cap 25 bars
  A breakout              120.65   -38.83     0.45    91.69    15.52   -19.83   16.91
  B pullback              146.04   -33.26    30.07    80.07    50.01   -25.34   -5.54
```

At a 10-bar cap the pullback arm is **-39% against +272%**. At 25 bars it wins
modestly on return and on drawdown (-33.3% vs -38.8%).

### 17.4 What this settles

**The screener was right and the entry trigger was also right — for short
holds.** The hypothesis that the 2.54x gate lift plus six backwards criteria
implied a broken entry does not survive its own test. The breakout trigger beats
the pullback trigger at 10, 15 and 20 bars on the same admissions, and collapses
it at a 10-bar cap.

Stated plainly, since it was asked for explicitly rather than optimised away:

- **As a short swing (10-15 bars): only the existing breakout entry works.**
  +0.17% / +0.37% net. The pullback profile is negative at 10 bars and weaker
  at 15. The Phase 16 profile is not reachable as a short swing.
- **As a 5-week position (25-40 bars): the pullback entry is mildly better**
  (+0.67 / +1.17 vs +0.58 / +1.04) with lower drawdown, and the standalone
  composite becomes marginally positive at 40 bars only.
- Phase 16's finding therefore describes a **different holding period**, not a
  better entry for the one being traded. Runs take eight weeks to pay and cost
  a 10% median excursion to sit through.

One incidental result worth recording: **Arm A is net-positive at every horizon
on 2022-2026** (+0.17 to +1.04, t 2.2-2.4), against the 2015-2021 walk-forward's
+0.48% (t=2.01) on the same base-rate definition. The existing entry does not
look worse in the recent window than it did in the old one. That is consistent
with PROVEN 12's 2.54x gate lift and cuts against the reading that the pipeline
is mistimed.

**Nothing adopted. Nothing validated.** Every number here is in-sample on a
window that is now discovery.

## Phase 18 — slots x signal count x risk: the honest ceiling

The deployment question, run with `AESPortfolioBacktester` throughout: risk-based
sizing, ATR(14) x 2.5 trailing stop, §6.2 ladder, §8 regime rules. Not §17's flat
~33% per position, which ignored where the stop sits.

Two windows, never pooled. **2022-2026 is discovery** (STATE.md §5).
Control: pool P1 at 3 slots / 1% reproduces the V0 fold table exactly
(+4.78%, mean DD 3.10%, worst 5.82%, folds to the decimal).

**Sizing convention:** `target_concurrent_positions` scales with slots at the
locked ratio (2.5/3), so base notional is 1/S of equity. Holding it at 2.5 while
raising slots to 15 would let one position take 40% of equity and make the
notional cap, not the risk budget, the operative constraint.

**Pools** (admission held at V0 20/5; only box geometry varies):

```
  pool   geometry                     2015-2021      2022-2026
  P1     baseline BoxParams()         120 sig/yr     166 sig/yr
  P2     big_range_min 0.05           180 sig/yr     246 sig/yr
  P3     sloped, drift 2.5            290 sig/yr     432 sig/yr
```

The same geometry yields ~40% more signals in the recent window.

### 18.1 The grid

```
  2015-2021                                        2022-2026 [DISCOVERY]
  pool  slots risk   ret%  meanDD  depl%  pos      ret%  meanDD  depl%  pos
  P1      3    1%    4.78    3.10   8.7   5/7      2.10    3.30   9.5   3/5
  P1      3    3%    8.88   11.25  24.3   3/7     11.03    8.90  26.7   4/5
  P1      6    1%    5.64    4.83  14.0   4/7      5.11    5.75  16.4   3/5
  P1      6    3%   10.60    8.79  25.2   3/7     15.38    9.02  29.9   4/5
  P1     10    1%    4.97    5.77  17.0   4/7      7.49    6.27  20.5   3/5
  P1     10    3%    5.50    6.98  19.6   4/7      7.78    7.32  23.1   4/5
  P1     15    1%    3.94    5.11  14.3   4/7      4.58    5.41  17.0   3/5
  P1     15    3%    4.03    5.17  14.4   4/7      4.70    5.51  17.1   3/5
  P2      6    3%   12.66   10.79  31.8   4/7      6.98   13.64  33.7   2/5
  P2     15    1%    6.62    6.91  22.7   4/7      7.07    7.84  24.5   2/5
  P3      3    3%    5.07   16.13  34.3   3/7    -13.17   18.53  34.2   1/5
  P3     10    3%   11.81   12.14  35.0   3/7     -0.11   13.36  39.7   1/5
  P3     15    3%    7.97   10.56  31.3   3/7      4.23    9.60  36.4   2/5
```

Full grid in `results/slot_sweep_2015-2021.json` and `..._2022-2026.json`.

### 18.2 What the grid says

1. **Six slots is the peak in both windows.** Three is under-deployed; ten and
   fifteen are worse. At fifteen, deployment *falls* (25%→14% on 2015-2021)
   because base notional is 1/S and the cap starts binding before the risk
   budget does.
2. **Risk 3% beats risk 1% at every slot count in both windows.** PROVEN 7
   confirmed with deployment free to move: 4.78→8.88 at three slots,
   5.64→10.60 at six.
3. **Deployment saturates at 30-38%** and never approaches the ~48% PROVEN 4
   estimated as necessary for 10%/yr. It reaches 10-15% anyway, because the
   traded subset carries a higher edge than the average signal — the +1.42pp
   selection gap measured in the Phase 15 correction.
4. **More signals still hurts, and hurts harder with deployment.** P3 (290/432
   per year) is negative in five of eight 2022-2026 cells, worst -13.17%. Sixth
   independent confirmation of PROVEN 5.
5. **The windows agree on rate, not on pool.** The best cells are P2 at 180/yr
   (old) and P1 at 166/yr (new) — both ~170-180 signals a year. P2 does not
   transfer (12.66→6.98); **P1 at 6 slots / 3% is the only cell above 10% in
   both windows** (10.60 / 15.38).
6. **Beta is not preserved.** PROVEN 2 claims 0.016-0.147 across "every fold and
   every configuration tested". At 3% risk it runs **0.26-0.40**. Scaling risk
   buys index exposure; that claim needs its scope narrowed to the 1% cells.

### 18.3 The ceiling, stated as a number

Best cell in each window, and the same cell's robustness:

```
                              2015-2021            2022-2026
  best-in-window cell      P2 6 slots 3%        P1 6 slots 3%
  annual return                 12.66%               15.38%
  compounded                    10.46%               13.80%
  alpha / beta              7.45 / 0.263        12.11 / 0.326
  mean DD / worst DD         10.79 / 15.40         9.02 / 13.48
  deployment                     31.8%                29.9%
  positive folds                  4/7                  4/5
  MEDIAN fold                    +3.11%              +10.83%
  mean ex-best fold              +4.98%               +5.86%
  mean ex-top-2 folds            -0.51%               +3.34%
  best fold's share of total       66%                  70%
```

**Yes, 12-15%/yr is reachable in-sample, at roughly 9-11% mean drawdown and
13-15% worst.** That is the honest headline and it is a real change from the
locked configuration's 4.78% at 3.10%.

**And it is carried by one year in both windows.** Remove the best fold and
12.66% becomes 4.98%, 15.38% becomes 5.86%. Remove two and the older window
goes negative. The dispersion is the same 2017/2021-and-2023 concentration
every other phase of this project has found; deployment multiplies it rather
than diversifying it.

**The cross-window-robust choice is P1 at 6 slots / 3% risk** — baseline
geometry, no loosening — delivering 10.60% and 15.38%, mean drawdown 8.79% and
9.02%. Against the locked config that is **roughly 2-3x the return for roughly
3x the drawdown and 4x the beta**, with fold consistency no better (3/7 and 4/5
against 5/7).

**What it costs, plainly:** to earn ~12%/yr you accept a ~9% average annual
drawdown, a ~13-15% worst year, beta near 0.33 rather than 0.07, and a median
year of −2.2% in the older window. The expectation you should plan on is not
12-15% but the ex-best-fold figure of **~6%/yr**, with 12-15% arriving only
when a 2017/2021/2023-type year does.

### 18.4 Status

Nothing adopted. `AES_HOLDOUT` was already spent (§16) and 2022-2026 is
discovery, so the 15.38% figure is in-sample twice over: the window is
discovery, and the cell is the best of 24. `FORWARD_CHECK_FROM` (2026-09-17)
remains the only clean evidence available.

## Phase 19 — the Early Sector Rotation blueprint, tested as a feature

Tested as a feature on the existing AES signal population rather than built as
a nine-phase system. Code: `research/sector_rotation.py`.

**Built:** §5.1 price/RS layer, §5.2 breadth layer, §3 lifecycle states via the
§9 classifier, §28 lead time. **Skipped:** §5.3 delivery (REJECTED #1 — six
formulations, two universes, no edge in any) and §5.4 derivatives (no clean
free historical OI). Sector indices are equal-weight composites of the AES
universe's own constituents, so index and breadth stay consistent and the
population matches the mid/smallcap names the signals come from. Survivorship
applies as everywhere else. 16 sectors, 49,584 sector-days.

State distribution: Dormant 42.9%, Leader 17.0%, Fading 15.6%, Emerging 12.7%,
Crowded 11.7%, Accumulating **0.2%**.

### 19.1 The "early" premise fails its own control

§28's measure, against the blueprint's own §14 requirement of a matched
non-event sample — random sector-days that are likewise not yet in the top
decile of 20-day excess return, same count, same 60-day horizon:

```
  window      arm        onsets  reached%   FP%   median lead   mean lead
  2015-2021   Emerging     1064     66.9    33.1       16.0        21.4
              placebo      1158     63.8    36.2       21.0        23.9
  2022-2026   Emerging      750     67.6    32.4       16.0        20.3
              placebo       819     64.7    35.3       21.0        23.8
```

Emerging reaches obvious momentum **3 percentage points** more often than a
random not-yet-obvious day. And its median lead is **shorter** than the
placebo's — 16 days against 21 — so it fires *closer* to the obvious move than
chance does, not earlier. Both windows agree. **The "early" premise is not
supported on this data.**

### 19.2 Edge by sector state, on the AES signals

Net of 0.585% cost, over the universe base rate, h=20.

```
  state          2015-2021                 2022-2026
                  n    edge      t          n    edge      t
  Dormant       206   +0.64    1.31       149   -0.47    0.12
  Emerging       65   -0.19    0.25        55   +0.90    1.02
  Leader        115   +1.50    1.90       107   +4.23    2.94
  Crowded        73   +0.65    0.98        56   +4.05    2.27
  Fading         40   -2.80   -1.20        42   -3.72   -1.97
  ALL           499   +0.46    1.87       410   +1.20    2.61
```

**Emerging is below the population average in both windows** (−0.19 vs +0.46;
+0.90 vs +1.20). The best state in both is **Leader** — the state the blueprint
argues is already too late. Fold by fold, Emerging beats the population average
in **2 of 7** folds (2015-2021) and **2 of 3** (2022-2026); per-fold Emerging
counts are 1-23, so most folds cannot measure it at all.

Emerging minus Fading is +2.61pp (t=1.07) and +4.62pp (t=2.14). The gap exists,
but it is **entirely the Fading side**: Emerging is below average in both
windows, so this is not "Emerging carries more edge", it is "Fading carries
less".

### 19.3 Silent Rotation loses to plain momentum, in both windows

§7 is the blueprint's stated central alpha hypothesis.

```
                                        2015-2021        2022-2026
  Silent Rotation                       -0.65  (n 28)    +0.32  (n 31)
  not Silent Rotation                   +0.52  (n 471)   +1.27  (n 379)
  plain momentum, sector top-30%        +0.61  (n 186)   +3.21  (n 157)
  plain momentum, sector bottom-30%     -1.05  (n 83)    -1.43  (n 67)
```

Silent Rotation is worse than its own complement and worse than plain sector
momentum, in both windows. Plain momentum separates cleanly (+0.61/+3.21 top
against −1.05/−1.43 bottom); the internals-based version does not.

### 19.4 Signal freshness does not rescue it

§27 alerts on Day 1, so the day-level test could dilute a short-lived effect.
Split by days since the sector entered Emerging:

```
                    2015-2021              2022-2026
  Emerging day 0-2   -2.46 (n 40)          +2.35 (n 32)
  Emerging day 3-9   -0.20 (n 19)          +0.02 (n 18)
  Emerging day 10+  +14.94 (n 6)           -5.18 (n 5)
```

The two windows point in **opposite directions** at every age bucket, and the
extreme values sit on n=5 and n=6. Noise, not a freshness effect.

### 19.5 The one consistent effect is the inverse one, and it dies at portfolio level

Fading is materially negative in both windows, which is the only result here
that repeats. Excluding Fading signals lifts per-signal edge by +0.30pp (6/8
folds) and +0.39pp (4/5 folds). Run through the portfolio at 6 slots / 3% risk:

```
                       2015-2021                    2022-2026
  baseline           10.60%  meanDD 8.79  3/7     15.38%  meanDD 9.02  4/5
  exclude Fading     10.89%  meanDD 8.89  3/7     14.45%  meanDD 9.53  4/5
  better in           3/7 folds, +0.29pp          1/5 folds, -0.93pp
  paired t                 0.12                        -1.39
```

It drops 9-11% of signals to gain nothing in the older window and to **lose**
0.93pp in the recent one. The per-signal uplift does not survive contact with
score ranking and six slots: the dropped signals were largely ones the
portfolio never traded.

### 19.6 Verdict — it does not separate, and this stops here

Per the standing rule, nothing proceeds to the blueprint's later phases.

- The "early" premise beats a placebo by 3pp and fires *later* than chance.
- Emerging carries no extra edge — below average in both windows.
- Silent Rotation, the central hypothesis, loses to plain sector momentum in
  both windows.
- The only repeatable effect is the inverse (avoid Fading) and it is worth
  nothing at portfolio level.

**Two implementation caveats, stated rather than buried.** First,
`Accumulating` is 0.2% of sector-days under this rule ordering, because
Emerging overrides it whenever acceleration is also high — so the blueprint's
"primary research target", the Accumulating → Emerging *transition*, is
essentially unobservable as a distinct event here. What was tested is
Emerging-as-a-state and Emerging-onset, both well populated, and neither shows
edge. Second, sector indices are equal-weight composites of today's
constituents, so early-period breadth is computed over survivors.

Neither caveat rescues the thesis: the blueprint's own §28 measure, run on
sector indices alone with no profitability claim attached, already fails
against a placebo, and that test is independent of both caveats.

## Phase 20 — the profit ladder, re-tested under the ATR stop

REJECTED #5 killed the ladder, but it ran under the SMA(10) stop. The re-test
reason is stated first, per the reading rule, and was verified before anything
was swept:

```
  stop                      positions  median hold  exit reasons
  SMA(10) spec                   273       6 bars   sma10 272, ladder1 100, ladder2 78, ladder3 26
  ATR 2.5x + 25-bar cap          206      17 bars   ladder1 110, ladder2 88, atr_trail 83,
                                                     time_stop 68, max_hold 49, ladder3 37
```

Under the SMA stop **272 of 273 positions died at the stop** — the ladder could
not bind because nothing survived to reach it. Under the ATR stop the median
hold is 17 bars and 235 ladder exits fire across 206 positions. The ladder is
cutting live positions for the first time, so REJECTED #5's finding does not
transfer. Config throughout: ATR 2.5x + 25-bar cap, equal-weight scoring,
6 slots / 3% risk (the cell the R figures were quoted from).

### 20.1 A measurement correction: the 0.65-0.72 R multiple was never the setups'

```
  window      variant        rows   row-mean R   positions   position R
  2015-2021   current ladder  440        0.652         206        0.184
              no ladder       199        0.212         199        0.212
  2022-2026   current ladder  435        0.720         192        0.164
              no ladder       191        0.215         191        0.215
```

A ladder splits one position into 2-4 trade rows and **every ladder row is
profitable by construction** — a tranche only sells because price reached its
trigger. Averaging R over rows therefore counts the winning fragments of a
position repeatedly and the losing remainder once. 0.652/0.720 is that row
average. The position-level R under the same ladder is **0.184 / 0.164**.

So the question "is 0.65-0.72 a property of the setups or an artefact of the
ladder" has a third answer: it is an artefact of the **measurement**. The
setups' true R is ~0.17, and removing the ladder raises it to ~0.21 — a real
15-31% improvement, on a much smaller base than the quoted figure implied.
(Every earlier R figure in this project that came from `trades_frame` row means
is inflated the same way; §18's tables are unaffected, they report returns.)

### 20.2 The right tail is truncated, in all three datasets

```
                        2015-2021                      2022-2026
  variant          p90win  maxwin  >20%  >30%    p90win  maxwin  >20%  >30%
  a current          13.0    22.8     1     0      14.1    27.0     3     0
  b no ladder        24.3    93.3    20     7      35.3    90.4    28    15
  c +12% 50/+20%     19.0    37.8    11     1      22.6    50.1    20     4
  d +15% 30% only    23.5    70.2    18     3      28.5    73.1    24     9
```

**Under the current ladder there is not one winner above +30% in either
window** — 398 positions, zero. Remove it and there are 7 and 15. The 41 real
trades say the same thing independently:

```
  variant            n   mean%   med%   win%    p75    p90     max  >20%  >30%   total%
  a current         39    1.47   2.85   64.1   10.9   15.0    18.8     0     0     57.3
  b no ladder       39    2.37   0.59   51.3   12.6   46.2    52.7     4     4     92.4
  c +12%            39    1.97   0.59   53.8   15.6   24.2    33.3     4     1     76.8
  d +15% 30%        39    2.31   0.59   53.8   13.6   38.9    42.5     4     3     90.1
```

Largest real trade under the current ladder: **+18.8%**. Without it: **+52.7%**,
and total return across the 39 goes 57.3% -> 92.4%.

**The setups do run. The ladder is the constraint on how far.** The recalled
"8-10% per trade" is an accurate description of what the ladder permits, not of
what the setups reach.

### 20.3 But deleting it is not the answer, and the windows disagree

Paired fold comparison against the current ladder:

```
  window      variant            better   mean diff   paired t   ann%      meanDD
  2015-2021   b no ladder          3/7       -0.13      -0.04   10.60->10.47   8.79->14.63
              c +12%               4/7       +4.04       1.09   10.60->14.64   8.79->11.58
              d +15% 30%           4/7       +0.59       0.23   10.60->11.19   8.79->12.83
  2022-2026   b no ladder          4/5      +13.70       1.98   15.38->29.08   9.02->11.52
              c +12%               4/5       +4.44       1.48   15.38->19.82   9.02->11.04
              d +15% 30%           4/5       +8.88       1.92   15.38->24.26   9.02->10.78
```

Full removal is the best variant in 2022-2026 (+13.70pp, t=1.98) and **worth
nothing in 2015-2021** (-0.13pp) while taking mean drawdown from 8.79% to
14.63% and worst from 11.92% to 20.19%. That is the same disagreement §18 found
between these windows, and it is why no single answer is available here.

**Variant (c) — first tranche moved to +12%/50%, +20%/20% retained — is the
only variant better than the current ladder in both windows**: +4.04pp and
+4.44pp, better in 4/7 and 4/5 folds, positive in **5/5** folds in 2022-2026,
and with the best ex-best-fold figure of any variant in both windows (7.40 and
7.60, against the current ladder's 6.17 and 5.86).

### 20.4 What the ladder actually buys, and the verdict

The ladder is not free and it is not useless — it is a trade:

```
                      win rate        median return      right tail
  current ladder      65.5 / 64.1%    +3.27 / +3.48%     capped ~23-27%
  no ladder           54.3 / 56.0%    +0.71 / +1.33%     to +93%
```

It buys roughly **9 points of win rate and 2pp of median return**, and pays for
them with the entire right tail and ~20% of true R. Which side of that is
preferable is a preference about the return distribution, not purely an
optimisation.

**Nothing adopted.** No variant clears conventional significance (best paired
t = 1.98) and the two windows disagree on the largest effect. Variant (c) is
recorded as **the strongest cross-window candidate this project has produced**
— better on return in both windows, better on ex-best-fold in both, 5/5
positive folds in one — and it should be the thing forward-tested from
`FORWARD_CHECK_FROM`, not adopted on this evidence.

REJECTED #5 stands as written **for the SMA stop**, and is superseded for the
ATR stop: under ATR the ladder demonstrably truncates the return distribution,
which is the opposite of the "survival curve unchanged" finding that killed it.
