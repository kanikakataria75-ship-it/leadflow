# Research log — why DAB failed, and what the data says instead

Everything here is reproducible from the modules in `research/`. Numbers are
measured on 395 NSE mid/small-cap symbols with ~670 trading days of NSE delivery
data (roughly Oct 2023 – Sep 2026), net of the costs and circuit modelling in
`config.py`.

**Headline.** The original strategy is anti-predictive and its core premise
(delivery percentage) has no measurable value. A mean-reversion signal does
work, but only at a **20-30 bar hold** — at the 10-bar horizon it cannot cover
realistic Indian delivery costs. At 25 bars the net edge is **+0.30% per trade
out-of-sample** after correct costs (section 8). That is the first positive
result here, and it is small, it is not a swing strategy any more, and the
holdout that produced it has since been consulted too often to stay clean.

---

## 1. Why the original strategy failed

### 1.1 Rules 1 and 2 are mechanically antagonistic

Delivery% is `deliverable qty ÷ TOTAL traded qty`. Rule 2 demands the
denominator doubles, so for delivery% to *also* rise 1.5×, deliverable quantity
must more than triple on the same bar.

| Measurement | Value |
|---|---|
| `corr(log volume_ratio, log delivery_ratio)` | **−0.257** |
| Median delivery ratio, quiet days | 1.015 |
| Median delivery ratio, 2×-volume days | **0.817** |
| Bars passing the other four rules that also had a same-day delivery spike | **0.88%** |

That produced **9 signals in 17 months** — not a tradeable sample. Reformulating
rule 1 as "spike within the prior 10 bars, breakout today" (`delivery_mode:
prior_window`) raised it to ~152. This fix was necessary but, as section 1.2
shows, nowhere near sufficient.

### 1.2 Every rule selects for extension, and extension mean-reverts

Reproduce with `python -m nifty_swing_bot.research.diagnostics`.

Forward **excess return vs the index**, measured from the next open — so market
direction is stripped out and only the rule's own skill remains. The comparison
that matters is against the universe base rate, not against zero.

| Rule | N | 10d excess | vs base rate |
|---|---|---|---|
| **Universe base rate** | 226,844 | **+0.231%** | — |
| 1. Delivery accumulation | 42,920 | +0.170% | −0.061 |
| 2. Volume surge | 20,835 | −0.003% | −0.234 |
| 3. Strong close | 59,238 | +0.187% | −0.044 |
| 4. Relative strength | 109,000 | +0.168% | −0.063 |
| 5. Breakout / VCP | 12,046 | +0.024% | −0.207 |
| **ALL FIVE (DAB)** | **591** | **−0.356%** | **−0.587** |

Read the last row carefully. DAB entries did not merely fail to add value — they
**underperformed a random pick from the same universe by 0.59 percentage points**
over 10 days. Every individual rule is at or below the base rate, and stacking
five anti-predictive filters compounds the damage rather than cancelling it.

The economics are unsurprising in hindsight. A stock that has (a) surged in
volume, (b) closed on its high, (c) outperformed for 10 days and (d) broken to
20-day highs is at maximum short-term extension. In the NSE mid/small-cap tier
that is precisely what reverts.

**It was not the tape.** Over the same window the benchmark rose and the
universe's own base rate was positive at every horizon (+0.554% raw over 10
bars). A long-only system had a following wind and still lost.

### 1.3 Delivery% showed no predictive power in any formulation

This is the finding that matters most, because delivery data was the entire
premise of the project. Six independent formulations were tested
(`python -m nifty_swing_bot.research.signal_lab`), none survived:

| Formulation | Train edge | Holdout edge | Verdict |
|---|---|---|---|
| Spike on a down day (accumulation into weakness) | −0.591% | +0.117% | no edge |
| Spike while price is flat (quiet absorption) | −0.723% | −0.328% | no edge |
| Absolute delivery% > 75 | −0.235% | +0.055% | no edge |
| 5d avg delivery > 1.25× its 20d avg | −0.598% | +0.088% | no edge |
| Recent spike + pullback in uptrend | −0.441% | +0.419% | no edge |
| Recent spike + RSI < 40 | −0.130% | +0.032% | no edge |

The decisive test was designed to isolate delivery's contribution: take the best
non-delivery signal and add a delivery filter to it.

| Signal | Train edge | Holdout edge | N (train) |
|---|---|---|---|
| RSI(2) dip in uptrend | **+0.715%** | **+0.345%** | 5,446 |
| The same setup **+ recent delivery spike** | **−0.380%** | +0.366% | 917 |

Adding delivery cut the sample by 83% and destroyed the edge. On this universe
and this window, delivery percentage is noise. It remains available in the data
layer and is still charted on the Stock Detail page, but nothing keys off it.

---

## 2. What does predict

Method: every candidate is scored on excess forward return against the universe
base rate, on a **date-separated train/holdout split**, with a Bonferroni
threshold reported for the number of hypotheses tested (22 → |t| > 3.05).

Everything that survived both periods is **mean reversion**. Everything
momentum-shaped failed.

| Candidate | Train edge (t) | Holdout edge (t) |
|---|---|---|
| RSI(2) < 10, above 50d MA | +0.715% (6.5) | +0.345% (3.5) |
| Pullback + RSI(2) < 15 | +0.714% (7.3) | +0.355% (4.0) |
| Above 50d MA, 3d return < −2% | +0.566% (8.4) | +0.344% (5.3) |
| RSI(14) < 30 | +0.517% (5.4) | +0.339% (3.6) |
| *Plain 20d breakout* | *−0.289% (−3.5)* | *−0.005%* |
| *DAB (reference)* | *−0.997%* | *−0.697%* |

A note on why **RSI(2)** and not RSI(14): reaching RSI(14) < 35 generally
requires enough sustained decline to drag price below its own 50-day average, so
"oversold *and* in an uptrend" is nearly an empty set on a slow oscillator. The
test is kept in the lab as `oversold_in_uptrend` to document this.

---

## 3. Why the mean-reversion strategy *still* lost

Implemented as `strategy/mr_strategy.py` (Pullback Reversion, "PBR") and run
through the same portfolio backtester. The first version lost badly, for a
reason worth recording.

| Configuration | Trades | Win% | Avg win / avg loss | PF | Bars held | Return |
|---|---|---|---|---|---|---|
| With reversion exit (close > 5d MA) | 1,414 | **55.7%** | **0.57** | 0.72 | 2.7 | **−41.50%** |
| Without it — hold the full 10 bars | 652 | 47.7% | **1.16** | 1.06 | 9.1 | **+7.95%** |

**The exit rule was the bug, and it was mine, not the strategy's.** Exiting the
moment price reverts feels right — the premise has played out — but it banks a
tiny gain at bar 2.7 while the position carried the full 2.5×ATR stop the whole
time. Win rate goes *up* and the account goes down: the textbook way to be right
often and lose anyway.

The deeper principle: **the exit horizon must match the horizon the edge was
measured over.** The entry edge was measured across 10 bars; exiting at bar 2.7
collects a fraction of it and pays the entire round-trip cost.

`use_reversion_exit` therefore defaults to `False`, with this reasoning recorded
in `config.py` so it does not get "fixed" back.

### A second bug worth recording

The backtester ranked same-day candidates by `deliv_spike_ratio`. PBR feature
frames have no such column, so every candidate scored 0 and ties broke
alphabetically — the backtest was partly measuring the alphabet. Strategies now
publish an explicit `rank_score`. This matters because these strategies generate
several times more signals than the position limit can take, so the *ranking*,
not the entry rule, decides most of what actually gets traded.

---

## 4. The result that settles it

Exposure turned out to dominate everything else, which is itself the finding.

| Configuration | Trades | PF | Return | Benchmark | Exposure |
|---|---|---|---|---|---|
| 10 positions, 6% heat | 671 | 1.05 | +7.25% | +23.81% | 57.9% |
| **20 positions, 12% heat** | 1,140 | 1.12 | **+26.47%** | +23.81% | 85.6% |
| 20 positions + index regime filter | 851 | 0.97 | −4.40% | +23.81% | 65.7% |
| 30 positions, 20% heat | 1,338 | 1.11 | +23.02% | +23.81% | 84.5% |

The 20-position variant appears to beat the index. It does not. That comparison
comes from measuring a partially-invested strategy against a fully-invested
benchmark. Adjusting for the beta the strategy actually carried:

| Period | Trades | Return | Benchmark | **Alpha** | PF |
|---|---|---|---|---|---|
| First half | 545 | +10.05% | +21.11% | **−7.97pp** | 1.08 |
| Second half (out-of-sample) | 615 | +6.35% | +10.77% | **−2.82pp** | 1.07 |
| Full period | 1,169 | +14.34% | +34.15% | **−14.85pp** | 1.06 |

*alpha = strategy return − (benchmark return × average exposure)*

**Negative in both halves.** The strategy is a worse way to own market exposure
than owning market exposure. Note also that the regime filter *hurt* — screening
out signals when the index sat below its 100-day average removed the best
mean-reversion opportunities, which is exactly when they occur.

---

## 5. The binding constraint: edge versus cost

| | Value |
|---|---|
| Round-trip cost (15bps slippage + 3bps brokerage per side, 10bps STT on sell) | **0.46%** |
| Best holdout edge over 10 bars | **+0.345%** |
| Net | **−0.115%** |

Every surviving signal has a holdout edge **smaller than the cost of capturing
it**. That single line explains why a statistically real signal (t = 3.5, 4,000+
holdout observations) produces a portfolio that cannot beat buy-and-hold. The
edge exists; it is just not big enough to pay the toll.

---

## 6. What would actually have to change

Ranked by how much of the gap each closes. None of these are validated — they
are the hypotheses the measurements point at.

1. **Hold longer, trade less.** Cost is fixed per round trip; edge should grow
   with horizon. The universe base rate rises from +0.231% (10 bars) to +1.296%
   (40 bars). The open question is whether the *signal's* edge grows faster than
   the base rate at those horizons — the diagnostics' hold-period sweep says it
   did not for DAB, and it has not yet been run for PBR. **This is the first
   thing to test.**
2. **Attack the cost side directly.** 46bps is a modelling assumption, not a
   law. STT (10bps on sell) is unavoidable on delivery trades, but Indian
   discount brokers charge ~₹20 flat per order rather than 3bps, and slippage on
   the most liquid quartile of the universe is well under 15bps. Re-running with
   a realistic cost model — and a `min_avg_turnover_cr` of 25–50 instead of 5 —
   could plausibly halve it. Cheapest experiment available.
3. **Stop competing with beta.** A long-only system in a rising market must beat
   the index to justify itself. The signals here have positive *excess* return
   but negative alpha once exposure is paid for. A market-neutral construction —
   long the mean-reversion candidates, short an index future or the momentum
   names the lab found to be negative — isolates the part that actually works.
   Caveat: shorting individual Indian equities beyond intraday is impractical for
   most retail accounts (SLB constraints), so the short leg realistically has to
   be an index future.
4. **Get more data.** 2.9 years is one regime. The delivery archive goes back
   much further; `python -m nifty_swing_bot.data.backfill --start 2018-01-01`
   would roughly triple the sample and let the walk-forward run enough folds to
   mean something.
5. **Reconsider the universe.** Excluding the Nifty 100 was justified for the
   *delivery* signal, which is now dead. Mean reversion may work better in large
   caps, where spreads are tighter and circuit risk is negligible — and where
   the cost assumption in (2) is most defensible.

### What the evidence does *not* support

- Any use of delivery percentage, in any of six formulations.
- Momentum/breakout entries at swing horizons in this tier. The strongest single
  result in the whole study is that buying extension here loses money. That is in
  principle a short signal, but see the shorting caveat above.
- Adding more filters to DAB. Its rules are individually anti-predictive; no
  combination of them recovers.

---

## 8. Realistic costs and holding period (the cost study)

Reproduce with `python -m nifty_swing_bot.research.cost_study`.

### 8.1 "Realistic" costs are *higher*, not lower

The first correction went the wrong way. The original model charged brokerage as
a percentage and STT on the **sell side only**. For a delivery (CNC) trade, STT
is **0.1% on both buy and sell**. Flat Rs 20 brokerage saves ~2bps on a Rs 1 lakh
position; the missing buy-side STT costs 10bps. Net: the honest model is worse.

Round-trip cost, % of notional:

| Scenario | Rs 25k | Rs 50k | Rs 1L | Rs 2L |
|---|---|---|---|---|
| 1. Original model | 0.460 | 0.460 | 0.460 | 0.460 |
| 2. Realistic charges | 0.775 | 0.649 | **0.585** | 0.554 |
| 3. + liquid quartile (6bps slippage) | 0.595 | 0.469 | **0.405** | 0.374 |
| 4. + zero brokerage | 0.406 | 0.374 | **0.358** | 0.350 |

Two things worth noting. Flat fees make **small positions much more expensive**
— 0.775% at Rs 25k versus 0.554% at Rs 2L — so position sizing now has a direct
cost consequence. And the best realistic case (0.358%) is only 22% cheaper than
the original guess, because STT and slippage dominate and neither is negotiable.

### 8.2 The liquidity restriction backfired

Restricting to the most liquid quartile cut cost by 31% (0.585% -> 0.405%). It
cut the **edge by 68%**:

| Universe | Signals (OOS) | Edge @20d | t | Cost | Net edge |
|---|---|---|---|---|---|
| Full mid/small cap (395) | 3,614 | **+0.738%** | 4.65 | 0.585 | **+0.153** |
| Liquid quartile (99) | 1,023 | +0.234% | 0.74 | 0.405 | −0.172 |

The t-statistic collapses from 4.65 to 0.74 — in the liquid quartile the effect
is no longer statistically distinguishable from nothing.

**Edge and cost are positively correlated.** The mean-reversion premium is
compensation for providing liquidity where liquidity is scarce. Restricting to
the names that are cheap to trade removes the very inefficiency being harvested.
This is the central lesson of the exercise, and it generalises: you cannot
cheaply harvest an illiquidity premium by trading only liquid things.

One honest caveat in the other direction. The liquid quartile's *gross* excess
over the index was higher (1.257% vs 0.665% at 10 days) because that group
simply outperformed over this window. A long-only book there would have earned
more — but from factor exposure, not selection skill, and you would get the same
by buying the whole quartile. The portfolio backtests reflect that and looked
good (alpha +13 to +83pp), but they swung wildly between adjacent hold periods
(+39%, +113%, +62% at holds 10/15/20 on the same 99 names) because 20 positions
drawn from 99 stocks is far too concentrated to be stable. Those numbers are not
trustworthy; the bar-level measurement, with 3,600 out-of-sample observations,
is.

### 8.3 Holding period: the hypothesis was right

Charges are paid once per round trip, so a longer hold amortises them — provided
the edge keeps growing. It does, up to a point.

Full universe, realistic costs (0.585% round trip), out-of-sample:

| Hold (bars) | Edge % | t | Net edge % | |
|---|---|---|---|---|
| 10 | +0.379 | 3.57 | −0.206 | negative |
| 15 | +0.457 | 3.43 | −0.128 | negative |
| **20** | **+0.738** | 4.65 | **+0.153** | **positive** |
| **25** | **+0.881** | 4.94 | **+0.296** | **positive — best** |
| **30** | **+0.863** | 4.45 | **+0.278** | **positive** |
| 40 | +0.501 | 2.13 | −0.085 | negative |

Net edge crosses zero between 15 and 20 bars, peaks around 25, and decays by 40
as the signal washes out into ordinary drift. At 20 bars the in-sample and
out-of-sample edges agree closely (0.675% vs 0.738%), which is more convincing
than the 10-bar case where they diverge sharply (0.753% vs 0.379%).

**This is the first genuinely positive, out-of-sample, net-of-realistic-cost
result in the project.**

### 8.4 What it costs to believe it

- **It is no longer a swing strategy.** A 20–30 bar hold is five to six weeks.
  That is position trading, and it changes risk, capital turnover and the
  emotional profile of the system entirely.
- **The magnitude is small.** +0.30% per trade at the optimum. With ~10 round
  trips per slot per year that is a few percent of annual alpha before any
  implementation friction — real, but not life-changing, and easily erased by
  worse fills than modelled.
- **Overlapping windows inflate every t-statistic** shown here. Treat them as
  ranking devices.
- **The holdout has now been consulted repeatedly** across this document. It is
  no longer a clean holdout. The next honest step is a genuine out-of-time test
  on data that has never been touched — which means extending the backfill.

### 8.5 Revised priorities

1. **Re-run the walk-forward at a 25-bar hold on the full universe.** This is
   the configuration the measurements support, and it has never been through
   the fold-by-fold validation.
2. **Extend history to 2018** (`--start 2018-01-01`). Everything above rests on
   three years and one regime, and the holdout is now spent.
3. **Drop the liquidity restriction.** It was a reasonable hypothesis and the
   data rejected it.
4. **Size positions larger and fewer.** Flat fees punish small tickets: 0.775%
   at Rs 25k versus 0.554% at Rs 2L is 22bps of pure structure, comparable to
   the entire edge.

---

## 7. Reproducing all of this

```bash
python -m nifty_swing_bot.research.diagnostics            # edge decomposition
python -m nifty_swing_bot.research.signal_lab             # hypothesis lab
python -m nifty_swing_bot.backtest.run_backtest --strategy dab
python -m nifty_swing_bot.backtest.run_backtest --strategy pbr
python -m nifty_swing_bot.research.cost_study        # costs + hold periods
```

Caveats that apply to every number above: overlapping forward windows inflate
t-statistics; a holdout consulted repeatedly stops being a holdout; and the
exposure and stop parameters in section 4 were chosen while looking at the full
period, which is why the split-sample alpha in that same section is the honest
number and the +26.47% is not.
