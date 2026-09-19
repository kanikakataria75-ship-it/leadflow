# AES — standing state

Current as of Phase 14. This is the recoverable summary: what is proven, what
is not, what has been tested and rejected, and the exact configuration that
walk-forward was run on. `RESEARCH_AES.md` holds the full narrative and the
working; this file is the index to it.

**Reading rule:** nothing in the REJECTED list should be re-tested without a
new reason that is stated first. Each entry records the number that killed
it, so the cost of re-running it is known in advance.

---

## 0. FROZEN FOR LIVE — 2026-09-17

`aes_scanner/live_config.py` is the single source of truth for what runs. It
differs from the §1 locked configuration in two places, both deliberate:

- **sizing** 6 slots / 3% risk (§18's only cell above 10% in both windows),
  not 3 slots / 1%.
- **exits** variant (c) — 50% @ +12%, 20% @ +20%, 30% trails on ATR 2.5x with
  the 25-bar cap — not the spec's +5%/+8%/+20% ladder (§20).

Conviction multipliers stay at 0.7/1.2: Phase 20 found score-to-edge is **not
monotone** (2 of 11 folds), and the configured values already beat both neutral
and steeper in both windows, so no scaling was adopted and none was changed.

**No further in-sample testing.** 2015-2021 was discovery from the start;
2022-2026 was spent as discovery by Phase 16. Everything from here comes from
the forward record the scanner accumulates from `FORWARD_CHECK_FROM`
(2026-09-17). Changing any frozen value restarts that record.

**The number most likely to be misremembered: position R is ~0.17, not 0.65.**
See §3d and `aes_scanner/README.md`.

## 1. The locked configuration

This is what Phase 9 walk-forwarded. Every value is a default in the named
dataclass unless stated.

| layer | setting |
|---|---|
| universe | 402 NSE mid/smallcap names (`niftymidcap150` + `niftysmallcap250`), Yahoo OHLCV, `auto_adjust=True` |
| screener (§1.1) | +20% in 10 sessions, within 5% of the 52-week high, close > ₹50, 20-day turnover ≥ ₹1 crore |
| box detector | `BoxParams()` — big box 10–45 bars at 15–25% range; small box 5–22 bars at 5–10%; `edge_basis="wick"` |
| watchlist | `WatchlistParams()` — `max_watch_without_box=90`, `box_stale_bars=22`, **`box_carry_bars=0`** |
| entry | §5 modes 1/2/3 unchanged (breakout, breakout+retest, box-bottom) |
| scoring | **equal weight, 0.25 × 4** (fast resolution, resistance absorption, rs_capture, prior cycles); buckets reject < 0.33, high conviction ≥ 0.66; trades `wait_and_watch` and above |
| sizing (§7) | risk-based, 1% of equity per trade, max 3 concurrent, target 2.5 slots, min risk/share 2% |
| **exit** | **ATR(14) × 2.5 chandelier trailing stop, closing basis**, plus big-box-bottom stop, plus spec ladder (50% @ +5%, 20% @ +8%, 20% @ +20%), plus §6.3 time stop (15 bars below +5%), plus **hard 25-bar cap** |
| costs | 0.585% round trip |
| regime (§8) | unchanged |

Single-pass result on 2015-2021 against the 2014-start cache: +38.17% total,
alpha +2.42%/yr, beta 0.047, max drawdown 5.81%, 6.09% mean capital
deployment, ~120 signals/yr, ~19 positions/yr, median hold 17 bars. **Read
the fold table below instead** — that single-pass figure carries the warm-up
artefact described there and is not the number to quote.

### Walk-forward, annual folds, nothing refitted

Eleven folds on the extended 2010-start data (`cache/ohlcv_2010`). The
2015-2021 numbers first reported in Phase 9 were computed against the
2014-start cache, where the 252-bar 52-week-high lookback was still filling
through early 2015 and suppressed admissions (2015: 91 signals then, 138
now). **Phase 9's headline was inflated by that warm-up artefact** and the
corrected figures are below.

```
  fold  sigs  traded  pos    ret%  alpha%   beta  maxDD%   win%  depl%   edge%      t
  2011    19      15    5   -2.19   -2.90  0.023    3.35   20.0   2.17   -3.86  -1.28   <- unusable
  2012    46      39   19   -0.30   -0.23  0.053    4.78   73.7   6.66   +1.87   1.83
  2013    32      26   18    7.34    6.64  0.023    1.92   77.8  11.85   +4.62   2.12
  2014   158     101   28   -1.85   -5.63  0.108    5.08   53.6  10.52   +3.59   2.69
  2015   138      94   24   -5.17   -3.01  0.134    7.16   37.5  10.35   -0.30   0.26
  2016    56      43   18    2.27    1.22  0.078    2.85   66.7   6.49   +1.53   1.17
  2017   132      73   28   10.33    5.15  0.147    5.82   60.7  12.61   +5.11   4.05
  2018    76      47   14    1.05    1.41  0.044    3.17   64.3   4.62   -6.03  -3.45   <- thin
  2019    23      19   12   -1.30   -1.58  0.048    2.14   66.7   5.60   -2.46  -0.79   <- thin
  2020    96      87   26    7.73    6.50  0.013    2.23   73.1   9.03   +2.29   1.68
  2021   291     189   32   15.39   11.63  0.086    3.48   81.2  14.51   +5.88   6.48
```

**No defensible fold set is significant:**

```
  fold set                          n   mean    t      p     pos    best3   ex-best3
  2011-2021 (all)                  11   3.03  1.59  0.143   6/11    100%     -0.15%
  2012-2021 (drop unusable 2011)   10   3.55  1.75  0.114   6/10     94%     +2.04%
  2014-2021 (drop thin 2011-13)     8   3.56  1.44  0.192    5/8    118%     -5.00%
  2015-2021 (the Phase 9 window)    7   4.33  1.60  0.160    5/7    110%     -3.15%
```

Per-trade edge across eleven folds: positive in 7/11, mean +1.11%,
**t=0.95, p=0.363**.

### Data

Two caches, deliberately not merged. `cache/ohlcv` (2014-start) produced
every result through Phase 11; `cache/ohlcv_2010` produced Phase 12. Over
the 2015-2021 overlap they agree to a **median relative close difference of
0.0000%** across 296 symbols, so the extension is an extension, not a
re-basing.

**Survivorship is the binding data limitation.** These are *today's* index
constituents, so a name present early is a survivor by construction:
53.3% of the universe existed in 2010, 59.1% in 2015, 80.1% in 2021,
100% in 2025. Pre-2015 folds see about half the universe.

**Corporate-action adjustment pre-2015 is sound** — better than later, in
fact (0.15 suspicious days per 10k bars in 2010-2014 vs 0.42 in 2015-2021),
and 11 sampled names with known pre-2015 splits showed a maximum move of
13.8% within ±4 days of any split date, zero suspect.

---

## 2. PROVEN

Findings that hold fold-by-fold, or are mechanical rather than fitted.

1. **The ATR trailing stop + hard cap is a genuine risk improvement.**
   Against the spec SMA(10) stop with the same ladder and cap: lower max
   drawdown in **7 of 7 folds** (3.10% vs 5.67%, paired t=−3.56, p=0.0119)
   and higher win rate in **7 of 7 folds** (68.8% vs 46.3%, paired t=7.17,
   p=0.0004). **Adopted as a risk component**, independent of entry logic.
   A trailing volatility stop resolving losers faster than a 10-day average
   is a mechanical property and behaves like one.

2. **Beta neutrality — at 1% risk only.** 0.016–0.147 across every fold and
   every configuration tested **at the locked 1% risk budget**. Phase 18
   narrowed this: at 3% risk per trade beta runs **0.26–0.40** in both windows.
   Scaling risk buys index exposure, so the beta-neutrality claim does not
   survive the sizing lever that PROVEN 7 recommends. Original text follows,
   and holds within its narrowed scope. Whatever this system earns, it is not index
   exposure. This distinguishes it from the prior DAB/MR study, whose
   apparent outperformance was beta from sitting 85% invested.

3. **The edge is horizon-dependent with a floor around 15 bars.**
   Like-for-like (vs universe base rate, net of cost): h=5 −0.00%,
   h=10 +0.70%, h=15 **+1.24%**, h=20 +1.21%. There is no edge at short
   holds, so faster turnover is not available as a remedy.

4. **Capital deployment, not holding period, is the binding constraint.**
   Mean exposure is **6.2%**, not the 28–45% that time-in-market suggests.
   At 6.2% and a 17-bar hold the measured edge implies ~1.1%/yr. Reaching
   10%/yr needs ~48% deployment — an eight-fold increase.

5. **Raising signal count does not raise return.** Three independent
   attempts, all lowering return and raising drawdown (see REJECTED 6–8).
   Consistent enough to treat as a property of this system: the marginal
   signal is worse than the average one, because score-ranking was already
   taking the good ones.

6. **The "+3.15% / 20-bar" headline was an absolute return, not an edge.**
   Decomposed: +3.73% gross − 0.58% cost = +3.15%, of which +1.26% is index
   drift and +0.68% is the universe base rate. Edge over a random entry in
   the same universe is **+1.21%, t=2.76**. A beta-0.03 book was never able
   to earn the other 1.94pp.

7. **Sizing scales cleanly; slots and signals do not.** With the signal set
   held byte-identical and slots at 3, raising risk-per-trade 1%→3% moves
   deployment 9.2%→24.5%, return 3.55%→8.41% and alpha 2.21%→4.88%, at mean
   drawdown 3.86%→11.50% (worst fold 7.16%→17.85%). Return and drawdown scale
   close to linearly — no hidden convexity either way — and concentration
   does **not** deteriorate (median top-1 share 26.1%→29.9%). This is the only
   lever tested that does what it says. It multiplies an unproven edge.

8. **The 20/5 admission gate is not arbitrary.** +1.83% net edge (t=9.32,
   n=4,660) in that cell against +0.57% at 15-20%/≤5% and +0.16% at
   ≥20%/5-10%. Relaxing either axis costs 69-91% of the edge.

9. **The box's binding gate is `big_min_bars`, not any range or duration
   ceiling.** Diagnosing all 39 real trades at their own entry bar: **27 fail
   on `big_min_bars`** with candidate box lengths of **1-9 bars**, and the
   range band is never evaluated for them. Only 4 fail on range and 2 on
   history length. The mechanism is mechanical rather than fitted: the box top
   is the running maximum over the anchor lookbacks, and at entry these names
   are *at* their running 90-bar maximum, so the anchor lands on roughly today
   and the box has no length to measure. This is the §7.2 "group 1 alignment"
   problem, and it is the only gate whose relaxation raises coverage without
   raising signal count (PROVEN 11).

10. **The 15-25% range band is wrong on the ceiling and right on the floor.**
   Decomposed: raising the ceiling 25%→35% moves box-at-entry 3→5 for +12
   signals/yr. Lowering the floor 15%→5% moves it **3→3** for +59 signals/yr,
   and costs fold return 4.78%→2.79% and positive folds 5/7→3/7. The floor's
   named counter-example does not survive contact: Colgate Palmolive at 8.8%
   is rejected on **duration** (a 3-bar candidate), not on the floor, so no
   floor setting recovers it.

11. **Coverage and signal count are separable, once.** Every lever tested
   before Phase 15 moved the two together. `big_min_bars` 10→6 moves
   box-at-entry 3→7 at **120.0→120.7 signals/yr** — flat — and improves fold
   consistency (p 0.063→0.039, positive folds 5/7→6/7). It still *lowers*
   mean fold return (4.78→3.84) and per-trade edge (2.39→2.19), so it is not
   adopted; but it is the one place where the usual coupling breaks, and the
   only geometry change that raises coverage without flooding the pipeline.

12. **The AES admission gate is a strong runner filter — measured, Phase 16.**
   Bars passing "+20% in 10 sessions and within 5% of the 52-week high" gain
   +30% within 25 sessions at **11.13%**, against a **4.38%** universe base
   rate — a **2.54x lift**, n=3,523 admissions over 2022-2026. This is the
   largest unconditional effect measured anywhere in this project, and it was
   never measured before because every phase tested the gate by what happened
   *after entry*, not by what the gate selects.

13. **Runs are preceded by pullbacks, not by breakouts** (Phase 16, and the
   only finding here that has ever held out of sample). Vol- and
   turnover-matched, +30%/25-session runners sit **below** their 20-EMA, low
   in their 40-bar range, with **weak** 20-day relative strength, **20.5%
   below** the 52-week high, on contracting volatility, ~75 bars after a prior
   20% move. Equal-weight composite AUC **0.599 discovery / 0.612 validation**
   on a symbol holdout fixed in advance, **21 of 21 features keeping their
   sign**. Directional, not a volatility proxy: the same composite scores
   -30% crashers at 0.444. See §16.

14. **Transfer to the traded population is not broken by liquidity.**
   Discovery edge by turnover quartile, each measured against *its own*
   universe base rate: Q1 +1.24%, Q2 −0.18%, Q3 +0.11%, **Q4 +1.02%** — flat
   to U-shaped, not concentrated in illiquid names. Q4 is where 24 of the
   user's 39 real trades sit. The naive version of this test (raw excess:
   Q1 3.00% vs Q4 1.64%) would have shown a false positive, because
   low-turnover names carry a much higher base rate (+1.17% vs +0.03%).
   `RESEARCH.md` §8.2's liquidity finding does not appear to transfer to
   this strategy. No quartile is individually significant (max t=1.65).

---

## 3. UNPROVEN

Pooled-significant but not stable fold-to-fold. Do not treat as established.

1. **The return edge.** On eleven folds: mean 3.03%, **t=1.59, p=0.143**,
   positive in 6/11, and the best three folds are **100%** of summed return
   (excluding them: −0.15%). Per-trade edge positive in 7/11, t=0.95,
   p=0.363. Extending the data from seven folds to eleven made every
   statistic weaker — the direction a partly-noise result moves when the
   sample grows.

2. **The ATR stop's *return* advantage** (as opposed to its risk advantage,
   which is proven). 4/7 folds, paired mean +1.68pp, **t=0.63, p=0.550**;
   the spec stop beats it by 9.3pp in 2020. The return ranking that made it
   best-of-nine was selection.

3. **The scoring system.** Treated as **overfit**. Three of four weighted
   factors were selected on the same sample they score; none holds its
   pooled sign in a majority of folds; the strongest factor (fast
   resolution, 0.40) is the least stable (3/7 folds). Equal weighting is
   used instead — not because it is better, but because it is not fitted.

4. **Everything traces to a few years.** Three independent analyses — the
   scoring-factor walk-forward (Phase 6), the portfolio walk-forward
   (Phase 9/12) and the breadth test (Phase 11/12) — land on 2017 and 2021
   as the carriers of pooled significance. Twelve years and 1,129 signals
   did not change that.

---

## 4. REJECTED — do not re-test without a new reason

| # | Tested | Result that killed it | Where |
|---|---|---|---|
| 1 | **Delivery %** — 6 formulations, plus as a filter on the best non-delivery signal | No edge in any formulation. Added to RSI(2)-dip: cut sample 83% and turned +0.715% into −0.380% | `RESEARCH.md` §1.3 |
| 2 | **Liquid-quartile restriction** | Cost −31%, edge −68%; t collapsed 4.65 → 0.74. Edge and cost are positively correlated — the premium *is* illiquidity | `RESEARCH.md` §8.2 |
| 3 | **Momentum / DAB entries, and SRDR** | Zero of 108 configurations closed the cost gap; shortfall widens monotonically with every cost assumption | `RESEARCH_SRDR.md` §10 |
| 4 | **Calibrated scoring weights** (0.40/0.25/0.20/0.15) | Separates out of sample in 1 fold of 4; three of four factors in-sample-selected; `prior_cycles` has a *negative* pooled r | `RESEARCH_AES.md` §6.2 |
| 5 | **Profit-ladder variants** (no ladder / +12% / +15%) | Survival curve unchanged — median hold 6 bars with the ladder *deleted*; 181/182 positions exit on the stop. Apparent gains were 2020-21 only; top-3 removal turned the best variant negative | §5.2 |
| ~~5~~ | **SUPERSEDED for the ATR stop — see §20.** REJECTED #5 ran under SMA(10), where 272 of 273 positions died at the stop with a 6-bar median hold, so the ladder could not bind. Under ATR 2.5x the median hold is 17 bars and the ladder is the dominant exit. Re-tested: it **truncates the return distribution** — zero winners above +30% across 398 positions in both windows, against 7 and 15 without it; largest of the 41 real trades +18.8% laddered vs +52.7% unladdered. REJECTED #5 stands for the SMA stop only | §20 |
| 6 | **Widening the stop by lengthening the SMA** (15/20/25) | Not a curve — −1.12 / −8.34 / +8.89, no trend. Distinct from adding a *buffer* to a fast average, which does work | §5.1, §5.3 |
| 7 | **Box carry-forward** (`box_carry_bars` 10–45) | Hit rate on 41 real trades 0 → 1 of 39. Return 38.17% → 31.57%. Implemented and tested; **left at 0** | §8.1, §8.4 |
| 8 | **Watchlist memory** (`max_watch_without_box` 90 → 750) | 4.5× signals, deployment 6.09% → 15.29% — the only thing that ever moved deployment — but return 38.17% → 22.50%, alpha 2.42 → 0.76, drawdown 5.81% → 14.60% | §8.4 |
| 9 | **More portfolio slots** (3 → 5 → 8) | Deployment 6.1% → 8.2% → 10.7%, return 37.8% → 18.6% → 21.3% | §5.4 |
| 10 | **Breadth / participation gate** (causal: trailing-63-session distinct names ≥ expanding median) | Gate retains 92% of signals; low-breadth folds +1.89% → +1.53%, edge −2.03% → −2.97%. Re-tested on twelve years: Q4 still **84% one year** (161/192 from 2021); excluding 2021 r=+0.016 p=0.694; **within-year r=+0.010, p=0.773** (n=770). The one independent high-breadth year added by the extension, **2014**, contradicts it — breadth 61, +1.94% mean signal excess, fold return **−1.85%**, alpha −5.63% | §11.2–11.4, §12.3 |
| 11 | **Admission as a gradient** (+20%/10d and within-5% softened to 15/5, 15/10) | The 20/5 cell is the best in the grid: +1.83% net edge (t=9.32) vs +0.57% at 15-20%/<=5% and +0.16% at >=20%/5-10%. Both conditions load-bearing. End to end: sig/yr 120→206, edge 2.39→1.79, fold return 4.78→1.19, p 0.063→0.424 | §14.1-14.2 |
| 12 | **Weak re-admission** (fresh 52-week high, no new 20% move) | sig/yr 120→259 (largest increase of anything tested), edge 2.39→1.53, fold return 4.78→2.40, drawdown 3.10→4.60 | §14.2 |
| 13 | **Notional position cap as a deployment lever** (15%→60% of equity) | Deployment moves 8.5%→9.4%. The cap never binds; the 1% risk budget exhausts first | §13 |
| 14 | **Geometry loosening** (`big_max_bars`, sloped channels, 15% range floor, post-parabolic shelves) | **Run in Phase 15** after the deferral premise was falsified on the axis that matters (box-at-entry is 3/39 under all five admission rules). Result: same pattern as 6–9. Every arm that raises box-at-entry materially also multiplies signals and cuts fold return; all five together reach 23/39 coverage and a **negative** fold return (−1.50%) at 3× the drawdown | §8.5, §15 |
| 15 | **`big_max_bars` 45→120** | Exactly zero effect on coverage at every value. All 27 duration rejections among the 39 real trades are at the **low** end (1–9 bars), not the high end. The two names it was built for cannot be recovered by it: Balu Forge and Orient Technologies have 88 and 44 bars of history at entry and are refused by `min_history_bars=120` | §15.2 |
| 16 | **`edge_basis="close"`** | Worse on every axis: box-at-entry 3→**2**, signals/yr 120→64, fold return 4.78→1.03. Close-only edges make boxes *narrower*, so more candidates fall below the range floor — the opposite of the "one spike breaks containment" premise | §15.3 |
| 17 | **Excluding the prior leg from range measurement** | Coverage unchanged at 3/39, edge/base 0.48→0.17. The anchored construction already excludes the leg: the box starts *at* the running-maximum bar, so the run-up that built the top is outside the window by design | §15.3 |
| 21 | **Early Sector Rotation blueprint** (§5.1 RS + §5.2 breadth + §3/§9 lifecycle states, tested as a feature on the AES signal population) | Does not separate, in either window. The §28 "early" premise beats a matched placebo by **3pp** on hit rate and fires with a *shorter* lead than a random not-yet-obvious day (16 vs 21 sessions), so it is not early. **Emerging is below the population average in both windows** (-0.19 vs +0.46; +0.90 vs +1.20); the best state is **Leader**, the one the blueprint calls too late. **Silent Rotation (§7, the central hypothesis) loses to plain sector momentum in both windows.** Signal freshness points opposite ways in the two windows on n=5-6. The only repeatable effect is the inverse — Fading is materially negative (-2.80, -3.72) — and excluding it is worth +0.29pp (t=0.12, 3/7 folds) and **-0.93pp** (t=-1.39, 1/5 folds) at portfolio level | §19 |
| 19 | **The Phase 16 runner composite as a standalone entry** | Top-decile, next-open, excess over base rate net of cost: **-0.35% at 10 bars, -0.22% at 15, -0.12% at 20, -0.02% at 25**, and only +0.43% at 40. At the one horizon that clears cost the median position is 10% underwater at its worst, half go below -10%, one in six below -20%. The excursion is larger than the edge | §17.1-17.2 |
| 22 | **Conviction-scaled sizing beyond the configured 0.7/1.2** | Score-to-edge is **not monotone**: monotone in 1/6 and 1/5 folds (2 of 11). Q4 beats Q1 in 8/10 folds (mean +2.71 / +1.91pp, rank corr +0.80 in both windows) so there is a direction, but 2025 reverses it by -14.06pp. At portfolio level the configured 0.7/1.2 beats neutral 1.0/1.0 (14.64 vs 14.55; **19.82 vs 11.67**) and steeper 0.5/1.5 (13.41; 15.64) in both windows, so it is kept unchanged rather than scaled | §21 |
| 20 | **Replacing the breakout trigger with the pullback profile** (same admission gate) | The breakout entry wins at every short horizon on identical admissions and near-identical signal counts (166 vs 168/yr): net **+0.17% vs -0.21%** at 10 bars, +0.37% vs +0.29% at 15, +0.60% vs +0.38% at 20. At a 10-bar cap the pullback arm returns **-39% against +272%**. It wins only at 25-40 bars | §17.3 |
| 18 | **Lowering the range floor** (15%→5%) | +59 signals/yr for **zero** coverage gain, fold return 4.78→2.79, positive folds 5/7→3/7. The floor is load-bearing; the *ceiling* is not (see PROVEN 10) | §15.3 |
| 25 | **Re-entry restricted to high-conviction positions stopped within 3 bars** (same mechanics; 24 cells) | **Vacuous — the population is empty under the live stop.** All 24 cells produce **zero** re-entries in both windows, so every cell is bit-identical to baseline. Cause is structural, not a gate bug: the ATR(14) × 2.5 trail fires only when price falls 2.5 ATR below the *running peak since entry*, and in the first three bars the peak is still near the entry, so a stop that early needs an immediate ~2.5 ATR collapse. Across **all** stop exits, `bars_held <= 3` occurs 4 times (2015-2021) and 3 times (2022-2026), and **none of those are high-conviction**; median stop-exit hold is 14 and 13 bars. Gate verified by relaxation: hc-only → 6 re-entries, any-bucket ≤8 bars → 3, hc AND ≤8 bars → 2, hc AND ≤3 bars → **0**. The fast-knockout population the rule was designed for is an artefact of the **SMA(10)** stop, which sits at price and can fire on bar 0 (7 of the user's 23 SMA10 stop-outs were ≤2 bars). Under the live trail it does not exist. Nothing to adopt and nothing to reject — the question is not answerable on this stop | §23b |
| 24 | **Re-entry after a stop-out** (N ∈ 5/10/15/20 bars × volume 1.5×/2×/none × cap 1/2, 24 cells). After an `atr_trail_stop` or `big_box_bottom_stop`, re-enter if price closes back above the level that fired, on volume. Everything else frozen | **Nominal pass, mechanism falsified — NOT adopted.** 1 distinct config (N5, 1.5× vol; cap 1 and 2 are identical because no name ever re-enters twice at N=5) clears the pre-registered bar in both windows: position edge 1.84→2.09 and 2.21→2.46, ex-best-fold 7.40→**8.56** and 8.32→**10.58**, mean drawdown *better* in both. **But attribution kills it.** The re-entry positions' own net P&L is **5.3%** and **3.7%** of the total equity gain; the other ~95% is slot reshuffling — 24 baseline positions vanish and 33 new ones appear as a re-entry occupies a slot and changes what else gets taken. Re-entry win rate is *below* baseline (45.5% / 36.4% vs 59.4% / 56.3%) and mean re-entry return in 2022-2026 is **+0.10%**, i.e. ~zero, while CAGR moves 19.49→21.98. A score-jitter null (20 seeds, ordering perturbation only) puts the 2022-2026 gain at just z≈1.1 above a null mean that is itself **+150k of the +221k** — two thirds of the "improvement" is reproducible by arbitrary perturbation. With 24 cells tested, ~1.2 false passes are expected at 5%. The mechanism under test (catching the second leg) is **not** what produced the number. Needs a null that adds N random extra positions before any re-entry claim is credible | §23 |
| 23 | **GAMMA — alternative daily structures at the box stage** (HH/HL at 2/3/4 swings; price above a rising EMA20; shallow-pullback continuation; and all three combined). Admission and everything downstream held frozen; the structure arms an episode **only** where the box detector found nothing | Bar was pre-registered before the run: raise signals/yr AND hold `edge_vs_base` AND hold ex-best-fold, in **both** windows. **All six arms fail it, all on the same term.** Every arm raises signals/yr (120→139–283 and 166→202–435) and every arm **loses ex-best-fold in 2022-2026**: 8.32 → −19.51 / −5.50 / +4.97 / −5.46 / −3.79 / −13.65. Positive folds 5/5 → 1–4/5. Strategy CAGR 19.49 → −7.55…18.42. Fifth attempt to raise signal count, fifth to fail the same way as REJECTED 8/12/18. **Two findings worth keeping:** (i) `rising_ma` is the only arm that *raises* per-trade edge vs base in both windows (−0.080→**+0.176**, t 1.10→2.26; 0.788→**+0.968**, t 2.89→4.80) while still destroying ex-best-fold — per-trade edge without portfolio survival; (ii) on the independent check the arms recover what the detector misses at scale — box 6/39, combined **27/39** of the real trades — so the structures *are* what is being traded, and it does not outperform. That is consistent with §16.7 and is evidence about the entries themselves, not only about GAMMA | §22 |

---

---

## 3a. THE STANDING CONTRADICTION (Phase 16)

The detector's gate is good and its discrimination is absent, and both are
measured:

- AES admissions run at 11.13% vs a 4.38% base rate (**2.54x**, PROVEN 12).
- Within a turnover- and volatility-matched comparison, **all** of the residual
  power separating runners from non-runners (AUC 0.598 of 0.599) sits in
  **twelve features the detector does not compute**, and six things it does
  compute point the wrong way: it requires proximity to the 52-week high
  (runners: 20.5% below), a breakout to the top of the range (runners: 0.42 of
  range), and a tight 15-25% box (runners: 29.6% wide).
- The 41 real trades sit on the **detector's** side of all six, not the
  runners' side — near the high, above the 20-EMA, high in range, strong RS,
  5 bars after a 20% move rather than 75.

The trader and the detector agree with each other and disagree with the runner
data on the same axes. That is the most economical explanation for why five
phases of tuning the detector toward the trader never moved the outcome.

**This is not yet a reason to change anything.** The runner profile predicts
the direction of a 30% expansion; it has not been shown to produce a positive
net edge after cost, and the 41 real trades were profitable (+1.95% to +4.85%
mean excess) doing the opposite, against a different and shorter objective.

## 3d. THE LADDER IS THE TAIL CONSTRAINT (Phase 20)

Re-tested under the ATR stop after REJECTED #5's premise was shown not to
transfer. Three things, the first of which corrects an earlier number here.

1. **The "average R multiple of 0.65-0.72" was a measurement artefact.** It is
   a mean over trade *rows*, and a ladder splits one position into 2-4 rows of
   which every ladder row is profitable by construction. Position-level R under
   the same ladder is **0.184 / 0.164**; without the ladder, **0.212 / 0.215**.
   Any R figure in this project taken from a `trades_frame` row mean is
   inflated the same way.
2. **The setups run; the ladder caps them.** Zero winners above +30% across 398
   positions under the current ladder, in both windows. Without it: 7 and 15.
   The 41 real trades agree independently — largest +18.8% laddered against
   +52.7% unladdered, total 57.3% against 92.4%.
3. **Deleting it is not the answer.** Full removal is +13.70pp in 2022-2026
   (t=1.98) and **-0.13pp** in 2015-2021 while taking mean drawdown 8.79% ->
   14.63%. The windows disagree, as in §18.

**Variant (c) — first tranche at +12%/50%, +20%/20% retained — is the only one
better in both windows** (+4.04pp and +4.44pp; 4/7 and 4/5 folds; 5/5 positive
folds in 2022-2026; best ex-best-fold in both). Not adopted — best paired
t is 1.98 — but it is the strongest cross-window candidate this project has
produced and is what `FORWARD_CHECK_FROM` should be spent on.

What the ladder buys, stated as the trade it is: ~9 points of win rate
(65% -> 56%) and ~2pp of median return, paid for with the entire right tail and
~20% of true R.

## 3c. THE CEILING, AS A NUMBER (Phase 18)

Slots x signal-count x risk, run through the real backtester with risk sizing
and the ATR stop. Control cell reproduces V0 exactly. Full grid in
`results/slot_sweep_*.json`, narrative in §18.

```
                            2015-2021          2022-2026 [DISCOVERY]
  best cell              P2 6 slots 3%        P1 6 slots 3%
  annual return               12.66%               15.38%
  mean DD / worst DD       10.79 / 15.40         9.02 / 13.48
  deployment                   31.8%                29.9%
  beta                         0.263                0.326
  positive folds                4/7                  4/5
  mean ex-best fold           +4.98%               +5.86%
  best fold's share             66%                  70%
```

**12-15%/yr is reachable in-sample, at ~9-11% mean drawdown and 13-15% worst.**
It is carried by one year in both windows: removing the best fold leaves ~5-6%,
removing two turns the older window negative.

**The cross-window-robust cell is P1 (baseline geometry) at 6 slots / 3%
risk** — the only cell above 10% in both windows (10.60% / 15.38%, mean DD
8.79% / 9.02%). Against the locked config: ~2-3x return, ~3x drawdown, ~4x
beta, fold consistency no better (3/7 and 4/5 vs 5/7).

**Plan on ~6%/yr, not 12-15%.** The ex-best-fold figure is the expectation; the
headline arrives only when a 2017/2021/2023-type year does.

Structural results: six slots is the peak in both windows; 3% risk beats 1% at
every slot count; deployment saturates at 30-38% and never reaches PROVEN 4's
~48%; fifteen slots *reduces* deployment because base notional is 1/S; and the
290-432 signal/yr pool is negative in five of eight recent-window cells — a
sixth confirmation of PROVEN 5.

## 3b. WHAT PHASE 17 SETTLED

The Phase 16 profile is real and does not pay.

- **Standalone it fails at every swing horizon** (REJECTED 19). AUC 0.61 buys
  1.01% gross over 40 bars against a 0.585% cost and a 10% median adverse
  excursion.
- **As a replacement entry trigger it is worse than what exists** (REJECTED 20),
  on the same admissions and the same signal count, at 10/15/20 bars.
- **§3a's implication was tested and did not hold.** The 2.54x gate lift plus
  six backwards criteria suggested the screener was right and the entry was
  mistimed. Run directly, the entry is not mistimed for the horizon being
  traded: the breakout trigger beats the pullback trigger at short holds and
  collapses it at a 10-bar cap. Phase 16 describes a **longer holding period**,
  not a better entry.
- **Incidental and worth keeping:** the existing entry is net-positive at every
  horizon on 2022-2026 (+0.17% to +1.04%, t=2.2-2.4), against +0.48% (t=2.01)
  on 2015-2021. It does not look worse in the recent window than the old one.

Reachability, stated rather than optimised away: **10-15 bar swing — only the
existing breakout entry. 25-40 bar position — the pullback entry, mildly.**

## 4a. REGIME — a real pattern that is not tradeable, and now contradicted

Phase 10 found fold performance tracking market breadth (r=+0.89 across seven
annual points) and index trend (r=+0.88), with no relationship to volatility
or drawdown. It was the only hypothesis in this project motivated by a
mechanism rather than a search.

Phases 11 and 12 tested it and it does not survive:

- Those correlations were computed on **annual means**. At signal level the
  relationship is r=+0.074; **within** year — the only variation a same-day
  gate could act on — it is **r=+0.010, p=0.773** (n=770).
- The top breadth quartile is **84% a single year** (161 of 192 from 2021),
  even after extending the data by four years specifically to add
  independent high-breadth observations.
- **2014, the one genuinely independent high-breadth year the extension
  added, points the wrong way**: breadth 61 (second-highest of eleven),
  +1.94% mean signal excess, fold return −1.85%, alpha −5.63%.

The mechanism may well be true — an accumulation-to-expansion system needs
expansions to exist. It is not *identifiable* on twelve years of Indian
mid/smallcap data in which the high-breadth regime occurred essentially once.
Treat it as a description of why the folds differ, not as a rule.

---

## 4b. THE BOTTLENECK, located

Five separate attempts have now moved the constraint around without moving
the number that matters. Coverage of the 41 real trades, by admission rule:

```
  rule                    on watchlist at entry    BOX AT ENTRY
  baseline 20/5                   25/39                3/39
  move relaxed 15/5               31/39                3/39
  widened 15/10                   31/39                3/39
  readmit on 52w high             33/39                3/39
  widened + readmit               33/39                3/39
```

Admission relaxation buys real coverage — 25→33 of 39, median gap 8→1
session — and the **box detector throws all of it away**. It recognises the
structure in 3 of 39 real entries regardless of what reaches it.

Not admission. Not scoring. Not the exit. Not sizing. **The box geometry.**

### Phase 15 answer: geometry is the bottleneck, and opening it does not pay

The five geometry arms were run against the same seven folds, with admission
held at V0 20/5 and only `BoxParams` changing. The harness reproduces the V0
row of `admission_variants.json` exactly, fold by fold.

```
  arm                          box@entry   sig/yr   ret_abs   edge/base   fold ret      p   meanDD
  V0 baseline                     3/39        120     +2.39      +0.48       +4.78  0.063     3.10
  (a) ceiling 25->35%             5/39        132     +2.24      +0.47       +3.87  0.152     3.52
  (a) floor 15->5%                3/39        179     +2.41      +0.45       +2.79  0.334     4.65
  (b) big_max_bars 120            3/39        111     +2.39      +0.41       +2.86  0.177     3.48
  (c) sloped channel 2.5          12/39       290     +2.13      +0.50       +2.08  0.231     5.77
  (d) edge_basis close            2/39         64     +1.88      +0.20       +1.03  0.648     3.24
  (e) exclude prior leg           3/39        132     +2.04      +0.17       +3.27  0.087     4.17
  (f) big_min_bars 6              7/39        121     +2.19      +0.31       +3.84  0.039     3.40
  ALL FIVE AS BRIEFED            23/39        319     +1.27      -0.37       -1.50  0.457     9.54
```

**The bottleneck was correctly located and it is not worth opening.** Coverage
is reachable — all five arms together take box-at-entry from 3 to 23 of 39,
against a ceiling of 25 set by what admission lets through. It costs the sign
of the fold return and triples the drawdown.

One result does not fit the earlier pattern and is recorded rather than
rounded off: the **sloped-channel arm holds its per-trade edge over the
universe base rate** (+0.50% at t=3.17, n=1,532, against +0.48% at t=2.01,
n=551) while quadrupling coverage. It still loses on the portfolio curve
(fold return 4.78→2.08, mean drawdown 3.10→5.77), which is what PROVEN 5
predicts: at three slots, 2.4x the signals means trading a different and
worse-ranked subset, not the same trades more often.

**A measurement error found while reproducing the baseline.** The field named
`edge` in `admission_variants.json` is `mean(fwd_gross_20) * 100 - cost` --
the **absolute** 20-bar net return, not an edge over anything. That is the
exact quantity PROVEN 6 warns about. `results/geometry_variants.json` reports
it as `ret_abs` and carries `edge_vs_base` alongside. The two can move in
opposite directions: arm (c) has `ret_abs` falling 2.39→2.13 while
`edge_vs_base` rises 0.48→0.50. **The four admission variants have not been
re-measured under the corrected definition**, so "the previous four all
lowered edge" is currently established only for the absolute-return version.

---

## 5. Protocol status

> **AES_HOLDOUT IS SPENT.** As of Phase 16 (2026-09-16) the 2022-2026 window
> has been read, exploratorily and in full, as *discovery* data for the runner
> study. This was authorised explicitly on the reasoning that 2015-2021 is
> exhausted and the market actually being traded is this one. **Nothing may
> cite AES_HOLDOUT as an untouched out-of-sample check again.** Every result
> in sections 1-4 that rests on "the holdout is unspent" now rests on nothing.
>
> What replaces it, defined in `aes/protocol.py` **before** any Phase 16 result
> was computed:
>
> - **`validation_symbols()`** — 25% of the universe (102 of 401 names), held
>   out by salted hash so the split cannot be nudged after the fact. Phase 16
>   discovery read only the other 75%. Spent once, on 2026-09-16.
> - **`FORWARD_CHECK_FROM = 2026-09-17`** — the only genuinely untouched
>   check left. Bars after this date do not exist yet and cannot be peeked at.
>   Anything Phase 16 produces that is worth trading should be paper-run from
>   here, not backtested into a window that has now been looked at.
>
> **The honest limitation of a symbol holdout:** the held-out names share the
> calendar with the discovery names, so market-wide regime is common to both.
> It controls for "did I fit these stocks"; it cannot control for "did I fit
> this market". 2022-2026 contains one major smallcap correction and one
> recovery, and a mean-reversion result from such a window is exactly what
> §4a warns about. Treat Phase 16 as regime-contingent until forward bars say
> otherwise.

- **AES_HOLDOUT was never consulted before Phase 16.** Boundary moved to **2022-07-05**
  (from 2022-01-01) after Phase 6 found 60 of 787 discovery signals placed
  entries up to 2022-07-04.
- Phase 9's rule — consult only if walk-forward holds fold-by-fold — was not
  met, and Phase 12 made the case weaker rather than stronger. **There is no
  longer a candidate whose validation the holdout would settle**, so it
  remains unspent. Extending history backwards was chosen precisely because
  it adds evidence without touching it.
- **Disclosed contamination:** Phase 1/2 box geometry was visually validated
  against a handful of 2022–2024 charts, and Phase 7 analysed 41 real trades
  from Jul 2024–Jan 2025. Both concern *detector geometry*, not strategy
  performance. No strategy parameter has been fitted to post-2022 data.

## 5a. Two harness traps, both live

Both were hit while reproducing the V0 baseline in Phase 15, and both silently
produce a *different, already-rejected* configuration rather than an error:

1. **`ScoreParams()` defaults to the calibrated 0.40/0.25/0.20/0.15 weights** --
   REJECTED #4. The locked configuration is equal weight, 0.25 x 4. Calling
   `attach_score(sig)` with no params scores with the rejected weights, which
   moves `traded` from 551 to 405 and every fold with it.
2. **`PortfolioParams()` defaults to `atr_stop_mult=None` and
   `max_hold_bars=None`** -- the spec SMA(10) stop with no cap. The locked
   configuration is ATR(14) x 2.5 plus a hard 25-bar cap (PROVEN 1). Taking
   the defaults reproduces §9.2's *spec* column, not §9.1: it lands on 1.16 /
   3.19 / -2.67 / -5.36 for 2015/2016/2018/2019, which are the spec-stop
   numbers to the decimal.

`research/geometry_sweep.py` names both explicitly as `EQUAL_WEIGHTS` and
`LOCKED_PORTFOLIO`. Any new harness should do the same, or reproduce a known
row before trusting its output. `research/aes_portfolio_run.py` still takes
both defaults and therefore does **not** run the locked configuration.

## 6. Code state

230 tests passing. `box_carry_bars` and the watchlist `trace` hook are
implemented, tested and default-off. Phase 15 added three more default-off
geometry options -- `allow_sloped` (+ `sloped_max_drift` / `sloped_min_drift`),
`exclude_prior_leg`, and `edge_basis="close"` -- all implemented, none adopted.
`research/geometry_sweep.py` is the sweep harness and reproduces the V0 row of
`admission_variants.json` fold by fold. `atr_stop_mult` / `sma_stop_buffer_pct`
are implemented with sizing that follows whichever stop is active. Two ruff
warnings remain, both pre-existing (`relative_strength.py`,
`test_aes_portfolio.py`).
