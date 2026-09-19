# Short-swing research: Indian mid/small caps

A hypothesis-driven search for a short-swing mechanism in the NIFTY Midcap 150 +
Smallcap 250 universe, with the validation discipline fixed in code before any
result was measured (`research/protocol.py`).

---

## 0. Protocol, fixed in advance

| Window | Dates | Purpose |
|---|---|---|
| DISCOVERY | 2015-01-01 → 2021-12-31 | Hypothesis generation, threshold choice. Unlimited looks. |
| VALIDATION | 2022-01-01 → 2024-06-30 | Candidate selection. Failure = rejection, not retuning. |
| HOLDOUT | 2024-07-01 → 2026-12-31 | Touched once, by one candidate, at the end. |

Boundaries were chosen for **regime coverage**, not performance — nothing had
been measured when they were set. Discovery spans the 2015-16 correction,
demonetisation, the 2017 melt-up, the 2018-19 small-cap bear, the COVID crash and
recovery, and the 2021 bull. Validation spans the 2022 rate shock and the 2023-24
recovery.

**Short-swing framework**, also fixed in advance: hard max 10 bars, mean ≤ 6,
median ≤ 5, ≥ 50% of trades closed within 5 bars. **Frequency floor**: ≥ 40
trades/year and ≥ 200 total.

---

## 1. Data and its limitations

400 of 401 constituents, median **3,133 bars** (~12.5 years, 2014-01 → 2026-09).
Benchmark: NIFTY Midcap 150 (`^CRSMID`); the Smallcap 250 index is not available
on the data source, so an equal-weighted universe return is used as the stricter
skill benchmark throughout the event studies.

### Data audit (`research/data_audit.py`)

| Check | Result |
|---|---|
| Duplicate dates, non-positive prices | 0 |
| OHLC inconsistencies | 17 symbols, 20 bars |
| Single-bar moves > 35% | 16 symbols, 24 bars |
| Stale runs > 5 bars | 13 symbols |
| **Symbols excluded from all research** | **30** (structural failures) |

### Survivorship bias — material, and only half measurable

| Date | Universe listed | Coverage |
|---|---|---|
| 2015-01-01 | 231 / 400 | **57.8%** |
| 2018-01-01 | 262 / 400 | 65.5% |
| 2022-01-01 | 320 / 400 | 80.0% |
| 2024-07-01 | 357 / 400 | 89.2% |

This counts only companies **not yet listed**. Names *delisted* or dropped from
the indices are absent from today's constituent list entirely and cannot be
counted from it at all. NSE's public archive does not serve point-in-time
constituent lists, so this cannot be fixed with the available data.

**Consequence, stated plainly:** discovery-period results are optimistically
biased by an unknown amount, and early-period results are computed on a smaller,
surviving subset. This is why the protocol places the evidential weight on
validation (80% coverage) and holdout (89%), and why no conclusion here rests on
2015-2017 performance alone.

### Look-ahead audit

Audited **structurally rather than by argument**. `find_lookahead` recomputes
every feature on history truncated at bar *t* and compares with the value
computed on the full series; any disagreement is leakage. This catches centred
windows, missing shifts, full-period normalisation and future-return leakage
generically.

**Result: PASSED.** Every feature reproduces exactly on truncated history, across
12 probe bars per symbol. The audit runs as a unit test
(`test_feature_builder_has_no_lookahead`) and on every event-study run.

One acknowledged deviation from point-in-time purity: **industry membership**
comes from the current constituent list, so a stock's peer group is assigned with
information not available historically. Industry classification is stable, so the
effect is second-order, but it is real and unquantified.

---

## 2. Phase 1 — why the previous strategy produced long holds

The predecessor (Delivery Accumulation Breakout) was diagnosed in `RESEARCH.md`.
Summary of what carried forward:

* Its five rules all select for **short-term extension**, which mean-reverts in
  this tier. Measured excess return of the composite signal was **−0.36%** over
  10 days against a universe base rate of +0.23% — worse than a random pick.
* Delivery percentage showed **no predictive value in six formulations**. The
  decisive test: adding a delivery filter to the best non-delivery signal moved
  its edge from +0.715% to −0.380%.
* Its edge only turned net-positive at **20-30 bar holds**, i.e. by becoming
  position trading. That is precisely the failure this study was asked to avoid.

**Decision: abandon the delivery premise.** Dropping it also removed the NSE
per-day bhavcopy constraint, which is what made 12 years of history — and
therefore genuine regime analysis — possible at all.

---

## 3. Phase 2 — hypotheses

Five mechanisms, each with a phenomenon, a reason, and a failure mode. All are
attempts to **separate information from flow**: information should be followed,
flow should be faded.

| # | Phenomenon | Why it should exist |
|---|---|---|
| H1 | Stock falls 5pp more than its sector while the sector is flat | Real business news moves peers too; a solo move is inventory |
| H2 | Breaks the 20-day low intraday, closes back above it | Stops and momentum sellers triggered, then trapped |
| H3 | Multi-day slide meets a heavy-volume bar closing strong | Heavy volume that fails to make a lower close = absorption |
| H4 | 1.5-ATR decline on **below**-average volume | In a thin book, price moves without information |
| H5 | Gaps down 2%+ while the sector does not | Overnight is where forced flow concentrates |

Controls that had to be **beaten, not matched**: plain RSI(2) < 10, an
unconditioned 3-day drop of 5%, and a 20-day breakout.

### Results (DISCOVERY, excess over equal-weighted universe, %)

| Hypothesis | events/yr | 1d | 3d | 10d |
|---|---|---|---|---|
| H4 liquidity vacuum | 806 | +0.187* | **+0.241*** | +0.156 |
| H1 idiosyncratic drop | 511 | +0.089 | +0.050 | +0.067 |
| H3 absorption | 766 | +0.031 | +0.020 | −0.083 |
| H2 failed breakdown | 586 | −0.047 | −0.027 | −0.164 |
| H5 orphan gap | 335 | −0.026 | **−0.407*** | −0.434 |
| *C1 plain RSI(2)* | 5,620 | +0.062* | +0.179* | +0.130* |
| *C2 "fell 5% in 3 days"* | 3,744 | +0.085* | +0.204* | +0.256* |
| *C3 breakout* | 2,775 | −0.167* | −0.337* | −0.225* |

`*` = |t| ≥ 3.

**What failed, and why it matters:**

* **H2 and H3 failed outright.** The trapped-seller squeeze and the absorption
  bar do not show up in the data at all.
* **H4 "worked" but barely beat a naive control** — +0.241% against C2's +0.204%
  for simply "the stock fell 5%". Four basis points of sophistication.
* **H5 was backwards, and that was the most useful result.** Unshared gap-downs
  *continue* falling. The information-vs-flow framing was right; the sign was
  wrong. This became a **filter**, not an entry.
* **C3 replicates the predecessor's finding** on a different universe and period:
  buying extension loses. Twice-replicated.

---

## 4. Phase 3 — where the mechanism actually lives

Binary hypotheses with hand-chosen thresholds were the wrong instrument. Sorting
on the *continuous* version of each signal found the mechanism (DISCOVERY, 3-bar
forward excess):

| Signal | Bucket 1 (worst) | Bucket 10 (best) | Monotonicity (Spearman) |
|---|---|---|---|
| **`rel_sector_3`** — return minus sector median | **+0.262%** (t=9.2) | −0.354% (t=−10.9) | **−0.96** |
| `vol_ratio` — participation | +0.100% | −0.255% | −0.95 |
| `ret_3_atr` — volatility-scaled move | +0.151% | −0.304% | −0.73 |
| `ret_3` — raw move | +0.180% | −0.334% | −0.58 |

A Spearman correlation of −0.96 across ten buckets is the strongest single piece
of evidence in the study. **Noise does not order outcomes monotonically.**
Sector-relative weakness orders them far more cleanly than raw weakness (−0.96
vs −0.58), which is the whole thesis in one number.

### The decisive contrast

Taking the single most extreme name in the universe each day:

| top N/day | ranked by `rel_sector_3` | ranked by `ret_3_atr` |
|---|---|---|
| 1 | **+0.405%** | **−0.193%** |
| 3 | +0.275% | +0.029% |
| 10 | +0.252% | +0.112% |
| 50 | +0.163% | +0.117% |

Same direction of price, opposite outcome. Ranking on *relative* weakness
concentrates the edge; ranking on *absolute* weakness **inverts** it, because the
biggest absolute decliner on a given day usually has real news. This is the
result the strategy is built on, and it is why the mechanism is not a generic
oversold-bounce rule: the signal is not that a stock fell, it is that it fell
**alone**.

### Interaction: which variable governs

```
                rel_sector q1   q2      q3      q4
ret_3_atr q1       +0.169    -0.032  -0.148  -0.286
ret_3_atr q3       +0.192    +0.205  +0.013  -0.182
```

The `rel_sector` column sets the sign regardless of absolute move size. H4's
volume story turned out much weaker than its marginal decile suggested — the
biggest declines on thin volume (+0.131) versus heavy volume (+0.119) are
effectively indistinguishable. Volume matters for fading *rallies*, not for
buying dips.

---

## 5. Phase 4 — the candidate

**Sector-Relative Dislocation Reversion (SRDR)** — `strategy/srdr.py`.

Filters, each a prediction of the thesis rather than a tuned parameter:

| Filter | Rationale |
|---|---|
| no orphan gap | H5 measured these continue falling (−0.407%, t>3) |
| not collapsing (> −3 ATR) | beyond that it is a repricing, not an inventory shock |
| sector calm (\|3d sector move\| < 4%) | peer comparison needs stable peers |
| trend intact (above 200-day) | below it, "cheap vs peers" may be a failing business |
| ~~liquid quartile~~ | **tested and rejected** — removing thin names removed the edge |

The last row matters: restricting to liquid names *reduced* the edge, replicating
an earlier independent finding. The premium is compensation for providing
liquidity, so it lives where liquidity is scarce.

### Filter stack results (DISCOVERY)

| topN | trades/yr | 3d edge | t | 5d edge | t |
|---|---|---|---|---|---|
| 1 | 245 | 0.414 | 3.49 | 0.638 | 4.09 |
| 2 | 490 | 0.373 | 4.78 | 0.517 | 5.09 |
| **3** | **734** | **0.409** | **6.73** | **0.543** | **6.90** |
| 5 | 1,223 | 0.378 | 8.28 | 0.448 | 7.67 |
| 8 | 1,953 | 0.320 | 9.13 | 0.397 | 8.85 |

Stable across an 8× range of selection width with rising t-statistics — a
**plateau, not a peak**. Parameters were chosen to sit inside it.

---

## 6. Phase 5 — the portfolio result, and the wall

Equal-weight, 20 positions, 5-bar hold, Rs 10 lakh book, full cost model.

| | DISCOVERY | VALIDATION |
|---|---|---|
| Total return | −12.06% | +3.07% |
| CAGR | −1.86% | +1.25% |
| **Benchmark CAGR** | **+13.41%** | **+26.87%** |
| Profit factor | 0.97 | 1.02 |
| Avg R | −0.012 | +0.008 |
| Trades | 3,571 (510/yr) | 1,301 (522/yr) |
| Mean / median hold | 4.83 / 5.0 bars | 4.85 / 5.0 bars |
| Costs | **Rs 553,581** | Rs 216,901 |
| Short-swing check | **PASS** | **PASS** |
| Frequency check | **PASS** | **PASS** |

**It qualifies as short-swing and clears the frequency floor, and it loses.**

The measurement and the backtest agree exactly, which is the important part: the
signal is real, and the arithmetic still does not work. At 20 equal-weight
positions on a Rs 10 lakh book each position is Rs 50,000, where the round-trip
cost is **0.65%** against a measured five-day edge of **0.54%**.

Costs consumed 55% of starting capital over seven years across 3,571 trades.

### The wall, now confirmed three independent ways

| Study | Measured edge | Round-trip cost | Net |
|---|---|---|---|
| DAB mean-reversion | +0.345% @ 10d | 0.46% | negative |
| Pullback Reversion | +0.345% @ 10d | 0.46% | positive only at 20-25d holds |
| **SRDR** | **+0.54% @ 5d** | **0.65%** | **negative** |

Indian **delivery** trading costs roughly 0.5-0.65% round trip, of which STT
alone is 0.20% (0.1% on *both* sides — a charge the project's original cost model
omitted) and slippage another 0.2-0.3%. Short-horizon cross-sectional equity
edges in this universe measure 0.2-0.6%. The edge does not clear the toll.

---

## 9. Phase 6 — robustness, and the verdict

108 configurations (top_n x positions x slippage x hold), run on a Rs 50 lakh
book so that flat fees were as favourable as they realistically get.

| Window | Configurations beating the benchmark |
|---|---|
| DISCOVERY | **0 / 108 (0%)** |
| VALIDATION | **5 / 108 (5%)** |

Five percent is what chance produces at 108 trials. The best discovery
configuration still trailed the benchmark by **2.29% CAGR**; the best validation
configuration beat it by 5.93%, but its neighbours in the same grid did not, and
all five winners sat at the longest hold tested.

### Robustness by axis (share beating the benchmark, DISCOVERY)

| Axis | Values tested | % positive | Median excess CAGR |
|---|---|---|---|
| top_n | 2 / 3 / 5 | 0% / 0% / 0% | −12.3 / −10.2 / −12.2 |
| positions | 10 / 15 / 20 | 0% / 0% / 0% | −12.6 / −11.0 / −11.2 |
| slippage | 5 / 8 / 12 / 15 bps | 0% across all | −7.1 / −9.9 / −12.5 / −14.1 |
| hold | 3 / 5 / 8 bars | 0% across all | −17.2 / −10.7 / −7.3 |

### Break-even does not exist within plausible assumptions

| Assumed slippage | DISCOVERY median excess | VALIDATION median excess |
|---|---|---|
| 5 bps (implausibly optimistic) | **−7.11%** | **−13.26%** |
| 8 bps | −9.92% | −14.39% |
| 12 bps | −12.53% | −20.09% |
| 15 bps (modelled) | −14.12% | −20.91% |

Even at 5 bps — better execution than a retail account will achieve on a
Rs 2 crore-turnover small cap — the strategy trails the index by 7 percentage
points a year.

Note the hold axis: the strategy gets steadily *less bad* the longer it holds
(−17.2 → −10.7 → −7.3). That is the same finding as the predecessor, reached
from a completely different direction. **The edge only begins to matter at
horizons that stop being short-swing.**

---

## 10. VERDICT: no candidate meets the standard

**SRDR is rejected, and no strategy in this study is recommended for trading.**

The holdout window has **not** been used. The protocol reserves it for a
candidate that survived validation; SRDR did not, so spending it would answer
nothing and would only burn the one piece of clean evidence left for future work.
It remains untouched and usable.

### What is nevertheless established

1. **A genuine, economically coherent short-horizon anomaly exists** in this
   universe: sector-relative dislocation reverts, with a monotone decile gradient
   (Spearman −0.96) and t-statistics up to 11 on hundreds of thousands of
   observations. This is not a marginal statistical curiosity.
2. **It is not large enough to trade at short horizons.** Measured edge peaks
   near 0.54% over five bars at the concentration a portfolio can actually hold.
   Indian delivery round-trip costs are 0.5-0.65%, of which 0.20% is STT charged
   on *both* sides and is not negotiable at any broker.
3. **The gap is structural, not a tuning failure.** Zero of 108 configurations
   closed it, and the shortfall widens monotonically with every cost assumption.

### Why this is a real conclusion rather than a failure to find one

Three independent studies, on two different universes, across different periods,
using different mechanisms, all terminated at the same arithmetic: **short-horizon
cross-sectional equity edges in Indian mid/small caps measure 0.2-0.6% per trade,
and delivery-settled round-trip costs are 0.5-0.65%.** The distribution of
available edge sits almost exactly on top of the cost of capturing it.

That is a finding about the market, not about the search.

### What would change the answer

* **A cost structure that is not delivery-settled.** STT drops from 0.10% to
  0.025% and applies to one side only for intraday. A mechanism that resolves
  within the session faces roughly a third of the friction. This is the single
  largest lever available and it requires intraday data, which this study did not
  have.
* **Holding longer.** The measured edge keeps growing past 10 bars. That answers
  a different question than the one asked here, and the previous study already
  showed it turns net-positive around 20-25 bars.
* **Not competing with beta.** Every long-only construction must beat a rising
  index. The signal has genuine cross-sectional skill (both tails work: the
  most-outperforming decile returns −0.354%), so a market-neutral implementation
  isolates it. The short leg is impractical in single Indian stocks, but an index
  future is not.
* **More history and point-in-time constituents**, which would remove the
  survivorship optimism that currently flatters the discovery window.

---

## 7. Research log — what was tried and what failed

| # | Hypothesis / change | Outcome |
|---|---|---|
| 1 | Delivery% spike, same-bar as volume surge | Mechanically self-defeating; 9 signals in 17 months |
| 2 | Delivery spike in prior window | Tradeable frequency, still no edge |
| 3 | Delivery in six further formulations | No edge in any; adding it to a working signal destroyed that signal |
| 4 | H2 trapped sellers / failed breakdown | No effect |
| 5 | H3 absorption bar | No effect |
| 6 | H4 liquidity vacuum (low-volume decline) | Real but barely beats "stock fell 5%" |
| 7 | H5 orphan gap as an entry | **Backwards** — became a filter instead |
| 8 | Binary thresholds on hypotheses | Wrong instrument; continuous sorting found the mechanism |
| 9 | Ranking on absolute weakness | **Inverts the edge** — the key negative result |
| 10 | Restricting to the liquid quartile | Cut cost 31%, cut edge 68% |
| 11 | Reversion-based exit (predecessor) | Raised win rate, destroyed win/loss ratio |
| 12 | Ranking by a column the strategy lacked | Silently sorted alphabetically — caught and fixed |

---

## 8. Reproducing

```bash
python -m nifty_swing_bot.research.event_study      # hypotheses + look-ahead audit
python -m nifty_swing_bot.research.conditioning     # decile / monotonicity / interactions
python -m nifty_swing_bot.research.refinement       # filter stacks
python -m nifty_swing_bot.research.srdr_study       # portfolio, discovery + validation
python -m nifty_swing_bot.research.sensitivity      # robustness surface
```
