# AES Daily Scanner — frozen 2026-09-17

Run it after the close:

```bash
python -m nifty_swing_bot.aes_scanner --equity 1000000 --deployed 180000
```

`--deployed` is the rupee value currently in open positions; supplying it lets
the report show sleeve usage against the 30% strategy target.

---

## What this is expected to do

**Rank and explain. It does not decide.**

Each evening it walks the 402-name mid/smallcap universe, applies the §1.1
screener, carries a persistent watchlist, runs the box detector on every
watched name, scores what it finds on four equal-weighted components, and
prints a sized trade plan for anything that triggers — entry reference, ATR
stop, both profit tranches, quantity, notional and rupee risk. It shows the
score's components individually so the ranking can be overruled, prints current
breadth as context, and lists **every name that failed a gate**, marked and
ranked below the clean ones, because a silent rejection cannot be checked.

Every run is persisted. From 2026-09-17 the forward record accumulates without
anyone having to remember to write it down.

### The frozen configuration

| layer | setting | why |
|---|---|---|
| geometry | baseline `BoxParams()` | every loosening tested in Phase 15 lowered fold return |
| sizing | 6 slots, 3% risk/trade | only cell above 10% annual in **both** windows (§18) |
| conviction | bucket multipliers 0.7 / 1.2 | score-to-edge is **not monotone**; the configured values already beat neutral and steeper in both windows (§20) |
| scoring | equal weight, 0.25 × 4 | the calibrated weights are REJECTED #4 |
| exits | ATR(14) × 2.5 trail, 25-bar cap, **50% @ +12%, 20% @ +20%, 30% trails** | variant (c), §20 |
| book | 30% strategy / 50% arbitrage / 20% gold, annual rebalance | §19 |

**Do not tune these against historical data.** 2015-2021 was discovery from the
start and 2022-2026 was spent as discovery by Phase 16. There is no in-sample
test left that can inform a change. If a value changes, the forward record
before and after are not the same experiment — record the change and restart
the record.

---

## What is unproven

Most of it. In order of how much it matters:

**1. Whether this makes money unsupervised.** The per-trade edge over a random
entry in the same universe is +0.48% (t=2.01) on 2015-2021 and +1.20% (t=2.61)
on 2022-2026. Neither window's *fold* return clears conventional significance,
and both are carried by one year: remove the best fold and 12.66% becomes
4.98%, 15.38% becomes 5.86%. **Plan on ~6%/yr from the strategy sleeve, not the
headline.**

**2. The exits being used here are a candidate, not an improvement.** Variant
(c) is better than the spec ladder in both windows (+4.04pp and +4.44pp, better
in 4/7 and 4/5 folds) and has the best ex-best-fold figure of any variant
tested. Its best paired t is 1.98. It was chosen to be forward-tested, not
because it is established.

**3. The scoring system is treated as overfit.** Three of the four calibrated
factors were selected on the sample they scored. Equal weighting is used
because it is *not fitted*, not because it is better. Score quartiles are not
monotone in edge (monotone in 2 of 11 folds), so the ranking is advisory.

**4. Nothing here is out-of-sample validated.** `AES_HOLDOUT` was spent as
discovery by Phase 16. The 25% symbol holdout was spent once. The only clean
evidence left is forward: bars dated 2026-09-17 or later.

**5. Survivorship.** The universe is today's index constituents, so early
history is a survivor sample — 53% of the universe existed in 2010, 59% in
2015, 80% in 2021.

---

## The number most likely to be misremembered

> **Position R multiple is ~0.17, not 0.65.**

Earlier work in this project quoted an "average R multiple of 0.65-0.72". That
figure is a mean over trade **rows**, and a profit ladder splits one position
into two to four rows of which **every ladder row is profitable by
construction** — a tranche only sells because price reached its trigger. The
row average therefore counts a position's winning fragments repeatedly and its
losing remainder once.

Measured per **position**:

| | row-mean R | position R |
|---|---|---|
| 2015-2021, spec ladder | 0.652 | **0.184** |
| 2022-2026, spec ladder | 0.720 | **0.164** |
| either window, no ladder | 0.212 / 0.215 | 0.212 / 0.215 |

Any R figure taken from a `trades_frame` row mean is inflated the same way.
When judging the forward record, aggregate to positions first.

Related, and the reason the exits changed: under the spec's +5%/+8% tranches
there was **not one winner above +30% across 398 positions** in either window.
The setups do run — the ladder was the constraint on how far.

---

## What the report shows

- **Breadth** — context only. It is not a gate (Phases 11/12 found it is not
  separable from a single year, and the one independent high-breadth year
  contradicted it).
- **Regime** — whether the benchmark is below its SMA(50). When it is, §8 takes
  high-conviction only and reduces sizes.
- **Book and deployment** — sleeve targets in rupees, and what is deployed
  against the 30% strategy sleeve. Above 100% of the sleeve the strategy draws
  on arbitrage, never on gold.
- **Candidates** by bucket, with the four score components shown separately.
- **Trade plans** — entry reference, stop and basis, risk per share, quantity,
  notional, rupee risk, and both tranche levels with the runner quantity.
- **Gate failures** — ranked nearest-miss first, with the stage and reason.

One honest difference from the backtest: it fills at the *next* bar's open,
which does not exist when the scan runs. The plan is priced off today's close
and labelled a reference. Stop distance and quantity move with the actual
opening price and should be recomputed against it.

---

## Files

| path | role |
|---|---|
| `live_config.py` | **the frozen configuration** — the single source of truth |
| `pipeline.py` | the scan itself |
| `sizing.py` | risk-per-share sizing and tranche levels |
| `report.py` | text rendering |
| `store.py` | SQLite persistence and the forward record |
| `audit.py` | full-universe funnel, one verdict per symbol |

`STATE.md` is the index of what is proven, unproven and rejected.
`RESEARCH_AES.md` holds the working.
