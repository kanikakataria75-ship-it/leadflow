# Nifty Swing Terminal

A research terminal for the **Delivery Accumulation Breakout (DAB)** strategy —
short swing trades (2–10 trading day holds) on the NSE mid- and small-cap tier,
built entirely on free data sources.

> **This is research tooling, not investment advice.** Nothing it produces is a
> recommendation to buy or sell any security. Read the
> [honest results](#honest-results) section before doing anything with it.

---

## What makes this different

Most retail strategies key off price and volume, which everyone already has.
This one adds **delivery percentage** — the fraction of a day's traded shares
that are actually taken to demat rather than squared off intraday. NSE publishes
it daily in the security bhavcopy, and it distinguishes *real accumulation* from
*intraday churn*. Two stocks can both trade 3× their normal volume; the one
where 60% of that went to delivery is being accumulated, the one at 15% is being
day-traded.

---

## The strategy

A signal fires on bar **T** only when **all five** conditions hold:

| # | Rule | Default |
|---|------|---------|
| 1 | **Delivery accumulation** — delivery% spiked above its 20-day baseline within the prior N bars | > 1.5×, within 10 bars |
| 2 | **Volume surge** — today's volume vs its 20-day average | > 2.0× |
| 3 | **Strong close** — close sits in the top of the day's high-low range | top 30% |
| 4 | **Relative strength** — stock's 10-day return beats the Nifty 500's | > 0 |
| 5 | **Breakout / VCP** — close above the prior 20-day high, *or* within 3% of it while ATR(5) < 0.7 × ATR(20) | — |

### Rule 1 differs from the naïve specification, on purpose

The original brief asked for the delivery spike **on the same bar** as the
volume surge. That formulation is close to self-defeating, and the data shows
why. Delivery% is `deliverable qty ÷ TOTAL traded qty`. Rule 2 demands the
denominator doubles, so for delivery% to *also* rise 1.5×, deliverable quantity
must more than triple on the same day.

Measured across 150 mid-caps over 17 months:

- `corr(log volume_ratio, log delivery_ratio) = **−0.257**` — they move *against*
  each other
- median delivery ratio on quiet days **1.015**, on 2×-volume days **0.817**
- of the 1,021 bars that passed the other four rules, only **0.88%** also had a
  same-day delivery spike — **9 signals in 17 months**, not a tradeable sample

So `delivery_mode` defaults to **`prior_window`**: the accumulation must happen
*before* the breakout. That is also what the strategy's own name describes —
Delivery Accumulation **Breakout** is a sequence, not a coincidence. Smart money
accumulates quietly on normal-volume days (when delivery% *can* spike), and
price breaks out afterwards. This yields ~152 signals over the same period.

The original same-bar rule is preserved as `delivery_mode: same_day` for
comparison. See `config.py` and the Settings page.

### Universe

Nifty 500 **minus Nifty 100** — the mid/small-cap tier — then a liquidity floor.

Large caps carry a structurally high baseline delivery% because institutions
hold them, which dilutes the signal. Measured over ~35 sessions:

| Tier | Median baseline delivery% | Within-stock dispersion (CV) |
|---|---|---|
| Nifty 100 | 54.7% | 0.151 |
| Nifty 500 ex-100 | **48.2%** | **0.194** (≈29% higher) |

Lower baseline *and* more dispersion, so a spike is both rarer and more
informative. Filters: ≥ ₹5 cr average daily turnover, ≥ ₹20 price, ≥ 120 bars
of history.

### Risk management

- **Stop** — `min(signal-day low, entry − 1.5 × ATR(14))`
- **Sizing** — risk 1% of equity per trade based on stop distance…
- **…capped** at 10% of equity in any one stock, applied *after* the risk
  formula. **Note the interaction:** the cap binds whenever the stop is tighter
  than `max_position_pct / risk_per_trade_pct` = 10%. With typical 4–6% stops,
  realised risk is therefore ~0.4–0.6%, not 1%. This is intentional (bounded
  small-cap concentration beats a uniform risk number) but you should know it.
- **Exits** — trailing stop on confirmed swing lows once past 1R, or a 10-bar
  max hold, whichever comes first
- **Portfolio** — max 10 concurrent positions, 6% total open heat

> A swing low is a *local minimum*. A rally that never pulls back contains none,
> so the trail cannot engage and `max_hold_days` is the real backstop for
> winners. In the backtest only 2 of 240 exits were trailing stops.

### Execution realism

Mid/small caps carry 5%/10%/20% circuit bands, so the backtester does **not**
assume clean stop fills:

- price **gaps through** the stop → fill at the open, plus extra slippage
- bar opens at the band and barely trades (**circuit-locked**) → no bid exists,
  the exit is deferred a bar and charged a further adverse move
- 15 bps slippage + 3 bps brokerage per side, 10 bps STT on sells
- position notional capped at 2% of the signal day's traded value

In the reference run, **5.0%** of exits were adverse fills.

---

## Setup

### 1. Python

```bash
pip install -r requirements.txt
```

Python 3.10+ (developed and tested on 3.14).

### 2. Anthropic API key (optional — only for the LLM layer)

Copy `.env.example` to `.env` in the project root and add your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get one at <https://console.anthropic.com/settings/keys>. Everything except the
AI narratives works without it. `.env` is git-ignored.

### 3. Backfill the data

This is the slow, network-heavy step; everything afterwards runs offline from
the cache. NSE serves one bhavcopy CSV per trading day and rate-limits, so a
3-year backfill takes roughly 30–40 minutes.

```bash
python -m nifty_swing_bot.data.backfill --start 2023-01-01
```

It is **resumable** — each trading day is cached separately, so an interruption
or a throttle loses at most the day in flight. Re-run to continue.

```bash
python -m nifty_swing_bot.data.backfill --status   # what's cached, and any gaps
```

### 4. Run the pipeline

```bash
python -m nifty_swing_bot.backtest.run_backtest       # backtest + stats + equity curve
python -m nifty_swing_bot.backtest.walk_forward       # rolling out-of-sample validation
python -m nifty_swing_bot.scanner.daily_scan --with-llm   # today's signals
```

### 5. Start the terminal

```bash
cd nifty_swing_bot/frontend && npm install && npm run build && cd ../..
uvicorn nifty_swing_bot.api.app:app --port 8000
```

Open <http://localhost:8000>.

For front-end development with hot reload, run the API on 8000 and, separately:

```bash
cd nifty_swing_bot/frontend && npm run dev     # http://localhost:5173, proxies /api
```

---

## Project layout

```
nifty_swing_bot/
  config.py                  All tunable parameters (pydantic; drives the Settings UI)
  data/
    cache.py                 TTL-aware parquet cache
    nse_session.py           Warmed-up NSE session (cookie dance, retries, throttle)
    universe.py              Nifty 500 − Nifty 100, liquidity filter
    fetch_ohlcv.py           yfinance OHLCV, incremental per-symbol caching
    fetch_delivery.py        NSE bhavcopy delivery%, one cached parquet per day
    backfill.py              Resumable bulk historical download
    store.py                 SQLite ledger: signals, outcomes, LLM insights
  strategy/
    indicators.py            ATR, rolling baselines, swing lows (look-ahead safe)
    dab_strategy.py          The five rules, stops, position sizing
  backtest/
    engine.py                Event-driven portfolio backtester
    metrics.py               Sharpe, Sortino, drawdown, profit factor, R stats
    run_backtest.py          Orchestration + CLI
    walk_forward.py          Rolling train/test validation
    bt_adapter.py            Cross-validation against backtesting.py
  scanner/daily_scan.py      Live signal scan + forward-outcome tracking
  llm/insight_generator.py   Anthropic insights, cached per symbol per day
  api/app.py                 FastAPI backend
  frontend/                  React + Vite terminal UI
  tests/                     44 unit tests
```

### Why a custom backtester

`backtesting.py` is strictly **single-asset**. Running it once per symbol and
stitching the results would let every symbol assume the full account balance, so
the equity curve, the concurrent-position cap and portfolio drawdown would all
be fiction. `vectorbt` handles multiple assets but needs numba callbacks for
this exit stack (swing-low trail gated at 1R + max-hold clock + circuit-aware
fills), which is hard to verify.

The custom engine simulates one shared account bar by bar, and — importantly —
the **same code path** serves the backtest, the walk-forward folds and the live
forward tracker, so there is no backtest/live skew.

`backtesting.py` is still used, as an *independent check* on the per-symbol
mechanics:

```bash
python -m nifty_swing_bot.backtest.bt_adapter
```

Result on the reference data — **5/5 symbols agree trade-for-trade**:

```
  APARINDS: AGREE
    matched trades      : 4
    custom engine only  : 0
    backtesting.py only : 0
    max entry price gap : Rs 0.0000
    stop exits compared : 2  (max gap Rs 0.0000)
    max-hold exits      : 2 excluded (known semantic difference)
```

Both engines select the same signals, enter on the same bars at identical
prices, and fill stops at identical prices. `max_hold` exits are excluded and
that exclusion is not a fudge: this engine closes a timed-out trade at the close
of the bar the clock expires on, which is what a trader watching the session
does. `backtesting.py` cannot express that — `trade.close()` queues a market
order filling at the *next* bar's open, and the only switch that would change it
(`trade_on_close=True`) would also move entries to the signal bar's close and
destroy the entry comparison that matters more. The two therefore exit timed-out
trades one bar apart by construction.

---

## Honest results

Reference run: 395 mid/small-cap symbols, 2025-01-03 → 2026-09-04, ₹10,00,000
starting capital, all costs and circuit modelling applied.

### Static backtest

| Metric | DAB | Nifty 500 |
|---|---|---|
| Total return | **−15.90%** | +2.40% |
| CAGR | −9.91% | — |
| Max drawdown | 19.77% | — |
| Sharpe | −1.34 | — |
| Win rate | 36.67% | — |
| Profit factor | **0.78** | — |
| Avg R multiple | **−0.114R** | — |
| Trades | 240 | — |

### Walk-forward (6 folds, 9-month train / 3-month test)

| Metric | Value |
|---|---|
| Out-of-sample trades | 268 |
| Combined OOS return | **−5.54%** |
| OOS avg R | −0.039R |
| OOS win rate | 41.4% |
| Profitable folds | 4 / 6 |

**Read this plainly: the strategy did not show a positive edge on 2.5 years of
NSE mid/small-cap data.** Both the static backtest and the walk-forward are
negative.

Two things worth noting before writing it off, and one before believing it:

- The optimiser could not find profitable parameters *in-sample* in most folds
  either (mean in-sample avg R ≈ 0.002). So this is **not** an overfitting
  story — there was no in-sample edge to overfit to. The grid simply contains no
  profitable region on this data.
- 55% of trades exited on `max_hold`, i.e. they timed out rather than resolving.
  The 10-bar clock may be too short for accumulation to play out, and the
  swing-low trail almost never engages. Those are the first things to test.
- The window is short (2.5 years) and covers one regime, in which the mid/small
  tier was broadly weak. That cuts both ways: it is not enough evidence to
  condemn the strategy either.

The terminal renders losing results exactly as plainly as winning ones. That is
the point of building it.

### Ideas the data points at

1. **Lengthen the hold.** With 55% timing out, test `max_hold_days` of 15–20.
2. **Trail differently.** A confirmed swing low is too rare to protect gains;
   try a chandelier/ATR trail so winners are actually ridden.
3. **Add a market-regime filter.** Long-only breakouts in a falling small-cap
   tape are a structural headwind — gate entries on the index trend.
4. **Widen the delivery window** beyond 10 bars, or require *multiple* spikes.

---

## The AES daily scanner

The production tool, in `aes_scanner/`. It replaces a Chartink + Excel +
TradingView loop; it does not replace the decision. Run it once after the
close:

```bash
python -m nifty_swing_bot.aes_scanner                 # scan, render, persist
python -m nifty_swing_bot.aes_scanner --equity 1500000
python -m nifty_swing_bot.aes_scanner --record        # the live forward record
```

Also available at `/aes` in the terminal UI, and over HTTP:

| endpoint | what it returns |
|---|---|
| `GET /api/aes/scan` | the latest evening's ranked candidates, every score component included |
| `GET /api/aes/history` | per-day scan metadata and the breadth series |
| `GET /api/aes/record` | the accumulating live record, with a closed/open summary |
| `GET /api/aes/chart/{date}/{symbol}` | the rendered box overlay |
| `POST /api/aes/scan` | trigger a background scan |

### What it does each evening

1. Refreshes OHLCV for the universe, runs the §1.1 screener, and rebuilds
   watchlist membership through the §1.3 state machine — admissions,
   hard-deletes, staleness. State is **recomputed from price, not stored**:
   the machine is a pure function of history, so nothing can drift between
   runs.
2. Detects the nested box on every watchlist name and reports big-box range
   and duration, small-box position, higher-lows count, share of closes above
   the midpoint, and where price sits in the box (0 = floor, 1 = top).
3. Computes the §3 context: nearest resistance with distance and age,
   absorbed/rejected prior-visit counts, daily/weekly/monthly breakout
   classification, and drawdown-conditional relative strength vs Nifty 500.
4. Ranks into the three §4 buckets using **equal weights**, not the Phase 3
   calibrated ones — Phase 6 found those overfit. Every component value is
   shown next to the score so the ranking can be overruled.
5. For anything the state machine actually triggered, computes the entry
   reference, the ATR 2.5× stop, size at 1% risk, and the rupee amounts.

Each run renders a box chart per candidate worth looking at and writes the
whole thing to `cache/aes_scans.sqlite`, so a live forward record accumulates
without anyone having to maintain it. Outcomes are replayed under the adopted
exit — ATR 2.5× trail with the 25-bar cap — so the live record is measured
the same way the backtest measured.

### What it does not do

It does not decide, and it is not validated as an unsupervised system. See
`STATE.md`: the risk components are established (the ATR trail lowered
drawdown and raised win rate in 7 of 7 walk-forward folds), the return edge is
not (three of eleven folds carry it, p=0.143 across all eleven). The scanner
exists because the ranking, the geometry and the risk arithmetic are useful
even where the return edge is unproven.

**Breadth is printed on every run and gates nothing.** Phase 11/12 showed it
is not separable from a single year — within-year r=+0.010, p=0.773 — so it
appears as context for how aggressive to be, with the caveat attached.

---

## Forward testing

The scanner accumulates a live record so real performance can be compared with
the backtest on identical terms:

```bash
python -m nifty_swing_bot.scanner.daily_scan --with-llm
```

Run it after the close — NSE publishes the bhavcopy around 18:00–19:00 IST;
before that, today's delivery file does not exist and the scan correctly finds
nothing. Signals go into `cache/terminal.db`, and every previously recorded
signal is walked forward using the **same exit rules the backtester applies**,
so `avg R` on the Dashboard is directly comparable to the backtest figure beside
it.

```bash
python -m nifty_swing_bot.scanner.daily_scan --track-only   # update outcomes only
```

Long stretches with zero signals are normal — the rules fire on roughly 0.4
names per day across ~400 stocks.

---

## The LLM insight layer

For each signal, Claude generates a company overview, a narrative explaining
which rule values fired and what they mean, an accumulation read, ranked risks,
and a risk-flagged assessment with a disclaimer. Output is structured through a
pydantic schema (`messages.parse`), so the UI always gets the same shape.

- Model: `claude-sonnet-5` (configurable in `config.yaml` or `ANTHROPIC_MODEL`)
- Fundamentals come from `yfinance.Ticker.info`, cached for a week
- **Insights are cached per symbol per day** in SQLite — re-running a scan costs
  nothing
- Third-party company descriptions are passed to the model as explicitly
  untrusted data, and the system prompt instructs it to ignore any instructions
  embedded in them

---

## Testing

```bash
python -m pytest nifty_swing_bot/tests -q
```

44 tests covering each DAB rule in isolation (break one condition, assert the
signal disappears), look-ahead discipline, stop and sizing arithmetic including
the position-cap interaction, and backtester mechanics — next-open fills, clean
stop fills at exactly −1R, gap-through fills worse than −1R, circuit-locked
deferral, the max-hold clock, trailing behaviour, position limits and the
cash/holdings accounting identity.

---

## Data sources

| Data | Source | Notes |
|---|---|---|
| OHLCV | Yahoo Finance via `yfinance` | `auto_adjust=True`, so splits don't fake gaps |
| Delivery % | `nsearchives.nseindia.com` security bhavcopy | one CSV per trading day |
| Index constituents | NSE archives `ind_nifty500list.csv` | weekly TTL |
| Fundamentals | `yfinance.Ticker.info` | best-effort; missing fields stay missing |
| Benchmark | `^CRSLDX` (Nifty 500) | falls back to `^CNX500`, then `^NSEI` |

NSE blocks naive scrapers, so `nse_session.py` warms a browser-like session
against the homepage for cookies before requesting archive files, re-warms on
401/403/429, and throttles with jitter between requests.

---

## Licence & disclaimer

For research and education. Not investment advice, not a recommendation, and no
warranty of accuracy. Indian equity markets carry substantial risk, and the
mid/small-cap tier this targets carries more of it — including circuit limits
that can make a position impossible to exit. Consult a SEBI-registered adviser
before trading. You are responsible for your own decisions.
