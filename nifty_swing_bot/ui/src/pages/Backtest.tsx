import { useState } from 'react'
import { api, useAsync, type Ladder, type Metrics } from '@/lib/api'
import { Col, DataTable, ErrorBox, Loading, Panel, Provenance, Segmented } from '@/components/ui'
import { TradeInspector } from '@/components/TradeInspector'
import { DivergingBars, Histogram, LineChart, StackedArea, UnderwaterChart } from '@/components/charts/Charts'
import { DASH, decimate, dirClass, fmt, int, pct, ratio } from '@/lib/format'
import { stagger } from '@/lib/motion'

const WINDOWS = ['2022-2026', '2015-2021'] as const
type Win = (typeof WINDOWS)[number]

/** The full metric set, in a deliberate reading order: return, then risk-adjusted,
 *  then shape of the distribution, then trade level. */
const METRIC_ROWS: { key: string; label: string; dp: number; pct?: boolean; dir?: boolean }[] = [
  { key: 'total_return_pct', label: 'Total return', dp: 2, pct: true, dir: true },
  { key: 'cagr_pct', label: 'CAGR', dp: 2, pct: true, dir: true },
  { key: 'volatility_pct', label: 'Volatility', dp: 2, pct: true },
  { key: 'downside_deviation_pct', label: 'Downside deviation', dp: 2, pct: true },
  { key: 'sharpe', label: 'Sharpe', dp: 2, dir: true },
  { key: 'sortino', label: 'Sortino', dp: 2, dir: true },
  { key: 'calmar', label: 'Calmar', dp: 2, dir: true },
  { key: 'omega', label: 'Omega', dp: 3, dir: true },
  { key: 'ulcer_index', label: 'Ulcer index', dp: 3 },
  { key: 'max_drawdown_pct', label: 'Max drawdown', dp: 2, pct: true, dir: true },
  { key: 'avg_drawdown_pct', label: 'Avg drawdown', dp: 2, pct: true, dir: true },
  { key: 'var_95_pct_daily', label: 'VaR 95% (daily)', dp: 3, pct: true },
  { key: 'cvar_95_pct_daily', label: 'CVaR 95% (daily)', dp: 3, pct: true },
  { key: 'skew', label: 'Skew', dp: 3 },
  { key: 'kurtosis', label: 'Excess kurtosis', dp: 3 },
  { key: 'beta', label: 'Beta', dp: 3 },
  { key: 'alpha_pct', label: 'Alpha', dp: 2, pct: true, dir: true },
  { key: 'win_rate_pct', label: 'Win rate', dp: 2, pct: true },
  { key: 'profit_factor', label: 'Profit factor', dp: 3, dir: true },
  { key: 'avg_r_multiple', label: 'Position R', dp: 3, dir: true },
]

function metricCell(m: Metrics, k: string, dp: number, asPct?: boolean) {
  const v = m[k]
  if (typeof v !== 'number' || !Number.isFinite(v)) return DASH
  return asPct ? pct(v, dp) : fmt(v, dp)
}

export default function Backtest() {
  const [win, setWin] = useState<Win>('2022-2026')
  const res = useAsync(() => api.results(), [])
  const ser = useAsync(() => api.series(win), [win])

  if (res.loading) return <div className="page"><Loading what="results" /></div>
  if (res.error) return <div className="page"><ErrorBox error={res.error} /></div>

  const w = res.data!.windows[win]
  const frozen: Ladder = w.ladders['variant_c_12_20']
  const old: Ladder | undefined = w.ladders['old_ladder_5_8_20']
  const s = ser.data

  const foldShares = frozen.strategy_folds_pct.map((v) => {
    const tot = frozen.strategy_folds_pct.reduce((a, b) => a + b, 0)
    return tot ? (v / tot) * 100 : null
  })

  const dates = s ? s.dates : []
  const keep = s ? decimate(dates.map((d, i) => ({ d, i })), 520, (o) => o.i) : []
  const idx = keep.map((o) => o.i)
  const pick = (arr: (number | null)[] | undefined) => (arr ? idx.map((i) => arr[i]) : [])
  const dDates = keep.map((o) => o.d)

  const foldCols: Col<{ year: number; strat: number; book: number; share: number | null }>[] = [
    { key: 'y', head: 'Fold', render: (r) => <span className="num">{r.year}</span> },
    { key: 's', head: 'Strategy %', num: true, render: (r) => <span className={r.strat >= 0 ? 'gain' : 'loss'}>{pct(r.strat, 2)}</span> },
    { key: 'sh', head: 'Share of total', num: true, render: (r) => (r.share == null ? DASH : pct(r.share, 1)) },
    { key: 'b', head: 'Book %', num: true, render: (r) => <span className={r.book >= 0 ? 'gain' : 'loss'}>{pct(r.book, 2)}</span> },
  ]
  const foldRows = frozen.fold_years.map((y, i) => ({
    year: y,
    strat: frozen.strategy_folds_pct[i],
    book: frozen.book_folds_pct[i],
    share: foldShares[i],
  }))

  return (
    <div className="page">
      <div className="page-head">
        <div className="row-between">
          <h1>Backtest</h1>
          <div className="row">
            <Provenance value={w.provenance} note={w.provenance_note} />
            <Segmented options={WINDOWS} value={win} onChange={setWin} />
          </div>
        </div>
        <p className="note">
          {w.start} → {w.end} · <span className="num">{int(w.signals)}</span> signals ·{' '}
          <span className="num">{int(frozen.n_positions)}</span> positions · frozen ladder variant (c)
        </p>
      </div>

      {/* ======================= THE BOOK — the headline =======================
          The Book is the complete portfolio: 30% strategy / 50% arbitrage /
          20% gold. The strategy sleeve is one third of it, so the Book's
          return is the result this page leads with and the sleeve is reported
          underneath as a component of it. */}
      <Panel>
        <div className="row-between" style={{ alignItems: 'flex-start', gap: 'var(--s-6)', flexWrap: 'wrap' }}>
          <div className="row" style={{ gap: 'var(--s-12)', alignItems: 'flex-start' }}>
            <div className="hero-metric reveal">
              <div className="l">Book · total return</div>
              <div className={`v ${dirClass(frozen.book_headline.total_return_pct)}`}>
                {pct(frozen.book_headline.total_return_pct, 2)}
              </div>
              <div className="note" style={{ marginTop: 4 }}>
                ex-best fold <span className="num">{pct(frozen.book_headline.ex_best_fold_pct, 2)}</span>
              </div>
            </div>
            <div className="hero-metric reveal" style={{ ['--delay' as string]: '90ms' }}>
              <div className="l">Book · CAGR</div>
              <div className={`v ${dirClass(frozen.book_headline.cagr_pct)}`}>
                {pct(frozen.book_headline.cagr_pct, 2)}
              </div>
              <div className="note" style={{ marginTop: 4 }}>
                best fold is <span className="num">{pct(frozen.book_headline.best_fold_share_pct, 1)}</span> of the summed return
              </div>
            </div>
          </div>
          <div style={{ maxWidth: 300 }}>
            <div className="eyebrow" style={{ marginBottom: 6 }}>What the book is</div>
            <p className="note">
              30% strategy · 50% arbitrage · 20% gold, annual rebalance. The strategy sleeve below is
              one component of this number, not the product&apos;s result.
            </p>
          </div>
        </div>
      </Panel>

      <Panel title="Book" sub="Complete portfolio, all three sleeves" flush>
        <div className="grid grid-4" style={{ gap: 0 }}>
          {[
            { l: 'Total return', v: pct(frozen.book.total_return_pct as number, 2), d: frozen.book.total_return_pct as number },
            { l: 'CAGR', v: pct(frozen.book.cagr_pct as number, 2), d: frozen.book.cagr_pct as number },
            { l: 'Volatility', v: pct(frozen.book.volatility_pct as number, 2) },
            { l: 'Downside deviation', v: pct(frozen.book.downside_deviation_pct as number, 2) },
            { l: 'Sharpe', v: fmt(frozen.book.sharpe as number, 2), d: frozen.book.sharpe as number },
            { l: 'Sortino', v: fmt(frozen.book.sortino as number, 2), d: frozen.book.sortino as number },
            { l: 'Max drawdown', v: pct(frozen.book.max_drawdown_pct as number, 2), d: frozen.book.max_drawdown_pct as number },
            { l: 'Positive folds', v: frozen.book_headline.positive_folds ?? DASH },
          ].map((m, i) => (
            <div key={m.l} className="funnel-stage reveal" style={stagger(i, 45)}>
              <div className="lbl">{m.l}</div>
              <div className={`n ${m.d === undefined ? '' : dirClass(m.d)}`} style={{ fontSize: 'var(--t-xl)' }}>{m.v}</div>
            </div>
          ))}
        </div>
      </Panel>

      {/* ------------------- the strategy sleeve, secondary ------------------- */}
      <Panel
        title="Strategy sleeve"
        sub="One sleeve of the book above — 30% of capital. Reported separately because its risk profile differs from the whole."
        flush
      >
        <div className="grid grid-4" style={{ gap: 0 }}>
          {[
            { l: 'Total return', v: pct(frozen.strategy_headline.total_return_pct, 2), d: frozen.strategy_headline.total_return_pct },
            { l: 'CAGR', v: pct(frozen.strategy_headline.cagr_pct, 2), d: frozen.strategy_headline.cagr_pct },
            { l: 'Ex-best fold', v: pct(frozen.strategy_headline.ex_best_fold_pct, 2), d: frozen.strategy_headline.ex_best_fold_pct },
            { l: 'Sharpe', v: fmt(frozen.strategy.sharpe as number, 2), d: frozen.strategy.sharpe as number },
            { l: 'Sortino', v: fmt(frozen.strategy.sortino as number, 2), d: frozen.strategy.sortino as number },
            { l: 'Max drawdown', v: pct(frozen.strategy.max_drawdown_pct as number, 2), d: frozen.strategy.max_drawdown_pct as number },
            { l: 'Positions', v: int(frozen.n_positions) },
            { l: 'Position R', v: ratio(frozen.position_r.value, 3), d: frozen.position_r.value },
          ].map((m, i) => (
            <div key={m.l} className="funnel-stage reveal" style={stagger(i, 40)}>
              <div className="lbl">{m.l}</div>
              <div className={`n ${m.d === undefined || m.d === null ? '' : dirClass(m.d)}`} style={{ fontSize: 'var(--t-lg)' }}>{m.v}</div>
            </div>
          ))}
        </div>
      </Panel>

      {ser.loading && <Panel title="Equity"><Loading what="series" /></Panel>}
      {s && (
        <>
          <Panel title="Equity curve" sub="Indexed to 100 at the first traded bar. One y-axis: the benchmark is indexed to the same base rather than given a second scale.">
            <LineChart
              dates={dDates}
              height={280}
              baseline={100}
              yFormat={(v) => fmt(v, 0)}
              tipFormat={(v) => fmt(v, 1)}
              series={[
                { label: 'Strategy sleeve', color: 'var(--cat-1)', values: pick(s.strategy_index) },
                { label: 'Total book', color: 'var(--cat-2)', values: pick(s.book_index) },
                { label: 'Nifty 500', color: 'var(--benchmark)', values: pick(s.benchmark_index), dashed: true, width: 1.5 },
              ]}
            />
          </Panel>

          <div className="grid grid-2">
            <Panel title="Underwater" sub="Drawdown from running peak, strategy sleeve">
              <UnderwaterChart dates={dDates} values={pick(s.drawdown_strategy_pct)} height={150} />
            </Panel>
            <Panel title="Rolling 12-month return" sub="252-session trailing window">
              <LineChart
                dates={decimate(s.rolling_12m_pct.dates.map((d, i) => ({ d, i })), 420, (o) => o.i).map((o) => o.d)}
                height={150}
                baseline={0}
                yFormat={(v) => `${fmt(v, 0)}%`}
                tipFormat={(v) => pct(v, 2)}
                series={[{
                  label: 'Rolling 12m',
                  color: 'var(--cat-1)',
                  values: decimate(s.rolling_12m_pct.values.map((v, i) => ({ v, i })), 420, (o) => o.i).map((o) => o.v),
                }]}
              />
            </Panel>
          </div>

          <div className="grid grid-2">
            <Panel title="Return by fold" sub="Each bar labelled with its share of the window's summed return — the concentration is the point">
              <DivergingBars labels={frozen.fold_years} values={frozen.strategy_folds_pct} shares={foldShares} height={200} />
            </Panel>
            <Panel title="Daily return distribution" sub="Strategy sleeve, percent per session">
              <Histogram values={s.daily_returns_pct} bins={40} height={200} colorBySign />
            </Panel>
          </div>

          <div className="grid grid-2">
            <Panel
              title="Trade R distribution"
              sub={`${int(s.r_multiples.length)} positions, ladder rows collapsed`}
            >
              <Histogram values={s.r_multiples} bins={30} height={200} unit="R" colorBySign />
              <p className="note" style={{ marginTop: 'var(--s-2)' }}>
                Position R is <span className="num">{ratio(frozen.position_r.value, 3)}</span>. The
                row-mean figure of{' '}
                <span className="num">{ratio(frozen.position_r.row_mean_do_not_use, 3)}</span> is{' '}
                <strong style={{ color: 'var(--warn)' }}>wrong</strong> — {frozen.position_r.why}
              </p>
            </Panel>
            <Panel title="Sleeve allocation over time" sub="Share of total book">
              <StackedArea
                dates={dDates}
                height={200}
                bands={[
                  { label: 'Strategy', color: 'var(--cat-1)', values: pick(s.sleeve_strategy) },
                  { label: 'Arbitrage', color: 'var(--cat-2)', values: pick(s.sleeve_arb) },
                  { label: 'Gold', color: 'var(--cat-3)', values: pick(s.sleeve_gold) },
                ]}
              />
            </Panel>
          </div>

          <Panel title="Folds" sub="Annual walk-forward folds with each fold's share of the summed return" flush>
            <DataTable cols={foldCols} rows={foldRows} rowKey={(r) => String(r.year)} />
          </Panel>
        </>
      )}

      {s && s.trades.length > 0 && (
        <TradeInspector trades={s.trades as Record<string, unknown>[]} />
      )}

      <Panel
        title="Full metric set"
        sub="Strategy sleeve beside total book. Both columns are the same window and the same frozen ladder."
        flush
      >
        <DataTable
          cols={[
            { key: 'm', head: 'Metric', render: (r: (typeof METRIC_ROWS)[number]) => r.label },
            {
              key: 'strategy', head: 'Strategy', num: true,
              render: (r) => (
                <span className={r.dir ? (Number(frozen.strategy[r.key]) >= 0 ? 'gain' : 'loss') : undefined}>
                  {metricCell(frozen.strategy, r.key, r.dp, r.pct)}
                </span>
              ),
            },
            {
              key: 'book', head: 'Book', num: true,
              render: (r) => (
                <span className={r.dir ? (Number(frozen.book[r.key]) >= 0 ? 'gain' : 'loss') : undefined}>
                  {metricCell(frozen.book, r.key, r.dp, r.pct)}
                </span>
              ),
            },
            ...(old
              ? [{
                  key: 'old', head: 'Strategy, old ladder', num: true,
                  title: 'The spec 5/8/20 ladder, for comparison. Not live.',
                  render: (r: (typeof METRIC_ROWS)[number]) => (
                    <span className="dim">{metricCell(old.strategy, r.key, r.dp, r.pct)}</span>
                  ),
                }]
              : []),
          ]}
          rows={METRIC_ROWS}
          rowKey={(r) => r.key}
          zebra
          foot="Risk-free 6.5%. Beta and alpha are against the Nifty 500. Position R is the position-level figure throughout."
        />
      </Panel>
    </div>
  )
}
