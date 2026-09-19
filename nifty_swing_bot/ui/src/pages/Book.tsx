import { useState } from 'react'
import { api, useAsync } from '@/lib/api'
import { Callout, Col, DataTable, ErrorBox, Loading, Panel, Provenance, Segmented } from '@/components/ui'
import { LineChart, StackedArea } from '@/components/charts/Charts'
import { Metric } from '@/components/terminal'
import { CapitalOrbit, type Sleeve } from '@/components/CapitalOrbit'
import { DASH, decimate, fmt, inr, int, pct, ratio } from '@/lib/format'

const WINDOWS = ['2022-2026', '2015-2021'] as const
type Win = (typeof WINDOWS)[number]

export default function Book() {
  const [win, setWin] = useState<Win>('2022-2026')
  const [sleeveSel, setSleeveSel] = useState<string | null>(null)
  const cfg = useAsync(() => api.config(), [])
  const res = useAsync(() => api.results(), [])
  const ser = useAsync(() => api.series(win), [win])

  if (ser.loading || res.loading) return <div className="page"><Loading what="book" /></div>
  if (ser.error) return <div className="page"><ErrorBox error={ser.error} /></div>

  const s = ser.data!
  const w = res.data!.windows[win]
  const frozen = w.ladders['variant_c_12_20']
  const bk = cfg.data?.book
  const diag = frozen.diagnostics

  const keep = decimate(s.dates.map((d, i) => ({ d, i })), 520, (o) => o.i)
  const idx = keep.map((o) => o.i)
  const dDates = keep.map((o) => o.d)
  const pick = (arr: (number | null)[]) => idx.map((i) => arr[i])

  const last = s.dates.length - 1
  const actual = {
    strategy: Number(s.sleeve_strategy[last] ?? 0),
    arb: Number(s.sleeve_arb[last] ?? 0),
    gold: Number(s.sleeve_gold[last] ?? 0),
  }
  const total = actual.strategy + actual.arb + actual.gold || 1

  type Row = { name: string; color: string; target: number; actual: number }
  const rows: Row[] = [
    { name: 'Strategy', color: 'var(--cat-1)', target: Number(bk?.w_strategy ?? 0.3), actual: actual.strategy },
    { name: 'Arbitrage', color: 'var(--cat-2)', target: Number(bk?.w_arbitrage ?? 0.5), actual: actual.arb },
    { name: 'Gold', color: 'var(--cat-3)', target: Number(bk?.w_gold ?? 0.2), actual: actual.gold },
  ]

  const allocCols: Col<Row>[] = [
    {
      key: 'name', head: 'Sleeve',
      render: (r) => (
        <span className="row" style={{ gap: 6 }}>
          <span className="legend-swatch block" style={{ background: r.color }} />
          {r.name}
        </span>
      ),
    },
    { key: 'target', head: 'Target', num: true, render: (r) => pct(r.target * 100, 1) },
    { key: 'actualw', head: 'Actual', num: true, render: (r) => pct((r.actual / total) * 100, 1) },
    {
      key: 'drift', head: 'Drift', num: true,
      render: (r) => {
        const d = (r.actual / total) * 100 - r.target * 100
        return <span className={Math.abs(d) < 0.5 ? 'flat' : d > 0 ? 'gain' : 'loss'}>{fmt(d, 2)}pp</span>
      },
    },
    { key: 'value', head: 'Value', num: true, render: (r) => inr(r.actual) },
  ]

  const corrCols: Col<Record<string, number | string>>[] = [
    { key: 'year', head: 'Year', render: (r) => <span className="num">{String(r.year ?? DASH)}</span> },
    {
      key: 'corr', head: 'Strategy ↔ gold', num: true,
      render: (r) => {
        const v = Number(r.correlation ?? r.corr ?? NaN)
        return <span className={Math.abs(v) < 0.25 ? 'gain' : 'loss'}>{ratio(v, 3)}</span>
      },
    },
    {
      key: 'sret', head: 'Strategy return', num: true,
      render: (r) => {
        const v = Number(r.strategy_ret_pct ?? NaN)
        return <span className={v >= 0 ? 'gain' : 'loss'}>{pct(v, 2)}</span>
      },
    },
  ]

  return (
    <div className="page">
      <div className="page-head">
        <div className="row-between">
          <h1>Book</h1>
          <div className="row">
            <Provenance value={w.provenance} note={w.provenance_note} />
            <Segmented options={WINDOWS} value={win} onChange={setWin} />
          </div>
        </div>
        <p className="note">
          The whole book, not just the traded sleeve. {pct(Number(bk?.w_strategy) * 100, 0)} strategy /{' '}
          {pct(Number(bk?.w_arbitrage) * 100, 0)} arbitrage at{' '}
          {pct(Number(bk?.arb_annual_rate) * 100, 2)} / {pct(Number(bk?.w_gold) * 100, 0)} gold,{' '}
          {String(bk?.rebalance)} rebalance, sized on the {String(bk?.strategy_capital_basis)} basis.
        </p>
      </div>

      <div className="grid grid-4">
        <Panel><Metric hero label="Book CAGR" value={frozen.book.cagr_pct as number} format={(v) => pct(v, 2)}
          tone={frozen.book.cagr_pct as number} sub={`ex-best fold ${pct(frozen.book_headline.ex_best_fold_pct, 2)}`} /></Panel>
        <Panel><Metric hero label="Mean deployment" value={diag.mean_deployment_pct} format={(v) => pct(v, 2)}
          sub={`peak ${pct(diag.max_deployment_pct, 1)} of book`} /></Panel>
        <Panel><Metric hero label="Rebalances" value={diag.rebalances} format={(v) => int(v)}
          sub={`${String(bk?.rebalance)} · ${int(s.rebalance_dates.length)} recorded`} /></Panel>
        <Panel><Metric hero label="Strategy ↔ gold" value={diag.strategy_gold_corr} format={(v) => ratio(v, 3)}
          sub="pooled realised correlation" /></Panel>
      </div>

      <div className="grid split-even">
        <Panel title="Capital orbit" sub="Outer ring is actual weight, inner ring is target. A drift shows as a gap." flush>
          <CapitalOrbit
            sleeves={rows.map((r): Sleeve => ({
              name: r.name, color: r.color, target: r.target,
              actual: r.actual / total, value: r.actual,
            }))}
            total={total}
            height={300}
            selected={sleeveSel}
            onSelect={setSleeveSel}
          />
          <div className="city-legend">
            {rows.map((r) => (
              <span className="legend-item" key={r.name}>
                <span className="legend-swatch block" style={{ background: r.color }} />
                {r.name} · <span className="num">{pct((r.actual / total) * 100, 1)}</span>
                <span className="dim"> / {pct(r.target * 100, 0)} target</span>
              </span>
            ))}
          </div>
        </Panel>
        <Panel title="Target vs actual" sub={`At ${s.dates[last]}`} flush>
          <DataTable cols={allocCols} rows={rows} rowKey={(r) => r.name}
            rowClass={(r) => (sleeveSel && sleeveSel !== r.name ? 'dimmed' : undefined)}
            foot={`Book value ${inr(total)}. Gold is never drawn on; the strategy may draw on arbitrage at T+1.`} />
        </Panel>
      </div>

      <Panel title="Sleeve allocation over time" sub="Share of total book. Rebalancing moves value between sleeves; sleeve returns are recorded separately from sleeve values.">
        <StackedArea
          dates={dDates}
          height={240}
          bands={[
            { label: 'Strategy', color: 'var(--cat-1)', values: pick(s.sleeve_strategy) },
            { label: 'Arbitrage', color: 'var(--cat-2)', values: pick(s.sleeve_arb) },
            { label: 'Gold', color: 'var(--cat-3)', values: pick(s.sleeve_gold) },
          ]}
        />
      </Panel>

      <div className="grid">
        <Panel
          title="Strategy deployment"
          sub={`Against its ${pct(Number(bk?.w_strategy) * 100, 0)} sleeve target. A diversifier that co-moves when the strategy struggles is not a diversifier, which is why correlation is reported per year below rather than pooled.`}
        >
          <LineChart
            dates={dDates}
            height={190}
            baseline={Number(bk?.w_strategy ?? 0.3) * 100}
            yFormat={(v) => `${fmt(v, 0)}%`}
            tipFormat={(v) => pct(v, 2)}
            series={[{ label: 'Deployment (% of book)', color: 'var(--cat-1)', values: pick(s.deployment_pct_of_book) }]}
          />
          <p className="note" style={{ marginTop: 'var(--s-2)' }}>
            The horizontal rule is the {pct(Number(bk?.w_strategy) * 100, 0)} sleeve target. Sessions
            above it are funded by drawing on arbitrage:{' '}
            <span className="num">{int(diag.borrow_days)}</span> borrow days (
            {pct(diag.borrow_day_pct, 2)}), peak{' '}
            <span className="num">{pct(diag.peak_borrow_pct_of_book, 2)}</span> of the book.
          </p>
        </Panel>
      </div>

      <div className="grid grid-2">
        <Panel title="Realised strategy ↔ gold correlation, per year" sub="Reported per year, not pooled" flush>
          <DataTable cols={corrCols} rows={frozen.gold_correlation_by_year} rowKey={(r, i) => String(r.year ?? i)} />
        </Panel>
        <Panel title="Rebalance history" sub={`${String(bk?.rebalance)} cadence`} flush>
          <DataTable
            cols={[
              { key: 'n', head: '#', num: true, render: (_r: string, i: number) => String(i + 1) },
              { key: 'd', head: 'Date', render: (r: string) => <span className="num">{r}</span> },
            ]}
            rows={s.rebalance_dates}
            rowKey={(r, i) => `${r}-${i}`}
            maxHeight={260}
            empty="No rebalance events recorded in this window."
          />
        </Panel>
      </div>

      <Callout>
        <strong>Deployment basis.</strong> Risk per trade is a fraction of the strategy sleeve, not
        of the whole book. Under the book basis the strategy borrows on 35–48% of sessions and takes
        essentially the entire arbitrage sleeve at its peak, which is a leveraged book rather than a{' '}
        {pct(Number(bk?.w_strategy) * 100, 0)} allocation.
      </Callout>
    </div>
  )
}
