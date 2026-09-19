import { useParams } from 'react-router-dom'
import { useAsync } from '@/lib/api'
import { Badge, Callout, Col, DataTable, ErrorBox, Loading, Panel, ScoreBars, Stat } from '@/components/ui'
import { DASH, decimate, fmt, inr, inrExact, int, pct, ratio } from '@/lib/format'

interface Plan {
  entry_ref: number
  stop: number
  stop_basis: string
  risk_per_share: number
  stop_pct: number
  qty: number
  notional: number
  risk_amount: number
  capped_by_notional: boolean
  tranches: { price: number; qty: number; trigger_pct: number }[]
  runner_qty: number
}

interface StockPayload {
  symbol: string
  company: string | null
  industry: string | null
  dates: string[]
  close: number[]
  high: number[]
  low: number[]
  atr14: number
  last_close: number
  in_latest_scan: boolean
  candidate: Record<string, unknown> | null
  box: { top: number | null; bottom: number | null; bars: number | null; range_pct: number | null; date: string | null } | null
  plan: Plan | null
  equity_basis: number
}

const SCORE_PARTS = ['sc_fast', 'sc_absorbed', 'sc_rs', 'sc_cycles'] as const

/** Price with the detected box drawn over it. The box is the structure the
 *  whole strategy is built on, so it is rendered as geometry on the chart
 *  rather than described in a caption. */
function PriceWithBox({ s }: { s: StockPayload }) {
  const W = 800
  const H = 300
  const M = { t: 12, r: 14, b: 24, l: 52 }
  const pts = decimate(s.dates.map((d, i) => ({ d, i })), 400, (o) => o.i)
  const idx = pts.map((p) => p.i)
  const close = idx.map((i) => s.close[i])
  const highs = idx.map((i) => s.high[i])
  const lows = idx.map((i) => s.low[i])

  const lo = Math.min(...lows, s.box?.bottom ?? Infinity)
  const hi = Math.max(...highs, s.box?.top ?? -Infinity)
  const pad = (hi - lo) * 0.06 || 1
  const y = (v: number) => M.t + (H - M.t - M.b) * (1 - (v - (lo - pad)) / ((hi + pad) - (lo - pad)))
  const x = (i: number) => M.l + ((W - M.l - M.r) * i) / Math.max(1, close.length - 1)

  const d = close.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join('')
  const ticks = 5
  const tickVals = Array.from({ length: ticks }, (_, k) => lo - pad + ((hi + pad - (lo - pad)) * k) / (ticks - 1))

  const boxTop = s.box?.top ?? null
  const boxBot = s.box?.bottom ?? null
  const boxStart = s.box?.bars ? Math.max(0, close.length - 1 - Number(s.box.bars)) : null

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height: H }}
      role="img" aria-label={`${s.symbol} price with detected box`}>
      {tickVals.map((t) => (
        <g key={t}>
          <line className="grid-line" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
          <text x={M.l - 6} y={y(t) + 3} textAnchor="end">{fmt(t, 0)}</text>
        </g>
      ))}

      {boxTop != null && boxBot != null && (
        <>
          <rect
            x={boxStart != null ? x(boxStart) : M.l}
            y={y(boxTop)}
            width={(W - M.r) - (boxStart != null ? x(boxStart) : M.l)}
            height={Math.max(1, y(boxBot) - y(boxTop))}
            fill="var(--cat-1)" fillOpacity={0.09}
            stroke="var(--cat-1)" strokeOpacity={0.45} strokeWidth={1}
          />
          <line x1={M.l} x2={W - M.r} y1={y(boxTop)} y2={y(boxTop)} stroke="var(--cat-1)" strokeWidth={1} strokeDasharray="4 3" />
          <line x1={M.l} x2={W - M.r} y1={y(boxBot)} y2={y(boxBot)} stroke="var(--loss)" strokeWidth={1} strokeDasharray="4 3" />
          <text x={W - M.r - 4} y={y(boxTop) - 4} textAnchor="end" style={{ fill: 'var(--cat-1)' }}>
            box top {fmt(boxTop, 2)}
          </text>
          <text x={W - M.r - 4} y={y(boxBot) + 11} textAnchor="end" style={{ fill: 'var(--loss)' }}>
            floor {fmt(boxBot, 2)}
          </text>
        </>
      )}

      <path className="series" d={d} stroke="var(--ink)" strokeWidth={1.6} />
      <line className="axis-line" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
      <text x={M.l} y={H - 7} textAnchor="start">{s.dates[0]}</text>
      <text x={W - M.r} y={H - 7} textAnchor="end">{s.dates[s.dates.length - 1]}</text>
    </svg>
  )
}

export default function StockDetail() {
  const { symbol = '' } = useParams()
  const q = useAsync<StockPayload>(
    () => fetch(`/api/stock/${encodeURIComponent(symbol)}`).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
      return r.json()
    }),
    [symbol],
  )

  if (q.loading) return <div className="page"><Loading what={symbol} /></div>
  if (q.error) return <div className="page"><ErrorBox error={q.error} /></div>
  const s = q.data!
  const c = s.candidate ?? {}
  const plan = s.plan

  const parts = SCORE_PARTS.map((k) => {
    const v = c[k] as unknown
    return typeof v === 'number' ? v : null
  })

  const scoreRows = SCORE_PARTS.map((k, i) => ({
    name: ({ sc_fast: 'fast resolution', sc_absorbed: 'resistance absorbed', sc_rs: 'RS capture', sc_cycles: 'prior cycles' } as Record<string, string>)[k] ?? k,
    value: parts[i],
    weight: 0.25,
    contribution: parts[i] != null ? parts[i]! * 0.25 : null,
  }))

  const scoreCols: Col<(typeof scoreRows)[number]>[] = [
    { key: 'n', head: 'Component', render: (r) => r.name },
    { key: 'v', head: 'Score', num: true, render: (r) => ratio(r.value, 3) },
    { key: 'w', head: 'Weight', num: true, render: (r) => ratio(r.weight, 2) },
    { key: 'c', head: 'Contribution', num: true, render: (r) => ratio(r.contribution, 4) },
  ]

  const tranCols: Col<Plan['tranches'][number]>[] = [
    { key: 'i', head: 'Tranche', render: (_t, i) => `#${i + 1}` },
    { key: 'trig', head: 'Trigger', num: true, render: (t) => `+${pct(t.trigger_pct, 0)}` },
    { key: 'px', head: 'Price', num: true, render: (t) => inrExact(t.price, 2) },
    { key: 'q', head: 'Qty', num: true, render: (t) => int(t.qty) },
    { key: 'val', head: 'Proceeds', num: true, render: (t) => inr(t.price * t.qty) },
  ]

  return (
    <div className="page">
      <div className="page-head">
        <div className="row-between">
          <div>
            <h1>{s.symbol}</h1>
            <p className="note">{s.company ?? DASH}{s.industry ? ` · ${s.industry}` : ''}</p>
          </div>
          <div className="row">
            {s.in_latest_scan ? (
              <Badge tone={c.bucket === 'high_conviction' ? 'accent' : 'default'}>
                {c.bucket === 'high_conviction' ? 'HIGH CONVICTION' : 'WAIT & WATCH'}
              </Badge>
            ) : (
              <Badge>NOT IN LATEST SCAN</Badge>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-4">
        <Panel><Stat label="Last close" value={inrExact(s.last_close, 2)} /></Panel>
        <Panel><Stat label="ATR(14)" value={fmt(s.atr14, 2)} sub={`${pct((s.atr14 / s.last_close) * 100, 2)} of price`} /></Panel>
        <Panel><Stat label="Score" value={ratio(c.score as number, 3)} sub={<ScoreBars parts={parts} />} /></Panel>
        <Panel><Stat label="Box range" value={s.box?.range_pct != null ? pct(Number(s.box.range_pct), 1) : DASH}
          sub={s.box?.bars != null ? `${int(Number(s.box.bars))} bars` : undefined} /></Panel>
      </div>

      <Panel title="Price and detected box" sub="The box top is the breakout level; the floor is the structural invalidation.">
        <PriceWithBox s={s} />
      </Panel>

      <div className="grid grid-2">
        <Panel title="Score breakdown" sub="Equal weight, 0.25 × 4. The calibrated weights are rejected — three of four factors were selected on the sample they score." flush>
          <DataTable cols={scoreCols} rows={scoreRows} rowKey={(r) => r.name}
            foot={`Total ${ratio(c.score as number, 3)} · bucket ${String(c.bucket ?? DASH)}`} />
        </Panel>

        <Panel
          title="Trade plan"
          sub={`Sized at ${pct(3, 0)} risk on ${inr(s.equity_basis)} equity, under the frozen parameters`}
        >
          {plan ? (
            <>
              <dl className="kv">
                <dt>Entry reference</dt><dd className="num">{inrExact(plan.entry_ref, 2)}</dd>
                <dt>Stop</dt><dd className="num loss">{inrExact(plan.stop, 2)} <span className="dim">({plan.stop_basis})</span></dd>
                <dt>Stop distance</dt><dd className="num">{pct(plan.stop_pct, 2)}</dd>
                <dt>Risk per share</dt><dd className="num">{inrExact(plan.risk_per_share, 2)}</dd>
                <dt>Quantity</dt><dd className="num">{int(plan.qty)}</dd>
                <dt>Notional</dt><dd className="num">{inr(plan.notional)}</dd>
                <dt>Risk amount</dt><dd className="num">{inr(plan.risk_amount)}</dd>
                <dt>Runner qty</dt><dd className="num">{int(plan.runner_qty)}</dd>
              </dl>
              <hr className="hairline" style={{ margin: 'var(--s-4) 0 var(--s-3)' }} />
              <DataTable cols={tranCols} rows={plan.tranches} rowKey={(_t, i) => String(i)}
                foot={`${int(plan.runner_qty)} shares trail on the ATR stop after the last tranche.`} />
              {plan.capped_by_notional && (
                <p className="note" style={{ marginTop: 'var(--s-2)', color: 'var(--warn)' }}>
                  Size was capped by the notional limit rather than the risk budget.
                </p>
              )}
            </>
          ) : (
            <div className="empty">
              No trade plan — this name is not in the latest scan, so there is no box floor to size
              against.
            </div>
          )}
        </Panel>
      </div>

      <Callout>
        <strong>Research tooling, not investment advice.</strong> Levels shown are what the frozen
        configuration would have used. Both backtest windows are discovery; the forward record is
        the only out-of-sample evidence.
      </Callout>
    </div>
  )
}
