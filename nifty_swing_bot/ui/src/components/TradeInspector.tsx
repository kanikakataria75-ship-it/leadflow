/** Trade inspector: positions, their event timeline, and the high-conviction losses.
 *
 * Everything here is a regrouping of the trade rows the engine emitted. A
 * position is the set of rows sharing (symbol, entry_date) -- the same
 * collapse `positions_from_rows` does server-side -- and each row becomes one
 * timeline event labelled with its own `exit_reason`. A ladder node appears
 * only when that ladder row exists. Nothing is interpolated, and no event is
 * invented to make a timeline look complete.
 */

import { useMemo, useState } from 'react'
import { Badge, Col, DataTable, Panel, Segmented } from '@/components/ui'
import { TradeTimeline, type TradeEvent } from '@/components/terminal'
import { dirClass, fmt, int, pct, ratio } from '@/lib/format'

interface Row {
  symbol: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  qty: number
  net_pnl: number
  r_multiple: number
  exit_reason: string
  bars_held: number
  score: number
  bucket: string
}

interface Position {
  key: string
  symbol: string
  entry_date: string
  entry_price: number
  bucket: string
  score: number
  qty: number
  net_pnl: number
  r_multiple: number
  bars_held: number
  last_exit: string
  reasons: string[]
  rows: Row[]
}

const TONE: Record<string, TradeEvent['tone']> = {
  ladder1: 'gain',
  ladder2: 'gain',
  ladder3: 'gain',
  atr_trail_stop: 'loss',
  big_box_bottom_stop: 'loss',
  time_stop: 'warn',
  max_hold: 'warn',
  end_of_data: 'warn',
}

const LABEL: Record<string, string> = {
  ladder1: 'Tranche 1',
  ladder2: 'Tranche 2',
  ladder3: 'Tranche 3',
  atr_trail_stop: 'Trailing stop',
  big_box_bottom_stop: 'Box floor stop',
  time_stop: 'Time stop',
  max_hold: 'Hold cap',
  end_of_data: 'End of data',
}

function group(rows: Row[]): Position[] {
  const m = new Map<string, Row[]>()
  for (const r of rows) {
    const k = `${r.symbol}|${r.entry_date}`
    if (!m.has(k)) m.set(k, [])
    m.get(k)!.push(r)
  }
  return [...m.entries()].map(([key, rs]) => {
    const sorted = [...rs].sort((a, b) => a.exit_date.localeCompare(b.exit_date))
    const first = sorted[0]
    return {
      key,
      symbol: first.symbol,
      entry_date: first.entry_date,
      entry_price: first.entry_price,
      bucket: first.bucket,
      score: first.score,
      qty: sorted.reduce((a, r) => a + r.qty, 0),
      net_pnl: sorted.reduce((a, r) => a + r.net_pnl, 0),
      // R at the POSITION level: the engine's per-row R weighted by the
      // quantity each row closed. Never the unweighted row mean, which a
      // profit ladder inflates by construction.
      r_multiple:
        sorted.reduce((a, r) => a + r.r_multiple * r.qty, 0) /
        Math.max(1, sorted.reduce((a, r) => a + r.qty, 0)),
      bars_held: Math.max(...sorted.map((r) => r.bars_held)),
      last_exit: sorted[sorted.length - 1].exit_date,
      reasons: sorted.map((r) => r.exit_reason),
      rows: sorted,
    }
  })
}

function eventsOf(p: Position): TradeEvent[] {
  const out: TradeEvent[] = [
    {
      label: 'Entry',
      date: p.entry_date,
      price: p.entry_price,
      qty: p.qty,
      tone: 'accent',
      detail: { bucket: p.bucket === 'high_conviction' ? 'high conviction' : 'wait & watch', score: ratio(p.score, 3) },
    },
  ]
  for (const r of p.rows) {
    out.push({
      label: LABEL[r.exit_reason] ?? r.exit_reason,
      date: r.exit_date,
      price: r.exit_price,
      qty: r.qty,
      pnl: r.net_pnl,
      tone: TONE[r.exit_reason] ?? 'accent',
      detail: { 'exit reason': r.exit_reason, 'bars held': int(r.bars_held), R: ratio(r.r_multiple, 3) },
    })
  }
  return out
}

const VIEWS = ['All positions', 'High-conviction losses'] as const
type View = (typeof VIEWS)[number]

export function TradeInspector({ trades }: { trades: Record<string, unknown>[] }) {
  const [view, setView] = useState<View>('All positions')
  const [open, setOpen] = useState<string | null>(null)

  const positions = useMemo(() => group(trades as unknown as Row[]), [trades])
  const hcLosses = useMemo(
    () => positions.filter((p) => p.bucket === 'high_conviction' && p.net_pnl < 0),
    [positions],
  )
  const rows = view === 'All positions' ? positions : hcLosses
  const sorted = useMemo(
    () => [...rows].sort((a, b) => (view === 'All positions' ? b.net_pnl - a.net_pnl : a.net_pnl - b.net_pnl)),
    [rows, view],
  )
  const sel = open ? positions.find((p) => p.key === open) ?? null : null

  const cols: Col<Position>[] = [
    { key: 'sym', head: 'Symbol', render: (p) => <span className="cell-sym">{p.symbol}</span> },
    { key: 'entry', head: 'Entry', render: (p) => <span className="num">{p.entry_date}</span> },
    { key: 'exit', head: 'Exit', render: (p) => <span className="num">{p.last_exit}</span> },
    {
      key: 'conv', head: 'Conviction',
      render: (p) => (
        <Badge tone={p.bucket === 'high_conviction' ? 'accent' : 'warn'}>
          {p.bucket === 'high_conviction' ? 'High' : 'Wait'}
        </Badge>
      ),
    },
    { key: 'score', head: 'Score', num: true, render: (p) => ratio(p.score, 3) },
    { key: 'bars', head: 'Bars', num: true, render: (p) => int(p.bars_held) },
    { key: 'legs', head: 'Legs', num: true, render: (p) => int(p.rows.length) },
    {
      key: 'pnl', head: 'Net P&L', num: true,
      render: (p) => <span className={dirClass(p.net_pnl)}>{fmt(p.net_pnl, 0)}</span>,
    },
    {
      key: 'r', head: 'R', num: true,
      render: (p) => <span className={dirClass(p.r_multiple)}>{ratio(p.r_multiple, 3)}</span>,
    },
    { key: 'why', head: 'Exit path', render: (p) => <span className="note">{p.reasons.join(' → ')}</span> },
  ]

  return (
    <Panel
      title="Trade inspector"
      sub={
        view === 'All positions'
          ? `${int(positions.length)} positions, ladder rows collapsed · click one for its timeline`
          : `${int(hcLosses.length)} positions the engine scored high conviction that still lost`
      }
      right={<Segmented options={VIEWS} value={view} onChange={setView} />}
      flush
    >
      {view === 'High-conviction losses' && (
        <div style={{ padding: 'var(--s-3) var(--s-4) 0' }}>
          <div className="callout">
            <strong>Audit view, not a strategy feature.</strong> These are positions that carried the
            engine&apos;s highest conviction at entry and still closed negative. Entry reasoning, exit
            reason and R are the engine&apos;s own record.
          </div>
        </div>
      )}

      <DataTable
        cols={cols}
        rows={sorted}
        rowKey={(p) => p.key}
        maxHeight={380}
        zebra
        rowClass={(p) => (open === p.key ? 'clickable expanded' : 'clickable')}
        empty={view === 'All positions' ? 'No positions in this window.' : 'No high-conviction losses in this window.'}
      />

      <div style={{ padding: 'var(--s-2) var(--s-4) var(--s-4)' }}>
        <div className="row" style={{ gap: 'var(--s-2)', marginBottom: 'var(--s-3)' }}>
          <span className="note">Inspect:</span>
          {sorted.slice(0, 12).map((p) => (
            <button
              key={p.key}
              className="input"
              style={{
                cursor: 'pointer',
                borderColor: open === p.key ? 'var(--accent-line)' : undefined,
                color: open === p.key ? 'var(--accent-bright)' : undefined,
              }}
              onClick={() => setOpen(open === p.key ? null : p.key)}
            >
              {p.symbol}
            </button>
          ))}
        </div>

        {sel ? (
          <>
            <div className="row-between" style={{ marginBottom: 'var(--s-2)' }}>
              <div className="row" style={{ gap: 'var(--s-3)' }}>
                <strong style={{ fontSize: 'var(--t-md)' }}>{sel.symbol}</strong>
                <Badge tone={sel.bucket === 'high_conviction' ? 'accent' : 'warn'}>
                  {sel.bucket === 'high_conviction' ? 'High conviction' : 'Wait & watch'}
                </Badge>
                <span className="note">score <span className="num">{ratio(sel.score, 3)}</span></span>
              </div>
              <div className="row" style={{ gap: 'var(--s-4)' }}>
                <span className="note">
                  Net <span className={`num ${dirClass(sel.net_pnl)}`}>{fmt(sel.net_pnl, 0)}</span>
                </span>
                <span className="note">
                  R <span className={`num ${dirClass(sel.r_multiple)}`}>{ratio(sel.r_multiple, 3)}</span>
                </span>
                <span className="note">
                  Return{' '}
                  <span className={`num ${dirClass(sel.rows[sel.rows.length - 1].exit_price - sel.entry_price)}`}>
                    {pct((sel.rows[sel.rows.length - 1].exit_price / sel.entry_price - 1) * 100, 2)}
                  </span>
                </span>
              </div>
            </div>
            <TradeTimeline events={eventsOf(sel)} />
          </>
        ) : (
          <p className="note">Pick a symbol above to see the events that actually fired on that position.</p>
        )}
      </div>
    </Panel>
  )
}

export default TradeInspector
