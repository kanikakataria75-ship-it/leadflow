import { Link } from 'react-router-dom'
import { api, useAsync } from '@/lib/api'
import { Badge, Callout, Col, DataTable, ErrorBox, Loading, Panel, Stat } from '@/components/ui'
import { Histogram, Sparkline } from '@/components/charts/Charts'
import { Metric } from '@/components/terminal'
import { DASH, fmt, int, pct, ratio, shortDate } from '@/lib/format'

interface Entry {
  scan_date?: string
  symbol?: string
  bucket?: string
  score?: number
  status?: string
  entry_date?: string
  entry_price?: number
  exit_date?: string
  exit_price?: number
  r_multiple?: number
  ret_pct?: number
  bars_held?: number
  exit_reason?: string
  [k: string]: unknown
}

export default function ForwardRecord() {
  const rec = useAsync(() => api.record(), [])
  const cfg = useAsync(() => api.config(), [])
  const hist = useAsync(() => api.scanHistory(), [])

  if (rec.loading) return <div className="page"><Loading what="forward record" /></div>
  if (rec.error) return <div className="page"><ErrorBox error={rec.error} /></div>

  const r = rec.data!
  const entries = (r.entries ?? []) as Entry[]
  const closed = entries.filter((e) => e.status === 'closed')
  const rs = closed.map((e) => (typeof e.r_multiple === 'number' ? e.r_multiple : null))

  const scans = (hist.data ?? []) as Record<string, unknown>[]
  const candSeries = scans.slice().reverse().map((s) => Number(s.candidate_count ?? 0))

  const cols: Col<Entry>[] = [
    { key: 'scan', head: 'Flagged', render: (e) => <span className="num">{e.scan_date ? shortDate(e.scan_date) : DASH}</span> },
    {
      key: 'sym', head: 'Symbol',
      render: (e) => e.symbol ? <Link className="cell-link cell-sym" to={`/stock/${encodeURIComponent(e.symbol)}`}>{e.symbol}</Link> : DASH,
    },
    {
      key: 'conv', head: 'Conviction',
      render: (e) => <Badge tone={e.bucket === 'high_conviction' ? 'accent' : 'default'}>{e.bucket === 'high_conviction' ? 'HIGH' : 'WAIT'}</Badge>,
    },
    { key: 'score', head: 'Score', num: true, render: (e) => ratio(e.score, 3) },
    {
      key: 'status', head: 'Status',
      render: (e) => <Badge tone={e.status === 'closed' ? 'default' : 'gain'}>{String(e.status ?? 'open').toUpperCase()}</Badge>,
    },
    { key: 'entry', head: 'Entry', num: true, render: (e) => fmt(e.entry_price, 2) },
    { key: 'exit', head: 'Exit', num: true, render: (e) => fmt(e.exit_price, 2) },
    { key: 'bars', head: 'Bars', num: true, render: (e) => int(e.bars_held) },
    {
      key: 'ret', head: 'Return', num: true,
      render: (e) => <span className={Number(e.ret_pct) >= 0 ? 'gain' : 'loss'}>{pct(e.ret_pct, 2)}</span>,
    },
    {
      key: 'r', head: 'R', num: true,
      render: (e) => <span className={Number(e.r_multiple) >= 0 ? 'gain' : 'loss'}>{ratio(e.r_multiple, 3)}</span>,
    },
    { key: 'why', head: 'Exit reason', render: (e) => <span className="muted">{String(e.exit_reason ?? DASH)}</span> },
  ]

  return (
    <div className="page">
      <div className="page-head">
        <div className="row-between">
          <h1>Forward Record</h1>
          <div className="row">
            <Badge tone="accent">SINCE {r.frozen_on}</Badge>
            <span className="note">day <span className="num">{int(cfg.data?.forward_record_days)}</span></span>
          </div>
        </div>
        <p className="note">
          Every scan&apos;s output over time: what was flagged, at what score, and what happened
          next. This is the only genuinely out-of-sample evidence this product has — both backtest
          windows are discovery.
        </p>
      </div>

      <Panel title="Experiment stages" sub="Only stages with recorded events are lit. Nothing is shown as complete that has not happened.">
        <div className="exp-rail">
          {[
            { k: 'Scan', n: scans.length },
            { k: 'Signal', n: entries.length },
            { k: 'Entry', n: entries.filter((e) => e.entry_date).length },
            { k: 'Position', n: r.n_open },
            { k: 'Exit', n: r.n_closed },
          ].map((st, i, arr) => (
            <div className="exp-step" key={st.k}>
              <div className="row" style={{ width: '100%', gap: 0, alignItems: 'center', justifyContent: 'center' }}>
                {i > 0 && <span className="exp-conn" />}
                <span className={st.n > 0 ? 'exp-dot on' : 'exp-dot'} style={{ margin: '0 6px' }} />
                {i < arr.length - 1 && <span className="exp-conn" />}
              </div>
              <div className="stat-label">{st.k}</div>
              <div className="num" style={{ fontSize: 'var(--t-md)', color: st.n > 0 ? 'var(--ink)' : 'var(--ink-4)' }}>
                {int(st.n)}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <Callout>
        <strong>The record begins at {r.frozen_on}.</strong> {r.note} If any frozen value changes,
        this record restarts: the period before a change and the period after it are not the same
        experiment, and splicing them would produce a number that describes neither.
      </Callout>

      <div className="grid grid-4">
        <Panel><Metric hero label="Entries" value={r.n_total} format={(v) => int(v)}
          sub={`${int(r.n_open)} open · ${int(r.n_closed)} closed`} /></Panel>
        <Panel>
          <Stat label="Win rate" value={r.win_rate_pct != null ? pct(r.win_rate_pct, 1) : DASH}
            tone={r.win_rate_pct != null ? r.win_rate_pct - 50 : undefined}
            sub={r.n_closed ? `on ${int(r.n_closed)} closed` : 'awaiting first close'} />
        </Panel>
        <Panel>
          <Stat label="Mean R" value={r.mean_r != null ? ratio(r.mean_r, 3) : DASH} tone={r.mean_r ?? undefined}
            sub="position level" />
        </Panel>
        <Panel>
          <Stat label="Backtest reference" value="0.20" sub="position R, 2022-2026 (discovery)" />
        </Panel>
      </div>

      {r.n_total === 0 ? (
        <Panel title="Live experiment">
          <div className="empty" style={{ padding: 'var(--s-16) var(--s-4)' }}>
            <div className="row" style={{ justifyContent: 'center', gap: 'var(--s-2)', marginBottom: 'var(--s-5)' }}>
              <span className="pulse" />
              <span className="eyebrow" style={{ color: 'var(--gain)' }}>Recording</span>
            </div>
            <div style={{ fontSize: 'var(--t-2xl)', fontWeight: 600, letterSpacing: 'var(--ls-tightest)', color: 'var(--ink)', marginBottom: 'var(--s-2)' }}>
              No entries yet
            </div>
            <div className="num" style={{ fontSize: 'var(--t-sm)', color: 'var(--accent-bright)', marginBottom: 'var(--s-4)' }}>
              Recording begins {r.frozen_on}
            </div>
            <div className="note" style={{ maxWidth: 520, margin: '0 auto' }}>
              The configuration was frozen on <span className="num">{r.frozen_on}</span> and the
              forward record starts from that date. Entries appear here as scans flag names and
              those names resolve. Nothing from the previous engine is imported — it was a different
              strategy, and carrying its record forward would misdescribe both.
            </div>
          </div>
        </Panel>
      ) : (
        <>
          <Panel title="Every flagged name" sub={`${int(entries.length)} entries since ${r.frozen_on}`} flush>
            <DataTable cols={cols} rows={entries} rowKey={(e, i) => `${e.symbol ?? 'x'}-${e.scan_date ?? i}`} zebra maxHeight={620} />
          </Panel>

          {closed.length > 0 && (
            <div className="grid grid-2">
              <Panel title="R distribution" sub={`${int(closed.length)} closed positions`}>
                <Histogram values={rs} bins={Math.min(24, Math.max(6, closed.length))} height={190} unit="R" colorBySign />
              </Panel>
              <Panel title="Running win rate" sub="Cumulative, as positions close">
                <div className="empty note" style={{ padding: 'var(--s-6)' }}>
                  Needs at least a handful of closed positions before a running rate says anything.
                </div>
              </Panel>
            </div>
          )}
        </>
      )}

      <Panel title="Scan history" sub="Candidate count per scan since the freeze">
        {scans.length ? (
          <div className="row" style={{ gap: 'var(--s-6)', alignItems: 'center' }}>
            <Sparkline values={candSeries} width={320} height={44} />
            <dl className="kv" style={{ minWidth: 220 }}>
              <dt>Scans recorded</dt><dd className="num">{int(scans.length)}</dd>
              <dt>Latest candidates</dt><dd className="num">{int(scans[0]?.candidate_count as number)}</dd>
              <dt>Latest breadth</dt><dd className="num">{scans[0]?.breadth != null ? pct(Number(scans[0].breadth), 1) : DASH}</dd>
            </dl>
          </div>
        ) : (
          <div className="empty">No scans recorded yet.</div>
        )}
      </Panel>
    </div>
  )
}
