import { Fragment, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, useAsync, type Candidate } from '@/lib/api'
import { Badge, Callout, Col, DataTable, Empty, ErrorBox, Loading, Panel, ScoreBars } from '@/components/ui'
import { Funnel, RegimeStrip, SignalPanel } from '@/components/terminal'
import { DASH, fmt, int, pct, ratio } from '@/lib/format'

const SCORE_PARTS = ['sc_fast', 'sc_absorbed', 'sc_rs', 'sc_cycles'] as const
const PART_LABELS: Record<string, string> = {
  sc_fast: 'Fast resolution',
  sc_absorbed: 'Resistance absorption',
  sc_rs: 'Relative strength',
  sc_cycles: 'Prior cycles',
}

function partsOf(c: Candidate): (number | null)[] {
  return SCORE_PARTS.map((k) => {
    const v = c[k] as unknown
    return typeof v === 'number' ? v : null
  })
}

interface Rejected {
  symbol?: string
  stage?: string
  reason?: string
  industry?: string
  score?: number
  [k: string]: unknown
}

export interface ScanJob {
  running: boolean
  error: string | null
  finishedAt: string | null
  start: () => void
}

export default function Scanner({ job }: { job?: ScanJob }) {
  const scan = useAsync(() => api.scan(), [job?.finishedAt ?? 'initial'])
  const hist = useAsync(() => api.scanHistory(), [job?.finishedAt ?? 'initial'])
  const [q, setQ] = useState('')
  const [open, setOpen] = useState<number | null>(null)
  const [stage, setStage] = useState<string | null>(null)

  const latest = (hist.data?.[0] ?? {}) as Record<string, unknown>
  const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

  const candidates = useMemo(() => {
    const rows = scan.data?.candidates ?? []
    if (!q.trim()) return rows
    const t = q.trim().toLowerCase()
    return rows.filter(
      (c) =>
        c.symbol.toLowerCase().includes(t) ||
        (c.industry ?? '').toLowerCase().includes(t) ||
        (c.company ?? '').toLowerCase().includes(t),
    )
  }, [scan.data, q])

  const rejected = (scan.data?.rejected ?? []) as Rejected[]

  /** Group rejections by the stage that stopped them — the engine's own
   *  `stage` field, never inferred. */
  const byStage = useMemo(() => {
    const m = new Map<string, Rejected[]>()
    for (const r of rejected) {
      const k = String(r.stage ?? 'unknown')
      if (!m.has(k)) m.set(k, [])
      m.get(k)!.push(r)
    }
    return [...m.entries()].sort((a, b) => b[1].length - a[1].length)
  }, [rejected])

  const shownRejects = stage ? rejected.filter((r) => String(r.stage ?? 'unknown') === stage) : rejected

  if (scan.loading) return <div className="page"><Loading what="scan" /></div>
  if (scan.error) return <div className="page"><ErrorBox error={scan.error} /></div>

  const cols: Col<Candidate>[] = [
    { key: 'rank', head: '#', num: true, width: 44, render: (_c, i) => <span className="dim">{i + 1}</span> },
    {
      key: 'sym', head: 'Symbol', width: 130,
      render: (c) => <Link className="cell-link cell-sym" to={`/stock/${encodeURIComponent(c.symbol)}`} onClick={(e) => e.stopPropagation()}>{c.symbol}</Link>,
    },
    { key: 'co', head: 'Company', render: (c) => <span className="muted">{c.company ?? DASH}</span> },
    { key: 'ind', head: 'Sector', render: (c) => <span className="muted">{c.industry ?? DASH}</span> },
    {
      key: 'bucket', head: 'Conviction',
      render: (c) => (
        <Badge tone={c.bucket === 'high_conviction' ? 'accent' : 'warn'}>
          {c.bucket === 'high_conviction' ? 'High' : 'Wait'}
        </Badge>
      ),
    },
    { key: 'score', head: 'Score', num: true, render: (c) => ratio(c.score, 3) },
    {
      key: 'parts', head: 'Components',
      title: 'fast resolution / absorption / relative strength / prior cycles — equal weight 0.25 × 4',
      render: (c) => <ScoreBars parts={partsOf(c)} />,
    },
    { key: 'mode', head: 'Entry', num: true, render: (c) => (c.entry_mode != null ? `M${c.entry_mode}` : DASH) },
    { key: 'gap', head: 'Gap d', num: true, render: (c) => int(c.admission_gap_days as number) },
    { key: 'top', head: 'Box top', num: true, render: (c) => fmt(c.box_top as number, 2) },
    { key: 'bot', head: 'Box floor', num: true, render: (c) => fmt(c.box_bottom as number, 2) },
    { key: 'bars', head: 'Bars', num: true, render: (c) => int(c.box_bars as number) },
    { key: 'range', head: 'Range %', num: true, render: (c) => pct(c.box_range_pct as number, 1) },
    {
      key: 'chart', head: 'Box',
      render: (c) =>
        scan.data?.scan_date ? (
          <a className="cell-link note" onClick={(e) => e.stopPropagation()}
            href={`/api/aes/chart/${scan.data.scan_date}/${encodeURIComponent(c.symbol)}`} target="_blank" rel="noreferrer">
            render →
          </a>
        ) : DASH,
    },
  ]

  return (
    <div className="page">
      <div className="page-head reveal">
        <div className="row-between">
          <div>
            <div className="eyebrow" style={{ marginBottom: 6 }}>Market scan</div>
            <h1 style={{ fontSize: 'var(--t-3xl)' }}>Radar</h1>
          </div>
          <div className="row" style={{ gap: 'var(--s-4)' }}>
            <span className="row" style={{ gap: 6 }}>
              <span className="pulse" />
              <span className="eyebrow">{scan.data?.scan_date ?? DASH}</span>
            </span>
            {job && (
              <button
                className="enter-btn"
                style={{ padding: '8px var(--s-5)', fontSize: 'var(--t-xs)' }}
                onClick={job.start}
                disabled={job.running}
                aria-busy={job.running}
              >
                {job.running ? (
                  <>
                    <span className="pulse" style={{ background: 'var(--accent)' }} />
                    Scanning…
                  </>
                ) : (
                  <>Run new scan <span aria-hidden="true">→</span></>
                )}
              </button>
            )}
          </div>
        </div>
        {job?.error && (
          <div className="callout" style={{ borderColor: 'rgba(224,85,97,0.35)', background: 'var(--loss-dim)' }}>
            <strong style={{ color: 'var(--loss)' }}>Scan could not start.</strong> {job.error}
          </div>
        )}
        {job?.running && (
          <div className="callout">
            <strong>Scan running.</strong> This is the scanner&apos;s own background job — the same
            one the evening tool runs. The table below still shows the last completed scan and will
            refresh when this one finishes.
          </div>
        )}
      </div>

      <Panel flush>
        <Funnel
          stages={[
            { label: 'Stocks scanned', value: num(latest.universe_size) },
            { label: 'Watching', value: num(latest.watchlist_size) },
            { label: 'Candidates', value: num(latest.candidate_count) },
            { label: 'Actionable', value: num(latest.actionable_count) },
          ]}
        />
      </Panel>

      <div className="grid split-even">
        <Panel title="Regime context"><RegimeStrip
          breadth={num(latest.breadth)}
          label={typeof latest.breadth_label === 'string' ? latest.breadth_label : null}
          pctile={num(latest.breadth_pctile)} /></Panel>
        <Panel title="Gate attrition" sub="Where names dropped out, by the stage that stopped them" flush>
          {byStage.length ? (
            <div style={{ padding: 'var(--s-4)', display: 'flex', flexDirection: 'column', gap: 'var(--s-3)' }}>
              {byStage.map(([k, rows]) => {
                const share = rows.length / Math.max(1, rejected.length)
                const on = stage === k
                return (
                  <button key={k} onClick={() => setStage(on ? null : k)}
                    style={{ background: 'transparent', border: 0, padding: 0, cursor: 'pointer', textAlign: 'left', color: 'inherit', font: 'inherit' }}>
                    <div className="row-between" style={{ marginBottom: 4 }}>
                      <span style={{ fontSize: 'var(--t-sm)', color: on ? 'var(--accent-bright)' : 'var(--ink-2)' }}>{k}</span>
                      <span className="num note">{int(rows.length)}</span>
                    </div>
                    <div className="meter"><div className="meter-fill" style={{ width: `${Math.max(share * 100, 2)}%` }} /></div>
                  </button>
                )
              })}
            </div>
          ) : (
            <Empty>No audit rows recorded for this scan.</Empty>
          )}
        </Panel>
      </div>

      <Callout>
        <strong>Breadth is context, not a gate.</strong> It groups and describes the regime; it was
        measured and found not to be tradeable as a filter, so nothing in the pipeline acts on it.
      </Callout>

      <Panel
        title="Ranked candidates"
        sub={`${int(candidates.length)} shown · click a row to decompose its score`}
        right={
          <input className="input" value={q} onChange={(e) => setQ(e.target.value)}
            placeholder="Filter symbol, company or sector…" style={{ width: 230 }} />
        }
        flush
      >
        {candidates.length ? (
          <div className="table-wrap" style={{ ['--table-max' as string]: '620px' } as React.CSSProperties}>
            <table className="data zebra">
              <thead>
                <tr>{cols.map((c) => <th key={c.key} className={c.num ? 'num' : undefined} title={c.title}>{c.head}</th>)}</tr>
              </thead>
              <tbody>
                {candidates.map((c, i) => (
                  <Fragment key={c.id}>
                    <tr
                      className={`clickable ${open === c.id ? 'expanded' : ''} ${c.actionable ? '' : 'dimmed'}`}
                      onClick={() => setOpen(open === c.id ? null : c.id)}
                    >
                      {cols.map((col) => (
                        <td key={col.key} className={col.num ? 'num' : undefined}>{col.render(c, i)}</td>
                      ))}
                    </tr>
                    {open === c.id && (
                      <tr>
                        <td colSpan={cols.length} style={{ background: 'var(--surface-2)', padding: 'var(--s-4)' }}>
                          <div className="grid grid-2" style={{ gap: 'var(--s-5)' }}>
                            <SignalPanel
                              components={SCORE_PARTS.map((k) => ({
                                key: k, label: PART_LABELS[k],
                                value: typeof c[k] === 'number' ? (c[k] as number) : null,
                                weight: 0.25,
                              }))}
                              total={c.score}
                              bucket={c.bucket}
                            />
                            <div>
                              <div className="eyebrow" style={{ marginBottom: 'var(--s-2)' }}>Structure</div>
                              <dl className="kv">
                                <dt>Box floor</dt><dd className="num">{fmt(c.box_bottom as number, 2)}</dd>
                                <dt>Box top</dt><dd className="num">{fmt(c.box_top as number, 2)}</dd>
                                <dt>Range</dt><dd className="num">{pct(c.box_range_pct as number, 2)}</dd>
                                <dt>Bars</dt><dd className="num">{int(c.box_bars as number)}</dd>
                                <dt>Admission gap</dt><dd className="num">{int(c.admission_gap_days as number)} d</dd>
                                <dt>Actionable</dt><dd>{c.actionable ? 'yes' : 'no'}</dd>
                              </dl>
                              <Link className="cell-link note" to={`/stock/${encodeURIComponent(c.symbol)}`} style={{ marginTop: 'var(--s-3)', display: 'inline-block' }}>
                                Open full detail →
                              </Link>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
              <tfoot>
                <tr><td colSpan={cols.length}>
                  Dimmed rows passed every gate but are not actionable today. Box renders open the detector&apos;s own chart for that scan.
                </td></tr>
              </tfoot>
            </table>
          </div>
        ) : (
          <Empty>No candidates in this scan.</Empty>
        )}
      </Panel>

      <Panel
        title="Gate failures"
        sub={stage ? `Filtered to “${stage}” — click the stage again to clear` : 'Names the pipeline looked at and rejected, with the stage and reason it recorded'}
        right={stage && <button className="input" style={{ cursor: 'pointer' }} onClick={() => setStage(null)}>Clear filter</button>}
        flush
      >
        {shownRejects.length ? (
          <DataTable
            cols={[
              { key: 'sym', head: 'Symbol', render: (r: Rejected) => <span className="cell-sym">{r.symbol ?? DASH}</span> },
              { key: 'ind', head: 'Sector', render: (r) => <span className="muted">{r.industry ?? DASH}</span> },
              { key: 'stage', head: 'Failed at', render: (r) => <Badge tone="warn">{String(r.stage ?? 'unknown')}</Badge> },
              { key: 'reason', head: 'Reason', render: (r) => <span className="muted">{String(r.reason ?? DASH)}</span> },
              { key: 'score', head: 'Score', num: true, render: (r) => (typeof r.score === 'number' ? ratio(r.score, 3) : DASH) },
            ]}
            rows={shownRejects}
            rowKey={(r, i) => `${r.symbol ?? 'x'}-${i}`}
            maxHeight={420}
            zebra
            foot="Rejection reasons are the engine's own audit rows. Nothing here is inferred."
          />
        ) : (
          <Empty>No audit rows recorded for this scan.</Empty>
        )}
      </Panel>
    </div>
  )
}
