/** Terminal-specific composite components.
 *
 * Every one of these renders values it is handed. None computes a score, a
 * return, a threshold or an outcome -- the arithmetic all happened in the
 * engine, and these only decide where a number sits and how it is labelled.
 */

import { useState, type ReactNode } from 'react'
import { useCountUp, stagger } from '@/lib/motion'
import { DASH, dirClass, fmt, int, pct, ratio, type Num } from '@/lib/format'

/* ======================================================================= */
/* Metric - a headline number with an optional count-up                     */
/* ======================================================================= */
export function Metric({
  label,
  value,
  format = (v) => fmt(v, 2),
  sub,
  tone,
  hero,
  countUp = true,
}: {
  label: string
  value: Num
  format?: (v: Num) => string
  sub?: ReactNode
  tone?: Num
  hero?: boolean
  countUp?: boolean
}) {
  const animated = useCountUp(countUp ? value : undefined)
  const shown = countUp ? animated : value
  return (
    <div className={hero ? 'stat stat-hero' : 'stat'}>
      <div className="stat-label">{label}</div>
      <div className={tone === undefined ? 'stat-value num' : `stat-value num ${dirClass(tone)}`}>
        {format(shown)}
      </div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

/* ======================================================================= */
/* Funnel - the pipeline, stage by stage                                    */
/* ----------------------------------------------------------------------- */
/* Each stage's bar width is its share of the FIRST stage, so the narrowing  */
/* is the actual attrition through the existing gates rather than a          */
/* decorative taper.                                                         */
/* ======================================================================= */
export interface Stage {
  label: string
  value: Num
  note?: string
}

export function Funnel({ stages }: { stages: Stage[] }) {
  const first = typeof stages[0]?.value === 'number' ? (stages[0].value as number) : 0
  return (
    <div className="funnel">
      {stages.map((s, i) => {
        const v = typeof s.value === 'number' ? s.value : null
        const share = first > 0 && v != null ? Math.max(0.02, v / first) : 0
        const prev = i > 0 && typeof stages[i - 1].value === 'number' ? (stages[i - 1].value as number) : null
        const dropped = prev != null && v != null ? prev - v : null
        return (
          <div className="funnel-stage reveal" key={s.label} style={stagger(i, 60)}>
            <div className="lbl">{s.label}</div>
            <FunnelNumber value={v} />
            <div className="funnel-bar" style={{ width: `${share * 100}%` }} />
            <div className="drop">
              {s.note ?? (dropped != null && dropped > 0 ? `−${int(dropped)} dropped` : dropped === 0 ? 'all carried' : ' ')}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function FunnelNumber({ value }: { value: number | null }) {
  const v = useCountUp(value ?? undefined, 620)
  return <div className="n">{value == null ? DASH : int(v)}</div>
}

/* ======================================================================= */
/* SignalPanel - why the existing system scored this what it did            */
/* ----------------------------------------------------------------------- */
/* The four components are shown separately with their own meters and the    */
/* engine's own weights. The total is the engine's total; this panel never    */
/* re-derives it, so the parts and the whole can never disagree here.         */
/* ======================================================================= */
export interface ScoreComponent {
  key: string
  label: string
  value: Num
  weight: number
}

export function SignalPanel({
  components,
  total,
  bucket,
  note,
}: {
  components: ScoreComponent[]
  total: Num
  bucket?: string
  note?: ReactNode
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-4)' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-3)' }}>
        {components.map((c, i) => {
          const v = typeof c.value === 'number' ? Math.max(0, Math.min(1, c.value)) : 0
          const has = typeof c.value === 'number'
          return (
            <div key={c.key} className="reveal" style={stagger(i, 55)}>
              <div className="row-between" style={{ marginBottom: 5 }}>
                <span className="eyebrow" style={{ letterSpacing: 'var(--ls-wide)' }}>{c.label}</span>
                <span className="row" style={{ gap: 'var(--s-3)' }}>
                  <span className="note">× {ratio(c.weight, 2)}</span>
                  <span className="num" style={{ fontSize: 'var(--t-sm)', color: has && v > 0 ? 'var(--ink)' : 'var(--ink-4)' }}>
                    {ratio(c.value, 3)}
                  </span>
                </span>
              </div>
              <div className="meter">
                <div className={v > 0 ? 'meter-fill' : 'meter-fill zero'} style={{ width: `${Math.max(v * 100, v > 0 ? 3 : 1.5)}%` }} />
              </div>
            </div>
          )
        })}
      </div>

      <hr className="hairline" />

      <div className="row-between">
        <div className="stat">
          <div className="stat-label">Conviction score</div>
          <div className="stat-value num">{ratio(total, 3)}</div>
        </div>
        {bucket && (
          <span className={bucket === 'high_conviction' ? 'badge badge-accent' : 'badge badge-warn'}>
            {bucket === 'high_conviction' ? 'High conviction' : 'Wait & watch'}
          </span>
        )}
      </div>

      {note && <p className="note">{note}</p>}
    </div>
  )
}

/* ======================================================================= */
/* TradeTimeline - the events that actually occurred on one position        */
/* ----------------------------------------------------------------------- */
/* Built from the trade rows the engine emitted. A ladder node appears only   */
/* if that ladder row exists; there is no placeholder for a tranche that      */
/* never fired.                                                              */
/* ======================================================================= */
export interface TradeEvent {
  label: string
  date: string
  price: Num
  qty?: Num
  pnl?: Num
  tone?: 'gain' | 'loss' | 'warn' | 'accent'
  detail?: Record<string, string>
}

export function TradeTimeline({ events }: { events: TradeEvent[] }) {
  const [open, setOpen] = useState<number | null>(null)
  if (!events.length) return <div className="empty">No trade events recorded.</div>
  const sel = open != null ? events[open] : null

  return (
    <div>
      <div className="timeline">
        <div className="timeline-rail" />
        <div className="timeline-events">
          {events.map((e, i) => (
            <button
              key={`${e.label}-${e.date}-${i}`}
              className="timeline-node reveal"
              style={stagger(i, 70)}
              onClick={() => setOpen(open === i ? null : i)}
              aria-expanded={open === i}
              aria-label={`${e.label} on ${e.date}`}
            >
              <span className={`timeline-dot ${e.tone ?? ''}`} />
              <span className="timeline-label">{e.label}</span>
              <span className="note num">{e.date}</span>
            </button>
          ))}
        </div>
      </div>
      {sel && (
        <div className="panel" style={{ marginTop: 'var(--s-3)' }}>
          <div className="panel-body">
            <div className="row-between" style={{ marginBottom: 'var(--s-2)' }}>
              <strong style={{ fontSize: 'var(--t-sm)' }}>{sel.label}</strong>
              <span className="note num">{sel.date}</span>
            </div>
            <dl className="kv">
              <dt>Price</dt>
              <dd className="num">{fmt(sel.price, 2)}</dd>
              {sel.qty != null && (<><dt>Quantity</dt><dd className="num">{int(sel.qty)}</dd></>)}
              {sel.pnl != null && (
                <><dt>Net P&amp;L</dt><dd className={`num ${dirClass(sel.pnl)}`}>{fmt(sel.pnl, 2)}</dd></>
              )}
              {Object.entries(sel.detail ?? {}).map(([k, v]) => (
                <span key={k} style={{ display: 'contents' }}>
                  <dt>{k}</dt>
                  <dd>{v}</dd>
                </span>
              ))}
            </dl>
          </div>
        </div>
      )}
    </div>
  )
}

/* ======================================================================= */
/* SignalCard - one candidate, as a signal rather than a table row          */
/* ======================================================================= */
export function SignalCard({
  symbol,
  company,
  sector,
  bucket,
  score,
  components,
  boxTop,
  boxBottom,
  actionable,
  onClick,
  index = 0,
}: {
  symbol: string
  company?: string | null
  sector?: string | null
  bucket: string
  score: Num
  components: (number | null)[]
  boxTop: Num
  boxBottom: Num
  actionable: boolean
  onClick?: () => void
  index?: number
}) {
  const s = typeof score === 'number' ? Math.max(0, Math.min(1, score)) : 0
  return (
    <button className="signal-card reveal" style={stagger(index)} onClick={onClick}>
      <div className="row-between" style={{ alignItems: 'flex-start' }}>
        <div>
          <div className="sym">{symbol}</div>
          <div className="co">{company ?? DASH}</div>
        </div>
        <span className={bucket === 'high_conviction' ? 'badge badge-accent' : 'badge badge-warn'}>
          {bucket === 'high_conviction' ? 'High' : 'Wait'}
        </span>
      </div>

      <div>
        <div className="row-between" style={{ marginBottom: 4 }}>
          <span className="eyebrow">Confidence</span>
          <span className="num" style={{ fontSize: 'var(--t-sm)' }}>{ratio(score, 3)}</span>
        </div>
        <div className="meter">
          <div className="meter-fill" style={{ width: `${Math.max(s * 100, 2)}%` }} />
        </div>
        <div className="score-bars" style={{ marginTop: 6 }} aria-hidden="true">
          {components.map((p, i) => {
            const v = typeof p === 'number' && Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : 0
            return <span key={i} className={v > 0 ? 'score-bar on' : 'score-bar'} style={{ height: `${Math.max(2, v * 14)}px` }} />
          })}
        </div>
      </div>

      <div className="row-between">
        <span className="note">{sector ?? DASH}</span>
        {actionable && <span className="badge badge-gain">Actionable</span>}
      </div>

      <div className="signal-box">
        <span className="note" style={{ letterSpacing: 'var(--ls-wide)' }}>BOX</span>
        <span className="loss">{fmt(boxBottom, 2)}</span>
        <span className="dim">→</span>
        <span className="gain">{fmt(boxTop, 2)}</span>
      </div>
    </button>
  )
}

/* ======================================================================= */
/* RegimeStrip - the existing breadth reading, drawn                        */
/* ======================================================================= */
export function RegimeStrip({
  breadth,
  label,
  pctile,
}: {
  breadth: Num
  label?: string | null
  pctile?: Num
}) {
  const v = typeof breadth === 'number' ? Math.max(0, Math.min(100, breadth)) : null
  return (
    <div>
      <div className="row-between" style={{ marginBottom: 6 }}>
        <span className="eyebrow">Market regime · breadth</span>
        <span className="row" style={{ gap: 'var(--s-3)' }}>
          <span className="num" style={{ fontSize: 'var(--t-md)' }}>{pct(breadth, 1)}</span>
          {label && <span className="badge">{label}</span>}
        </span>
      </div>
      <div className="meter" style={{ height: 6 }}>
        <div className="meter-fill" style={{ width: `${v ?? 0}%` }} />
      </div>
      <p className="note" style={{ marginTop: 6 }}>
        {pctile != null && <>{pct(pctile, 0)} percentile of its own history · </>}
        breadth is context, not a gate — nothing in the pipeline acts on it
      </p>
    </div>
  )
}
