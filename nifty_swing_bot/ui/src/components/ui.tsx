/** Shared primitives. Every page is built from these, so a change here is a
 *  change everywhere and the terminal cannot drift into two visual languages.
 */

import type { ReactNode } from 'react'
import { DASH, dirClass, fmt, pct, pctSigned, type Num } from '@/lib/format'

/* ------------------------------------------------------------------ Panel */
export function Panel({
  title,
  sub,
  right,
  children,
  flush,
}: {
  title?: ReactNode
  sub?: ReactNode
  right?: ReactNode
  children: ReactNode
  flush?: boolean
}) {
  return (
    <section className="panel">
      {(title || right) && (
        <header className="panel-head">
          <div>
            <div className="panel-title">{title}</div>
            {sub && <div className="note" style={{ marginTop: 2 }}>{sub}</div>}
          </div>
          {right && <div className="row">{right}</div>}
        </header>
      )}
      <div className={flush ? 'panel-body flush' : 'panel-body'}>{children}</div>
    </section>
  )
}

/* ------------------------------------------------------------------ Badge */
export function Badge({
  children,
  tone = 'default',
  title,
}: {
  children: ReactNode
  tone?: 'default' | 'warn' | 'accent' | 'gain' | 'loss'
  title?: string
}) {
  const cls = tone === 'default' ? 'badge' : `badge badge-${tone}`
  return (
    <span className={cls} title={title}>
      {children}
    </span>
  )
}

/** Provenance is never implicit. Every window states what it is, in the UI. */
export function Provenance({ value, note }: { value: string; note?: string }) {
  const validated = value.toUpperCase() === 'VALIDATED'
  return (
    <Badge tone={validated ? 'gain' : 'warn'} title={note}>
      {value.toUpperCase()}
    </Badge>
  )
}

/* ------------------------------------------------------------------- Stat */
export function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  tone?: Num
}) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={tone === undefined ? 'stat-value num' : `stat-value num ${dirClass(tone)}`}>
        {value}
      </div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

/** A headline return and its ex-best-fold twin, bound together.
 *
 *  This component exists so the pairing cannot be forgotten: there is no way to
 *  render a headline in this terminal without passing the twin, because the
 *  twin is a required prop. In both windows a single fold is 48-69% of the
 *  summed return, so a headline shown alone is a lie by omission. */
export function HeadlineReturn({
  label,
  value,
  exBest,
  shareOfBest,
  dp = 2,
}: {
  label: string
  value: Num
  exBest: Num
  shareOfBest?: Num
  dp?: number
}) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="row" style={{ alignItems: 'baseline', gap: 0 }}>
        <span className={`stat-value num ${dirClass(value)}`}>{pct(value, dp)}</span>
        <span className="twin">
          <span className="twin-label">ex-best fold</span>
          <span className={`num ${dirClass(exBest)}`}>{pct(exBest, dp)}</span>
        </span>
      </div>
      {shareOfBest != null && (
        <div className="stat-sub">
          best fold is <span className="num">{pct(shareOfBest, 1)}</span> of the summed return
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ Table */
export interface Col<T> {
  key: string
  head: ReactNode
  num?: boolean
  width?: number
  render: (row: T, i: number) => ReactNode
  title?: string
}

export function DataTable<T>({
  cols,
  rows,
  rowKey,
  maxHeight,
  zebra,
  rowClass,
  empty = 'No rows',
  foot,
}: {
  cols: Col<T>[]
  rows: T[]
  rowKey: (row: T, i: number) => string
  maxHeight?: number
  zebra?: boolean
  rowClass?: (row: T, i: number) => string | undefined
  empty?: ReactNode
  foot?: ReactNode
}) {
  if (!rows.length) return <div className="empty">{empty}</div>
  return (
    <div
      className="table-wrap"
      style={maxHeight ? ({ ['--table-max' as string]: `${maxHeight}px` } as React.CSSProperties) : undefined}
    >
      <table className={zebra ? 'data zebra' : 'data'}>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.key} className={c.num ? 'num' : undefined} style={c.width ? { width: c.width } : undefined} title={c.title}>
                {c.head}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={rowKey(r, i)} className={rowClass?.(r, i)}>
              {cols.map((c) => (
                <td key={c.key} className={c.num ? 'num' : undefined}>
                  {c.render(r, i)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
        {foot && (
          <tfoot>
            <tr>
              <td colSpan={cols.length}>{foot}</td>
            </tr>
          </tfoot>
        )}
      </table>
    </div>
  )
}

/* ------------------------------------------------------------- Score bars */
/** The four score components, decomposed inline rather than summed away.
 *  Equal weight 0.25 x 4, so four bars of equal width is the honest picture. */
export function ScoreBars({ parts }: { parts: (number | null | undefined)[] }) {
  return (
    <span className="score-bars" title={parts.map((p) => fmt(p, 2)).join(' / ')}>
      {parts.map((p, i) => {
        const v = typeof p === 'number' && Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : 0
        return (
          <span
            key={i}
            className={v > 0 ? 'score-bar on' : 'score-bar'}
            style={{ height: `${Math.max(2, v * 14)}px` }}
          />
        )
      })}
    </span>
  )
}

/* ----------------------------------------------------------------- States */
export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}

export function Loading({ what = 'data' }: { what?: string }) {
  return <div className="empty">Loading {what}…</div>
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div className="callout" style={{ borderColor: 'rgba(224,85,97,0.35)', background: 'var(--loss-dim)' }}>
      <strong style={{ color: 'var(--loss)' }}>Could not load.</strong> {error}
    </div>
  )
}

export function Callout({ children }: { children: ReactNode }) {
  return <div className="callout">{children}</div>
}

/* -------------------------------------------------------------- Segmented */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: readonly T[]
  value: T
  onChange: (v: T) => void
}) {
  return (
    <div className="seg" role="tablist">
      {options.map((o) => (
        <button key={o} role="tab" aria-selected={o === value} className={o === value ? 'on' : undefined} onClick={() => onChange(o)}>
          {o}
        </button>
      ))}
    </div>
  )
}

/* -------------------------------------------------------------- Delta cell */
export function Delta({ v, dp = 2 }: { v: Num; dp?: number }) {
  return <span className={dirClass(v)}>{v == null ? DASH : pctSigned(v, dp)}</span>
}
