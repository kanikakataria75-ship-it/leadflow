/** The chart system. Hand-built SVG, no chart library.
 *
 * Every chart in the terminal is drawn here so they share one house style
 * rather than inheriting a library's defaults. The rules, applied uniformly:
 *
 *   - 2px series strokes, recessive 1px grid, hairline axes
 *   - ONE y-axis, always. Two measures of different scale become two charts or
 *     get indexed to a common base; there is no dual-axis chart in this app
 *   - a crosshair + tooltip on every time series, a per-mark tooltip on bars
 *     and cells; interactivity is the default, not an enhancement
 *   - a legend whenever there are two or more series, so identity is never
 *     carried by colour alone
 *   - green/red appear ONLY where the quantity is P&L direction
 *   - 2px surface-coloured gaps between stacked fills so bands stay separable
 */

import { useCallback, useId, useMemo, useRef, useState } from 'react'
import { DASH, fmt, monthYear, pct, type Num } from '@/lib/format'

/* ====================================================================== */
/* scales                                                                  */
/* ====================================================================== */
export interface Box {
  w: number
  h: number
  top: number
  right: number
  bottom: number
  left: number
}

const BOX: Box = { w: 800, h: 240, top: 12, right: 14, bottom: 24, left: 46 }

const plotW = (b: Box) => b.w - b.left - b.right
const plotH = (b: Box) => b.h - b.top - b.bottom

function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return [lo]
  const span = hi - lo
  const raw = span / count
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const norm = raw / mag
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag
  const start = Math.ceil(lo / step) * step
  const out: number[] = []
  for (let v = start; v <= hi + step * 1e-9; v += step) out.push(Number(v.toFixed(10)))
  return out
}

function extent(series: (Num[])[], pad = 0.04): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const s of series)
    for (const v of s) {
      if (typeof v === 'number' && Number.isFinite(v)) {
        if (v < lo) lo = v
        if (v > hi) hi = v
      }
    }
  if (!Number.isFinite(lo)) return [0, 1]
  if (lo === hi) return [lo - 1, hi + 1]
  const p = (hi - lo) * pad
  return [lo - p, hi + p]
}

/* ====================================================================== */
/* tooltip                                                                 */
/* ====================================================================== */
interface TipRow { label: string; value: string; color?: string }
interface TipState { x: number; y: number; title: string; rows: TipRow[] }

function Tooltip({ tip }: { tip: TipState | null }) {
  if (!tip) return null
  return (
    <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 12 }}>
      <div className="tooltip-title">{tip.title}</div>
      {tip.rows.map((r, i) => (
        <div className="tooltip-row" key={i}>
          <span className="k">
            {r.color && <span className="legend-swatch" style={{ background: r.color, width: 8, height: 8, borderRadius: 2 }} />}
            {r.label}
          </span>
          <span className="num">{r.value}</span>
        </div>
      ))}
    </div>
  )
}

/* ====================================================================== */
/* Legend                                                                  */
/* ====================================================================== */
export function Legend({ items, block }: { items: { label: string; color: string }[]; block?: boolean }) {
  if (items.length < 2) return null
  return (
    <div className="legend" style={{ marginBottom: 'var(--s-2)' }}>
      {items.map((it) => (
        <span className="legend-item" key={it.label}>
          <span className={block ? 'legend-swatch block' : 'legend-swatch'} style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  )
}

/* ====================================================================== */
/* LineChart - indexed time series, one y-axis                             */
/* ====================================================================== */
export interface LineSeries {
  label: string
  color: string
  values: Num[]
  dashed?: boolean
  width?: number
}

export function LineChart({
  dates,
  series,
  height = 240,
  yFormat = (v: number) => fmt(v, 0),
  tipFormat = (v: Num) => fmt(v, 2),
  baseline,
}: {
  dates: string[]
  series: LineSeries[]
  height?: number
  yFormat?: (v: number) => string
  tipFormat?: (v: Num) => string
  baseline?: number
}) {
  const b: Box = { ...BOX, h: height }
  const [tip, setTip] = useState<TipState | null>(null)
  const ref = useRef<SVGSVGElement>(null)
  const [lo, hi] = useMemo(() => extent(series.map((s) => s.values)), [series])
  const n = dates.length
  const x = useCallback((i: number) => b.left + (plotW(b) * i) / Math.max(1, n - 1), [b, n])
  const y = useCallback((v: number) => b.top + plotH(b) * (1 - (v - lo) / (hi - lo || 1)), [b, lo, hi])
  const ticks = useMemo(() => niceTicks(lo, hi, 5), [lo, hi])

  const path = (vals: Num[]) => {
    let d = ''
    let pen = false
    vals.forEach((v, i) => {
      if (typeof v !== 'number' || !Number.isFinite(v)) { pen = false; return }
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`
      pen = true
    })
    return d
  }

  const onMove = (e: React.MouseEvent) => {
    const svg = ref.current
    if (!svg || !n) return
    const r = svg.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * b.w
    const i = Math.round(((px - b.left) / plotW(b)) * (n - 1))
    if (i < 0 || i >= n) return setTip(null)
    setTip({
      x: e.clientX,
      y: e.clientY,
      title: dates[i],
      rows: series.map((s) => ({ label: s.label, value: tipFormat(s.values[i]), color: s.color })),
    })
  }

  const hoverIdx = tip ? dates.indexOf(tip.title) : -1
  const xLabels = useMemo(() => {
    const step = Math.max(1, Math.floor(n / 6))
    // `i === n - 1` is forced so the series always ends on a dated tick, but it
    // duplicates the label whenever the last index also lands on a step.
    const keep = new Set<number>()
    for (let i = 0; i < n; i += step) keep.add(i)
    if (n) keep.add(n - 1)
    return [...keep].sort((a, b) => a - b).map((i) => ({ i, d: dates[i] }))
  }, [dates, n])

  return (
    <div style={{ position: 'relative' }}>
      <Legend items={series.map((s) => ({ label: s.label, color: s.color }))} />
      <svg
        ref={ref}
        className="chart"
        viewBox={`0 0 ${b.w} ${b.h}`}
        preserveAspectRatio="none"
        style={{ height }}
        onMouseMove={onMove}
        onMouseLeave={() => setTip(null)}
        role="img"
        aria-label={series.map((s) => s.label).join(', ')}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={b.left} x2={b.w - b.right} y1={y(t)} y2={y(t)} />
            <text x={b.left - 6} y={y(t) + 3} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        {baseline !== undefined && baseline >= lo && baseline <= hi && (
          <line className="zero-line" x1={b.left} x2={b.w - b.right} y1={y(baseline)} y2={y(baseline)} />
        )}
        {xLabels.map(({ i, d }) => (
          <text key={i} x={x(i)} y={b.h - 7} textAnchor={i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'}>
            {monthYear(d)}
          </text>
        ))}
        {series.map((s) => (
          <path
            key={s.label}
            className="series"
            d={path(s.values)}
            stroke={s.color}
            strokeWidth={s.width ?? 2}
            strokeDasharray={s.dashed ? '4 3' : undefined}
          />
        ))}
        {hoverIdx >= 0 && (
          <>
            <line className="crosshair" x1={x(hoverIdx)} x2={x(hoverIdx)} y1={b.top} y2={b.h - b.bottom} />
            {series.map((s) => {
              const v = s.values[hoverIdx]
              if (typeof v !== 'number' || !Number.isFinite(v)) return null
              return <circle key={s.label} cx={x(hoverIdx)} cy={y(v)} r={3.5} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
            })}
          </>
        )}
        <line className="axis-line" x1={b.left} x2={b.w - b.right} y1={b.h - b.bottom} y2={b.h - b.bottom} />
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

/* ====================================================================== */
/* Underwater - drawdown, filled below zero                                */
/* ====================================================================== */
export function UnderwaterChart({ dates, values, height = 130 }: { dates: string[]; values: Num[]; height?: number }) {
  const b: Box = { ...BOX, h: height }
  const gid = useId()
  const [tip, setTip] = useState<TipState | null>(null)
  const ref = useRef<SVGSVGElement>(null)
  const lo = Math.min(-1, ...values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v)))
  const n = dates.length
  const x = (i: number) => b.left + (plotW(b) * i) / Math.max(1, n - 1)
  const y = (v: number) => b.top + plotH(b) * (v / (lo || -1))
  const ticks = niceTicks(lo, 0, 4)

  let d = `M${x(0)},${y(0)}`
  values.forEach((v, i) => {
    const val = typeof v === 'number' && Number.isFinite(v) ? v : 0
    d += `L${x(i).toFixed(2)},${y(val).toFixed(2)}`
  })
  d += `L${x(n - 1)},${y(0)}Z`

  const onMove = (e: React.MouseEvent) => {
    const svg = ref.current
    if (!svg || !n) return
    const r = svg.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * b.w
    const i = Math.round(((px - b.left) / plotW(b)) * (n - 1))
    if (i < 0 || i >= n) return setTip(null)
    setTip({ x: e.clientX, y: e.clientY, title: dates[i], rows: [{ label: 'drawdown', value: pct(values[i], 2), color: 'var(--loss)' }] })
  }

  return (
    <div style={{ position: 'relative' }}>
      <svg ref={ref} className="chart" viewBox={`0 0 ${b.w} ${b.h}`} preserveAspectRatio="none" style={{ height }}
        onMouseMove={onMove} onMouseLeave={() => setTip(null)} role="img" aria-label="Drawdown from peak">
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--loss)" stopOpacity="0.02" />
            <stop offset="100%" stopColor="var(--loss)" stopOpacity="0.30" />
          </linearGradient>
        </defs>
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={b.left} x2={b.w - b.right} y1={y(t)} y2={y(t)} />
            <text x={b.left - 6} y={y(t) + 3} textAnchor="end">{fmt(t, 0)}%</text>
          </g>
        ))}
        <path d={d} fill={`url(#${gid})`} stroke="var(--loss)" strokeWidth={1.5} />
        <line className="zero-line" x1={b.left} x2={b.w - b.right} y1={y(0)} y2={y(0)} />
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

/* ====================================================================== */
/* StackedArea - sleeve allocation over time                               */
/* ====================================================================== */
export function StackedArea({
  dates,
  bands,
  height = 220,
  asPercent = true,
}: {
  dates: string[]
  bands: { label: string; color: string; values: Num[] }[]
  height?: number
  asPercent?: boolean
}) {
  const b: Box = { ...BOX, h: height }
  const [tip, setTip] = useState<TipState | null>(null)
  const ref = useRef<SVGSVGElement>(null)
  const n = dates.length

  const totals = useMemo(
    () => dates.map((_, i) => bands.reduce((a, s) => a + (Number(s.values[i]) || 0), 0)),
    [dates, bands],
  )
  const norm = (i: number, v: Num) => {
    const t = totals[i] || 1
    const val = Number(v) || 0
    return asPercent ? (val / t) * 100 : val
  }
  const hi = asPercent ? 100 : Math.max(...totals, 1)
  const x = (i: number) => b.left + (plotW(b) * i) / Math.max(1, n - 1)
  const y = (v: number) => b.top + plotH(b) * (1 - v / hi)

  const paths = useMemo(() => {
    const cum = new Array(n).fill(0)
    return bands.map((s) => {
      const lower = [...cum]
      s.values.forEach((v, i) => { cum[i] += norm(i, v) })
      let d = ''
      for (let i = 0; i < n; i++) d += `${i ? 'L' : 'M'}${x(i).toFixed(2)},${y(cum[i]).toFixed(2)}`
      for (let i = n - 1; i >= 0; i--) d += `L${x(i).toFixed(2)},${y(lower[i]).toFixed(2)}`
      return { d: d + 'Z', color: s.color, label: s.label }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bands, n, hi])

  const onMove = (e: React.MouseEvent) => {
    const svg = ref.current
    if (!svg || !n) return
    const r = svg.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * b.w
    const i = Math.round(((px - b.left) / plotW(b)) * (n - 1))
    if (i < 0 || i >= n) return setTip(null)
    setTip({
      x: e.clientX, y: e.clientY, title: dates[i],
      rows: bands.map((s) => ({ label: s.label, value: asPercent ? pct(norm(i, s.values[i]), 1) : fmt(s.values[i], 0), color: s.color })),
    })
  }

  return (
    <div style={{ position: 'relative' }}>
      <Legend items={bands.map((s) => ({ label: s.label, color: s.color }))} block />
      <svg ref={ref} className="chart" viewBox={`0 0 ${b.w} ${b.h}`} preserveAspectRatio="none" style={{ height }}
        onMouseMove={onMove} onMouseLeave={() => setTip(null)} role="img" aria-label="Sleeve allocation over time">
        {niceTicks(0, hi, 4).map((t) => (
          <g key={t}>
            <line className="grid-line" x1={b.left} x2={b.w - b.right} y1={y(t)} y2={y(t)} />
            <text x={b.left - 6} y={y(t) + 3} textAnchor="end">{asPercent ? `${fmt(t, 0)}%` : fmt(t, 0)}</text>
          </g>
        ))}
        {/* 2px surface gap between bands so adjacent fills stay separable */}
        {paths.map((p) => (
          <path key={p.label} d={p.d} fill={p.color} fillOpacity={0.82} stroke="var(--surface)" strokeWidth={2} />
        ))}
        <line className="axis-line" x1={b.left} x2={b.w - b.right} y1={b.h - b.bottom} y2={b.h - b.bottom} />
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

/* ====================================================================== */
/* DivergingBars - per-year return, green/red because it IS P&L direction   */
/* ====================================================================== */
export function DivergingBars({
  labels,
  values,
  height = 190,
  shares,
}: {
  labels: (string | number)[]
  values: Num[]
  height?: number
  shares?: Num[]
}) {
  const b: Box = { ...BOX, h: height, left: 46, bottom: 30 }
  const [tip, setTip] = useState<TipState | null>(null)
  const nums = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  const lo = Math.min(0, ...nums)
  const hi = Math.max(0, ...nums)
  const pad = (hi - lo) * 0.08 || 1
  const y = (v: number) => b.top + plotH(b) * (1 - (v - (lo - pad)) / ((hi + pad) - (lo - pad)))
  const n = labels.length
  const slot = plotW(b) / Math.max(1, n)
  const bw = Math.min(46, slot * 0.62)

  return (
    <div style={{ position: 'relative' }}>
      <svg className="chart" viewBox={`0 0 ${b.w} ${b.h}`} preserveAspectRatio="none" style={{ height }}
        role="img" aria-label="Return by year">
        {niceTicks(lo - pad, hi + pad, 4).map((t) => (
          <g key={t}>
            <line className="grid-line" x1={b.left} x2={b.w - b.right} y1={y(t)} y2={y(t)} />
            <text x={b.left - 6} y={y(t) + 3} textAnchor="end">{fmt(t, 0)}%</text>
          </g>
        ))}
        <line className="zero-line" x1={b.left} x2={b.w - b.right} y1={y(0)} y2={y(0)} />
        {labels.map((lab, i) => {
          const v = values[i]
          if (typeof v !== 'number' || !Number.isFinite(v)) return null
          const cx = b.left + slot * i + slot / 2
          const top = v >= 0 ? y(v) : y(0)
          const h = Math.max(1.5, Math.abs(y(v) - y(0)))
          const color = v >= 0 ? 'var(--gain)' : 'var(--loss)'
          return (
            <g key={String(lab)}
              onMouseEnter={(e) =>
                setTip({
                  x: e.clientX, y: e.clientY, title: String(lab),
                  rows: [
                    { label: 'return', value: pct(v, 2), color },
                    ...(shares && shares[i] != null ? [{ label: 'share of total', value: pct(shares[i], 1) }] : []),
                  ],
                })
              }
              onMouseLeave={() => setTip(null)}
            >
              <rect x={cx - bw / 2} y={top} width={bw} height={h} rx={2} fill={color} fillOpacity={0.85} />
              <text x={cx} y={b.h - 16} textAnchor="middle">{lab}</text>
              <text x={cx} y={b.h - 5} textAnchor="middle" style={{ fill: 'var(--ink-4)' }}>
                {shares && shares[i] != null ? `${fmt(shares[i], 0)}%` : ''}
              </text>
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

/* ====================================================================== */
/* Histogram                                                               */
/* ====================================================================== */
export function Histogram({
  values,
  bins = 34,
  height = 180,
  unit = '%',
  colorBySign = false,
}: {
  values: Num[]
  bins?: number
  height?: number
  unit?: string
  colorBySign?: boolean
}) {
  const b: Box = { ...BOX, h: height, bottom: 26 }
  const [tip, setTip] = useState<TipState | null>(null)
  const nums = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  const { counts, lo, step } = useMemo(() => {
    if (!nums.length) return { counts: [] as number[], lo: 0, step: 1 }
    const mn = Math.min(...nums)
    const mx = Math.max(...nums)
    const st = (mx - mn) / bins || 1
    const c = new Array(bins).fill(0)
    for (const v of nums) c[Math.min(bins - 1, Math.max(0, Math.floor((v - mn) / st)))]++
    return { counts: c, lo: mn, step: st }
  }, [nums, bins])

  if (!counts.length) return <div className="empty">No observations</div>
  const maxC = Math.max(...counts)
  const slot = plotW(b) / bins
  const y = (c: number) => b.top + plotH(b) * (1 - c / maxC)
  const zeroX = b.left + ((0 - lo) / (step * bins)) * plotW(b)

  return (
    <div style={{ position: 'relative' }}>
      <svg className="chart" viewBox={`0 0 ${b.w} ${b.h}`} preserveAspectRatio="none" style={{ height }}
        role="img" aria-label="Distribution">
        {niceTicks(0, maxC, 3).map((t) => (
          <g key={t}>
            <line className="grid-line" x1={b.left} x2={b.w - b.right} y1={y(t)} y2={y(t)} />
            <text x={b.left - 6} y={y(t) + 3} textAnchor="end">{fmt(t, 0)}</text>
          </g>
        ))}
        {counts.map((c, i) => {
          const x0 = lo + i * step
          const mid = x0 + step / 2
          const color = colorBySign ? (mid >= 0 ? 'var(--gain)' : 'var(--loss)') : 'var(--cat-1)'
          return (
            <rect key={i} x={b.left + slot * i + 1} y={y(c)} width={Math.max(1, slot - 2)} height={Math.max(0, b.h - b.bottom - y(c))}
              rx={1.5} fill={color} fillOpacity={0.8}
              onMouseEnter={(e) => setTip({ x: e.clientX, y: e.clientY, title: `${fmt(x0, 2)}${unit} … ${fmt(x0 + step, 2)}${unit}`, rows: [{ label: 'count', value: fmt(c, 0) }] })}
              onMouseLeave={() => setTip(null)} />
          )
        })}
        {lo < 0 && (
          <line className="zero-line" x1={zeroX} x2={zeroX} y1={b.top} y2={b.h - b.bottom} />
        )}
        <line className="axis-line" x1={b.left} x2={b.w - b.right} y1={b.h - b.bottom} y2={b.h - b.bottom} />
        <text x={b.left} y={b.h - 6} textAnchor="start">{fmt(lo, 1)}{unit}</text>
        <text x={b.w - b.right} y={b.h - 6} textAnchor="end">{fmt(lo + step * bins, 1)}{unit}</text>
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

/* ====================================================================== */
/* SectorTreemap - THE market visual                                       */
/* ---------------------------------------------------------------------- */
/* Area = share of the traded universe. Colour = live signal density, on a   */
/* single-hue sequential ramp. Both encode real quantities; it is static     */
/* unless hovered. This replaces a rotating 3D bar ring that encoded nothing */
/* ====================================================================== */
export interface TreeCell { label: string; value: number; intensity: number; meta?: Record<string, string> }

function squarify(items: TreeCell[], w: number, h: number) {
  // Simple slice-and-dice with alternating orientation: deterministic, stable
  // between renders, and adequate at this cell count. A full squarified layout
  // buys better aspect ratios and costs reproducibility across data updates.
  const total = items.reduce((a, i) => a + i.value, 0) || 1
  const out: (TreeCell & { x: number; y: number; w: number; h: number })[] = []
  let x = 0
  let y = 0
  let rowW = w
  let rowH = h
  let horizontal = w > h
  let remaining = total
  for (const it of items) {
    const frac = it.value / (remaining || 1)
    if (horizontal) {
      const cw = rowW * frac
      out.push({ ...it, x, y, w: cw, h: rowH })
      x += cw
      rowW -= cw
    } else {
      const ch = rowH * frac
      out.push({ ...it, x, y, w: rowW, h: ch })
      y += ch
      rowH -= ch
    }
    remaining -= it.value
    horizontal = rowW > rowH
  }
  return out
}

export function SectorTreemap({ cells, height = 300 }: { cells: TreeCell[]; height?: number }) {
  const [tip, setTip] = useState<TipState | null>(null)
  const W = 800
  const H = height
  const laid = useMemo(() => squarify(cells, W, H), [cells, H])
  const ramp = ['var(--seq-1)', 'var(--seq-2)', 'var(--seq-3)', 'var(--seq-4)', 'var(--seq-5)']
  const colorOf = (t: number) => ramp[Math.min(ramp.length - 1, Math.max(0, Math.round(t * (ramp.length - 1))))]

  if (!cells.length) return <div className="empty">No universe data</div>

  return (
    <div style={{ position: 'relative' }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height }}
        role="img" aria-label="Sector map: area is universe weight, colour is signal density">
        {laid.map((c) => {
          const big = c.w > 62 && c.h > 30
          return (
            <g key={c.label}
              onMouseEnter={(e) => setTip({
                x: e.clientX, y: e.clientY, title: c.label,
                rows: Object.entries(c.meta ?? {}).map(([k, v]) => ({ label: k, value: v })),
              })}
              onMouseLeave={() => setTip(null)}>
              <rect x={c.x + 1} y={c.y + 1} width={Math.max(0, c.w - 2)} height={Math.max(0, c.h - 2)}
                fill={colorOf(c.intensity)} stroke="var(--surface)" strokeWidth={2} rx={2} />
              {big && (
                <>
                  <text x={c.x + 8} y={c.y + 17} style={{ fill: 'var(--ink)', fontFamily: 'var(--font-ui)', fontSize: 11, fontWeight: 600 }}>
                    {c.label.length > 20 ? `${c.label.slice(0, 19)}…` : c.label}
                  </text>
                  <text x={c.x + 8} y={c.y + 30} style={{ fill: 'var(--ink-2)', fontSize: 9.5 }}>
                    {fmt(c.value, 0)} names
                  </text>
                </>
              )}
            </g>
          )
        })}
      </svg>
      <div className="row" style={{ marginTop: 'var(--s-2)', justifyContent: 'space-between' }}>
        <span className="note">Area = share of universe · colour = live signal density</span>
        <span className="row" style={{ gap: 4 }}>
          <span className="note">low</span>
          {ramp.map((c) => <span key={c} className="legend-swatch block" style={{ background: c, width: 16 }} />)}
          <span className="note">high</span>
        </span>
      </div>
      <Tooltip tip={tip} />
    </div>
  )
}

/* ====================================================================== */
/* Sparkline - inline, no axes, for table cells                            */
/* ====================================================================== */
export function Sparkline({ values, color = 'var(--cat-1)', width = 92, height = 22 }: { values: Num[]; color?: string; width?: number; height?: number }) {
  const nums = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  if (nums.length < 2) return <span className="dim">{DASH}</span>
  const lo = Math.min(...nums)
  const hi = Math.max(...nums)
  const d = values
    .map((v, i) => {
      if (typeof v !== 'number' || !Number.isFinite(v)) return ''
      const x = (width * i) / (values.length - 1)
      const y = height - 2 - (height - 4) * ((v - lo) / (hi - lo || 1))
      return `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join('')
  return (
    <svg width={width} height={height} style={{ display: 'block' }} aria-hidden="true">
      <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
    </svg>
  )
}
