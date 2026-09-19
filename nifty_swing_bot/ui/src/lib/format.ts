/** Number formatting. Every figure in the terminal passes through here.
 *
 * Two rules the rest of the app relies on:
 *   1. A missing number renders as an em-dash, never as 0, NaN or "null".
 *      A zero and an absence mean different things on a trading screen.
 *   2. Anything returned here is meant to sit in a `.num` cell, which is
 *      monospace with tabular figures. Padding or aligning in JS would fight
 *      that, so nothing here pads.
 */

export const DASH = '—'

export type Num = number | null | undefined

const isNum = (v: Num): v is number => typeof v === 'number' && Number.isFinite(v)

export function fmt(v: Num, dp = 2): string {
  if (!isNum(v)) return DASH
  return v.toLocaleString('en-IN', { minimumFractionDigits: dp, maximumFractionDigits: dp })
}

export function pct(v: Num, dp = 2): string {
  if (!isNum(v)) return DASH
  return `${fmt(v, dp)}%`
}

/** Percent with an explicit sign, for anything directional. */
export function pctSigned(v: Num, dp = 2): string {
  if (!isNum(v)) return DASH
  return `${v > 0 ? '+' : ''}${fmt(v, dp)}%`
}

export function signed(v: Num, dp = 2): string {
  if (!isNum(v)) return DASH
  return `${v > 0 ? '+' : ''}${fmt(v, dp)}`
}

/** Indian rupee, in lakh/crore when the magnitude warrants it. */
export function inr(v: Num, dp = 0): string {
  if (!isNum(v)) return DASH
  const a = Math.abs(v)
  const sign = v < 0 ? '-' : ''
  if (a >= 1e7) return `${sign}₹${fmt(a / 1e7, 2)}Cr`
  if (a >= 1e5) return `${sign}₹${fmt(a / 1e5, 2)}L`
  return `${sign}₹${fmt(a, dp)}`
}

/** Exact rupees, for a trade plan where the number is actionable. */
export function inrExact(v: Num, dp = 2): string {
  if (!isNum(v)) return DASH
  return `₹${fmt(v, dp)}`
}

export function int(v: Num): string {
  if (!isNum(v)) return DASH
  return Math.round(v).toLocaleString('en-IN')
}

export function ratio(v: Num, dp = 3): string {
  return isNum(v) ? fmt(v, dp) : DASH
}

/** The CSS class that carries P&L direction. Green/red mean money, only. */
export function dirClass(v: Num): string {
  if (!isNum(v) || v === 0) return 'flat'
  return v > 0 ? 'gain' : 'loss'
}

export function shortDate(iso: string): string {
  if (!iso) return DASH
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: '2-digit' })
}

export function monthYear(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-GB', { month: 'short', year: '2-digit' })
}

/** Decimate a long series for rendering without changing its shape.
 *  Keeps first and last, and never drops an extremum inside a bucket. */
export function decimate<T>(xs: T[], target: number, value: (t: T) => number): T[] {
  if (xs.length <= target) return xs
  const step = xs.length / target
  const out: T[] = []
  for (let i = 0; i < target; i++) {
    const lo = Math.floor(i * step)
    const hi = Math.min(xs.length, Math.floor((i + 1) * step))
    let pick = lo
    let best = -Infinity
    for (let j = lo; j < hi; j++) {
      const m = Math.abs(value(xs[j]))
      if (m > best) { best = m; pick = j }
    }
    out.push(xs[pick])
  }
  if (out[0] !== xs[0]) out[0] = xs[0]
  out[out.length - 1] = xs[xs.length - 1]
  return out
}
