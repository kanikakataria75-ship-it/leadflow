/** Capital orbit: the book's three sleeves as proportional arcs around a core.
 *
 * The geometry is the allocation. Each arc's angular span is that sleeve's
 * share of the book, the core carries the book's total value, and the ring
 * radius is constant so nothing but angle encodes weight. Target and actual are
 * drawn as two concentric rings so a drift is visible as a gap rather than as a
 * number the reader has to subtract for themselves.
 *
 * No new allocation logic: the weights come from the frozen BookParams and the
 * actual sleeve values come from the persisted series, both unchanged.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { useReducedMotion } from '@/lib/motion'
import { inr, pct } from '@/lib/format'

export interface Sleeve {
  name: string
  color: string
  target: number
  actual: number
  value: number
}

const TAU = Math.PI * 2

export function CapitalOrbit({
  sleeves,
  total,
  height = 300,
  onSelect,
  selected,
}: {
  sleeves: Sleeve[]
  total: number
  height?: number
  onSelect?: (name: string | null) => void
  selected?: string | null
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const reduced = useReducedMotion()
  const [hover, setHover] = useState<{ x: number; y: number; s: Sleeve } | null>(null)
  const progress = useRef(reduced ? 1 : 0)

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number) => {
      ctx.clearRect(0, 0, w, h)
      const cx = w / 2
      const cy = h / 2
      const R = Math.min(w, h) * 0.34
      const p = progress.current

      // A slight vertical squash reads as a ring seen at an angle rather than
      // a flat pie -- depth without a perspective camera.
      const squash = 0.62
      const ring = (radius: number, from: number, to: number, color: string, width: number, alpha: number) => {
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(1, squash)
        ctx.beginPath()
        ctx.arc(0, 0, radius, from, to)
        ctx.strokeStyle = color
        ctx.globalAlpha = alpha
        ctx.lineWidth = width
        ctx.lineCap = 'butt'
        ctx.stroke()
        ctx.restore()
      }

      // faint full circles as the track
      ring(R, 0, TAU, 'rgba(255,255,255,0.05)', 22, 1)
      ring(R * 0.72, 0, TAU, 'rgba(255,255,255,0.035)', 10, 1)

      let aT = -Math.PI / 2
      let aA = -Math.PI / 2
      for (const s of sleeves) {
        const dim = selected && selected !== s.name ? 0.28 : 1
        const spanT = TAU * s.target * p
        const spanA = TAU * s.actual * p

        // outer ring: ACTUAL
        ring(R, aA, aA + spanA, s.color, 22, 0.9 * dim)
        // inner ring: TARGET
        ring(R * 0.72, aT, aT + spanT, s.color, 10, 0.4 * dim)

        // a 2px surface gap so adjacent arcs stay separable
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(1, squash)
        ctx.beginPath()
        ctx.arc(0, 0, R, aA + spanA - 0.004, aA + spanA)
        ctx.strokeStyle = 'var(--surface)'
        ctx.lineWidth = 24
        ctx.restore()

        aT += spanT
        aA += spanA
      }

      // core
      ctx.beginPath()
      ctx.arc(cx, cy, R * 0.44, 0, TAU)
      const g = ctx.createRadialGradient(cx, cy - R * 0.2, 0, cx, cy, R * 0.5)
      g.addColorStop(0, 'rgba(76,141,255,0.16)')
      g.addColorStop(1, 'rgba(76,141,255,0.02)')
      ctx.fillStyle = g
      ctx.fill()
      ctx.strokeStyle = 'rgba(110,166,255,0.22)'
      ctx.lineWidth = 1
      ctx.stroke()

      ctx.textAlign = 'center'
      ctx.fillStyle = 'rgba(154,162,179,0.9)'
      ctx.font = '600 9px Schibsted Grotesk, system-ui, sans-serif'
      ctx.fillText('BOOK', cx, cy - 10)
      ctx.fillStyle = 'rgba(233,236,243,0.95)'
      ctx.font = '500 17px JetBrains Mono, monospace'
      ctx.fillText(inr(total * p), cx, cy + 11)

      if (p < 1) progress.current = Math.min(1, p + 0.045)
    },
    [sleeves, total, selected],
  )

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    let raf = 0
    let stopped = false
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const resize = () => {
      const r = wrap.getBoundingClientRect()
      canvas.width = Math.max(1, Math.floor(r.width * dpr))
      canvas.height = Math.max(1, Math.floor(r.height * dpr))
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(wrap)
    const loop = () => {
      if (stopped) return
      const r = wrap.getBoundingClientRect()
      draw(ctx, r.width, r.height)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => {
      stopped = true
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [draw])

  /** Which sleeve is under the pointer: angle within the ring band. */
  const pick = (mx: number, my: number): Sleeve | null => {
    const wrap = wrapRef.current
    if (!wrap) return null
    const r = wrap.getBoundingClientRect()
    const cx = r.width / 2
    const cy = r.height / 2
    const R = Math.min(r.width, r.height) * 0.34
    const dx = mx - cx
    const dy = (my - cy) / 0.62
    const dist = Math.hypot(dx, dy)
    if (dist < R - 16 || dist > R + 16) return null
    let a = Math.atan2(dy, dx) + Math.PI / 2
    if (a < 0) a += TAU
    let acc = 0
    for (const s of sleeves) {
      const span = TAU * s.actual
      if (a >= acc && a < acc + span) return s
      acc += span
    }
    return null
  }

  return (
    <div className="orbit" ref={wrapRef} style={{ height }}>
      <canvas ref={canvasRef} aria-hidden="true" />
      <div
        className="orbit-hit"
        onPointerMove={(e) => {
          const r = wrapRef.current!.getBoundingClientRect()
          const s = pick(e.clientX - r.left, e.clientY - r.top)
          setHover(s ? { x: e.clientX, y: e.clientY, s } : null)
        }}
        onPointerLeave={() => setHover(null)}
        onClick={() => onSelect?.(hover ? (hover.s.name === selected ? null : hover.s.name) : null)}
        role="presentation"
      />
      {hover && (
        <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 12 }}>
          <div className="tooltip-title">{hover.s.name}</div>
          <div className="tooltip-row">
            <span className="k">target</span>
            <span className="num">{pct(hover.s.target * 100, 1)}</span>
          </div>
          <div className="tooltip-row">
            <span className="k">actual</span>
            <span className="num">{pct(hover.s.actual * 100, 1)}</span>
          </div>
          <div className="tooltip-row">
            <span className="k">value</span>
            <span className="num">{inr(hover.s.value)}</span>
          </div>
        </div>
      )}
    </div>
  )
}
