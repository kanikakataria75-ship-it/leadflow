/** The Market City.
 *
 * A structured isometric view of the traded universe. It is not a particle
 * field: every element is placed deliberately and every dimension is a real
 * quantity the backend already serves.
 *
 *   district        one sector. Districts are laid out on a packed grid and
 *                   sized by the sector's share of the universe, so a large
 *                   district IS a large part of the book of names.
 *   district hue    identity only -- a locator, not a measurement. Assigned in
 *                   fixed order to the five largest sectors; everything else
 *                   takes the neutral slate. Hues are never cycled, so a hue
 *                   always means the same sector.
 *   tower           one stock. Towers are placed on the district's own grid in
 *                   deterministic order -- the city does not reshuffle between
 *                   renders of identical data.
 *   tower height    the candidate's existing conviction score. A name the scan
 *                   did not surface has no score, so it renders as a low
 *                   neutral plinth rather than being given an invented height.
 *   tower bloom     ALSO the existing score. Glow radius and alpha scale with
 *                   conviction, so height and light carry the same number
 *                   twice rather than encoding two different things.
 *   illumination    actionable, exactly as the backend flags it. Green is not
 *                   in the sector palette, so "lit green" can only ever mean
 *                   actionable and never "this is the healthcare district".
 *
 * Atmosphere is depth-cued, not decorative: distant districts sit behind more
 * fog and lose contrast, which is what makes the plan read as a receding
 * surface. Labels are drawn last, over their own backing plates, so no amount
 * of glow behind them costs a single character of legibility.
 *
 * Why an isometric canvas rather than three.js: a few hundred extruded boxes
 * with no lighting model, no textures and a fixed camera. An axonometric
 * projection with painter's-algorithm depth sorting renders it at 60fps in
 * ~13KB; three.js would add ~600KB for the same picture.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useReducedMotion } from '@/lib/motion'

export interface CityStock {
  symbol: string
  sector: string
  /** Existing conviction score in [0,1], or null when not in the scan. */
  score: number | null
  actionable: boolean
}

export interface CitySector {
  industry: string
  symbols: number
  weight_pct: number
  candidates: number
}

/** Sector identity hues, validated against this surface:
 *
 *   node scripts/validate_palette.js "#3D8BFD,#C2703A,#8F66CC,#B8873A,#CE6FB0" --mode dark
 *     [PASS] Lightness band  · [PASS] Chroma floor · [PASS] CVD separation
 *     [PASS] Normal-vision floor (worst adjacent dE 18.9) · [PASS] Contrast
 *
 * Green and red are deliberately absent: green means ACTIONABLE on this canvas
 * and red/green mean P&L everywhere else in the terminal. A sector can never
 * borrow either. */
const SECTOR_HUES: [number, number, number][] = [
  [61, 139, 253],  // blue
  [194, 112, 58],  // orange
  [143, 102, 204], // violet
  [184, 135, 58],  // amber
  [206, 111, 176], // magenta
]
const NEUTRAL_HUE: [number, number, number] = [124, 140, 168]

const rgba = (c: [number, number, number], a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a})`
const mix = (c: [number, number, number], t: [number, number, number], k: number): [number, number, number] =>
  [c[0] + (t[0] - c[0]) * k, c[1] + (t[1] - c[1]) * k, c[2] + (t[2] - c[2]) * k]

interface Tower {
  symbol: string
  gx: number
  gy: number
  h: number
  score: number | null
  actionable: boolean
}

interface District {
  name: string
  x: number
  y: number
  w: number
  d: number
  symbols: number
  candidates: number
  hue: [number, number, number]
  towers: Tower[]
  cx: number
  cy: number
  poly: [number, number][]
}

const ISO_X = Math.cos(Math.PI / 6)
const ISO_Y = Math.sin(Math.PI / 6)

function hash(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return Math.abs(h)
}

function layout(sectors: CitySector[], stocks: CityStock[]): District[] {
  const ordered = [...sectors].sort((a, b) => b.symbols - a.symbols)
  const bySector = new Map<string, CityStock[]>()
  for (const s of stocks) {
    if (!bySector.has(s.sector)) bySector.set(s.sector, [])
    bySector.get(s.sector)!.push(s)
  }

  const total = ordered.reduce((a, s) => a + s.symbols, 0) || 1
  const PLAN = 26
  const out: District[] = []
  let cursorX = 0
  let cursorY = 0
  let rowDepth = 0

  ordered.forEach((sec, rank) => {
    const share = sec.symbols / total
    const side = Math.max(3, Math.round(Math.sqrt(share * PLAN * PLAN * 1.35)))
    const w = side
    const d = Math.max(3, Math.round(side * 0.82))

    if (cursorX + w > PLAN) {
      cursorX = 0
      cursorY += rowDepth + 1.4
      rowDepth = 0
    }

    const own = (bySector.get(sec.industry) ?? []).slice().sort((a, b) => {
      if ((b.score ?? -1) !== (a.score ?? -1)) return (b.score ?? -1) - (a.score ?? -1)
      return a.symbol.localeCompare(b.symbol)
    })

    const cells: { gx: number; gy: number }[] = []
    for (let gy = 0; gy < d - 1; gy++) for (let gx = 0; gx < w - 1; gx++) cells.push({ gx, gy })
    const n = Math.min(cells.length, Math.max(own.length, Math.round(sec.symbols * 0.5)))

    const towers: Tower[] = []
    for (let i = 0; i < n; i++) {
      const cell = cells[i]
      const stock = own[i]
      const scored = stock && stock.score != null
      const hsh = hash(stock?.symbol ?? `${sec.industry}#${i}`)
      towers.push({
        symbol: stock?.symbol ?? '',
        gx: cell.gx + 0.5,
        gy: cell.gy + 0.5,
        h: scored ? 0.55 + (stock!.score as number) * 2.6 : 0.16 + (hsh % 7) * 0.035,
        score: stock?.score ?? null,
        actionable: Boolean(stock?.actionable),
      })
    }

    out.push({
      name: sec.industry, x: cursorX, y: cursorY, w, d,
      symbols: sec.symbols, candidates: sec.candidates,
      // Fixed-order assignment by rank. Sectors beyond the palette take the
      // neutral rather than a generated hue, so no two sectors ever share one.
      hue: rank < SECTOR_HUES.length ? SECTOR_HUES[rank] : NEUTRAL_HUE,
      towers, cx: 0, cy: 0, poly: [],
    })
    cursorX += w + 1.4
    rowDepth = Math.max(rowDepth, d)
  })
  return out
}

export function MarketCity({
  sectors,
  stocks,
  height = 460,
  onSelectSector,
  focus,
  interactive = true,
  showLabels = true,
}: {
  sectors: CitySector[]
  stocks: CityStock[]
  height?: number | string
  onSelectSector?: (sector: string | null) => void
  focus?: string | null
  interactive?: boolean
  showLabels?: boolean
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const reduced = useReducedMotion()
  /** The hovered district, as STATE -- it changes only when the pointer moves
   *  from one district to another, so it re-renders a handful of times per
   *  session rather than on every pointer event. */
  const [hoverD, setHoverD] = useState<District | null>(null)
  /** Live pointer/hover, as a REF. `draw` reads this so that moving the mouse
   *  never changes the identity of the draw callback. That mattered: when
   *  `hover` was a dependency of `draw`, every pointer event produced a new
   *  `draw`, which re-ran the animation effect, which called `resize()` --
   *  and assigning `canvas.width` blanks the canvas. The result was a visible
   *  flicker on every mouse move. */
  const hoverRef = useRef<{ x: number; y: number; d: District } | null>(null)
  const tipRef = useRef<HTMLDivElement>(null)

  const districts = useMemo(() => layout(sectors, stocks), [sectors, stocks])
  const cam = useRef({ x: 0, y: 0, z: 1, tx: 0, ty: 0, tz: 1 })
  /** Pointer/scroll parallax, in grid units. Deliberately tiny. */
  const par = useRef({ x: 0, y: 0, tx: 0, ty: 0 })
  /** Pointer light, in canvas pixels. `p` is its strength, damped so the lamp
   *  fades in and out rather than popping when the pointer enters or leaves. */
  const light = useRef({ x: -1e4, y: -1e4, p: 0, tx: -1e4, ty: -1e4, tp: 0 })
  const districtsRef = useRef<District[]>(districts)
  districtsRef.current = districts
  // Everything `draw` reads lives in a ref, so `draw` is created once and the
  // render loop is never torn down while the user is interacting with it.
  const focusRef = useRef<string | null | undefined>(focus)
  focusRef.current = focus
  const labelsRef = useRef<boolean>(showLabels)
  labelsRef.current = showLabels

  const extent = useMemo(() => {
    let mx = 0
    let my = 0
    for (const d of districts) {
      mx = Math.max(mx, d.x + d.w)
      my = Math.max(my, d.y + d.d)
    }
    return { mx: mx || 1, my: my || 1 }
  }, [districts])
  const extentRef = useRef(extent)
  extentRef.current = extent


  useEffect(() => {
    const d = districts.find((x) => x.name === focus)
    if (d) {
      cam.current.tx = -(d.x + d.w / 2)
      cam.current.ty = -(d.y + d.d / 2)
      cam.current.tz = 1.85
    } else {
      cam.current.tx = -extent.mx / 2
      cam.current.ty = -extent.my / 2
      cam.current.tz = 1
    }
    if (reduced) {
      cam.current.x = cam.current.tx
      cam.current.y = cam.current.ty
      cam.current.z = cam.current.tz
    }
  }, [focus, districts, extent, reduced])

  // Scroll parallax: a fraction of a grid unit as the panel moves through the
  // viewport. Capped so the city never appears to slide out from under its own
  // labels.
  useEffect(() => {
    if (reduced || !interactive) return
    const onScroll = () => {
      const el = wrapRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      const centred = (r.top + r.height / 2 - window.innerHeight / 2) / window.innerHeight
      par.current.ty = Math.max(-0.5, Math.min(0.5, centred)) * 0.55
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [reduced, interactive])

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number) => {
      const c = cam.current
      const k = reduced ? 1 : 0.085
      c.x += (c.tx - c.x) * k
      c.y += (c.ty - c.y) * k
      c.z += (c.tz - c.z) * k
      par.current.x += (par.current.tx - par.current.x) * (reduced ? 1 : 0.08)
      par.current.y += (par.current.ty - par.current.y) * (reduced ? 1 : 0.08)
      const L = light.current
      const lk = reduced ? 1 : 0.16
      L.x += (L.tx - L.x) * lk
      L.y += (L.ty - L.y) * lk
      L.p += (L.tp - L.p) * (reduced ? 1 : 0.1)

      // ---- atmosphere: a vertical wash, dark at the horizon ---------------
      ctx.clearRect(0, 0, w, h)
      const sky = ctx.createLinearGradient(0, 0, 0, h)
      sky.addColorStop(0, 'rgba(12,18,32,0.55)')
      sky.addColorStop(0.45, 'rgba(10,15,26,0.22)')
      sky.addColorStop(1, 'rgba(5,7,12,0)')
      ctx.fillStyle = sky
      ctx.fillRect(0, 0, w, h)

      // a soft glow at the horizon line, where the plan recedes
      const halo = ctx.createRadialGradient(w / 2, h * 0.3, 0, w / 2, h * 0.3, Math.max(w, h) * 0.62)
      halo.addColorStop(0, 'rgba(61,139,253,0.055)')
      halo.addColorStop(1, 'rgba(61,139,253,0)')
      ctx.fillStyle = halo
      ctx.fillRect(0, 0, w, h)

      const extent = extentRef.current
      const span = extent.mx + extent.my
      const projW = span * ISO_X
      const projH = span * ISO_Y + 3.2 * 0.62
      const unit = Math.min(w / projW, h / projH) * 0.96 * c.z
      const ox = w / 2
      const oy = h / 2 - span * ISO_Y * unit * 0.06

      const proj = (gx: number, gy: number, gz = 0): [number, number] => {
        const X = gx + c.x + par.current.x
        const Y = gy + c.y + par.current.y
        return [ox + (X - Y) * ISO_X * unit, oy + (X + Y) * ISO_Y * unit - gz * unit * 0.62]
      }

      // ---- ground grid ----------------------------------------------------
      ctx.strokeStyle = 'rgba(110,166,255,0.04)'
      ctx.lineWidth = 1
      const g0 = -2
      const g1 = Math.max(extent.mx, extent.my) + 2
      for (let i = g0; i <= g1; i += 2) {
        const a = proj(i, g0); const b = proj(i, g1)
        const p = proj(g0, i); const q = proj(g1, i)
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke()
      }

      // Roof highlights are collected here and composited in ONE additive pass
      // after the city is drawn. They are never mixed into the roof's own fill,
      // because that fill's alpha encodes the conviction score -- brightening
      // it with a cursor would make the light look like data.
      const specs: { x: number; y: number; r: number; s: number; c: [number, number, number] }[] = []
      const order = [...districtsRef.current].sort((a, b) => a.x + a.y - (b.x + b.y))
      const focused = focusRef.current ?? null
      const hoverName = hoverRef.current?.d.name ?? null

      // Depth fog: districts further back (lower x+y) lose contrast toward the
      // sky colour. This is what makes the plan read as receding rather than
      // as a flat pattern.
      const depthOf = (d: District) => (d.x + d.y) / (extent.mx + extent.my || 1)

      for (const d of order) {
        const sel = focused && d.name !== focused
        const dim = sel ? 0.26 : 1
        const lit = d.name === hoverName || d.name === focused
        // Capped well below the old 0.42: fog is a depth cue, and past ~0.3 it
        // desaturates the district hue to the point where sector identity is
        // lost, which is the one thing the colour is carrying.
        const fog = 0.26 - depthOf(d) * 0.26
        const hue = mix(d.hue, [18, 26, 42], fog)

        const p0 = proj(d.x, d.y)
        const p1 = proj(d.x + d.w, d.y)
        const p2 = proj(d.x + d.w, d.y + d.d)
        const p3 = proj(d.x, d.y + d.d)
        d.poly = [p0, p1, p2, p3]
        d.cx = (p0[0] + p2[0]) / 2
        d.cy = (p0[1] + p2[1]) / 2

        // platform, tinted by the sector hue at low alpha
        ctx.beginPath()
        ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1])
        ctx.lineTo(p2[0], p2[1]); ctx.lineTo(p3[0], p3[1]); ctx.closePath()
        ctx.fillStyle = rgba(hue, (lit ? 0.30 : 0.19) * dim)
        ctx.fill()
        ctx.strokeStyle = rgba(hue, (lit ? 0.95 : 0.5) * dim)
        ctx.lineWidth = lit ? 1.7 : 1
        ctx.stroke()

        // district ground-glow when hovered or focused
        if (lit && !sel) {
          const gg = ctx.createRadialGradient(d.cx, d.cy, 0, d.cx, d.cy, unit * d.w * 0.9)
          gg.addColorStop(0, rgba(d.hue, 0.16))
          gg.addColorStop(1, rgba(d.hue, 0))
          ctx.fillStyle = gg
          ctx.fill()
        }

        const ts = [...d.towers].sort((a, b) => a.gx + a.gy - (b.gx + b.gy))
        for (const t of ts) {
          const bx = d.x + t.gx
          const by = d.y + t.gy
          const s = 0.4
          const top = t.h
          const A = proj(bx - s, by - s, top)
          const B = proj(bx + s, by - s, top)
          const C = proj(bx + s, by + s, top)
          const D = proj(bx - s, by + s, top)
          const Bb = proj(bx + s, by - s, 0)
          const Cb = proj(bx + s, by + s, 0)
          const Db = proj(bx - s, by + s, 0)

          const scored = t.score != null
          const sc = (t.score ?? 0) as number

          // ---- bloom: radius and alpha both scale with conviction ---------
          if (scored && !sel) {
            const rad = unit * (0.9 + sc * 2.4)
            const bloom = ctx.createRadialGradient(C[0], A[1], 0, C[0], A[1], rad)
            const bc: [number, number, number] = t.actionable ? [79, 211, 165] : d.hue
            bloom.addColorStop(0, rgba(bc, (0.1 + sc * 0.3) * dim))
            bloom.addColorStop(1, rgba(bc, 0))
            ctx.fillStyle = bloom
            ctx.beginPath()
            ctx.arc(C[0], A[1], rad, 0, Math.PI * 2)
            ctx.fill()
          }

          const faceL = t.actionable ? mix([24, 96, 74], [18, 26, 42], fog) : mix(hue, [10, 14, 22], 0.40)
          const faceR = t.actionable ? mix([16, 72, 56], [18, 26, 42], fog) : mix(hue, [8, 11, 18], 0.54)
          const roof = t.actionable
            ? mix([79, 211, 165], [18, 26, 42], fog * 0.7)
            : scored
              ? mix(hue, [255, 255, 255], sc * 0.22)
              : mix(hue, [14, 20, 32], 0.52)

          ctx.beginPath()
          ctx.moveTo(D[0], D[1]); ctx.lineTo(C[0], C[1])
          ctx.lineTo(Cb[0], Cb[1]); ctx.lineTo(Db[0], Db[1]); ctx.closePath()
          ctx.fillStyle = rgba(faceL, dim)
          ctx.fill()

          ctx.beginPath()
          ctx.moveTo(C[0], C[1]); ctx.lineTo(B[0], B[1])
          ctx.lineTo(Bb[0], Bb[1]); ctx.lineTo(Cb[0], Cb[1]); ctx.closePath()
          ctx.fillStyle = rgba(faceR, dim)
          ctx.fill()

          ctx.beginPath()
          ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1])
          ctx.lineTo(C[0], C[1]); ctx.lineTo(D[0], D[1]); ctx.closePath()
          ctx.fillStyle = rgba(roof, (scored ? 0.55 + sc * 0.45 : 0.46) * dim)
          ctx.fill()

          // distance from the pointer light to this roof, for the additive pass
          if (L.p > 0.01 && !sel) {
            const rx = C[0]
            const ry = A[1]
            const dist = Math.hypot(rx - L.x, ry - L.y)
            const reach = unit * 7.5
            if (dist < reach) {
              const falloff = (1 - dist / reach) ** 2
              specs.push({
                x: rx, y: ry,
                r: unit * (0.8 + (scored ? sc : 0.2) * 1.5),
                s: falloff * L.p * (scored ? 0.5 + sc * 0.5 : 0.28),
                c: t.actionable ? [120, 240, 200] : mix(d.hue, [255, 255, 255], 0.5) as [number, number, number],
              })
            }
          }

          if (scored && sc > 0.25) {
            ctx.strokeStyle = rgba(t.actionable ? [110, 240, 200] : mix(d.hue, [255, 255, 255], 0.4), 0.55 * dim)
            ctx.lineWidth = 1
            ctx.stroke()
          }
        }
      }

      // ---- pointer light: one additive pass over the finished city --------
      if (L.p > 0.01) {
        ctx.save()
        ctx.globalCompositeOperation = 'lighter'

        // the lamp itself: a soft pool of light on the plan
        const reach = Math.min(w, h) * 0.42
        const lamp = ctx.createRadialGradient(L.x, L.y, 0, L.x, L.y, reach)
        lamp.addColorStop(0, `rgba(96,150,240,${0.13 * L.p})`)
        lamp.addColorStop(0.45, `rgba(70,115,200,${0.05 * L.p})`)
        lamp.addColorStop(1, 'rgba(60,100,180,0)')
        ctx.fillStyle = lamp
        ctx.fillRect(0, 0, w, h)

        // specular on the roofs the light is passing over
        for (const sp of specs) {
          const g2 = ctx.createRadialGradient(sp.x, sp.y, 0, sp.x, sp.y, sp.r)
          g2.addColorStop(0, rgba(sp.c, 0.5 * sp.s))
          g2.addColorStop(1, rgba(sp.c, 0))
          ctx.fillStyle = g2
          ctx.beginPath()
          ctx.arc(sp.x, sp.y, sp.r, 0, Math.PI * 2)
          ctx.fill()
        }
        ctx.restore()
      }

      // ---- labels last, on their own plates -------------------------------
      // Placed largest-district-first with a greedy overlap test. A label that
      // cannot be placed without colliding is dropped rather than stacked --
      // two overlapping plates are less legible than one plate and a dot, and
      // the dropped name is still one hover away in the tooltip.
      if (labelsRef.current) {
        const placed: { x: number; y: number; w: number; h: number }[] = []
        const byImportance = [...order].sort((a, b) => {
          if (a.name === focused) return -1
          if (b.name === focused) return 1
          if (a.name === hoverName) return -1
          if (b.name === hoverName) return 1
          return b.symbols - a.symbols
        })

        for (const d of byImportance) {
          const sel = focused && d.name !== focused
          const dim = sel ? 0.34 : 1
          const label = (d.name.length > 20 ? `${d.name.slice(0, 19)}…` : d.name).toUpperCase()
          const sub = `${d.symbols} names${d.candidates ? ` · ${d.candidates} cand` : ''}`

          ctx.font = '600 10px Schibsted Grotesk, system-ui, sans-serif'
          const tw = ctx.measureText(label).width
          ctx.font = '400 9px JetBrains Mono, monospace'
          const sw = ctx.measureText(sub).width
          const bw = Math.max(tw, sw, 50) + 14
          const bh = 29
          const bx = d.cx - bw / 2
          const by = d.cy - bh / 2 - 2

          const clash = placed.some(
            (r) => bx < r.x + r.w && bx + bw > r.x && by < r.y + r.h && by + bh > r.y,
          )
          const must = d.name === focused || d.name === hoverName
          if (clash && !must) {
            // marker only: the district stays findable without a plate
            ctx.beginPath()
            ctx.arc(d.cx, d.cy, 2.5, 0, Math.PI * 2)
            ctx.fillStyle = rgba(d.hue, 0.7 * dim)
            ctx.fill()
            continue
          }
          placed.push({ x: bx, y: by, w: bw, h: bh })

          ctx.fillStyle = `rgba(8,11,18,${0.86 * dim})`
          ctx.beginPath()
          const rr = 4
          ctx.moveTo(bx + rr, by)
          ctx.arcTo(bx + bw, by, bx + bw, by + bh, rr)
          ctx.arcTo(bx + bw, by + bh, bx, by + bh, rr)
          ctx.arcTo(bx, by + bh, bx, by, rr)
          ctx.arcTo(bx, by, bx + bw, by, rr)
          ctx.closePath()
          ctx.fill()
          ctx.strokeStyle = rgba(d.hue, (must ? 0.8 : 0.42) * dim)
          ctx.lineWidth = 1
          ctx.stroke()

          ctx.textAlign = 'center'
          ctx.font = '600 10px Schibsted Grotesk, system-ui, sans-serif'
          ctx.fillStyle = `rgba(237,240,246,${0.97 * dim})`
          ctx.fillText(label, d.cx, by + 13)
          ctx.font = '400 9px JetBrains Mono, monospace'
          ctx.fillStyle = `rgba(160,170,188,${0.92 * dim})`
          ctx.fillText(sub, d.cx, by + 24)
        }
      }
    },
    [reduced],
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
      const cw = Math.max(1, Math.floor(r.width * dpr))
      const ch = Math.max(1, Math.floor(r.height * dpr))
      // Assigning width/height CLEARS the canvas even when the value is
      // unchanged, so this only writes when the size actually moved.
      if (canvas.width === cw && canvas.height === ch) return
      canvas.width = cw
      canvas.height = ch
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
    if (reduced) {
      const r = wrap.getBoundingClientRect()
      draw(ctx, r.width, r.height)
    } else {
      raf = requestAnimationFrame(loop)
    }

    return () => {
      stopped = true
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [draw, reduced])

  const pick = (mx: number, my: number): District | null => {
    const order = [...districtsRef.current].sort((a, b) => b.x + b.y - (a.x + a.y))
    for (const d of order) {
      const p = d.poly
      if (p.length !== 4) continue
      let inside = false
      for (let i = 0, j = 3; i < 4; j = i++) {
        const [xi, yi] = p[i]
        const [xj, yj] = p[j]
        if (yi > my !== yj > my && mx < ((xj - xi) * (my - yi)) / (yj - yi) + xi) inside = !inside
      }
      if (inside) return d
    }
    return null
  }

  return (
    <div className="city" ref={wrapRef} style={{ height }}>
      <canvas ref={canvasRef} aria-hidden="true" />
      {interactive && (
        <div
          className="city-hit"
          onPointerMove={(e) => {
            const r = wrapRef.current!.getBoundingClientRect()
            const mx = e.clientX - r.left
            const my = e.clientY - r.top
            // pointer parallax, capped at a third of a grid unit
            par.current.tx = ((mx / r.width - 0.5) * 2) * 0.34
            light.current.tx = mx
            light.current.ty = my
            light.current.tp = 1
            if (light.current.p === 0) {
              // first entry: place the lamp under the cursor rather than
              // sliding it in from wherever it was left
              light.current.x = mx
              light.current.y = my
            }
            const d = pick(mx, my)
            hoverRef.current = d ? { x: e.clientX, y: e.clientY, d } : null
            // Move the tooltip by touching the node directly. Routing this
            // through state would re-render on every pointer event for a
            // change React cannot batch away.
            if (tipRef.current && d) {
              tipRef.current.style.left = `${e.clientX + 14}px`
              tipRef.current.style.top = `${e.clientY + 12}px`
            }
            // State changes only when the district under the pointer changes.
            setHoverD((prev) => (prev?.name === (d?.name ?? undefined) ? prev : d))
          }}
          onPointerLeave={() => {
            par.current.tx = 0
            light.current.tp = 0
            hoverRef.current = null
            setHoverD(null)
          }}
          onClick={() => {
            const d = hoverRef.current?.d
            if (!d) {
              onSelectSector?.(null)
              return
            }
            onSelectSector?.(d.name === focus ? null : d.name)
          }}
          role="presentation"
        />
      )}
      {hoverD && (
        <div
          className="tooltip"
          ref={tipRef}
          style={{
            left: (hoverRef.current?.x ?? 0) + 14,
            top: (hoverRef.current?.y ?? 0) + 12,
          }}
        >
          <div className="tooltip-title">{hoverD.name}</div>
          <div className="tooltip-row">
            <span className="k">names in universe</span>
            <span className="num">{hoverD.symbols}</span>
          </div>
          <div className="tooltip-row">
            <span className="k">candidates today</span>
            <span className="num">{hoverD.candidates}</span>
          </div>
          <div className="tooltip-row">
            <span className="k" style={{ color: 'var(--ink-4)' }}>
              {hoverD.name === focus ? 'click to zoom out' : 'click to focus'}
            </span>
            <span />
          </div>
        </div>
      )}
    </div>
  )
}
