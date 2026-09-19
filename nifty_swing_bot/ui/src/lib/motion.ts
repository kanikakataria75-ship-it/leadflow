/** Motion primitives.
 *
 * Every one of these checks `prefers-reduced-motion` and degrades to the final
 * state immediately rather than to a shorter animation. The brief's own test is
 * that the product still reads as a serious terminal with animation off, so
 * "off" has to mean the number is simply there, not that it arrives faster.
 */

import { useEffect, useRef, useState } from 'react'

/** True when the OS asks for reduced motion. Live, not read once. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
  )
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const on = () => setReduced(mq.matches)
    mq.addEventListener?.('change', on)
    return () => mq.removeEventListener?.('change', on)
  }, [])
  return reduced
}

/** Count a number up on mount.
 *
 * Returns the final value immediately under reduced motion, and also whenever
 * the target is not a finite number -- a count-up animating toward NaN would
 * render a flickering dash, which is worse than no animation at all.
 */
export function useCountUp(target: number | null | undefined, ms = 700): number | null | undefined {
  const reduced = useReducedMotion()
  const [v, setV] = useState<number | null | undefined>(
    () => (reduced || typeof target !== 'number' || !Number.isFinite(target) ? target : 0),
  )
  const raf = useRef(0)

  useEffect(() => {
    if (reduced || typeof target !== 'number' || !Number.isFinite(target)) {
      setV(target)
      return
    }

    // requestAnimationFrame does not fire in a hidden or occluded tab. Without
    // this guard a terminal loaded in a background tab renders every animated
    // figure as an em-dash until it is focused -- the number is correct in the
    // payload and absent on screen, which is the worst failure mode this UI
    // has. When the page is not visible, skip straight to the real value.
    if (typeof document !== 'undefined' && document.hidden) {
      setV(target)
      return
    }

    const t0 = performance.now()
    const from = 0
    let done = false
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / ms)
      // easeOutCubic: fast arrival, no overshoot. A spring would bounce past
      // the real figure, which on a P&L number is briefly a lie.
      const e = 1 - Math.pow(1 - p, 3)
      setV(from + (target - from) * e)
      if (p < 1) raf.current = requestAnimationFrame(tick)
      else done = true
    }
    raf.current = requestAnimationFrame(tick)

    // Belt and braces: if the frame loop never ran (tab backgrounded mid-mount,
    // throttled, or the page was never painted), land on the true value anyway.
    const failsafe = window.setTimeout(() => {
      if (!done) setV(target)
    }, ms + 250)

    return () => {
      cancelAnimationFrame(raf.current)
      window.clearTimeout(failsafe)
    }
  }, [target, ms, reduced])

  return v
}

/** Pointer parallax, clamped to [-1, 1] and written as CSS vars on a ref.
 *  Multipliers live in CSS (`.par-1/2/3`) so the movement band is defined in
 *  one place rather than per component. */
export function useParallax<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const reduced = useReducedMotion()

  useEffect(() => {
    if (reduced) return
    const el = ref.current
    if (!el) return
    let frame = 0
    const onMove = (e: PointerEvent) => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        const r = el.getBoundingClientRect()
        const x = ((e.clientX - r.left) / r.width - 0.5) * 2
        const y = ((e.clientY - r.top) / r.height - 0.5) * 2
        el.style.setProperty('--px', String(Math.max(-1, Math.min(1, x)).toFixed(3)))
        el.style.setProperty('--py', String(Math.max(-1, Math.min(1, y)).toFixed(3)))
      })
    }
    const onLeave = () => {
      el.style.setProperty('--px', '0')
      el.style.setProperty('--py', '0')
    }
    el.addEventListener('pointermove', onMove)
    el.addEventListener('pointerleave', onLeave)
    return () => {
      cancelAnimationFrame(frame)
      el.removeEventListener('pointermove', onMove)
      el.removeEventListener('pointerleave', onLeave)
    }
  }, [reduced])

  return ref
}

/** Staggered entrance delay, in ms, capped so a long list never makes the
 *  reader wait for row 40. */
export function stagger(i: number, step = 45, cap = 360): React.CSSProperties {
  return { ['--delay' as string]: `${Math.min(i * step, cap)}ms` } as React.CSSProperties
}
