/** Floating command bar + ⌘K palette.
 *
 * Replaces the vertical sidebar. Navigation is a single compact glass rail that
 * floats over the content, and every route plus the scan action is reachable
 * from the palette.
 *
 * The palette's "Run new scan" entry calls the scanner's OWN existing endpoint.
 * It is the same background job the evening tool starts -- this is a second
 * trigger for it, not a second implementation of it.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { api, useAsync, type Candidate } from '@/lib/api'

export const ROUTES: { to: string; label: string; group: string }[] = [
  { to: '/dashboard', label: 'Dashboard', group: 'Live' },
  { to: '/scanner', label: 'Scanner', group: 'Live' },
  { to: '/forward', label: 'Forward Record', group: 'Live' },
  { to: '/backtest', label: 'Backtest', group: 'Research' },
  { to: '/book', label: 'Book', group: 'Portfolio' },
  { to: '/settings', label: 'Settings', group: 'System' },
]

interface Action {
  id: string
  label: string
  hint?: string
  group: string
  run: () => void
}

export function CommandBar({ onRunScan, scanning }: { onRunScan?: () => void; scanning?: boolean }) {
  const [open, setOpen] = useState(false)
  const nav = useNavigate()
  const { data: cfg } = useAsync(() => api.config(), [])
  const { data: scan } = useAsync(() => api.scan(), [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
      }
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const t = cfg?.exits

  return (
    <>
      <header className="cmdbar">
        <div className="cmdbar-inner">
          <NavLink to="/" className="cmd-brand" aria-label="LeadFlow home">
            <span className="brand-mark" aria-hidden="true" />
            <span className="cmd-brand-name">LeadFlow</span>
          </NavLink>

          <nav className="cmd-nav" aria-label="Sections">
            {ROUTES.map((r) => (
              <NavLink key={r.to} to={r.to} className={({ isActive }) => (isActive ? 'active' : undefined)}>
                {r.label}
              </NavLink>
            ))}
          </nav>

          <div className="cmd-right">
            {t && (
              <span className="cmd-config num" title={t.ladder_tradeoff}>
                {cfg?.sizing.slots}&nbsp;slots · {cfg?.sizing.risk_per_trade_pct}%&nbsp;risk · ATR×{t.atr_mult}
              </span>
            )}
            <button className="cmd-k" onClick={() => setOpen(true)} aria-label="Open command palette">
              <span className="dim">Search</span>
              <kbd>⌘K</kbd>
            </button>
          </div>
        </div>
      </header>

      {open && (
        <Palette
          onClose={() => setOpen(false)}
          candidates={scan?.candidates ?? []}
          onNavigate={(to) => {
            nav(to)
            setOpen(false)
          }}
          onRunScan={
            onRunScan
              ? () => {
                  onRunScan()
                  setOpen(false)
                }
              : undefined
          }
          scanning={scanning}
        />
      )}
    </>
  )
}

function Palette({
  onClose,
  onNavigate,
  onRunScan,
  candidates,
  scanning,
}: {
  onClose: () => void
  onNavigate: (to: string) => void
  onRunScan?: () => void
  candidates: Candidate[]
  scanning?: boolean
}) {
  const [q, setQ] = useState('')
  const [i, setI] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const actions: Action[] = useMemo(() => {
    const out: Action[] = ROUTES.map((r) => ({
      id: r.to,
      label: r.label,
      hint: r.group,
      group: 'Navigate',
      run: () => onNavigate(r.to),
    }))
    if (onRunScan) {
      out.unshift({
        id: 'scan',
        label: scanning ? 'Scan already running' : 'Run new scan',
        hint: scanning ? 'in progress' : 'starts the existing scan job',
        group: 'Actions',
        run: () => {
          if (!scanning) onRunScan()
        },
      })
    }
    for (const c of candidates.slice(0, 40)) {
      out.push({
        id: `stock-${c.symbol}`,
        label: c.symbol,
        hint: c.company ?? c.industry ?? 'candidate',
        group: 'Stocks in latest scan',
        run: () => onNavigate(`/stock/${encodeURIComponent(c.symbol)}`),
      })
    }
    return out
  }, [candidates, onNavigate, onRunScan, scanning])

  const filtered = useMemo(() => {
    const t = q.trim().toLowerCase()
    if (!t) return actions.slice(0, 12)
    return actions
      .filter((a) => a.label.toLowerCase().includes(t) || (a.hint ?? '').toLowerCase().includes(t))
      .slice(0, 14)
  }, [q, actions])

  useEffect(() => setI(0), [q])

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setI((n) => Math.min(filtered.length - 1, n + 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setI((n) => Math.max(0, n - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      filtered[i]?.run()
    }
  }

  let lastGroup = ''

  return (
    <div className="palette-backdrop" onClick={onClose} role="presentation">
      <div
        className="palette"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
      >
        <input
          ref={inputRef}
          className="palette-input"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={onKey}
          placeholder="Search LeadFlow — routes, actions, stocks in the latest scan…"
        />
        <div className="palette-list">
          {filtered.length === 0 && <div className="empty">No matches.</div>}
          {filtered.map((a, n) => {
            const head = a.group !== lastGroup ? a.group : null
            lastGroup = a.group
            return (
              <div key={a.id}>
                {head && <div className="palette-group">{head}</div>}
                <button
                  className={n === i ? 'palette-item on' : 'palette-item'}
                  onMouseEnter={() => setI(n)}
                  onClick={() => a.run()}
                >
                  <span>{a.label}</span>
                  {a.hint && <span className="note">{a.hint}</span>}
                </button>
              </div>
            )
          })}
        </div>
        <div className="palette-foot">
          <span className="note">↑↓ navigate · ↵ open · esc close</span>
        </div>
      </div>
    </div>
  )
}
