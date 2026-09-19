import { useCallback, useEffect, useRef, useState } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import { api } from '@/lib/api'
import { CommandBar } from '@/components/CommandBar'
import Home from '@/pages/Home'
import Dashboard from '@/pages/Dashboard'
import Scanner from '@/pages/Scanner'
import Backtest from '@/pages/Backtest'
import Book from '@/pages/Book'
import ForwardRecord from '@/pages/ForwardRecord'
import StockDetail from '@/pages/StockDetail'
import Settings from '@/pages/Settings'
import Login from '@/pages/Login'

/** Scan state lives at the app root so the command palette and the Scanner page
 *  drive the SAME job. Both call the scanner's existing POST endpoint and both
 *  read the existing status endpoint; nothing about a scan is reimplemented in
 *  the frontend. */
export function useScanJob() {
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [finishedAt, setFinishedAt] = useState<string | null>(null)
  const poll = useRef<number | null>(null)

  const stop = useCallback(() => {
    if (poll.current) window.clearInterval(poll.current)
    poll.current = null
  }, [])

  const tick = useCallback(async () => {
    try {
      const s = await api.scanStatus()
      setRunning(s.running)
      setError(s.error)
      if (!s.running) {
        setFinishedAt(s.finished_at)
        stop()
      }
    } catch {
      /* transient: keep polling, the next tick will tell us */
    }
  }, [stop])

  const start = useCallback(async () => {
    setError(null)
    try {
      await api.startScan()
      setRunning(true)
      stop()
      poll.current = window.setInterval(tick, 2500)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [tick, stop])

  // Pick up a scan that some other client already started.
  useEffect(() => {
    void tick()
    return stop
  }, [tick, stop])

  return { running, error, finishedAt, start }
}

/** Session state. The server is the authority -- this only mirrors it so the
 *  UI knows whether to render the terminal or the gate. */
function useSession() {
  const [state, setState] = useState<'checking' | 'in' | 'out'>('checking')
  const refresh = useCallback(async () => {
    try {
      const r = await fetch('/api/auth/status')
      const j = await r.json()
      setState(j?.authenticated ? 'in' : 'out')
    } catch {
      setState('out')
    }
  }, [])
  useEffect(() => {
    void refresh()
  }, [refresh])
  return { state, refresh, signOut: async () => {
    await fetch('/api/auth/logout', { method: 'POST' }).catch(() => {})
    setState('out')
  } }
}

export default function App() {
  const loc = useLocation()
  const job = useScanJob()
  const session = useSession()

  if (session.state === 'checking') {
    return <div className="empty" style={{ paddingTop: '22vh' }}>Loading…</div>
  }
  if (session.state === 'out') {
    return <Login onSignedIn={session.refresh} />
  }

  // The entry screen is full-bleed: the command bar floats over the market
  // city rather than sitting above it in a shell.
  if (loc.pathname === '/') {
    return (
      <Routes>
        <Route path="/" element={<Home />} />
      </Routes>
    )
  }

  return (
    <div className="shell">
      <CommandBar onRunScan={job.start} scanning={job.running} />
      {/* keyed on pathname so each page re-runs its entrance animation */}
      <div key={loc.pathname} style={{ display: 'contents' }}>
        <Routes location={loc}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/scanner" element={<Scanner job={job} />} />
          <Route path="/forward" element={<ForwardRecord />} />
          <Route path="/backtest" element={<Backtest />} />
          <Route path="/book" element={<Book />} />
          <Route path="/stock/:symbol" element={<StockDetail />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </div>
    </div>
  )
}
