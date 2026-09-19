import { useEffect, useRef, useState } from 'react'

/** The gate.
 *
 * The backend refuses every API call without a session cookie, so this screen
 * is the only thing a signed-out visitor can reach. It shows no figures, no
 * scan state and no configuration — an unauthenticated viewer learns the
 * product's name and nothing else about the book.
 */
export default function Login({ onSignedIn }: { onSignedIn: () => void }) {
  const [pw, setPw] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const ref = useRef<HTMLInputElement>(null)

  useEffect(() => {
    ref.current?.focus()
  }, [])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      const r = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: pw }),
      })
      if (r.ok) {
        onSignedIn()
        return
      }
      const j = await r.json().catch(() => ({}))
      setError(j?.detail ?? `Sign-in failed (${r.status}).`)
    } catch {
      setError('Could not reach the server.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="home" style={{ justifyContent: 'center' }}>
      <div className="home-inner" style={{ gap: 'var(--s-8)' }}>
        <div className="reveal" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-3)', alignItems: 'center' }}>
          <div className="hero-kicker">Market Intelligence System</div>
          <h1 className="hero-title" style={{ fontSize: 'var(--t-4xl)' }}>
            Lead<span className="accent">Flow</span>
          </h1>
        </div>

        <form
          onSubmit={submit}
          className="panel reveal"
          style={{ ['--delay' as string]: '120ms', width: 'min(380px, 100%)', padding: 'var(--s-6)' }}
        >
          <label className="stat-label" htmlFor="pw" style={{ display: 'block', marginBottom: 'var(--s-2)' }}>
            Password
          </label>
          <input
            id="pw"
            ref={ref}
            className="input"
            type="password"
            autoComplete="current-password"
            value={pw}
            onChange={(e) => setPw(e.target.value)}
            style={{ width: '100%', fontSize: 'var(--t-base)', padding: '10px 12px' }}
            aria-invalid={!!error}
            aria-describedby={error ? 'pw-error' : undefined}
          />

          {error && (
            <div
              id="pw-error"
              className="callout"
              role="alert"
              style={{
                marginTop: 'var(--s-3)',
                borderColor: 'rgba(224,85,97,0.35)',
                background: 'var(--loss-dim)',
              }}
            >
              <strong style={{ color: 'var(--loss)' }}>{error}</strong>
            </div>
          )}

          <button
            type="submit"
            className="enter-btn"
            disabled={busy || !pw}
            style={{ width: '100%', marginTop: 'var(--s-4)', justifyContent: 'center' }}
          >
            {busy ? 'Checking…' : 'Sign in'}
          </button>

          <p className="note" style={{ marginTop: 'var(--s-4)', textAlign: 'center' }}>
            Private research terminal. Not investment advice.
          </p>
        </form>
      </div>
    </div>
  )
}
