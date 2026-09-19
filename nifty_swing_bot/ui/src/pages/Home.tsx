import { Link } from 'react-router-dom'
import { api, useAsync } from '@/lib/api'
import { MarketCity, type CityStock } from '@/components/MarketCity'
import { CommandBar } from '@/components/CommandBar'
import { Funnel } from '@/components/terminal'
import { useParallax } from '@/lib/motion'
import { DASH, int } from '@/lib/format'

/** The entry screen.
 *
 * The city behind the wordmark is the real universe: real sectors, sized by
 * their real symbol counts, with the live scan's candidates raised as lit
 * towers. Every counter under it comes from the same scan-history row the
 * Scanner page reads. When the scan store is empty the counters read as dashes
 * rather than zeros -- an absent scan and a scan that found nothing are
 * different facts and the entry screen should not blur them.
 */
export default function Home() {
  const scan = useAsync(() => api.scan(), [])
  const uni = useAsync(() => api.universe(), [])
  const hist = useAsync(() => api.scanHistory(), [])
  const cfg = useAsync(() => api.config(), [])
  const parallax = useParallax<HTMLDivElement>()

  const latest = (hist.data?.[0] ?? {}) as Record<string, unknown>
  const candidates = scan.data?.candidates ?? []
  const sectors = uni.data?.sectors ?? []

  const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

  const stocks: CityStock[] = candidates.map((c) => ({
    symbol: c.symbol,
    sector: c.industry ?? 'Unclassified',
    score: c.score,
    actionable: Boolean(c.actionable),
  }))

  return (
    <div className="home" ref={parallax}>
      <div className="home-city par-1">
        <MarketCity
          sectors={sectors}
          stocks={stocks}
          height="100%"
          interactive={false}
          showLabels={false}
        />
      </div>

      <CommandBar />

      <div className="home-inner">
        <div
          className="reveal par-2"
          style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-4)', alignItems: 'center' }}
        >
          <div className="hero-kicker">Market Intelligence System</div>
          <h1 className="hero-title">
            Lead<span className="accent">Flow</span>
          </h1>
          <div className="hero-flow">
            Accumulation <span className="arrow">→</span> Expansion
          </div>
          <p className="hero-tag">Systematic. Disciplined. Evidence driven.</p>
        </div>

        <div
          className="panel reveal par-3"
          style={{ ['--delay' as string]: '160ms', maxWidth: 1000, width: '100%' }}
        >
          <div className="panel-head">
            <div className="panel-title">
              Market pipeline
              {scan.data?.scan_date && (
                <span className="note" style={{ marginLeft: 8 }}>scan {scan.data.scan_date}</span>
              )}
            </div>
            <span className="row" style={{ gap: 6 }}>
              <span className="pulse" />
              <span className="eyebrow">Live</span>
            </span>
          </div>
          <div className="panel-body flush">
            <Funnel
              stages={[
                { label: 'Stocks scanned', value: num(latest.universe_size) },
                { label: 'Watchlist', value: num(latest.watchlist_size) },
                { label: 'Candidates', value: num(latest.candidate_count) },
                { label: 'Actionable', value: num(latest.actionable_count) },
              ]}
            />
          </div>
        </div>

        <div
          className="reveal"
          style={{
            ['--delay' as string]: '260ms',
            display: 'flex', flexDirection: 'column', gap: 'var(--s-4)', alignItems: 'center',
          }}
        >
          <Link className="enter-btn" to="/dashboard">
            Enter terminal <span aria-hidden="true">→</span>
          </Link>
          <p className="note" style={{ maxWidth: 560 }}>
            Configuration frozen <span className="num">{cfg.data?.frozen_on ?? DASH}</span> · forward
            record day <span className="num">{int(cfg.data?.forward_record_days)}</span>. Research
            tooling, not investment advice.
          </p>
        </div>

        <div className="home-strip reveal" style={{ ['--delay' as string]: '340ms' }}>
          <div>
            <h4>Research driven</h4>
            <p className="note">Evidence over opinion.</p>
          </div>
          <div>
            <h4>Risk controlled</h4>
            <p className="note">Survive to compound.</p>
          </div>
          <div>
            <h4>Process oriented</h4>
            <p className="note">Consistency over emotion.</p>
          </div>
          <div>
            <h4>Long-term edge</h4>
            <p className="note">Built, not found.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
