import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, useAsync, type Candidate, type Sector } from '@/lib/api'
import { Badge, Callout, ErrorBox, HeadlineReturn, Loading, Panel, Provenance } from '@/components/ui'
import { Funnel, Metric, RegimeStrip, SignalCard, SignalPanel } from '@/components/terminal'
import { MarketCity, type CityStock } from '@/components/MarketCity'
import { stagger, useParallax } from '@/lib/motion'
import { DASH, inr, int, pct, ratio } from '@/lib/format'

/** The scan store's own column names for the four score components. */
const SCORE_PARTS = ['sc_fast', 'sc_absorbed', 'sc_rs', 'sc_cycles'] as const
const PART_LABELS: Record<string, string> = {
  sc_fast: 'Fast resolution',
  sc_absorbed: 'Resistance absorption',
  sc_rs: 'Relative strength',
  sc_cycles: 'Prior cycles',
}

function partsOf(c: Candidate): (number | null)[] {
  return SCORE_PARTS.map((k) => {
    const v = c[k] as unknown
    return typeof v === 'number' ? v : null
  })
}

export default function Dashboard() {
  const cfg = useAsync(() => api.config(), [])
  const scan = useAsync(() => api.scan(), [])
  const rec = useAsync(() => api.record(), [])
  const res = useAsync(() => api.results(), [])
  const uni = useAsync(() => api.universe(), [])
  const hist = useAsync(() => api.scanHistory(), [])
  const nav = useNavigate()
  const parallax = useParallax<HTMLDivElement>()
  const [sel, setSel] = useState<Candidate | null>(null)
  const [focusSector, setFocusSector] = useState<string | null>(null)

  if (cfg.loading || scan.loading) return <div className="page"><Loading what="terminal" /></div>
  if (cfg.error) return <div className="page"><ErrorBox error={cfg.error} /></div>

  const candidates = scan.data?.candidates ?? []
  const actionable = candidates.filter((c) => Boolean(c.actionable))
  const window = res.data?.windows?.['2022-2026']
  const frozen = window?.ladders?.['variant_c_12_20']
  const latest = (hist.data?.[0] ?? {}) as Record<string, unknown>
  const bk = cfg.data?.book
  const startCapital = 1_000_000
  const active = sel ?? candidates[0] ?? null

  const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

  const sectors = (uni.data?.sectors ?? []) as Sector[]
  const cityStocks: CityStock[] = candidates.map((c) => ({
    symbol: c.symbol,
    sector: c.industry ?? 'Unclassified',
    score: c.score,
    actionable: Boolean(c.actionable),
  }))

  return (
    <div className="page" ref={parallax}>
      {/* ---------------- hero ---------------- */}
      <div className="page-head reveal">
        <div className="row-between">
          <div>
            <div className="eyebrow" style={{ marginBottom: 6 }}>Market status</div>
            <h1 style={{ fontSize: 'var(--t-3xl)' }}>Command Center</h1>
          </div>
          <div className="row">
            <span className="row" style={{ gap: 6 }}>
              <span className="pulse" />
              <span className="eyebrow">Scan {scan.data?.scan_date ?? DASH}</span>
            </span>
          </div>
        </div>
      </div>

      {/* ---------------- headline metrics ---------------- */}
      <div className="grid grid-4">
        <Panel><Metric hero label="Actionable today" value={actionable.length} format={(v) => int(v)}
          sub={`${int(candidates.length)} candidates surfaced`} /></Panel>
        <Panel><Metric hero label="Forward record" value={rec.data?.n_total ?? 0} format={(v) => int(v)}
          sub={`day ${int(cfg.data?.forward_record_days)} · ${int(rec.data?.n_open)} open / ${int(rec.data?.n_closed)} closed`} /></Panel>
        <Panel><Metric hero label="Forward win rate" value={rec.data?.win_rate_pct ?? null}
          format={(v) => (v == null ? DASH : pct(v, 1))}
          tone={rec.data?.win_rate_pct != null ? rec.data.win_rate_pct - 50 : undefined}
          sub={rec.data?.n_closed ? `${int(rec.data.n_closed)} closed positions` : 'no closed positions yet'} /></Panel>
        <Panel><Metric hero label="Forward mean R" value={rec.data?.mean_r ?? null}
          format={(v) => (v == null ? DASH : ratio(v, 3))} tone={rec.data?.mean_r ?? undefined}
          sub="position level, not row mean" /></Panel>
      </div>

      {rec.data && rec.data.n_total === 0 && (
        <Callout>
          <strong>The forward record starts here.</strong> It begins at{' '}
          <span className="num">{rec.data.frozen_on}</span> and is currently empty. {rec.data.note}
        </Callout>
      )}

      {/* ---------------- universe + pipeline ---------------- */}
      <div className="grid split">
        <Panel
          title="Market structure"
          sub="Sector districts, sized by their share of the universe. Tower height is the existing conviction score; green towers are actionable."
          right={<span className="note">{focusSector ? `focused · ${focusSector}` : 'click a district to focus'}</span>}
          flush
        >
          <MarketCity
            sectors={sectors}
            stocks={cityStocks}
            height={380}
            focus={focusSector}
            onSelectSector={setFocusSector}
          />
          <div className="city-legend">
            <span className="legend-item">
              <span className="legend-swatch block" style={{ background: 'rgba(110,166,255,0.85)' }} />
              candidate · height = score
            </span>
            <span className="legend-item">
              <span className="legend-swatch block" style={{ background: 'rgba(79,211,165,0.85)' }} />
              actionable
            </span>
            <span className="legend-item">
              <span className="legend-swatch block" style={{ background: 'rgba(120,136,164,0.4)' }} />
              in universe, not surfaced today
            </span>
          </div>
        </Panel>

        <div className="grid" style={{ gap: 'var(--s-4)', alignContent: 'start' }}>
          <Panel title="Pipeline" sub="Where today's universe stands" flush>
            <Funnel
              stages={[
                { label: 'Universe', value: num(latest.universe_size) },
                { label: 'Watchlist', value: num(latest.watchlist_size) },
                { label: 'Candidates', value: num(latest.candidate_count) },
                { label: 'Actionable', value: num(latest.actionable_count) },
              ]}
            />
          </Panel>
          <Panel title="Regime">
            <RegimeStrip
              breadth={num(latest.breadth)}
              label={typeof latest.breadth_label === 'string' ? latest.breadth_label : null}
              pctile={num(latest.breadth_pctile)}
            />
          </Panel>
        </div>
      </div>

      {/* ---------------- candidates + signal panel ---------------- */}
      <div className="grid split-wide">
        <Panel
          title="Today's candidates"
          sub="Ranked by the existing conviction score. Components shown inline rather than collapsed into one number."
          right={<a className="cell-link note" href="/scanner">Full scanner →</a>}
        >
          {candidates.length ? (
            <div className="grid grid-3" style={{ gap: 'var(--s-3)' }}>
              {candidates.slice(0, 9).map((c, i) => (
                <SignalCard
                  key={c.id}
                  index={i}
                  symbol={c.symbol}
                  company={c.company}
                  sector={c.industry}
                  bucket={c.bucket}
                  score={c.score}
                  components={partsOf(c)}
                  boxTop={c.box_top as number}
                  boxBottom={c.box_bottom as number}
                  actionable={Boolean(c.actionable)}
                  onClick={() => setSel(c)}
                />
              ))}
            </div>
          ) : (
            <div className="empty">No candidates in the latest scan.</div>
          )}
        </Panel>

        <div className="grid" style={{ gap: 'var(--s-4)', alignContent: 'start' }}>
          <Panel
            title="What LeadFlow sees"
            sub={active ? `${active.symbol} · score decomposition` : 'Select a candidate'}
            right={active && (
              <button className="input" style={{ cursor: 'pointer' }} onClick={() => nav(`/stock/${encodeURIComponent(active.symbol)}`)}>
                Open →
              </button>
            )}
          >
            {active ? (
              <SignalPanel
                components={SCORE_PARTS.map((k) => ({
                  key: k,
                  label: PART_LABELS[k],
                  value: typeof active[k] === 'number' ? (active[k] as number) : null,
                  weight: 0.25,
                }))}
                total={active.score}
                bucket={active.bucket}
                note="Equal weight, 0.25 × 4. The calibrated weights are rejected — three of four factors were selected on the sample they score. This panel displays the engine's components and the engine's total; it does not recompute either."
              />
            ) : (
              <div className="empty">No candidate selected.</div>
            )}
          </Panel>

          <Panel title="Book allocation" sub={`${pct(Number(bk?.w_strategy) * 100, 0)} / ${pct(Number(bk?.w_arbitrage) * 100, 0)} / ${pct(Number(bk?.w_gold) * 100, 0)}, ${bk?.rebalance} rebalance`}>
            {[
              { name: 'Strategy', w: Number(bk?.w_strategy ?? 0.3), color: 'var(--cat-1)' },
              { name: 'Arbitrage', w: Number(bk?.w_arbitrage ?? 0.5), color: 'var(--cat-2)' },
              { name: 'Gold', w: Number(bk?.w_gold ?? 0.2), color: 'var(--cat-3)' },
            ].map((r, i) => (
              <div key={r.name} className="reveal" style={{ ...stagger(i, 60), marginBottom: 'var(--s-3)' }}>
                <div className="row-between" style={{ marginBottom: 4 }}>
                  <span className="row" style={{ gap: 6 }}>
                    <span className="legend-swatch block" style={{ background: r.color }} />
                    <span style={{ fontSize: 'var(--t-sm)' }}>{r.name}</span>
                  </span>
                  <span className="num note">{pct(r.w * 100, 0)} · {inr(r.w * startCapital)}</span>
                </div>
                <div className="meter">
                  <div className="meter-fill" style={{ width: `${r.w * 100}%`, background: r.color }} />
                </div>
              </div>
            ))}
            <p className="note">Gold is a diversifier and is never drawn on. The strategy sleeve may draw on arbitrage at T+1.</p>
          </Panel>

          {frozen && window && (
            <Panel
              title="Backtest reference"
              sub="2022-2026, frozen ladder"
              right={<Provenance value={window.provenance} note={window.provenance_note} />}
            >
              <HeadlineReturn
                label="Strategy CAGR"
                value={frozen.strategy_headline.cagr_pct}
                exBest={frozen.strategy_headline.ex_best_fold_pct}
                shareOfBest={frozen.strategy_headline.best_fold_share_pct}
              />
              <hr className="hairline" style={{ margin: 'var(--s-3) 0' }} />
              <dl className="kv">
                <dt>Sharpe</dt><dd className="num">{ratio(frozen.strategy.sharpe as number, 2)}</dd>
                <dt>Max drawdown</dt><dd className="num loss">{pct(frozen.strategy.max_drawdown_pct as number, 2)}</dd>
                <dt>Position R</dt><dd className="num">{ratio(frozen.position_r.value, 3)}</dd>
              </dl>
              <p className="note" style={{ marginTop: 'var(--s-3)' }}>
                Backtest, not live. The forward record above is the only out-of-sample evidence.
              </p>
            </Panel>
          )}
        </div>
      </div>

      <div className="row" style={{ justifyContent: 'flex-end' }}>
        <Badge>{int(uni.data?.n)} names in universe</Badge>
      </div>
    </div>
  )
}
