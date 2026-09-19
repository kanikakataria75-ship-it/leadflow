import { api, useAsync } from '@/lib/api'
import { Badge, Callout, Col, DataTable, ErrorBox, Loading, Panel, Stat } from '@/components/ui'
import { DASH, int, pct, ratio } from '@/lib/format'

interface Row { k: string; v: string; note?: string }

export default function Settings() {
  const cfg = useAsync(() => api.config(), [])
  const health = useAsync(() => api.health(), [])

  if (cfg.loading) return <div className="page"><Loading what="configuration" /></div>
  if (cfg.error) return <div className="page"><ErrorBox error={cfg.error} /></div>

  const c = cfg.data!
  const locked = health.data?.config_locked !== false

  const cols: Col<Row>[] = [
    { key: 'k', head: 'Parameter', render: (r) => r.k },
    { key: 'v', head: 'Value', num: true, render: (r) => r.v },
    { key: 'n', head: 'Why', render: (r) => <span className="note">{r.note ?? ''}</span> },
  ]

  const geometry: Row[] = [
    { k: 'Box geometry', v: 'baseline', note: 'Every loosening tested lowered fold return; sloped/range/duration variants are implemented and default-off.' },
    { k: 'Admission move', v: '+20% / 10d', note: 'The 20/5 cell is the best in its grid; both conditions are load-bearing.' },
    { k: 'Admission proximity', v: 'within 5% of 52w high', note: 'Softening to 15/5 or 15/10 cut edge from +1.83% to +0.57%.' },
  ]

  const sizing: Row[] = [
    { k: 'Slots', v: int(c.sizing.slots), note: 'The only cell above 10% annual in both windows.' },
    { k: 'Risk per trade', v: pct(c.sizing.risk_per_trade_pct, 0), note: 'Paired with the slot count; changing one without the other changes the policy.' },
    { k: 'Conviction multipliers', v: c.sizing.conviction_multipliers.map((m) => ratio(m, 1)).join(' / '), note: 'Score-to-edge is not monotone, so nothing steeper was adopted; 0.7/1.2 beat both 1.0/1.0 and 0.5/1.5.' },
  ]

  const exits: Row[] = [
    { k: 'Trailing stop', v: `ATR(${c.exits.atr_period}) × ${ratio(c.exits.atr_mult, 1)}`, note: 'The one component with fold-by-fold evidence behind it. Beat every SMA variant: 37.84 vs 26.77 best buffer.' },
    { k: 'Hard hold cap', v: `${int(c.exits.max_hold_bars)} bars`, note: 'Capped beats uncapped, 37.84 vs 24.25.' },
    ...c.exits.ladder.map((l, i) => ({
      k: `Tranche ${i + 1}`,
      v: `${pct(l.fraction_pct, 0)} @ +${pct(l.trigger_pct, 0)}`,
      note: i === 0 ? 'The spec +5%/+8% tranches truncate the return distribution: zero winners above +30% across 398 positions.' : undefined,
    })),
    { k: 'Remainder', v: `${pct(c.exits.remainder_trails_pct, 0)} trails`, note: 'Runs on the ATR trail until the stop or the cap.' },
  ]

  const book: Row[] = [
    { k: 'Strategy sleeve', v: pct(Number(c.book.w_strategy) * 100, 0) },
    { k: 'Arbitrage sleeve', v: pct(Number(c.book.w_arbitrage) * 100, 0), note: `Accrues at ${pct(Number(c.book.arb_annual_rate) * 100, 2)} annual, T+1 redemption.` },
    { k: 'Gold sleeve', v: pct(Number(c.book.w_gold) * 100, 0), note: 'Marked to GOLDBEES. Never drawn on.' },
    { k: 'Rebalance', v: String(c.book.rebalance), note: 'BookParams() defaults to quarterly — that default silently produced a different book in an earlier result file, and is now guarded.' },
    { k: 'Sizing basis', v: String(c.book.strategy_capital_basis), note: 'Risk is a fraction of the sleeve, not the whole book.' },
  ]

  const scoring: Row[] = Object.entries(c.scoring.weights).map(([k, v]) => ({
    k: k.replace(/_/g, ' '),
    v: ratio(v, 2),
  }))

  return (
    <div className="page">
      <div className="page-head">
        <div className="row-between">
          <h1>Settings</h1>
          <Badge tone={locked ? 'accent' : 'loss'}>{locked ? 'FROZEN — MATCHES' : 'CONFIG DRIFTED'}</Badge>
        </div>
        <p className="note">
          The frozen configuration, read-only. Frozen{' '}
          <span className="num">{c.frozen_on}</span>.
        </p>
      </div>

      <div className="grid split-even">
        <Panel flush>
          <div className="core-ring">
            <div className="core-badge">
              <div>
                <div className="eyebrow" style={{ color: 'var(--accent-bright)', marginBottom: 2 }}>
                  {locked ? 'Frozen' : 'Drifted'}
                </div>
                <div className="num" style={{ fontSize: 'var(--t-sm)' }}>{c.frozen_on}</div>
                <div className="note" style={{ marginTop: 2 }}>day {int(c.forward_record_days)}</div>
              </div>
            </div>
          </div>
          <div className="city-legend" style={{ justifyContent: 'center' }}>
            <span className="note">
              {locked
                ? 'The running configuration matches the freeze. The server refuses to start if it does not.'
                : 'The running configuration does NOT match the freeze.'}
            </span>
          </div>
        </Panel>

        <Panel title="Frozen parameters" sub="Read-only. These are the values every result on this terminal was produced under." flush>
          <div className="grid grid-2" style={{ gap: 0 }}>
            {[
              { v: int(c.sizing.slots), l: 'slots' },
              { v: pct(c.sizing.risk_per_trade_pct, 0), l: 'risk / trade' },
              { v: `ATR x ${ratio(c.exits.atr_mult, 1)}`, l: `period ${int(c.exits.atr_period)}` },
              { v: `${int(c.exits.max_hold_bars)} bar`, l: 'hold cap' },
              { v: c.sizing.conviction_multipliers.map((m) => ratio(m, 1)).join(' / '), l: 'conviction multipliers' },
              { v: `${pct(Number(c.book.w_strategy) * 100, 0)} / ${pct(Number(c.book.w_arbitrage) * 100, 0)} / ${pct(Number(c.book.w_gold) * 100, 0)}`, l: `book · ${String(c.book.rebalance)}` },
            ].map((x, i) => (
              <div key={x.l} className="funnel-stage reveal" style={{ ['--delay' as string]: `${i * 50}ms` }}>
                <div className="lbl">{x.l}</div>
                <div className="n" style={{ fontSize: 'var(--t-xl)' }}>{x.v}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <Callout>
        <strong>{c.warning}</strong> The server refuses to start on a configuration that does not match the freeze, so a drifted
        deploy fails loudly rather than quietly serving numbers from a different experiment.
      </Callout>

      {!locked && (
        <div className="callout" style={{ borderColor: 'rgba(224,85,97,0.35)', background: 'var(--loss-dim)' }}>
          <strong style={{ color: 'var(--loss)' }}>The running configuration does not match the freeze.</strong>{' '}
          {String(health.data?.config_error ?? '')}
        </div>
      )}

      <div className="grid grid-4">
        <Panel><Stat label="Frozen on" value={<span className="num">{c.frozen_on}</span>} sub="forward record epoch" /></Panel>
        <Panel><Stat label="Days live" value={int(c.forward_record_days)} sub="since the freeze" /></Panel>
        <Panel><Stat label="Record entries" value={int(c.forward_record_entries)} sub={c.forward_record_since ?? 'none yet'} /></Panel>
        <Panel><Stat label="Ladder" value={c.exits.ladder_name} sub={c.exits.ladder_status} /></Panel>
      </div>

      <Callout>
        <strong>Ladder trade-off, stated rather than buried.</strong> {c.exits.ladder_tradeoff}
      </Callout>

      <div className="grid grid-2">
        <Panel title="Geometry & admission" flush><DataTable cols={cols} rows={geometry} rowKey={(r) => r.k} /></Panel>
        <Panel title="Sizing" flush><DataTable cols={cols} rows={sizing} rowKey={(r) => r.k} /></Panel>
      </div>

      <div className="grid grid-2">
        <Panel title="Exits" flush><DataTable cols={cols} rows={exits} rowKey={(r) => r.k} /></Panel>
        <Panel title="Book" flush><DataTable cols={cols} rows={book} rowKey={(r) => r.k} /></Panel>
      </div>

      <div className="grid grid-2">
        <Panel title="Scoring weights" sub={c.scoring.note} flush>
          <DataTable cols={cols} rows={scoring} rowKey={(r) => r.k} />
        </Panel>
        <Panel title="Raw freeze record">
          <pre style={{
            margin: 0, fontFamily: 'var(--font-mono)', fontSize: 'var(--t-xs)',
            color: 'var(--ink-2)', whiteSpace: 'pre-wrap', lineHeight: 1.6,
          }}>
            {c.describe}
          </pre>
          <p className="note" style={{ marginTop: 'var(--s-3)' }}>
            Artefacts:{' '}
            {Object.entries((health.data?.artefacts ?? {}) as Record<string, boolean>).map(([k, v]) => (
              <span key={k} style={{ marginRight: 10 }}>
                <span style={{ color: v ? 'var(--gain)' : 'var(--loss)' }}>{v ? '●' : '○'}</span> {k}
              </span>
            )) || DASH}
          </p>
        </Panel>
      </div>
    </div>
  )
}
