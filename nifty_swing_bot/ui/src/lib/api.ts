/** The LeadFlow API client.
 *
 * There is exactly one backend and these are all of its endpoints. The DAB
 * router is not mounted server-side, so there is no path from this file to the
 * old signal ledger even by accident.
 */

import { useEffect, useState } from 'react'

export interface Concentration {
  mean: number | null
  median: number | null
  ex_best_period: number | null
  ex_top2: number | null
  best_period_share_pct: number | null
  positive: string | null
}

export interface Headline {
  total_return_pct: number | null
  cagr_pct: number | null
  ex_best_fold_pct: number | null
  best_fold_share_pct: number | null
  mean_fold_pct: number | null
  median_fold_pct: number | null
  positive_folds: string | null
}

export type Metrics = Record<string, number | string | null>

export interface Ladder {
  params: Record<string, number>
  n_positions: number
  n_traded_rows: number
  strategy: Metrics
  book: Metrics
  strategy_headline: Headline
  book_headline: Headline
  fold_years: number[]
  strategy_folds_pct: number[]
  book_folds_pct: number[]
  position_r: {
    value: number | null
    basis: string
    row_mean_do_not_use: number | null
    why: string
  }
  diagnostics: Record<string, number>
  gold_correlation_by_year: Record<string, number | string>[]
}

export interface WindowResults {
  provenance: string
  provenance_note: string
  start: string
  end: string
  signals: number
  ladders: Record<string, Ladder>
}

export interface Results {
  generated: string
  frozen_on: string
  config: string
  windows: Record<string, WindowResults>
}

export interface Series {
  start: string
  end: string
  dates: string[]
  strategy_index: (number | null)[]
  book_index: (number | null)[]
  benchmark_index: (number | null)[]
  drawdown_strategy_pct: (number | null)[]
  drawdown_book_pct: (number | null)[]
  sleeve_strategy: (number | null)[]
  sleeve_arb: (number | null)[]
  sleeve_gold: (number | null)[]
  book_value: (number | null)[]
  deployment_pct_of_book: (number | null)[]
  rebalance_dates: string[]
  rolling_12m_pct: { dates: string[]; values: (number | null)[] }
  daily_returns_pct: (number | null)[]
  r_multiples: (number | null)[]
  trades: Record<string, unknown>[]
  gold_correlation_by_year: Record<string, number | string>[]
  n_positions: number
  n_signals: number
  reconciliation: boolean | null
  ladder: string
}

export interface Config {
  product: string
  frozen_on: string
  forward_record_days: number
  forward_record_entries: number
  forward_record_since: string | null
  describe: string
  warning: string
  sizing: { slots: number; risk_per_trade_pct: number; conviction_multipliers: number[] }
  exits: {
    atr_period: number
    atr_mult: number
    max_hold_bars: number
    ladder: { trigger_pct: number; fraction_pct: number }[]
    remainder_trails_pct: number
    ladder_name: string
    ladder_status: string
    ladder_tradeoff: string
  }
  book: Record<string, number | string | boolean>
  scoring: { weights: Record<string, number>; note: string }
}

export interface Candidate {
  id: number
  scan_date: string
  symbol: string
  company: string | null
  industry: string | null
  bucket: string
  score: number
  actionable: number | boolean
  entry_mode: number | null
  box_top: number | null
  box_bottom?: number | null
  box_bars?: number | null
  box_range_pct?: number | null
  [k: string]: unknown
}

export interface ScanPayload {
  scan_date: string | null
  candidates: Candidate[]
  rejected: Record<string, unknown>[]
}

export interface ForwardRecord {
  frozen_on: string
  note: string
  entries: Record<string, unknown>[]
  n_total: number
  n_open: number
  n_closed: number
  win_rate_pct: number | null
  mean_r: number | null
}

export interface Sector {
  industry: string
  symbols: number
  weight_pct: number
  candidates: number
  mean_score: number | null
  max_score: number | null
  signal_density_pct: number
}

export interface Universe {
  available: boolean
  n: number
  sectors: Sector[]
}

export interface ScanStatus {
  running: boolean
  started_at: string | null
  finished_at: string | null
  error: string | null
  candidates: number | null
  actionable: number | null
}

async function post<T>(path: string): Promise<T> {
  const r = await fetch(path, { method: 'POST', headers: { Accept: 'application/json' } })
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`
    try {
      const j = await r.json()
      if (j?.detail) detail = String(j.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return (await r.json()) as T
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { headers: { Accept: 'application/json' } })
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`
    try {
      const j = await r.json()
      if (j?.detail) detail = String(j.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return (await r.json()) as T
}

export const api = {
  config: () => get<Config>('/api/config'),
  results: () => get<Results>('/api/results'),
  series: (w: string) => get<Series>(`/api/series/${encodeURIComponent(w)}`),
  windows: () => get<{ name: string; provenance: string; start: string; end: string; signals: number }[]>('/api/windows'),
  scan: () => get<ScanPayload>('/api/scan/latest'),
  scanHistory: () => get<Record<string, unknown>[]>('/api/scan/history'),
  record: () => get<ForwardRecord>('/api/record'),
  universe: () => get<Universe>('/api/universe'),
  health: () => get<Record<string, unknown>>('/api/health'),

  // The scan trigger and its progress poll are the scanner's OWN existing
  // endpoints, called unchanged. The UI starts the same background job the
  // evening tool starts; it does not reimplement any part of a scan.
  startScan: () => post<{ started: boolean }>('/api/aes/scan'),
  scanStatus: () => get<ScanStatus>('/api/aes/scan/status'),
}

export interface Async<T> {
  data: T | null
  error: string | null
  loading: boolean
}

/** One fetch hook. Deliberately minimal -- no cache layer, because every page
 *  here wants the current state of a small payload, and a stale-cache bug on a
 *  trading screen is worse than a refetch. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): Async<T> {
  const [state, setState] = useState<Async<T>>({ data: null, error: null, loading: true })
  useEffect(() => {
    let alive = true
    setState((s) => ({ ...s, loading: true, error: null }))
    fn()
      .then((d) => alive && setState({ data: d, error: null, loading: false }))
      .catch((e: Error) => alive && setState({ data: null, error: e.message, loading: false }))
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return state
}
