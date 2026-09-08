/**
 * CR-AW decision 8: the Setup v2 card — verdict for quote under / over max and
 * for no listed structure; analogue wording ("N analogues within the
 * similarity ceiling", never "K="); band selection; KNN tab rows.
 *
 * Run:  cd react_today_setup && npm test
 */
import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { SetupV2Card } from '../setupv2/SetupV2App'
import TradeTab from '../setupv2/components/TradeTab'
import KnnFactorsTab from '../setupv2/components/KnnFactorsTab'

function cell(band, mean_pnl, win_rate, lo, hi, n) {
  return { band, mean_pnl, win_rate, wilson_lo: lo, wilson_hi: hi, n, n_dates: n, baseline_mean: mean_pnl + 0.03, beat_baseline: -0.03, threshold: 0.05, partition: 'train', mean_width_actual: 10 }
}

const NEAR = cell('near', 1.9116, 0.6842, 0.5254, 0.8092, 38)
const MID  = cell('mid',  1.2349, 0.5588, 0.3945, 0.7112, 34)
const FAR  = cell('far',  1.9937, 0.4839, 0.3197, 0.6516, 31)

function card(overrides = {}) {
  const base = {
    ok: true, date: '2026-09-03', ticker: 'SPX',
    context: { spot: 7704.25, spot_source: 'bars', regime: 'magnet-above', implied_move: 24.27 },
    wall: { price_es: 7806.175, gex_b: 810.3, sigma: 2.5496, band: 'far', regime: 'magnet-above', above_spot: true },
    structure: {
      template_id: 'debit_spread_to_target', listed: true, listed_reason: null, direction: 'call',
      legs: [
        { side: 'long',  type: 'call', strike_es: 7796.175, strike_spx: 7775, opra: 'SPX260918C07775000', bid: 41.7, ask: 42.4, mid: 42.05, quote_valid: true, stale_quote: false, quote_minute: '06:34' },
        { side: 'short', type: 'call', strike_es: 7806.175, strike_spx: 7785, opra: 'SPX260918C07785000', bid: 37.6, ask: 38.3, mid: 37.95, quote_valid: true, stale_quote: false, quote_minute: '06:34' },
      ],
      short_strike_spx: 7785, long_strike_spx: 7775, width_nominal: 10, width_actual: 10,
      expiry: '2026-09-25', dte_calendar: 22, sessions_target: 15, expiry_target: '2026-09-25', expiry_listed: true, expiry_dte_target: 15,
    },
    quote: { net_debit: 4.1, quote_minute: '06:34', quote_valid: true, stale_quote: false, market_implied: 0.41, entry_minute_pt: '06:34',
             by_minute: [{ minute: '06:34', net_debit: 4.1, valid: true }, { minute: '06:59', net_debit: 4.2, valid: true }], window_pt: ['06:30', '07:00'], warnings: [] },
    fair_value: { n: 65, width: 10, fair: 5.206, max_price: 4.72, boot_lo: 4.05, boot_hi: 6.39, full_payout_rate: 0.5077, any_payout_rate: 0.5385, basis: 't15', valuation_horizon_sessions: 15, n_computed: 68, n_valued: 65, n_no_t15_close: 3, n_no_horizon_close: 0, horizon_mix: { '5': 34, '20': 28, '60': 6 }, n_boot: 1000, seed: 20260903 },
    pnl: { expected: 1.08, lo: -0.076, hi: 2.264, fee_pts: 0.026, fee_per_contract_per_leg: 0.65 },
    verdict: { code: 'enter', text: 'quote is under the max — enter at the open' },
    band: { today: 'far', sigma: 2.5496, im_open_straddle: 53.34, table_spot: 7670.175, cr_id: 'CR-AR', rows: [NEAR, MID, FAR], all: null, today_cell: FAR },
    analogues: { k: 70, k_with_outcomes: 68, touch_rate: 0.7353, touch_ci: [0.6199, 0.8255], mean_days_to_reach: 4.84, close_at_wall_rate: 0.1029, full_payout_rate: 0.5077, any_payout_rate: 0.5385, similarity_ceiling: 5.0 },
    structural_probability: {
      touch_rate: 0.7353, post_touch: { pattern_label: 'stepping-stone', same_bucket_n: 25, total_touchers: 50, advisory: { n: 25 },
        fractions: { t1: { below: 0.2, at: 0.2, above: 0.6 }, t5: { below: 0.3, at: 0.1, above: 0.6 }, t15: { below: 0.36, at: 0.08, above: 0.56 } } },
    },
    knn: {
      config_version: 'v3', distance_ceiling: 5.0, half_life_months: 18, corpus_n: 394, analogue_n: 70,
      factors: [
        { key: 'cluster_1_signed_distance_sigma', label: 'Distance to wall', group: 'Wall geometry', lo_label: 'closer', hi_label: 'farther', today: 1.0, populated: true, weight: 3.0, percentile: 34, band_lo: 0.8, band_hi: 1.3, band_lo_pct: 22, band_hi_pct: 52, in_band: true, corpus_n: 394, analogue_n: 70 },
        { key: 'dominance_30plus', label: '30+ DTE share', group: 'Gamma by expiry', lo_label: 'little', hi_label: 'lots', today: 35, populated: true, weight: 0.75, percentile: 72, band_lo: 10, band_hi: 25, band_lo_pct: 20, band_hi_pct: 50, in_band: false, corpus_n: 394, analogue_n: 70 },
        { key: 'atm_iv_percentile', label: 'IV rank', group: 'Volatility surface', lo_label: 'low', hi_label: 'high', today: null, populated: false, weight: 1.0, percentile: null, band_lo: null, band_hi: null, band_lo_pct: null, band_hi_pct: null, in_band: null, corpus_n: 0, analogue_n: 0 },
      ],
      match_quality: { in_band: 1, total: 2, outliers: ['dominance_30plus'] },
    },
    manage: { enter_under_max: 4.72, dte_calendar: 22, sessions_target: 15, expiry: '2026-09-25',
      watches: [
        { key: 'wall_half_life', label: 'Wall watch', status: 'untested', value: null, note: 'needs the wall table (CR-AT)' },
        { key: 'vol_state', label: 'Vol state', status: 'untested', value: { implied_move: 24.27, implied_move_percentile: 7.87 }, note: 'IV rank / VRP not populated' },
      ] },
    stamp: { cr_id: 'CR-AR', cell: 'far · debit · hold to close', n: 31, rerun_date: '2026-09-07', next_run: '2026-10-01', fees_included: true, fee_per_contract_per_leg: 0.65, fee_pts: 0.026, quote_minute_pt: '06:34' },
    warnings: [],
  }
  return { ...base, ...overrides }
}

describe('Setup v2 card — verdict (decision 5)', () => {
  it('quote under max → enter at the open, gauge shows quote / max / fair', () => {
    render(<TradeTab card={card()} apiBase="" />)
    const v = screen.getByTestId('verdict')
    expect(v.dataset.code).toBe('enter')
    expect(v.textContent).toContain('enter at the open')
    expect(screen.getByTestId('max-price').textContent).toContain('4.7')
    expect(screen.getByTestId('fair-value').textContent).toBe('5.2')
    expect(screen.getByTestId('price-gauge').textContent).toContain('quote now 4.10')
    expect(screen.getByTestId('expected-pnl').textContent).toContain('+1.1')
    expect(screen.getByTestId('market-implied').textContent).toBe('41%')
    expect(screen.getByTestId('expiry-tag').textContent).toBe('15 sessions (25 Sep)')
    expect(screen.getByTestId('structure-name').textContent).toContain('25 Sep')
  })

  it('quote above max → skip, never wait', () => {
    const c = card({ quote: { ...card().quote, net_debit: 5.6, market_implied: 0.56 }, verdict: { code: 'skip', text: 'skip — quote above max' } })
    render(<TradeTab card={c} apiBase="" />)
    const v = screen.getByTestId('verdict')
    expect(v.dataset.code).toBe('skip')
    expect(v.textContent).toContain('quote above max')
    expect(v.textContent.toLowerCase()).not.toContain('wait')
  })

  it('no listed structure → the message, no legs, verdict none', () => {
    const c = card({
      structure: { ...card().structure, listed: false, listed_reason: 'no_listed_structure', legs: [] },
      quote: { ...card().quote, net_debit: null, market_implied: null, by_minute: [] },
      verdict: { code: 'no_structure', text: 'no listed structure at this expiry' },
    })
    render(<TradeTab card={c} apiBase="" />)
    expect(screen.getByTestId('no-listed-structure')).toBeInTheDocument()
    expect(screen.queryByTestId('structure-name')).toBeNull()
    expect(screen.getByTestId('verdict').textContent).toBe('no listed structure at this expiry')
  })
})

describe('Setup v2 card — analogue wording and band', () => {
  it('says "analogues within the similarity ceiling" and never "K="', () => {
    render(<SetupV2Card card={card()} apiBase="" />)
    const text = screen.getByTestId('setup-v2-card').textContent
    expect(text).toContain('70 analogues within the similarity ceiling')
    expect(text).toContain('68 with a computed outcome')
    expect(text).not.toMatch(/K=/)
    expect(text).not.toMatch(/edge ×/i)
    expect(text).not.toContain('supported')
    expect(text).not.toContain('low-confidence')
  })

  it('outlines today\'s band and states the T+15 basis, exclusions and the at-wall rate separately', () => {
    render(<TradeTab card={card()} apiBase="" />)
    expect(screen.getByTestId('band-far').className).toContain('today')
    expect(screen.getByTestId('band-near').className).not.toContain('today')
    expect(screen.getByTestId('band-far').textContent).toContain('+1.99')
    expect(screen.getByTestId('band-fact').textContent).toContain('Today is far')
    const fvBox = screen.getByTestId('fair-value').parentElement.textContent
    expect(fvBox).toContain('at T+15 sessions')
    expect(fvBox).toContain('68 with a computed outcome, 3 without a T+15 close excluded')
    const facts = screen.getByTestId('analogue-facts').textContent
    expect(facts).toContain('Finished above 7785')
    expect(facts).toContain('51%')
    expect(facts).toContain('at or above the wall at T+15')
    expect(facts).toContain('Closed at the wall')
    expect(facts).toContain('10%')
  })

  it('stamp carries the reference, cell, re-run, next run and fees', () => {
    render(<TradeTab card={card()} apiBase="" />)
    const s = screen.getByTestId('stamp').textContent
    expect(s).toContain('CR-AR')
    expect(s).toContain('far · debit · hold to close')
    expect(s).toContain('n 31')
    expect(s).toContain('7 Sep 2026')
    expect(s).toContain('1 Oct')
    expect(s).toContain('$0.65 / contract / leg')
  })

  it('untested watches render grey and say untested', () => {
    render(<TradeTab card={card()} apiBase="" />)
    const w = screen.getByTestId('watch-vol_state')
    expect(w.className).toBe('untested')
    expect(w.textContent).toContain('untested')
    expect(w.textContent).toContain('8th percentile')
  })

  it('detail toggle reveals post-touch bars and the quote strip', () => {
    render(<TradeTab card={card()} apiBase="" />)
    expect(screen.queryByTestId('detail-panel')).toBeNull()
    fireEvent.click(screen.getByTestId('detail-toggle'))
    expect(screen.getByTestId('post-touch-bars')).toBeInTheDocument()
    expect(screen.getByTestId('quote-strip')).toBeInTheDocument()
  })
})

describe('Setup v2 — KNN factors tab (decision 6)', () => {
  it('lists every factor with today, weight, percentile, band and read; null shows not populated', () => {
    render(<KnnFactorsTab card={card()} />)
    const dist = screen.getByTestId('factor-cluster_1_signed_distance_sigma').textContent
    expect(dist).toContain('+1.00 IM')
    expect(dist).toContain('×3.0')
    expect(dist).toContain('34th')
    expect(dist).toContain('Inside the analogue band')
    const out = screen.getByTestId('factor-dominance_30plus').textContent
    expect(out).toContain('Outside the analogue band')
    const np = screen.getByTestId('factor-atm_iv_percentile').textContent
    expect(np).toContain('not populated')
    expect(screen.getByTestId('match-quality').textContent).toContain('1 of 2 populated factors inside the analogue band')
    expect(screen.getByTestId('match-quality').textContent).toContain('30+ DTE share is the outlier')
    expect(screen.getByTestId('knn-tab').textContent).toContain('similarity ceiling of 5')
    expect(screen.getByTestId('knn-tab').textContent).not.toMatch(/K=/)
  })

  it('tab switch from Trade to KNN factors', () => {
    render(<SetupV2Card card={card()} apiBase="" />)
    expect(screen.getByTestId('trade-tab')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: /KNN factors/ }))
    expect(screen.getByTestId('knn-tab')).toBeInTheDocument()
    expect(screen.queryByTestId('trade-tab')).toBeNull()
  })
})
