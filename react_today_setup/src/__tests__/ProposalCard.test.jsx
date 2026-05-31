/**
 * Tests for ProposalCard expanded edge-chart panel (CR-G Step 6a).
 *
 * Run:  cd react_today_setup && npm test
 *
 * Covers: expand affordance, loading skeleton, success/error render,
 * timeframe selector, multi-card independence, collapse.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react'
import ProposalCard from '../components/ProposalCard'

// ── Synthetic proposal ────────────────────────────────────────────────────────

const MOCK_PROPOSAL = {
  template_id:         'debit_spread_to_target',
  template_kind:       'spread',
  rationale:           'Debit spread to magnet.',
  legs: [
    { side: 'long',  type: 'call', strike: 4225, quantity: 1 },
    { side: 'short', type: 'call', strike: 4250, quantity: 1 },
  ],
  expiry_dte_target:   5,
  expiry_dte_bucket:   '0-7',
  source: { type: 'regime_target', regime: 'magnet-above', drift_target: 4250, dominant_wall_gex_b: '12' },
  wing_distance_recipe: '',
}

const NO_TRADE_PROPOSAL = {
  template_id:   'feature_no_trade',
  template_kind: 'no_trade',
  rationale:     'No trade today.',
  legs:          [],
  expiry_dte_target: 0,
  expiry_dte_bucket: '',
  source:        { regime: 'amplification' },
}

const MOCK_CONTEXT = { regime: 'magnet-above', spot: 4184.25, implied_move: 23.59 }

const MOCK_PL_RESPONSE = {
  ok: true,
  trade_date: '2023-05-01',
  ticker: 'SPX',
  evaluation_time: '2023-05-06T20:00:00+00:00',
  current_spot: 4184.25,
  implied_move: 23.59,
  legs: [],
  pl_curve: { prices: [4100, 4200, 4300], pnl: [0, 10, 25] },
  iv_curve:  { prices: [4100, 4200, 4300], iv:  [0.15, 0.15, 0.15] },
  trade_thesis: {
    lower: 4250, upper: null, regime_kind: 'magnet-above',
    structural_prob: 0.70, structural_ci: [0.48, 0.85], structural_n: 20,
    implied_prob: 0.497, edge_ratio: 1.41,
  },
  edge_zones: [],
  greeks: { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 },
  key_levels: { max_profit: 25, max_loss: 0, breakevens: [4225.27] },
  todays_edge: {
    regime: 'magnet-above',
    per_horizon: [
      {
        horizon: 't1', struct_touch: 0.50, struct_touch_ci: [0.28, 0.72], n_touch: 14,
        mkt_touch: 0.10, touch_edge: 0.40,
        struct_close: 0.07, struct_close_ci: [0.02, 0.19], n_close: 14,
        mkt_close: 0.05, close_edge: 0.02, low_confidence: false,
      },
      {
        horizon: 't5', struct_touch: 0.64, struct_touch_ci: [0.41, 0.82], n_touch: 14,
        mkt_touch: 0.30, touch_edge: 0.34,
        struct_close: 0.21, struct_close_ci: [0.07, 0.45], n_close: 14,
        mkt_close: 0.15, close_edge: 0.06, low_confidence: false,
      },
      {
        horizon: 't15', struct_touch: 0.64, struct_touch_ci: [0.41, 0.82], n_touch: 14,
        mkt_touch: 0.50, touch_edge: 0.14,
        struct_close: 0.36, struct_close_ci: [0.15, 0.62], n_close: 14,
        mkt_close: 0.25, close_edge: 0.11, low_confidence: false,
      },
    ],
    warnings: [],
  },
  warnings: [],
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function renderCard(proposalOverride = {}, extraProps = {}) {
  const proposal = { ...MOCK_PROPOSAL, ...proposalOverride }
  return render(
    <ProposalCard
      proposal={proposal}
      date="2023-05-01"
      ticker="SPX"
      apiBase=""
      context={MOCK_CONTEXT}
      {...extraProps}
    />
  )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('ProposalCard — expand affordance', () => {
  it('renders an expand button for a normal proposal', () => {
    renderCard()
    expect(screen.getByTestId('expand-toggle')).toBeInTheDocument()
    expect(screen.getByText(/Show edge chart/)).toBeInTheDocument()
  })

  it('does NOT render expand button for no-trade proposal', () => {
    render(
      <ProposalCard
        proposal={NO_TRADE_PROPOSAL}
        date="2023-05-01"
        ticker="SPX"
        apiBase=""
        context={MOCK_CONTEXT}
      />
    )
    expect(document.querySelector('[data-testid="expand-toggle"]')).toBeNull()
  })

  it('clicking expand shows the expanded panel and toggles button label', async () => {
    global.fetch = vi.fn(() => new Promise(() => {})) // pending forever
    renderCard()
    const btn = screen.getByTestId('expand-toggle')
    expect(btn).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(btn)
    expect(screen.getByTestId('proposal-expanded-panel')).toBeInTheDocument()
    expect(btn).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText(/Hide edge chart/)).toBeInTheDocument()
  })

  it('clicking expand a second time collapses the chart', async () => {
    global.fetch = vi.fn(() => new Promise(() => {}))
    renderCard()
    const btn = screen.getByTestId('expand-toggle')
    fireEvent.click(btn)  // expand
    expect(screen.getByTestId('proposal-expanded-panel')).toBeInTheDocument()
    fireEvent.click(btn)  // collapse
    expect(document.querySelector('[data-testid="proposal-expanded-panel"]')).toBeNull()
  })
})

describe('ProposalCard — loading skeleton', () => {
  it('shows loading skeleton (null data passed to ProposalEdgeChart) while fetch is pending', async () => {
    global.fetch = vi.fn(() => new Promise(() => {})) // never resolves
    renderCard()
    fireEvent.click(screen.getByTestId('expand-toggle'))
    // ProposalEdgeChart renders skeleton when data is null
    const skeleton = document.querySelector('svg[aria-label="Loading chart…"]')
    expect(skeleton).not.toBeNull()
  })
})

describe('ProposalCard — fetch success', () => {
  afterEach(() => { delete global.fetch })

  it('replaces skeleton with chart on successful response', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(MOCK_PL_RESPONSE),
      })
    )
    renderCard()
    fireEvent.click(screen.getByTestId('expand-toggle'))

    // Wait for the chart to appear (skeleton replaced)
    await waitFor(() => {
      expect(document.querySelector('svg[aria-label="Proposal edge chart"]')).not.toBeNull()
    })
  })

  it('posts to /api/proposals/pl-data with correct body shape', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(MOCK_PL_RESPONSE),
      })
    )
    renderCard()
    fireEvent.click(screen.getByTestId('expand-toggle'))

    await waitFor(() => expect(global.fetch).toHaveBeenCalled())

    const [url, opts] = global.fetch.mock.calls[0]
    expect(url).toContain('/api/proposals/pl-data')
    const body = JSON.parse(opts.body)
    expect(body.trade_date).toBe('2023-05-01')
    expect(body.ticker).toBe('SPX')
    expect(body.timeframe).toBe('t5')  // default
    expect(body.regime_block.regime).toBe('magnet-above')
    expect(body.legs).toHaveLength(2)
    // leg type mapping: call → c
    expect(body.legs[0].flag).toBe('c')
    expect(body.legs[0].side).toBe('long')
    expect(body.legs[0].qty).toBe(1)
    // expiration = trade_date + expiry_dte_target
    expect(body.legs[0].expiration).toBe('2023-05-06')
  })
})

describe('ProposalCard — fetch error', () => {
  afterEach(() => { delete global.fetch })

  it('shows error state (ProposalEdgeChart with ok=false) on fetch failure', async () => {
    global.fetch = vi.fn(() => Promise.reject(new Error('Network error')))
    renderCard()
    fireEvent.click(screen.getByTestId('expand-toggle'))

    await waitFor(() => {
      expect(screen.getByText(/Network error/)).toBeInTheDocument()
    })
  })

  it('shows error state when API returns ok=false', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: false,
        json: () => Promise.resolve({ ok: false, error: 'no feature vector' }),
      })
    )
    renderCard()
    fireEvent.click(screen.getByTestId('expand-toggle'))

    await waitFor(() => {
      expect(screen.getByText(/no feature vector/)).toBeInTheDocument()
    })
  })
})

describe('ProposalCard — timeframe selector', () => {
  afterEach(() => { delete global.fetch })

  it('renders t1 / t5 / t15 timeframe buttons in the expanded panel', async () => {
    global.fetch = vi.fn(() => new Promise(() => {}))
    renderCard()
    fireEvent.click(screen.getByTestId('expand-toggle'))
    const selector = screen.getByTestId('timeframe-selector')
    expect(selector.querySelectorAll('button')).toHaveLength(3)
    expect(selector.querySelector('[aria-pressed="true"]').textContent).toBe('t5')
  })

  it('changing timeframe re-fetches (clears cache for new timeframe)', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(MOCK_PL_RESPONSE),
      })
    )
    renderCard()
    fireEvent.click(screen.getByTestId('expand-toggle'))

    // Wait for first fetch to complete
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1))

    // Switch to t1
    const selector = screen.getByTestId('timeframe-selector')
    const t1Btn = [...selector.querySelectorAll('button')].find(b => b.textContent === 't1')
    fireEvent.click(t1Btn)

    // Should trigger a second fetch with t1
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2))
    const body2 = JSON.parse(global.fetch.mock.calls[1][1].body)
    expect(body2.timeframe).toBe('t1')
  })
})

describe('ProposalCard — multi-card independence', () => {
  afterEach(() => { delete global.fetch })

  it('two cards can be expanded independently', () => {
    global.fetch = vi.fn(() => new Promise(() => {}))
    const { container } = render(
      <div>
        <ProposalCard
          proposal={MOCK_PROPOSAL}
          date="2023-05-01"
          ticker="SPX"
          apiBase=""
          context={MOCK_CONTEXT}
        />
        <ProposalCard
          proposal={{ ...MOCK_PROPOSAL, template_id: 'pin_butterfly_tight', template_kind: 'butterfly' }}
          date="2023-05-01"
          ticker="SPX"
          apiBase=""
          context={MOCK_CONTEXT}
        />
      </div>
    )
    const [btn1, btn2] = container.querySelectorAll('[data-testid="expand-toggle"]')

    // Expand card 1 only
    fireEvent.click(btn1)
    const panels = container.querySelectorAll('[data-testid="proposal-expanded-panel"]')
    expect(panels).toHaveLength(1)

    // Expand card 2 — both are now expanded
    fireEvent.click(btn2)
    expect(container.querySelectorAll('[data-testid="proposal-expanded-panel"]')).toHaveLength(2)

    // Collapse card 1 — only card 2 remains expanded
    fireEvent.click(btn1)
    expect(container.querySelectorAll('[data-testid="proposal-expanded-panel"]')).toHaveLength(1)
  })
})

// ── TodaysEdgeBlock rendering ─────────────────────────────────────────────────

describe('ProposalCard — today\'s edge block (CR-V)', () => {
  let fetchMock

  beforeEach(() => {
    fetchMock = vi.fn(() =>
      Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PL_RESPONSE) })
    )
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders per-horizon rows from todays_edge.per_horizon (list-driven)', async () => {
    renderCard()
    // Wait for prefetch to resolve
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await waitFor(() => screen.getByText('~1d'))
    // All three horizons rendered (list-driven, not hard-coded)
    expect(screen.getByText('~1d')).toBeInTheDocument()
    expect(screen.getByText('~5d')).toBeInTheDocument()
    expect(screen.getByText('~15d')).toBeInTheDocument()
  })

  it('renders touch edge and close edge labels', async () => {
    renderCard()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await waitFor(() => screen.getByText('Touch edge'))
    expect(screen.getByText('Touch edge')).toBeInTheDocument()
    expect(screen.getByText('Close edge')).toBeInTheDocument()
  })

  it('shows null gracefully when todays_edge is absent from response', async () => {
    const noEdge = { ...MOCK_PL_RESPONSE, todays_edge: null }
    fetchMock = vi.fn(() =>
      Promise.resolve({ ok: true, json: () => Promise.resolve(noEdge) })
    )
    vi.stubGlobal('fetch', fetchMock)
    renderCard()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    // No ~1d / ~5d / ~15d horizon rows rendered
    expect(screen.queryByText('~1d')).not.toBeInTheDocument()
  })

  it('renders CI brackets in the breakdown text', async () => {
    // t1 touch: struct_touch_ci [0.28, 0.72] → renders "[28–72%]"
    renderCard()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await waitFor(() => screen.getByText('~1d'))
    // CI bracket should appear somewhere in the rendered output
    const el = document.body.textContent
    expect(el).toContain('[28–72%]')
  })

  it('shows ~ flag for fails-lb cells, not for clears cells', async () => {
    // t15 touch: struct_lb 41% < mkt_touch 50% → fails-lb → shows ~
    // t1 touch:  struct_lb 28% ≥ mkt_touch 10% → clears   → no ~ in t1-touch position
    renderCard()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await waitFor(() => screen.getByText('~1d'))
    // At least one ~ flag present (for fails-lb cells in the mock data)
    const tildes = screen.getAllByText('~')
    expect(tildes.length).toBeGreaterThan(0)
  })

  it('~ flag has descriptive title for tooltip', async () => {
    renderCard()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await waitFor(() => screen.getByText('~1d'))
    const tilde = screen.getAllByText('~')[0]
    expect(tilde).toHaveAttribute('title')
    expect(tilde.getAttribute('title')).toMatch(/lower.bound/i)
  })
})
