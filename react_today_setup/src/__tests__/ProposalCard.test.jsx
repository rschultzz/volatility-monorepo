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
