/**
 * Tests for ProposalEdgeChart (CR-G Step 5b).
 *
 * Run:  cd packages/web-shared && npm test
 *
 * Uses a synthetic Step 4 response with enough structure to exercise all
 * rendering branches. No live API call.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ProposalEdgeChart from '../ProposalEdgeChart.jsx'

// ── Synthetic Step 4 response ─────────────────────────────────────────────────
// Covers: magnet-above (one-sided upper tail), 2 edge zones, 1 breakeven.

function makePrices(lo, hi, n = 40) {
  return Array.from({ length: n }, (_, i) => lo + (hi - lo) * (i / (n - 1)))
}

const PRICES = makePrices(4100, 4300, 40)
const PNLS   = PRICES.map(p => Math.max(0, Math.min(25, p - 4225)))
const IV     = PRICES.map(() => 0.15)

const MOCK_DATA = {
  ok: true,
  trade_date: '2023-05-01',
  ticker: 'SPX',
  evaluation_time: '2023-05-05T20:00:00+00:00',
  current_spot: 4184.25,
  implied_move: 23.59,
  legs: [
    { strike: 4225, expiration: '2023-05-05', flag: 'c', side: 'long',  qty: 1, iv: 0.15, initial_value: 0.0 },
    { strike: 4250, expiration: '2023-05-05', flag: 'c', side: 'short', qty: 1, iv: 0.15, initial_value: 0.0 },
  ],
  pl_curve: { prices: PRICES, pnl: PNLS },
  iv_curve:  { prices: PRICES, iv:  IV   },
  trade_thesis: {
    lower: 4187.0,
    upper: null,
    regime_kind: 'magnet-above',
    structural_prob: 0.70,
    structural_ci: [0.48, 0.85],
    structural_n: 20,
    implied_prob: 0.497,
    edge_ratio: 1.41,
  },
  edge_zones: [
    {
      lower: 4145.0, upper: 4185.0,
      classification: 'strong-negative',
      n_bins: 8, avg_edge_ratio: 0.0, min_structural_n: 20,
      representative: {
        lower: 4145.0, upper: 4150.0,
        structural_prob: 0.0, structural_n: 20,
        structural_ci: [0.0, 0.16],
        implied_prob: 0.02, edge_ratio: 0.0,
        classification: 'strong-negative',
      },
    },
    {
      lower: 4185.0, upper: 4220.0,
      classification: 'moderate-positive',
      n_bins: 7, avg_edge_ratio: 2.1, min_structural_n: 3,  // thin!
      representative: {
        lower: 4185.0, upper: 4190.0,
        structural_prob: 0.05, structural_n: 3,
        structural_ci: [0.01, 0.24],
        implied_prob: 0.021, edge_ratio: 2.42,
        classification: 'moderate-positive',
      },
    },
  ],
  greeks: { delta: 0.0, gamma: 0.0, theta: 0.0, vega: 0.0, rho: 0.0 },
  key_levels: { max_profit: 25.0, max_loss: 0.0, breakevens: [4225.27] },
  warnings: ['using atmiv for all legs (per-leg smile interpolation not yet implemented)'],
}

const NULL_DATA = null

const ERROR_DATA = { ok: false, error: 'db connect failed: connection refused' }

// Non-range regime: both bounds null → no structural_range band, no edge zones
const NO_RANGE_DATA = {
  ...MOCK_DATA,
  trade_thesis: { ...MOCK_DATA.trade_thesis, lower: null, upper: null, regime_kind: 'amplification' },
  edge_zones: [],
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function renderChart(props = {}) {
  return render(<ProposalEdgeChart data={MOCK_DATA} width={800} height={400} {...props} />)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('ProposalEdgeChart', () => {
  describe('loading / null state', () => {
    it('renders skeleton when data is null', () => {
      render(<ProposalEdgeChart data={null} width={800} height={400} />)
      const skeleton = document.querySelector('svg[aria-label="Loading chart…"]')
      expect(skeleton).not.toBeNull()
    })

    it('renders error message when data.ok is false', () => {
      render(<ProposalEdgeChart data={ERROR_DATA} width={800} height={400} />)
      expect(screen.getByText(/db connect failed/)).toBeInTheDocument()
    })
  })

  describe('default layer visibility', () => {
    it('renders the P/L curve (default ON)', () => {
      renderChart()
      expect(document.querySelector('[data-testid="pl-curve"]')).not.toBeNull()
    })

    it('renders the current-spot line (default ON)', () => {
      renderChart()
      expect(document.querySelector('[data-testid="spot-line"]')).not.toBeNull()
    })

    it('renders edge zone bands (default ON)', () => {
      renderChart()
      const negZones = document.querySelectorAll('[data-testid="edge-zone-strong-negative"]')
      const posZones = document.querySelectorAll('[data-testid="edge-zone-moderate-positive"]')
      expect(negZones.length + posZones.length).toBeGreaterThan(0)
    })

    it('does NOT render IV curve by default (default OFF)', () => {
      renderChart()
      expect(document.querySelector('[data-testid="iv-curve"]')).toBeNull()
    })

    it('does NOT render breakeven lines by default (default OFF)', () => {
      renderChart()
      expect(document.querySelector('[data-testid="breakeven-0"]')).toBeNull()
    })

    it('does NOT render P/L marker lines by default (default OFF)', () => {
      renderChart()
      expect(document.querySelector('[data-testid="max-profit-line"]')).toBeNull()
      expect(document.querySelector('[data-testid="max-loss-line"]')).toBeNull()
    })
  })

  describe('layer toggles', () => {
    it('toggling P/L Curve off removes the polyline', () => {
      renderChart()
      const btn = screen.getByText('P/L Curve').closest('button')
      fireEvent.click(btn)  // ON → OFF
      expect(document.querySelector('[data-testid="pl-curve"]')).toBeNull()
    })

    it('toggling IV on renders the IV curve polyline', () => {
      renderChart()
      const btn = screen.getByText('IV').closest('button')
      fireEvent.click(btn)  // OFF → ON
      expect(document.querySelector('[data-testid="iv-curve"]')).not.toBeNull()
    })

    it('toggling Breakevens on renders breakeven lines', () => {
      renderChart()
      const btn = screen.getByText('Breakevens').closest('button')
      fireEvent.click(btn)  // OFF → ON
      expect(document.querySelector('[data-testid="breakeven-0"]')).not.toBeNull()
    })

    it('toggling P/L Markers on renders max-profit and max-loss lines', () => {
      renderChart()
      const btn = screen.getByText('P/L Markers').closest('button')
      fireEvent.click(btn)  // OFF → ON
      expect(document.querySelector('[data-testid="max-profit-line"]')).not.toBeNull()
      expect(document.querySelector('[data-testid="max-loss-line"]')).not.toBeNull()
    })

    it('toggling Edge Zones off removes all zone rects', () => {
      renderChart()
      const btn = screen.getByText('Edge Zones').closest('button')
      fireEvent.click(btn)  // ON → OFF
      expect(document.querySelectorAll('[data-testid^="edge-zone-"]')).toHaveLength(0)
    })

    it('calls onLayerToggle callback when layer is toggled', () => {
      const onLayerToggle = vi.fn()
      render(<ProposalEdgeChart data={MOCK_DATA} width={800} height={400} onLayerToggle={onLayerToggle} />)
      fireEvent.click(screen.getByText('IV').closest('button'))
      expect(onLayerToggle).toHaveBeenCalledWith(expect.objectContaining({ iv_curve: true }))
    })
  })

  describe('non-range regime (amplification)', () => {
    it('does NOT render structural-range band when both bounds are null', () => {
      render(<ProposalEdgeChart data={NO_RANGE_DATA} width={800} height={400} />)
      expect(document.querySelector('[data-testid="structural-range-band"]')).toBeNull()
    })

    it('P/L curve still renders for non-range regime', () => {
      render(<ProposalEdgeChart data={NO_RANGE_DATA} width={800} height={400} />)
      expect(document.querySelector('[data-testid="pl-curve"]')).not.toBeNull()
    })

    it('no edge zone bands when edge_zones is empty', () => {
      render(<ProposalEdgeChart data={NO_RANGE_DATA} width={800} height={400} />)
      expect(document.querySelectorAll('[data-testid^="edge-zone-"]')).toHaveLength(0)
    })
  })

  describe('thin-n zone annotation', () => {
    it('thin-n zone (min_structural_n < 5) renders with ⚠ annotation', () => {
      renderChart()
      // The moderate-positive zone has min_structural_n=3 — should render ⚠
      // Find SVG text elements inside the moderate-positive zone group
      const zoneGroup = document.querySelector('[data-testid="edge-zone-moderate-positive"]')
      expect(zoneGroup).not.toBeNull()
      expect(zoneGroup.textContent).toContain('⚠')
    })

    it('normal zone (min_structural_n ≥ 5) does NOT render ⚠', () => {
      renderChart()
      const zoneGroup = document.querySelector('[data-testid="edge-zone-strong-negative"]')
      expect(zoneGroup).not.toBeNull()
      expect(zoneGroup.textContent).not.toContain('⚠')
    })
  })

  describe('hover tooltip', () => {
    it('tooltip appears on mouseover within plot area', () => {
      renderChart()
      const svg = document.querySelector('svg[aria-label="Proposal edge chart"]')
      // getBoundingClientRect is stubbed to 800×400
      fireEvent.mouseMove(svg, { clientX: 400, clientY: 200 })
      expect(document.querySelector('[data-testid="hover-tooltip"]')).not.toBeNull()
    })

    it('tooltip disappears on mouseleave', () => {
      renderChart()
      const svg = document.querySelector('svg[aria-label="Proposal edge chart"]')
      fireEvent.mouseMove(svg, { clientX: 400, clientY: 200 })
      expect(document.querySelector('[data-testid="hover-tooltip"]')).not.toBeNull()
      fireEvent.mouseLeave(svg)
      expect(document.querySelector('[data-testid="hover-tooltip"]')).toBeNull()
    })

    it('tooltip is absent outside the plot area (near left padding)', () => {
      renderChart()
      const svg = document.querySelector('svg[aria-label="Proposal edge chart"]')
      // x=10 is inside PAD.left (68) — should NOT trigger tooltip
      fireEvent.mouseMove(svg, { clientX: 10, clientY: 200 })
      expect(document.querySelector('[data-testid="hover-tooltip"]')).toBeNull()
    })
  })

  describe('click popover', () => {
    it('clicking on an edge zone shows the zone popover', () => {
      renderChart()
      const svg = document.querySelector('svg[aria-label="Proposal edge chart"]')
      // Click somewhere in the price range that maps to an edge zone
      // x=250 → price ≈ 4100 + (250-68)/732 * 200 ≈ 4150 (inside strong-negative zone)
      fireEvent.click(svg, { clientX: 250, clientY: 200 })
      expect(document.querySelector('[data-testid="zone-popover"]')).not.toBeNull()
    })

    it('zone popover is dismissed by clicking the ✕ button', () => {
      renderChart()
      const svg = document.querySelector('svg[aria-label="Proposal edge chart"]')
      fireEvent.click(svg, { clientX: 250, clientY: 200 })
      const closeBtn = document.querySelector('[data-testid="zone-popover"] button')
      expect(closeBtn).not.toBeNull()
      fireEvent.click(closeBtn)
      expect(document.querySelector('[data-testid="zone-popover"]')).toBeNull()
    })
  })

  describe('magnet-above one-sided trade thesis', () => {
    it('structural_range band starts at drift_target (lower=4187) and extends to right axis', () => {
      renderChart()
      const band = document.querySelector('[data-testid="structural-range-band"]')
      expect(band).not.toBeNull()
      // The band's x attribute should be to the right of center (drift_target > spot)
      const x = parseFloat(band.getAttribute('x'))
      // PAD.left (68) + plotW * ((4187 - 4100) / 200) ≈ 68 + 732 * 0.435 ≈ 386
      expect(x).toBeGreaterThan(68)
      // Band width should extend to the right axis edge
      const w = parseFloat(band.getAttribute('width'))
      expect(w).toBeGreaterThan(0)
      // x + width should reach the right edge (PAD.left + plotW = 68 + 732 = 800 - 52 = 748)
      // With clipping, x + w should approximately equal plotW + PAD.left
      expect(x + w).toBeCloseTo(748, -1)  // within ±10
    })
  })

  describe('mobile width (380px)', () => {
    it('renders without overflow at 380px width', () => {
      render(<ProposalEdgeChart data={MOCK_DATA} width={380} height={400} />)
      // Should render the chart container without throwing
      expect(document.querySelector('[data-testid="proposal-edge-chart"]')).not.toBeNull()
      const svg = document.querySelector('svg[aria-label="Proposal edge chart"]')
      expect(svg).not.toBeNull()
      expect(svg.getAttribute('width')).toBe('380')
    })

    it('layer toggle chips wrap and remain accessible at 380px', () => {
      render(<ProposalEdgeChart data={MOCK_DATA} width={380} height={400} />)
      // All 7 layer chips should be present regardless of width
      const chips = document.querySelectorAll('[role="group"] button')
      expect(chips).toHaveLength(7)
    })
  })
})
