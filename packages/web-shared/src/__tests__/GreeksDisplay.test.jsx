/**
 * Tests for GreeksDisplay component (CR-G Step 7).
 *
 * Run:  cd packages/web-shared && npx vitest run
 *
 * Covers: all-5-greeks render, ≈ 0 for zero values (expected at expiration),
 * null → renders nothing, info tooltip on focus, mobile flex-wrap.
 */
import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import GreeksDisplay from '../GreeksDisplay.jsx'

const TYPICAL_GREEKS = {
  delta:  0.5023,
  gamma:  0.0142,
  theta: -0.1875,
  vega:   0.8901,
  rho:    0.2345,
}

// At expiration theta, vega, gamma, rho are 0; delta is step-function
const EXPIRY_GREEKS = {
  delta: 1.0,
  gamma: 0.0,
  theta: 0.0,
  vega:  0.0,
  rho:   0.0,
}

describe('GreeksDisplay', () => {
  it('renders nothing when greeks is null', () => {
    const { container } = render(
      <GreeksDisplay greeks={null} evaluationTime={null} />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders all 5 greek chips when greeks object is provided', () => {
    render(<GreeksDisplay greeks={TYPICAL_GREEKS} evaluationTime="2023-05-05T00:00:00" />)
    expect(screen.getByTestId('greek-delta')).toBeInTheDocument()
    expect(screen.getByTestId('greek-gamma')).toBeInTheDocument()
    expect(screen.getByTestId('greek-theta')).toBeInTheDocument()
    expect(screen.getByTestId('greek-vega')).toBeInTheDocument()
    expect(screen.getByTestId('greek-rho')).toBeInTheDocument()
  })

  it('formats typical greek values correctly', () => {
    render(<GreeksDisplay greeks={TYPICAL_GREEKS} evaluationTime={null} />)
    // delta: 4dp
    expect(screen.getByTestId('greek-delta').textContent).toMatch('0.5023')
    // gamma: 4dp
    expect(screen.getByTestId('greek-gamma').textContent).toMatch('0.0142')
    // theta: $-0.19 (rounded) — negative dollar/day
    expect(screen.getByTestId('greek-theta').textContent).toMatch('-$0.19')
    // vega: 4dp
    expect(screen.getByTestId('greek-vega').textContent).toMatch('0.8901')
    // rho: 4dp
    expect(screen.getByTestId('greek-rho').textContent).toMatch('0.2345')
  })

  it('renders ≈ 0 for zero theta and vega (expected at expiration)', () => {
    render(<GreeksDisplay greeks={EXPIRY_GREEKS} evaluationTime="2023-05-05T00:00:00" />)
    expect(screen.getByTestId('greek-theta').textContent).toContain('≈ 0')
    expect(screen.getByTestId('greek-vega').textContent).toContain('≈ 0')
    expect(screen.getByTestId('greek-gamma').textContent).toContain('≈ 0')
    expect(screen.getByTestId('greek-rho').textContent).toContain('≈ 0')
    // delta should NOT be ≈ 0 when it's 1.0
    expect(screen.getByTestId('greek-delta').textContent).toContain('1.0000')
  })

  it('renders ≈ 0 for a full zero greeks object', () => {
    const zeros = { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 }
    render(<GreeksDisplay greeks={zeros} evaluationTime={null} />)
    const allChips = ['delta', 'gamma', 'theta', 'vega', 'rho']
    allChips.forEach(key => {
      expect(screen.getByTestId(`greek-${key}`).textContent).toContain('≈ 0')
    })
  })

  it('info tooltip is hidden by default', () => {
    render(<GreeksDisplay greeks={TYPICAL_GREEKS} evaluationTime={null} />)
    expect(screen.queryByTestId('greeks-info-tooltip')).toBeNull()
  })

  it('info tooltip appears on focus of ⓘ button', () => {
    render(<GreeksDisplay greeks={TYPICAL_GREEKS} evaluationTime="2023-05-05T00:00:00" />)
    const btn = screen.getByTestId('greeks-info-btn')
    fireEvent.focus(btn)
    const tooltip = screen.getByTestId('greeks-info-tooltip')
    expect(tooltip).toBeInTheDocument()
    expect(tooltip.textContent).toMatch(/expiration/)
    expect(tooltip.textContent).toMatch(/theta and vega/)
  })

  it('info tooltip disappears on blur', () => {
    render(<GreeksDisplay greeks={TYPICAL_GREEKS} evaluationTime={null} />)
    const btn = screen.getByTestId('greeks-info-btn')
    fireEvent.focus(btn)
    expect(screen.getByTestId('greeks-info-tooltip')).toBeInTheDocument()
    fireEvent.blur(btn)
    expect(screen.queryByTestId('greeks-info-tooltip')).toBeNull()
  })

  it('container uses flexWrap:wrap so chips can reflow on narrow screens', () => {
    render(<GreeksDisplay greeks={TYPICAL_GREEKS} evaluationTime={null} />)
    const container = screen.getByTestId('greeks-display')
    expect(container.style.flexWrap).toBe('wrap')
  })

  it('evaluation date appears in tooltip when evaluationTime is provided', () => {
    render(<GreeksDisplay greeks={TYPICAL_GREEKS} evaluationTime="2023-05-05T21:00:00" />)
    const btn = screen.getByTestId('greeks-info-btn')
    fireEvent.focus(btn)
    expect(screen.getByTestId('greeks-info-tooltip').textContent).toMatch('2023-05-05')
  })
})
