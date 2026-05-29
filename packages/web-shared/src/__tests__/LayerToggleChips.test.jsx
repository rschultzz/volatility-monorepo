/**
 * Tests for LayerToggleChips (CR-G Step 5a).
 *
 * Run:  cd packages/web-shared && npm test
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import LayerToggleChips from '../LayerToggleChips.jsx'

const LAYERS = [
  { id: 'pl_curve',   label: 'P/L Curve',  defaultVisible: true  },
  { id: 'edge_zones', label: 'Edge Zones',  defaultVisible: true  },
  { id: 'iv_curve',   label: 'IV',          defaultVisible: false },
]

describe('LayerToggleChips', () => {
  it('renders a chip for each layer', () => {
    render(<LayerToggleChips layers={LAYERS} />)
    expect(screen.getByText('P/L Curve')).toBeInTheDocument()
    expect(screen.getByText('Edge Zones')).toBeInTheDocument()
    expect(screen.getByText('IV')).toBeInTheDocument()
  })

  it('reflects defaultVisible in aria-pressed', () => {
    render(<LayerToggleChips layers={LAYERS} />)
    const plBtn = screen.getByText('P/L Curve').closest('button')
    const ivBtn = screen.getByText('IV').closest('button')
    expect(plBtn).toHaveAttribute('aria-pressed', 'true')
    expect(ivBtn).toHaveAttribute('aria-pressed', 'false')
  })

  it('uncontrolled mode: click toggles visibility and calls onChange', () => {
    const onChange = vi.fn()
    render(<LayerToggleChips layers={LAYERS} onChange={onChange} />)
    const ivBtn = screen.getByText('IV').closest('button')

    // IV starts OFF (defaultVisible: false)
    expect(ivBtn).toHaveAttribute('aria-pressed', 'false')

    fireEvent.click(ivBtn)
    expect(ivBtn).toHaveAttribute('aria-pressed', 'true')
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ iv_curve: true }))
  })

  it('uncontrolled mode: second click toggles back to original state', () => {
    const onChange = vi.fn()
    render(<LayerToggleChips layers={LAYERS} onChange={onChange} />)
    const plBtn = screen.getByText('P/L Curve').closest('button')

    fireEvent.click(plBtn)  // ON → OFF
    expect(plBtn).toHaveAttribute('aria-pressed', 'false')
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ pl_curve: false }))

    fireEvent.click(plBtn)  // OFF → ON
    expect(plBtn).toHaveAttribute('aria-pressed', 'true')
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ pl_curve: true }))
  })

  it('controlled mode: respects external value prop, does not update internally', () => {
    const onChange = vi.fn()
    const value = { pl_curve: false, edge_zones: true, iv_curve: true }
    const { rerender } = render(<LayerToggleChips layers={LAYERS} value={value} onChange={onChange} />)

    const plBtn = screen.getByText('P/L Curve').closest('button')
    expect(plBtn).toHaveAttribute('aria-pressed', 'false')

    fireEvent.click(plBtn)
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ pl_curve: true }))
    // Controlled: the button aria-pressed should NOT change (parent owns state)
    expect(plBtn).toHaveAttribute('aria-pressed', 'false')

    // Parent updates value → re-render with new state
    rerender(<LayerToggleChips layers={LAYERS} value={{ ...value, pl_curve: true }} onChange={onChange} />)
    expect(plBtn).toHaveAttribute('aria-pressed', 'true')
  })

  it('renders zero chips for empty layers array', () => {
    const { container } = render(<LayerToggleChips layers={[]} />)
    expect(container.querySelectorAll('button')).toHaveLength(0)
  })

  it('md size applies minHeight ≥ 44 for tap target compliance', () => {
    render(<LayerToggleChips layers={LAYERS} size="md" />)
    const btn = screen.getByText('P/L Curve').closest('button')
    // computed style in jsdom doesn't evaluate px but we can check the inline style
    expect(btn.style.minHeight).toBe('44px')
  })
})
