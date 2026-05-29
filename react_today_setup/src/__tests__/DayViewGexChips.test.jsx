/**
 * Tests for DayView GEX layer toggle chips (CR-G Step 6b).
 *
 * Run:  cd react_today_setup && npm test
 *
 * Verifies: chips render, default state (both ON), toggle off/on interactions.
 * MiniPriceChart visual side-effects (price lines) are not tested here — those
 * are covered by filterClusters unit tests in packages/web-shared.
 */
import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import DayView from '../components/DayView'

// Minimal landscapeData with one pos cluster and one neg cluster.
const LANDSCAPE = {
  confluences: [
    { center_price: 4300, max_gex:  5e9, quality: 'pin' },
    { center_price: 4100, max_gex: -3e9, quality: 'pin' },
  ],
  bucket_summary: {},
  regime: { regime: 'magnet-above', drift_target: 4300 },
}

function renderDayView(dateOverride = '2023-05-01') {
  return render(
    <DayView
      label="Anchor"
      date={dateOverride}
      ticker="SPX"
      apiBase=""
      landscapeData={LANDSCAPE}
      regime="magnet-above"
      autoRegime="magnet-above"
      flag={null}
      allowPairFlag={false}
      onRegimeFlag={() => {}}
      onPromote={() => {}}
      onDemote={() => {}}
      onDeleteFlag={() => {}}
      onPairFlag={() => {}}
    />
  )
}

describe('DayView — GEX layer chips', () => {
  it('renders the GEX chip container when a date is provided', () => {
    renderDayView()
    expect(document.querySelector('[data-testid="gex-layer-chips"]')).not.toBeNull()
  })

  it('does NOT render GEX chips when date is null / absent', () => {
    render(
      <DayView
        label="Selected"
        date={null}
        ticker="SPX"
        apiBase=""
        landscapeData={null}
        regime={null}
        autoRegime={null}
        flag={null}
        allowPairFlag={false}
        onRegimeFlag={() => {}}
        onPromote={() => {}}
        onDemote={() => {}}
        onDeleteFlag={() => {}}
        onPairFlag={() => {}}
      />
    )
    expect(document.querySelector('[data-testid="gex-layer-chips"]')).toBeNull()
  })

  it('renders Pos GEX and Neg GEX chips', () => {
    renderDayView()
    expect(screen.getByText('Pos GEX')).toBeInTheDocument()
    expect(screen.getByText('Neg GEX')).toBeInTheDocument()
  })

  it('both chips are ON by default (aria-pressed=true)', () => {
    renderDayView()
    const posBtn = screen.getByText('Pos GEX').closest('button')
    const negBtn = screen.getByText('Neg GEX').closest('button')
    expect(posBtn).toHaveAttribute('aria-pressed', 'true')
    expect(negBtn).toHaveAttribute('aria-pressed', 'true')
  })

  it('toggling Pos GEX off reflects in aria-pressed', () => {
    renderDayView()
    const posBtn = screen.getByText('Pos GEX').closest('button')
    fireEvent.click(posBtn)
    expect(posBtn).toHaveAttribute('aria-pressed', 'false')
  })

  it('toggling Neg GEX off does not affect Pos GEX chip', () => {
    renderDayView()
    const negBtn = screen.getByText('Neg GEX').closest('button')
    const posBtn = screen.getByText('Pos GEX').closest('button')
    fireEvent.click(negBtn)
    expect(negBtn).toHaveAttribute('aria-pressed', 'false')
    expect(posBtn).toHaveAttribute('aria-pressed', 'true')  // unaffected
  })

  it('toggling a chip off then back on restores ON state', () => {
    renderDayView()
    const posBtn = screen.getByText('Pos GEX').closest('button')
    fireEvent.click(posBtn)  // OFF
    expect(posBtn).toHaveAttribute('aria-pressed', 'false')
    fireEvent.click(posBtn)  // ON
    expect(posBtn).toHaveAttribute('aria-pressed', 'true')
  })

  it('renders 2 chips total (no extra chips)', () => {
    renderDayView()
    const chips = document.querySelector('[data-testid="gex-layer-chips"]')
    const buttons = chips.querySelectorAll('button')
    expect(buttons).toHaveLength(2)
  })
})
