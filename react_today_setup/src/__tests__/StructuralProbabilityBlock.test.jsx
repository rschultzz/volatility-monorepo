/**
 * CR-AV decision 2 / 5: the post-touch section renders an advisory block
 * (label · n · timeframe direction %) and no badge implying support or caution.
 *
 * Run:  cd react_today_setup && npm test
 */
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import StructuralProbabilityBlock from '../components/StructuralProbabilityBlock'

function sp(overrides = {}, ptOverrides = {}) {
  return {
    outcome_status: 'ok',
    regime_kind: 'magnet-above',
    k: 70, k_with_outcomes: 70,
    touch_rate: 0.7, close_rate: 0.5, touch_ci_lower: 0.58, touch_ci_upper: 0.8,
    post_touch: {
      filter_mode: 'strict',
      pattern_label: 'stepping-stone',
      same_bucket_n: 23, total_touchers: 50,
      fractions: {
        t1:  { below: 0.2, at: 0.2, above: 0.6 },
        t5:  { below: 0.3, at: 0.09, above: 0.61 },
        t15: { below: 0.36, at: 0.08, above: 0.56 },
      },
      wilson_cis: {
        t1:  { below: [0.1, 0.4], at: [0.1, 0.4], above: [0.4, 0.8] },
        t5:  { below: [0.15, 0.5], at: [0.02, 0.2], above: [0.4, 0.8] },
        t15: { below: [0.2, 0.55], at: [0.02, 0.25], above: [0.37, 0.73] },
      },
      advisory_only: true,
      advisory: { pattern_label: 'stepping-stone', n: 23, n_pooled: 50, timeframe: 't5', direction: 'above',
                  fraction: 0.61, wilson_lo: 0.4, wilson_hi: 0.8 },
      ...ptOverrides,
    },
    ...overrides,
  }
}

describe('StructuralProbabilityBlock — post-touch advisory block (CR-AV)', () => {
  it('renders label + n + timeframe direction fraction from the advisory block', () => {
    render(<StructuralProbabilityBlock sp={sp()} dte={5} />)
    const adv = screen.getByTestId('pt-advisory')
    expect(adv.textContent).toBe('post-touch pattern: stepping-stone · n=23 · t5 above 61%')
  })

  it('stepping-stone and mixed render the same block shape, only the label differs', () => {
    const { unmount } = render(<StructuralProbabilityBlock sp={sp()} dte={5} />)
    const a = screen.getByTestId('pt-advisory').textContent
    unmount()
    render(<StructuralProbabilityBlock sp={sp({}, { pattern_label: 'mixed',
      advisory: { pattern_label: 'mixed', n: 23, n_pooled: 50, timeframe: 't5', direction: 'above', fraction: 0.61 } })} dte={5} />)
    const b = screen.getByTestId('pt-advisory').textContent
    expect(a.replace('stepping-stone', 'mixed')).toBe(b)
  })

  it('never renders a supported / low-confidence / mixed-pattern badge', () => {
    for (const label of ['stepping-stone', 'touch-and-reject', 'mixed', 'touch-and-pin']) {
      const { container, unmount } = render(<StructuralProbabilityBlock sp={sp({}, { pattern_label: label,
        advisory: { pattern_label: label, n: 23, timeframe: 't5', direction: 'above', fraction: 0.61 } })} dte={5} />)
      const text = container.textContent
      expect(text).not.toMatch(/supported/i)
      expect(text).not.toMatch(/low-confidence/i)
      expect(text).not.toMatch(/no clear direction/i)
      expect(text).not.toMatch(/Direction signal/i)
      unmount()
    }
  })

  it('falls back to raw post_touch fields when the server advisory block is absent', () => {
    render(<StructuralProbabilityBlock sp={sp({}, { advisory: undefined })} dte={15} />)
    // t15 row for a 15-DTE trade; direction from the advisory is absent → no fraction part
    expect(screen.getByTestId('pt-advisory').textContent).toBe('post-touch pattern: stepping-stone · n=23')
  })

  it('renders no advisory when the payload is not marked advisory_only (legacy payload)', () => {
    const { container } = render(<StructuralProbabilityBlock sp={sp({}, { advisory_only: undefined, advisory: undefined })} dte={5} />)
    expect(container.querySelector('[data-testid="pt-advisory"]')).toBeNull()
    expect(container.textContent).not.toMatch(/supported/i)
  })
})
