/**
 * Tests for MiniPriceChart GEX layer filtering (CR-G Step 6b).
 *
 * filterClusters is a pure function extracted from MiniPriceChart for
 * testability — it separates cluster visibility logic from lightweight-charts
 * side-effects (which are hard to test in jsdom).
 */
import { describe, it, expect } from 'vitest'
import { filterClusters } from '../MiniPriceChart.jsx'

const POS_CLUSTER  = { center_price: 4300, max_gex:  5e9 }
const NEG_CLUSTER  = { center_price: 4100, max_gex: -3e9 }
const ZERO_CLUSTER = { center_price: 4200, max_gex:  0   }  // zero counts as positive

describe('filterClusters', () => {
  it('returns all clusters when both flags are true (default behaviour)', () => {
    const result = filterClusters([POS_CLUSTER, NEG_CLUSTER], true, true)
    expect(result).toHaveLength(2)
  })

  it('returns only positive clusters when showPosGex=true, showNegGex=false', () => {
    const result = filterClusters([POS_CLUSTER, NEG_CLUSTER, ZERO_CLUSTER], true, false)
    expect(result).toHaveLength(2)
    result.forEach(c => expect(Number(c.max_gex)).toBeGreaterThanOrEqual(0))
  })

  it('returns only negative clusters when showPosGex=false, showNegGex=true', () => {
    const result = filterClusters([POS_CLUSTER, NEG_CLUSTER, ZERO_CLUSTER], false, true)
    expect(result).toHaveLength(1)
    expect(result[0]).toBe(NEG_CLUSTER)
  })

  it('returns empty array when both flags are false', () => {
    const result = filterClusters([POS_CLUSTER, NEG_CLUSTER], false, false)
    expect(result).toHaveLength(0)
  })

  it('handles null / undefined clusters gracefully', () => {
    expect(filterClusters(null, true, true)).toHaveLength(0)
    expect(filterClusters(undefined, true, true)).toHaveLength(0)
  })

  it('handles empty clusters array', () => {
    expect(filterClusters([], true, true)).toHaveLength(0)
  })
})
