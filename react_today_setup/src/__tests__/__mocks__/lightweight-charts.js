// Minimal lightweight-charts stub for jsdom test environment.
// MiniPriceChart is imported transitively via web-shared/index.js but is not
// rendered in ProposalCard tests — this stub prevents module evaluation errors.

export const ColorType = { Solid: 'solid' }
export const LineStyle = { Dashed: 1, Dotted: 2 }

function noop() {}

const mockSeries = {
  setData: noop,
  createPriceLine: function() { return {} },
  removePriceLine: noop,
  coordinateToPrice: function() { return 0 },
}

const mockTimeScale = {
  setVisibleRange: noop,
}

const mockPriceScale = {
  setVisibleRange: noop,
}

const mockChart = {
  addSeries: function() { return mockSeries },
  applyOptions: noop,
  remove: noop,
  timeScale: function() { return mockTimeScale },
  priceScale: function() { return mockPriceScale },
}

export function createChart() { return mockChart }

// Series type tags (passed as first arg to addSeries)
export const CandlestickSeries = 'CandlestickSeries'
export const LineSeries = 'LineSeries'
