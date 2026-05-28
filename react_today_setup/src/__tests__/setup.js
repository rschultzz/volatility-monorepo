import '@testing-library/jest-dom'

// Stub ResizeObserver — not available in jsdom
globalThis.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Stub SVGElement.getBoundingClientRect for ProposalEdgeChart
if (typeof SVGElement !== 'undefined') {
  SVGElement.prototype.getBoundingClientRect = () => ({
    left: 0, top: 0, right: 800, bottom: 400, width: 800, height: 400,
  })
}

// Stub lightweight-charts so MiniPriceChart (imported transitively) doesn't crash
globalThis.__lwc_stub__ = true
