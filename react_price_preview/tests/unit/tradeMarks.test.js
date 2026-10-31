// CR-DR item 2 — My trades primitive: fill time inside its bar, hit test. Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { TRADE_LONG_RGB, TRADE_SHORT_RGB, TradeMarksPrimitive, timeX, tradeRgb } from '../../src/components/tradePrimitives.js'

// a fake time scale: bar i at x = 100 + 10 i, spacing 10
const T0 = 1761811740 // any shifted epoch on a minute
const times = [0, 1, 2, 3, 4].map((i) => T0 + 60 * i)
const ts = {
  timeToCoordinate: (t) => { const i = times.indexOf(t); return i < 0 ? null : 100 + 10 * i },
  options: () => ({ barSpacing: 10 }),
}

test('timeX: a fill inside its bar, left edge at :00, centre at :30', () => {
  assert.equal(timeX(ts, times, 60, times[2]), 115)
  assert.equal(timeX(ts, times, 60, times[2] + 30), 120)
  assert.equal(timeX(ts, times, 60, times[2] + 17), 115 + 10 * 17 / 60)
  assert.equal(timeX(ts, times, 60, times[0] - 1), null) // before the data
})

test('palette: blue long, slate short (C1)', () => {
  assert.equal(tradeRgb('long'), TRADE_LONG_RGB)
  assert.equal(tradeRgb('short'), TRADE_SHORT_RGB)
  assert.equal(TRADE_LONG_RGB, '96,165,250')
  assert.equal(TRADE_SHORT_RGB, '148,163,184')
})

test('paint records entry / exit hits at the fill price; hitTest picks the nearest within 10 px', () => {
  const prim = new TradeMarksPrimitive()
  const series = { priceToCoordinate: (p) => 1000 - p } // 1 px per point
  const ctx = new Proxy({}, { get: (o, k) => (k in o ? o[k] : () => {}), set: (o, k, v) => { o[k] = v; return true } })
  const chart = { timeScale: () => ts }
  const d = {
    times, step: 60, selectedId: null,
    trades: [{ id: 48, direction: 'short', entryT: times[1] + 17, entryPrice: 655.75, exitT: times[3] + 12, exitPrice: 642, lines: [] }],
  }
  prim.paint({ context: ctx, horizontalPixelRatio: 1, verticalPixelRatio: 1 }, chart, series, d)
  const hits = prim.hits()
  assert.deepEqual(hits.map((h) => h.kind), ['entry', 'exit'])
  assert.equal(hits[0].priceY, 1000 - 655.75)
  assert.equal(hits[1].x, 125 + 10 * 12 / 60)
  assert.equal(prim.hitTest({ x: hits[1].x + 3, y: hits[1].y - 3 }), 48)
  assert.equal(prim.hitTest({ x: hits[1].x + 12, y: hits[1].y }), null)
  assert.equal(prim.hitTest(null), null)
})
