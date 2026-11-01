// CR-DR item 3b — "at entry" context helpers. Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { entryContext, entryUntil, ptToShifted, shiftedToPt } from '../../src/tradeContext.js'
import { utcEpochShowingZoneTime } from '../../src/flowShared.js'

// 2026-10-30 07:30 PDT = 14:30 UTC
const T0 = Date.UTC(2026, 9, 30, 14, 30) / 1000
const N = 19 // bars 07:30 … 07:48 (the until replay cut at 07:49 returns only bars that start before it)
const closes = Array.from({ length: N }, (_, i) => 7650 + (i % 3))
const cellBars = { t: closes.map((_, i) => T0 + 60 * i), h: closes.map((c) => c + 0.5), l: closes.map((c) => c - 0.5), c: closes }
const mark = (kind, minute, price) => ({ kind, time: T0 + 60 * minute, price, confirm_time: T0 + 60 * (minute + 4), aggressor: 400,
  opposite: 50, merged_vol: 0, median: 100, break_time: null, held_min: 0 })
const cells = { segments: [{ name: 'rth', exact: true, start: T0 - 3600, end: T0 + 390 * 60, bars: cellBars,
  marks: [mark('buyers_absorbed', 2, 7654.5), mark('sellers_absorbed', 3, 7640)], cells: { t: [], p: [], buy: [], sell: [] } }] }
const candles = closes.map((c, i) => ({ time: T0 + 60 * i, open: c, high: c + 0.5, low: c - 0.5, close: c }))
const v2Bars = closes.map((_, i) => ({ time: T0 + 60 * i, delta: i, vol: 100 }))
const entryShifted = ptToShifted('2026-10-30T07:49:00')

test('until = the entry minute start; PT strings <-> shifted epochs', () => {
  assert.equal(entryUntil('2026-10-30T07:49:17'), '2026-10-30T07:49:00')
  assert.equal(entryUntil(null), null)
  assert.equal(ptToShifted('2026-10-30T07:49:17'), Date.UTC(2026, 9, 30, 7, 49, 17) / 1000)
  assert.equal(shiftedToPt(ptToShifted('2026-10-30T07:40:00')), '2026-10-30T07:40:00')
  assert.equal(utcEpochShowingZoneTime(T0), ptToShifted('2026-10-30T07:30:00'))
})

test('no anchor: no box; levels near the entry price and the last 5 complete minutes', () => {
  const c = entryContext({ cells, v2Bars, bars: candles, anchorShifted: null, entryShifted, entryPrice: 7655.75, tradeDate: '2026-10-30' })
  assert.equal(c.box, null)
  assert.equal(c.cutoff, '07:48')
  assert.deepEqual(c.levels.map((l) => [l.kind, l.price, l.dist]), [['buyers_absorbed', 7654.5, -1.25]]) // 7640 is > 5 pts away
  assert.deepEqual(c.last5, { minutes: 5, delta: 14 + 15 + 16 + 17 + 18, vol: 500, from: '07:44', to: '07:48' })
})

test('with an anchor the chart code yields a box over the replayed bars only', () => {
  const c = entryContext({ cells, v2Bars, bars: candles, anchorShifted: ptToShifted('2026-10-30T07:40:00'), entryShifted,
    entryPrice: 7655.75, tradeDate: '2026-10-30' })
  assert.ok(c.box, 'a box')
  assert.equal(c.box.from, '07:40')
  assert.ok(c.box.to <= '07:48')
  assert.ok(c.box.verdict && typeof c.box.verdict.label === 'string')
})
