// CR-DN — Flow v3 ranges as segments: set end (item 1).
// Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { anchorReadout, anchoredCvd, edgeReadout, endFromPick, endWindowText, flattenBars } from '../../src/flowV3.js'

const T0 = 1_777_642_200 // a UTC minute
const id = (t) => t
const tick = (price) => Math.round(price / 0.25)
const bars = (closes) => ({
  t: closes.map((_, i) => T0 + 60 * i), h: closes.map((c) => c + 0.5), l: closes.map((c) => c - 0.5), c: closes,
})
// cells: [minute, price, buy, sell]
const payload = (closes, cells, exact = true) => ({
  segments: [{
    name: 'rth', exact, start: T0, end: T0 + 390 * 60, bars: bars(closes), marks: null,
    cells: { t: cells.map((c) => T0 + 60 * c[0]), p: cells.map((c) => tick(c[1])), buy: cells.map((c) => c[2]), sell: cells.map((c) => c[3]) },
  }],
})
const rows = (deltas) => deltas.map((d, i) => ({ time: T0 + 60 * i, delta: d, vol: Math.abs(d) * 2 }))
const hm = (t) => `${String(Math.floor(((t - T0) / 60 + 409) / 60)).padStart(2, '0')}:${String(((t - T0) / 60 + 409) % 60).padStart(2, '0')}`

test('anchoredCvd / anchorReadout with an end: nothing after the end; without one, unchanged', () => {
  const r = rows([5, 1, 2, 3, 4, 10])
  const closes = [100, 101, 102, 103, 104, 110]
  assert.deepEqual(anchoredCvd(r, T0 + 60), [null, 0, 2, 5, 9, 19])
  assert.deepEqual(anchoredCvd(r, T0 + 60, T0 + 180), [null, 0, 2, 5, null, null])
  const full = anchorReadout(r, closes, T0 + 60)
  const cut = anchorReadout(r, closes, T0 + 60, T0 + 180)
  assert.equal(full.delta, 19)
  assert.equal(full.minutes, 4)
  assert.equal(cut.delta, 5)
  assert.equal(cut.dPrice, 2)
  assert.equal(cut.minutes, 2)
  assert.equal(cut.vol, 10)
  assert.equal(cut.cpp, 5)
})

test('edgeReadout with an end: bars after it are ignored, the box ends on the end bar', () => {
  // the 110 close (minute 3) would grow the box; with the end at minute 2 it is never seen
  const p = payload([100, 103, 99, 110, 100], [[1, 103, 5, 1], [3, 110, 50, 0]])
  const b = flattenBars(p)
  const full = edgeReadout(p, b, T0, id, 15)
  const cut = edgeReadout(p, b, T0, id, 15, T0 + 120)
  assert.equal(full.top.price, 110)
  assert.equal(full.end, null)
  assert.equal(cut.top.price, 103)
  assert.equal(cut.to, T0 + 120)
  assert.equal(cut.end, T0 + 120)
  assert.equal(cut.top.contracts, 5)
})

test('endFromPick: after the anchor, not after the last bar; a 5-min pick keeps the whole candle', () => {
  const a = T0
  assert.equal(endFromPick(T0 + 480, 60, a, T0 + 600), T0 + 480)
  assert.equal(endFromPick(T0, 60, a, T0 + 600), null) // at the anchor
  assert.equal(endFromPick(T0 - 60, 60, a, T0 + 600), null) // before it
  assert.equal(endFromPick(T0 + 660, 60, a, T0 + 600), null) // after the last bar
  assert.equal(endFromPick(T0 + 300, 300, a, T0 + 900), T0 + 540) // 5-min candle 300..599 -> its last minute
  assert.equal(endFromPick(T0 + 600, 300, a, T0 + 720), T0 + 720) // last candle still forming -> the last bar
  assert.equal(endFromPick(T0 + 60, 60, null, T0 + 600), null) // no anchor
})

test('endWindowText: "06:49 → 06:57 · 8 min"', () => {
  assert.equal(endWindowText(T0, T0 + 480, hm), '06:49 → 06:57 · 8 min')
  assert.equal(endWindowText(T0, null, hm), '')
})
