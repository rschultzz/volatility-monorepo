// CR-DC — Flow v3 helpers. Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  ALPHA_MIN,
  ALPHA_SPAN,
  anchoredCvd,
  anchorReadout,
  buildHeatColumns,
  chartMarks,
  flattenCells,
  floorIndex,
  lastSwingAnchor,
  makeShifter,
  mergeRows,
  percentileAlpha,
  profileTotals,
  rollingDelta,
  sessionOpenAnchor,
} from '../../src/flowV3.js'

const T = 1_000_000_020 * 0 + 1_777_617_000 // a shifted 06:30
const rows = (deltas, vols) => deltas.map((d, i) => ({ time: T + 60 * i, delta: d, vol: vols ? vols[i] : Math.abs(d ?? 0) * 2, cvd: i }))
const id = (t) => t

test('anchored CVD: 0 at the anchor, accumulates from the next bar, null before', () => {
  const r = rows([5, -3, 4, 10, -2])
  assert.deepEqual(anchoredCvd(r, T + 120), [null, null, 0, 10, 8])
  assert.deepEqual(anchoredCvd(r, T), [0, -3, 1, 11, 9])
})

test('anchored CVD: anchor between candle times snaps to the candle at or before it (5-min charts)', () => {
  const r = rows([5, -3, 4, 10, -2])
  assert.deepEqual(anchoredCvd(r, T + 150), [null, null, 0, 10, 8])
})

test('anchored CVD: no anchor = the v2 CVD; anchor before the data = all null', () => {
  const r = rows([5, -3])
  assert.deepEqual(anchoredCvd(r, null), [0, 1])
  assert.deepEqual(anchoredCvd(r, T - 600), [null, null])
})

test('anchored CVD: bars without data stay null and do not reset the sum', () => {
  const r = rows([5, null, 4, 6])
  assert.deepEqual(anchoredCvd(r, T), [0, null, 4, 10])
})

test('anchor readout: delta, price change, minutes, contracts per point', () => {
  const r = rows([5, -3, 4, 10, null], [10, 20, 30, 40, 0])
  const closes = [100, 99, 101, 103, 103]
  assert.deepEqual(anchorReadout(r, closes, T + 60), { anchorTime: T + 60, delta: 14, dPrice: 4, minutes: 2, vol: 70, cpp: 17.5, fromSeg: null, toSeg: null })
  assert.equal(anchorReadout(r, [100, 100, 100, 100], T)?.cpp, null) // Δ price 0 -> no cpp
  assert.equal(anchorReadout(r, closes, null), null)
  const segs = r.map((x, i) => ({ ...x, seg: i < 2 ? 'eth' : 'rth' }))
  assert.deepEqual([anchorReadout(segs, closes, T).fromSeg, anchorReadout(segs, closes, T).toSeg], ['eth', 'rth'])
})

test('rolling net delta over N minutes', () => {
  const r = rows([1, 2, 3, 4, 5, 6])
  assert.deepEqual(rollingDelta(r, 3), [1, 3, 6, 9, 12, 15])
  const r5 = [0, 1, 2].map((i) => ({ time: T + 300 * i, delta: 10 * (i + 1) })) // 5-min candles, N = 10 min
  assert.deepEqual(rollingDelta(r5, 10), [10, 30, 50])
  assert.deepEqual(rollingDelta(rows([1, null, 3]), 5), [1, null, 4])
})

test('shifter: one offset per hour, equal to the shift function', () => {
  let calls = 0
  const s = makeShifter((t) => {
    calls += 1
    return t - 7 * 3600
  })
  assert.equal(s(3600 * 10 + 5), 3600 * 3 + 5)
  assert.equal(s(3600 * 10 + 59), 3600 * 3 + 59)
  assert.equal(calls, 1)
})

const payload = {
  segments: [
    {
      name: 'rth', start: T, end: T + 600,
      cells: { t: [T, T, T + 60, T + 60, T + 300], p: [400, 401, 400, 402, 400], buy: [0, 4, 2, 0, 8], sell: [6, 0, 1, 3, 0] },
      marks: [{ kind: 'sellers_absorbed', time: T + 60, price: 100, break_time: T + 300 }, { kind: 'buyers_absorbed', time: T + 120, price: 101, break_time: null }],
    },
    { name: 'eth', start: T - 3600, end: T, cells: { t: [T - 60], p: [399], buy: [1], sell: [1] }, marks: null },
  ],
}

test('heat columns: one column per candle, per-segment percentile opacity, 0.5-pt merge', () => {
  const flat = flattenCells(payload, id)
  const times = [T, T + 60, T + 120]
  const cols = buildHeatColumns(flat, times, 60)
  assert.deepEqual([...cols[1][0].p], [400, 401])
  assert.deepEqual([...cols[1][1].p], [400, 402])
  assert.equal(cols[1][2], null) // T+300 is outside the last candle's minute
  // rth sell distribution: [1, 3, 6] -> 6 is the 100th percentile, 1 the 33rd
  assert.ok(Math.abs(cols[1][0].sa[0] - (ALPHA_MIN + ALPHA_SPAN)) < 1e-6)
  assert.ok(Math.abs(cols[1][1].sa[0] - (ALPHA_MIN + ALPHA_SPAN / 3)) < 1e-6)
  assert.equal(cols[1][0].sa[1], 0)
  // 0.5 pt: ticks 400 and 401 merge into 400
  assert.deepEqual([...cols[2][0].p], [400])
  assert.deepEqual([cols[2][0].buy[0], cols[2][0].sell[0]], [4, 6])
})

test('heat columns: 5-min candles sum their minutes', () => {
  const cols = buildHeatColumns(flattenCells(payload, id), [T, T + 300], 300)
  assert.deepEqual([...cols[1][0].p], [400, 401, 402])
  assert.deepEqual([...cols[1][0].buy], [2, 4, 0])
  assert.deepEqual([...cols[1][0].sell], [7, 0, 3])
  assert.deepEqual([...cols[1][1].buy], [8])
})

test('percentile alpha: zero for empty values', () => {
  assert.equal(percentileAlpha([1, 2], 0), 0)
  assert.equal(percentileAlpha([], 5), 0)
})

test('profile: totals from the anchor on, merged to 0.5 pt', () => {
  const flat = flattenCells(payload, id)
  assert.deepEqual(profileTotals(flat, T + 60), [{ p: 400, buy: 10, sell: 1 }, { p: 402, buy: 0, sell: 3 }])
  assert.deepEqual(profileTotals(flat, -Infinity).length, 4)
  assert.deepEqual(mergeRows(profileTotals(flat, T), 2), [{ p: 400, buy: 14, sell: 7 }, { p: 402, buy: 0, sell: 3 }])
})

test('chart marks: only segments with marks, snapped, line to break or segment end', () => {
  const m = chartMarks(payload, [T, T + 60, T + 120, T + 180, T + 240, T + 300], id)
  assert.equal(m.length, 2)
  assert.deepEqual([m[0].idx, m[0].endIdx, m[1].idx, m[1].endShifted], [1, 5, 2, T + 600])
})

test('anchors: last swing = end of the last confirmed leg; session open = earliest segment start', () => {
  const times = [T - 3600, T, T + 60, T + 120]
  assert.equal(lastSwingAnchor([{ end_time: T + 60 }, { end_time: T + 125 }], times, id), T + 120)
  assert.equal(lastSwingAnchor([], times, id), null)
  assert.equal(sessionOpenAnchor(payload.segments, times, id), T - 3600)
  assert.equal(floorIndex([1, 3, 5], 4), 1)
})
