// CR-DS item 4 — read history: light edgeReadout, levels as of a minute, the minute-by-minute replay (no later bars).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { absorptionLevels, boxVerdict, edgeReadout, flattenBars, readHistory, readHistoryText } from '../../src/flowV3.js'

const T0 = 1_777_642_200
const id = (t) => t
const tick = (price) => Math.round(price / 0.25)
const at = (m) => T0 + 60 * m
const mark = (kind, price, m, extra = {}) => ({ kind, price, time: at(m), confirm_time: at(m + 4), aggressor: 500, opposite: 50, merged_vol: 0, break_time: null, ...extra })
const payload = (closes, marks = [], cells = []) => ({
  segments: [{
    name: 'rth', source: 'ironbeam', exact: true, start: T0, end: T0 + 390 * 60,
    bars: { t: closes.map((_, i) => at(i)), h: closes.map((c) => c + 0.5), l: closes.map((c) => c - 0.5), c: closes },
    marks, cells: { t: cells.map((c) => at(c[0])), p: cells.map((c) => tick(c[1])), buy: cells.map((c) => c[2]), sell: cells.map((c) => c[3]) },
  }],
})
const rowsOf = (closes) => closes.map((_, i) => ({ time: at(i), delta: 10, vol: 1000, seg: 'rth' }))
// box 100–103 after minute 3, price back near the top, buyers absorbed 104.25 at minute 5 (at the top of the 104.5 high)
const CLOSES = [103, 102, 101, 100, 101, 102, 103, 104, 103, 104, 103, 104, 103.5, 104, 104, 103.75, 104, 102, 101, 101.5]
const hist = (closes, end = null) => {
  const p = payload(closes, [mark('buyers_absorbed', 104.25, 5)])
  const rows = rowsOf(closes)
  return readHistory({ payload: p, bars: flattenBars(p), rows, closes, anchorShifted: T0, endShifted: end, shift: id, holdMin: 15,
    breakHoldMin: 5, candleTimes: closes.map((_, i) => at(i)), segments: p.segments, baseline: null, tradeDate: '2026-10-30', ptMinOf: () => 0 })
}

test('light edgeReadout = full readout without edge counts / probe contracts', () => {
  const closes = [100, 101, 99, 100, 103.5, 100.5, 100, 108, 108.5, 109, 109, 109, 109, 110, 109.5]
  const p = payload(closes, [], [[1, 100, 5, 5], [4, 103.5, 9, 1], [8, 108.5, 3, 3]])
  const full = edgeReadout(p, flattenBars(p), T0, id, 5, null, 2)
  const light = edgeReadout(p, flattenBars(p), T0, id, 5, null, 2, { counts: false })
  const strip = (b) => ({ from: b.from, to: b.to, high: b.high, low: b.low, lastClose: b.lastClose, top: b.top.price, bottom: b.bottom.price, probes: (b.probes || []).map((x) => [x.at, x.dir, x.extreme]) })
  assert.deepEqual(light.frozen.map(strip), full.frozen.map(strip))
  assert.deepEqual(strip(light), strip(full))
  assert.deepEqual(light.pending, full.pending)
  assert.equal(light.minutes, null)
  assert.equal(light.top.contracts, null)
  assert.ok(full.minutes)
})

test('absorptionLevels as of a minute: unconfirmed marks left out, later breaks not yet broken', () => {
  const p = payload(CLOSES, [mark('buyers_absorbed', 104.25, 5, { break_time: at(15) }), mark('sellers_absorbed', 100, 9)])
  const times = CLOSES.map((_, i) => at(i))
  const at10 = absorptionLevels(p, times, id, 8, at(10))
  assert.deepEqual(at10.map((l) => [l.kind, l.broken]), [['buyers_absorbed', false]]) // sellers confirm at 13
  const at16 = absorptionLevels(p, times, id, 8, at(16))
  assert.deepEqual(at16.map((l) => [l.kind, l.broken]), [['buyers_absorbed', true], ['sellers_absorbed', false]])
})

test('replay: SHORT READ minutes grouped into one range per box; history up to a minute ignores later bars', () => {
  const h = hist(CLOSES)
  assert.equal(h.minutes, CLOSES.length)
  assert.ok(h.ranges.length >= 1)
  assert.equal(h.ranges[0].side, 'short')
  assert.equal(h.ranges[0].box, T0)
  assert.ok(h.ranges[0].from >= at(10)) // not before the box is 10 min old
  for (let k = 8; k < CLOSES.length; k += 1) {
    // same history whether the later bars exist (set end at k) or not (payload cut at k)
    assert.deepEqual(hist(CLOSES, at(k)).ranges, hist(CLOSES.slice(0, k + 1)).ranges, `minute ${k}`)
  }
  // the replay's last minute = the live verdict at that end
  const p = payload(CLOSES, [mark('buyers_absorbed', 104.25, 5)])
  const r = edgeReadout(p, flattenBars(p), T0, id, 15, at(14), 5)
  const ro = { delta: 140, vol: 14000, dPrice: 1, minutes: 14, fromSeg: 'rth', toSeg: 'rth' }
  const live = boxVerdict({ box: { ...r, readout: ro, ended: { kind: 'current' } }, pct: null, levels: absorptionLevels(p, CLOSES.map((_, i) => at(i)), id, 8, at(15)), tradeDate: '2026-10-30' })
  const last = hist(CLOSES, at(14)).ranges.at(-1)
  assert.equal(live.kind, 'short')
  assert.equal(last.to, at(14))
})

test('readHistoryText', () => {
  const hh = (t) => `m${(t - T0) / 60}`
  assert.equal(readHistoryText([], hh), 'Read live: none')
  assert.equal(readHistoryText([{ from: at(10), to: at(14), side: 'short' }, { from: at(20), to: at(20), side: 'long' }], hh), 'Read live: m10–m14 (short) · m20 (long)')
})
